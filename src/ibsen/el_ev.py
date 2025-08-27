import numpy as np
# from matplotlib import pyplot as plt
from numpy import pi, sin, cos, exp
import astropy.units as u
from astropy import constants as const
from scipy.integrate import trapezoid, cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
# import time
# from joblib import Parallel, delayed
from ibsen.utils import  beta_from_g, loggrid, t_avg_func, wrap_grid, trapz_loglog
import matplotlib.pyplot as plt

from ibsen.transport_solvers.transport_on_ibs_solvers import solve_for_n, nonstat_characteristic_solver
# import xarray as xr
# from pathlib import Path


from ibsen.ibs import IBS


G = float(const.G.cgs.value)
K_BOLTZ = float(const.k_B.cgs.value)
HBAR = float(const.hbar.cgs.value)
C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)
M_E = float(const.m_e.cgs.value)
E_ELECTRON = 4.803204e-10
MC2E = M_E * C_LIGHT**2
R_ELECTRON = E_ELECTRON**2 / MC2E

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)

ERG_TO_EV = 6.24E11 # 1 erg = 6.24E11 eV
DAY = 86400.
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

def syn_loss(ee, B):
    """
    Synchrotron losses, dE/dt.
    https://www.mpi-hd.mpg.de/personalhomes/frieger/HEA5.pdf
    
    Parameters
    ----------
    ee : np.ndarray
        Electron energy [eV].
    B : np.ndarray
        Magnetic field [cgs].

    Returns
    -------
    np.ndarray
        Synchrotron losses, dE/dt [eV/s]

    """
    # return -4e5 * B**2 * (ee/1e10)**2 # that was Denys expression ???
    return -2.5e5 * B**2 * (ee/1e10)**2 # that's what I get???

def gfunc(x, a = -0.362, b = 0.826, alpha = 0.682, beta = 1.281): 
    """
    Auxillary function from 1310.7971
    """
    return (1 +  (a * x**alpha) / (1 + b * x**beta) )**(-1)
    
def Gani(u_, c = 6.13):
    """
    G_ani from 1310.7971, eq. (35)
    """
    cani = c
    return cani * u_ * np.log(1 + 2.16 * u_ / cani) / (1 + cani * u_ / 0.822)

def Giso(u_, c = 4.62):
    """
    G_iso from 1310.7971, eq. (38)
    """
    ciso = c
    return ciso * u_ * np.log(1 + 0.722 * u_ / ciso) / (1 + ciso * u_ / 0.822)

def Giso_full(x):
    """
    Full G_iso from 1310.7971, which is G_iso times gfunc
    """
    return Giso(x, 5.68) * gfunc(x)

def ic_loss(Ee, Topt, Ropt, dist): # Ee in eV !!!!!!
    """
    Inverse Compton (isotropic) losses dE/de 

    Parameters
    ----------
    Ee : np.ndarray
        Electron energy [eV].
    Topt : np.ndarray
        Effective temperature of a star [K].
    Ropt : np.ndarray
        Star radius [cm].
    dist : np.ndarray
        Distance from the star.

    Returns
    -------
        np.ndarray.
        IC losses for a single electron dE/dt [eV/s].

    """
    kappa = (Ropt / 2 / dist)**2
    T_me = (K_BOLTZ * Topt / MC2E)
    Ee_me = Ee  / ERG_TO_EV / MC2E
    coef = 2 * R_ELECTRON**2 * M_E**3 * C_LIGHT**4 * kappa * T_me**2 / pi / HBAR**3 # 1/s
    Edot = coef * MC2E * ERG_TO_EV * Giso_full(4 * T_me * Ee_me) #[eV/s]
    return -Edot #[eV/s]

    
def t_adiab(dist, eta_flow):
    """
    Adiabatic (?) time defined as t_ad = eta_flow * dist / c.

    Parameters
    ----------
    dist : np.ndarray
        Characteristic distance, e.g. the binary separation [cm].
    eta_flow : np.ndarray
        Proportionality coefficient.

    Returns
    -------
    np.ndarray
        t_ad [s].

    """
    return eta_flow * dist / C_LIGHT

def ecpl(E, ind, ecut, norm):
    """
    Exponential cut-off power-law.
    norm * E^(-ind) * exp(-E/ecut).
    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    ecut : np.ndarray
        Exponential cut-off energy.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Exponential cut-off power-law.

    """
    return norm * E**(-ind) * exp(-E / ecut)

def pl(E, ind, norm):
    """
    Power law norm * E^(-ind).

    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Power-law.

    """
    return norm * E**(-ind)

def secpl(E, ind, ecut, beta_e, norm):
    """
    Super-exponential cut-off power-law.
    norm * E^(-ind) * exp(-(E/ecut)^beta_e).
    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    ecut : np.ndarray
        Exponential cut-off energy.
    beta_e : np.ndarray
        Super-exponential index.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Super-exponential cut-off power-law.

    """
    return norm * E**(-ind) * exp(- (E / ecut)**beta_e )



def total_loss(ee, B, Topt, Ropt, dist, eta_flow, eta_syn, eta_IC):
    """
    Total losses dE/dt = eta_syn * syn_loss + eta_IC * ic_loss - E/t_ad.
    Negative! For a single electron.

    Parameters
    ----------
    ee : np.ndarray
        Electron energy [eV].
    B : np.ndarray
        Magnetic field [cgs].
    Topt : np.ndarray
        Optical star effective temperature [K].
    Ropt : np.ndarray
        Optical star radius [cm].
    dist : np.ndarray
        Distance from the optical star [cm].
    eta_flow : np.ndarray
        Proportionality coefficient for the adiabatic time.
        If > 1e10, adiabatic losses are neglected.
    eta_syn : np.ndarray
        Proportionality coefficient for the synchrotron losses.
    eta_IC : np.ndarray
        Proportionality coefficient for the inverse Compton losses.

    Returns
    -------
    np.ndarray
        Total loss for a single electron dE/dt [eV/s].

    """
    eta_flow_max = np.max(np.asarray(eta_flow))
    if eta_flow_max < 1e10:
        return eta_syn * syn_loss(ee, B) + eta_IC * ic_loss(ee, Topt, Ropt, dist) - ee / t_adiab(dist, eta_flow)
    else:
        return eta_syn * syn_loss(ee, B) + eta_IC * ic_loss(ee, Topt, Ropt, dist)
    
def stat_distr(Es, Qs, Edots):
    """
    calculates the stationary spectrum of electrons: the solution of
    d(n * Edot)/dE = Q(E), which is
        
    n(E) = 1/|Edot| \int_E ^\infty Q(E') dE' 
                                                
    Parameters
    ----------
    Es : np.ndarray
        A grid of electron energies.
    Qs : np.ndarray
        The right-hand side Q calculated at a grid Es.
    Edots : np.ndarray
        The total losses Edot calculated at a grid Es.

    Returns
    -------
    np.ndarray
        The stationary electron spectrum.

    """
    
    integral_here = np.zeros(Es.size)+1
    energies_back = Es[::-1]
    # Q_func_array = Q_func(energies_back)
    integral_here = -cumulative_trapezoid(Qs[::-1], energies_back, initial = 0)[::-1]
    return 1 / np.abs(Edots) * integral_here


def stat_distr_with_leak(Es, Qs, Edots, Ts, mode = 'ivp'):
    """  
    Calculates the stationary spectrum with leakage: the solution of
    d(n * Edot)/dE + n/T(E) = Q(E), which is
        
    n(E) = 1/|Edot| \int_E ^\infty Q(E') K(E, E') dE', where
    K(E, E') = exp( \int_E^E' dE''/ (Edot(E'') * T(E'') )  )

    Parameters
    ----------
    Es : np.ndarray
        A grid of electron energies.
    Qs : np.ndarray
        The right-hand side Q calculated on a grid Es.
    Edots : np.ndarray
        The total losses Edot calculated on a grid Es.
    Ts : np.ndarray
        The leakage times T calculated on a grid Es.
    mode: Optional, str
        How to solve the equation. If mode == /'ivp/' (default), it is solved
        with scipy.integrate.solve_ivp. Honest, but sometimes doesnt work 
        and usually slow. If mode == /'analyt/', the general solution for the
        equation is used. Faster, but sometimes wrong hehehhhehehe

    Returns
    -------
    np.ndarray
        The stationary electron spectrum on a grid Es.

    """
    if mode == 'ivp':
        spl_q = interp1d(Es, Qs)
        spl_edot = interp1d(Es, Edots)
        spl_T = interp1d(Es, Ts)
        q_ = lambda en: spl_q(en)
        edot_ = lambda en: spl_edot(en)
        T_ = lambda en: spl_T(en)
        
        def ode_rhs(e_, u):
            edot_val = edot_(e_)
            T_val = T_(e_)
            q_val = q_(e_)
            return q_val - u / (edot_val * T_val)
        
        # Solve backwards: from e_max to e_min
        sol = solve_ivp(ode_rhs, [Es[-1], Es[0]], [0], t_eval=Es[::-1], method='RK45',
                        rtol=1e-3, atol=1e-40)
        
        u_numeric = sol.y[0][::-1]        
        n_numeric = u_numeric / Edots
        return n_numeric
    if mode == 'analyt':
        
        inv_Tf = 1 / (Ts * Edots)
        # First, compute ∫_∞^{e} (1 / (T * f)) de
        inner_int = cumulative_trapezoid(inv_Tf[::-1], Es[::-1], initial=0)[::-1]  
        inner_int_spl = interp1d(Es, inner_int, kind='linear')    
        epr = Es
        ee, eepr = np.meshgrid(Es, epr, indexing='ij')
        inner_int2d = inner_int_spl(eepr) - inner_int_spl(ee)
        integrand = Qs * np.exp(inner_int2d)
        outer_int2d = cumulative_trapezoid(integrand[::-1, ::-1], epr[::-1], initial=0, axis=1)
        outer_int = np.diag(outer_int2d)[::-1]
        n_analytic = outer_int / Edots
        return n_analytic
    
    
def evolved_e_advection(s_, edot_func, f_inject_func,
              tot_loss_args, f_args, vel_func, v_args, emin_grid = 1e9, 
              emax_grid = 5.1e14):  
    """
    Calculates the electron spectrum on a grid of positions s_ along the
    IBS, taking into account advection and cooling. The solution is obtained
    with the finite-difference method for the transport equation:
    v(s) * dn/ds + d(edot * n)/de = f_inject(s, e).

    Parameters
    ----------
    s_ : np.ndarray of shape (Ns,)
        The grid of positions along the IBS [cm]. Internally
        the equation is solved on a grid 0 < s < 1.01* max(s_) with Ns=601 points.
    edot_func : callable with signature edot_func(s, e, *tot_loss_args)
        The total losses dE/dt [eV/s] for a single electron.
    f_inject_func : callable with signature f_inject_func(s, e, *f_args)
        The injection function dNdot/dsde [1/s/cm/eV].
    tot_loss_args : tuple
        Optional arguments for edot_func.
    f_args : tuple
        Optional arguments for f_inject_func.
    vel_func : callable with signature vel_func(s, *v_args)
        Bulk motion velocity along the IBS [cm/s].
    v_args : tuple
        Optional arguments for vel_func.
    emin_grid : float, optional
        The min of energy grid to solve the equaion on.
         Internally the equation is solved from emin_grid/10 to emax_grid*2.
           The default is 1e9.
    emax_grid : float, optional
        The max of energy grid to solve the equaion on.
         Internally the equation is solved from emin_grid/10 to emax_grid*2.
           The default is 5.1e14.

    Returns
    -------
    dNe_deds_IBS : np.ndarray of shape (Ns, Ne)
        The electron spectra dNedot/deds on the grid s_ and on a grid of energies
        between emin_grid and emax_grid.
    e_vals : np.ndarray of shape (Ne,)
        The electron energy grid on which the solution is provided.

    """    
    # we calculate it on e-grid emin / extend_d < e <  emax * extend_u
    # and hope that zero boundary conditions will be not important
    
    # extend_u = 10; extend_d = 10; 
    extend_u = 2; extend_d = 10; 
    Ns, Ne_dec = 601, 123 
    # Ns, Ne = 201, 203 
 
    e_vals_sample = loggrid(emin_grid / extend_d, emax_grid * extend_u,
                            Ne_dec)
    
    ### now we'll look where the f_inject is not negligible and only there
    ### will we solve the equation 
    ### Currently it's set so that the eq is solved at all energies 
    ssmesh, emesh = np.meshgrid(s_, e_vals_sample, indexing='ij')
    f_sample = f_inject_func(ssmesh, emesh, *f_args)
    f_sample_sed = f_sample[1, :] * e_vals_sample**2
    e_where_good = np.where(f_sample_sed > np.max(f_sample_sed) / (-1e6) )
    s_vals = np.linspace(0, np.max(s_)*1.01, Ns)
    e_vals = e_vals_sample[e_where_good]
    dNe_deds = solve_for_n(v_func = vel_func, edot_func = edot_func,
                        f_func = f_inject_func,
                        v_args = v_args, 
                        edot_args = tot_loss_args,
                        f_args = f_args,
                        s_grid = s_vals, e_grid = e_vals, 
                        method = 'FDM_cons', bound = 'neun')
    # #### Only leave the part of the solution between emin_grid < e < emax_grid 
    ind_int = np.logical_and(e_vals <= emax_grid, e_vals >= emin_grid)
    e_vals = e_vals[ind_int]
    dNe_deds = dNe_deds[:, ind_int]
    #### and evaluate the values on the IBS grid previously obtained:dNe_de_IBS, e_vals
    interp_x = interp1d(s_vals, dNe_deds, axis=0, kind='linear', fill_value='extrapolate')
    dNe_deds_IBS = interp_x(s_)
    dNe_deds_IBS[dNe_deds_IBS <= 0] = np.min(dNe_deds_IBS[dNe_deds_IBS>0]) / 3.14
    
    return dNe_deds_IBS, e_vals

def evolved_e_nonstat_1zone(t_start, t_stop,  
                            q_func, edot_func, 
                          init='stat',
                          e_grid=None,
                          emin=1e9, emax=5.1e14, coef_down=10, ne_dec=101,
                          eps_big=3e-3, eps_small=1e-3, dt_min=1e-2*DAY,
                          dt_max=5*DAY, dt_first=None, adaptive_dt=False):
    """
    A solver for a non-stationary equation of electrons population cooling n(e, t):
    dn/dt + d(edot_func(e, t) * n)/de = q_func(e, t). It uses the solver for the equation
    with stationary edot and q, and then applies it multiple times.

    Please note: the energy grid e_grid that you provide OR the energy grid
    that is calculated from emin, emax, coef_down, ne_dec is used to calculate
    the EDGES of energy bins, while the solution is calulated on the
    CENTRES of the bins (these centre energies are returned as e_bins).

    Parameters
    ----------
    t_start : float
        Start time of evolution.
    t_stop : float
        End time of evolution.
    q_func : callable
        Injection function [1/s], of signature q_func(e, t).
    edot_func : callable
        Cooling function [eV/s], of signature edot_func(e, t).
    init :  optional
        Desctibes the initial distribution n(e, t=t_start). You can set it to:
        - 'zero' -- n(e, t=t_start) = 0
        - 'stat' -- n(e, t=t_start) = stationary distribution at t=t_start
        - tuple (e, n0) -- where e is a grid of energies and n0 is the initial spectrum
        The default is 'stat'.
    e_grid : 1d-array, optional
        The energy grid to calculate the evolution on. The default is None.
    emin : float, optional
        If e_grid is not set, this is used as the energy grid minimum 
        (but see `coef_down`). The default is 1e9.
    emax : float, optional
        If e_grid is not set, this is used as the energy grid maximum. The default is 5.1e14.
    coef_down : float, optional
        If e_grid is not set, this is used to calculate the REAL minimum
        energy: emin_real = emin/coef_down. The default is 10.
    ne_dec : int, optional
        If e_grid is not set, this is used as a number of energy nods
        per decade. The default is 101.
    eps_big : float, optional
        If adaptive_dt=True, this is treated as the critical relative error
        to refine a time step dt. The default is 3e-3.
    eps_small : float, optional
        If adaptive_dt=True, this is treated as the critical relative error
        to increase a time step dt. Should be < eps_big (not forced).
        The default is 1e-3.
    dt_min : float, optional
        If adaptive_dt=True, this is treated as the floor for a time step dt.
        Should be < dt_max (not forced).
          If adaptive_dt=False, this is the constant time step.
            The default is 1e-2*DAY.
    dt_max : float, optional
        If adaptive_dt=True, this is treated as the ceiling for a time step dt.
         Should be > dt_min (not forced). The default is 5*DAY.
    dt_first : float, optional
        The first time step. Caution: you should not make it too large, it should
        be << t_stop-t_start.
          The default is None (which sets actual dt_first=dt_min).
    adaptive_dt : bool, optional
        If False, the time evolution proceeds with a constant dt=dt_min.
         If True, the adaptive time step is chosen by comparing the solution
        of the equation with one step dt and two half-steps dt/2.
             The default is False.

    Raises
    ------
    ValueError
        if the initial spectrum (e, n0) is not reognized
        as having a required form.

    Returns
    -------
    ts : np. 1d-array 
        The time grid of evolution.
    e_bins : np. 1d-array 
        The energy grid of evolution (energy bins CENTRES).
    dNe_des : np. 2d-array 
        The 2d-array of electron spectra n(e, t) at each time step
        (energy bins CENTRES).
        dNe_des[i, j] = n(e_bins[j], ts[i]).
    edots_avg : np. 2d-array 
        average cooling function at each time step (on energy bins EDGES).
    q_avg : np. 2d-array 
        average injection function at each time step (on energy bins EDGES).
    Sorryyyyyyyy for the mess with edges and centres of energy bins.

    """
    if e_grid is None:
        e_grid = loggrid(emin/coef_down, emax, ne_dec)
        
    if isinstance(init, str):
        if init == 'zero':
            _n_prev =  0 * e_grid
        elif init == 'stat':
           _n_prev = stat_distr(Es = e_grid,  
                                Qs = q_func(e_grid, t_start),
                                Edots = edot_func(e_grid, t_start)
                                ) 
        e_prev = e_grid
    elif isinstance(init, tuple) or isinstance(init, list):
        e_prev, _n_prev = init
    else:
        raise ValueError('init should be only zero or stat or tuple (e, n0)')
        
    tt_now = t_start
    # as initial dt, take 1/3rd of max evolving time
    # dt = np.max(-e_grid/edot_func(e_grid, t_start)) / 3. 
    if dt_first is None:
        dt_first = dt_min
    dt = dt_first
    dNe_des = [  ]
    ts = [  ]
    edots_avg = []
    q_avg = []
    
    while tt_now < t_stop:
        # print(tt_now/DAY)
        #### first, let's do one step
        _edot_e_1 = t_avg_func(edot_func, tt_now, tt_now+dt, 3)
        _q_e_1 = t_avg_func(q_func, tt_now, tt_now+dt, 3)

        e_bins, n_1step = nonstat_characteristic_solver(t_evol=dt,
                        test_energies=e_grid,
                        edot_func=_edot_e_1,
                        Q_func = _q_e_1,
                        init_cond = (e_prev, _n_prev),
                        )
        if not adaptive_dt:
            tt_now += dt
            # e_grids.append(e_bins)
            dNe_des.append(n_1step)
            ts.append(tt_now)
            e_prev, _n_prev = e_bins, n_1step
            edots_avg.append(_edot_e_1(e_grid))
            q_avg.append(_q_e_1(e_grid))

        if adaptive_dt:
            ##### if adaprive t-step, do two half-steps and compare with 
            ##### the result from 1 whole step
            _edot_e_1of2 = t_avg_func(edot_func, tt_now, tt_now+dt/2, 3)
            _edot_e_2of2 = t_avg_func(edot_func, tt_now+dt/2, tt_now+dt,3)
            _q_e_1of2 = t_avg_func(q_func, tt_now, tt_now+dt/2, 3)
            _q_e_2of2 = t_avg_func(q_func, tt_now+dt/2, tt_now+dt, 3)
            
            e_bins1of2, n_1of2step = nonstat_characteristic_solver(t_evol=dt/2,
                            test_energies=e_grid,
                            edot_func=_edot_e_1of2,
                            Q_func = _q_e_1of2,
                            init_cond = (e_prev, _n_prev),)
            
            e_bins2of2, n_2of2step = nonstat_characteristic_solver(t_evol=dt/2,
                            test_energies=e_grid,
                            edot_func=_edot_e_2of2,
                            Q_func = _q_e_2of2,
                            init_cond = (e_bins1of2, n_1of2step),)
            
            # #### now compare e-SEDs (in 1 step) with n (in 2 half-steps )
            e_mask = (e_bins2of2 > emin) | (e_bins2of2 < emax)
            xarr = e_bins2of2[e_mask] # cut the edges just in case
            char_value = np.max(n_2of2step[e_mask] * xarr**2)
            # logsed2 = np.log10()
            _norm_diff = (
                (
                trapezoid( (n_2of2step[e_mask] * xarr**2 - n_1step[e_mask] * xarr**2)**2,
                                   xarr
                          ) 
                )**0.5 / (np.max(xarr) - np.min(xarr)) 
                )
            _norm_n2step = ( (trapezoid( (n_2of2step[e_mask] * xarr**2)**2, xarr) 
                              )**0.5 / (np.max(xarr) - np.min(xarr)) 
                            )
    
            eps_here = _norm_diff / (_norm_n2step + 1e-20 * char_value)
            # print(eps_here)
            
            if (eps_here >= eps_big) and dt > dt_min:
                dt = max(dt/2, dt_min)
                continue
            else:
                tt_now += dt
                # e_grids.append(e_bins)
                dNe_des.append(n_1step)
                ts.append(tt_now)
                e_prev, _n_prev = e_bins2of2, n_2of2step
                edots_avg.append(_edot_e_1(e_grid))
                q_avg.append(_q_e_1(e_grid))
                if eps_here <= eps_small:
                    dt = min(dt * 1.3, dt_max)
                    
                
    ts, dNe_des, edots_avg, q_avg = [np.array(ar_) for ar_ in (ts, dNe_des,
                                                            edots_avg, q_avg)]
    
    return ts, e_bins, dNe_des, edots_avg, q_avg
    
def evolved_e(cooling, r_SP, ss, rs, thetas, edot_func, f_inject_func,
              tot_loss_args, f_args, vel_func = None, v_args = None, emin = 1e9, 
              emax = 5.1e14, t_func = None, t_args = None, eta_flow_func = None, 
              eta_flow_args = None):
    """
    The function previously used to calculate the electron spectrum on the IBS
    taking into account cooling, providing a single interface (and by the interface 
    I mean this function) fot various cooling modes which are possible. But the initial
    idea was that it requires only the necessary parameters, because you don't have
    to know much about the IBS and winds actually.

    Parameters
    ----------
    cooling : str
        A string describing the cooling mode. It can be:
        - 'no' -- no cooling, the electron spectrum is just the injected one
        - 'stat_apex' -- stationary cooling, but the injection function is averaged
          over the IBS (i.e. f_inject(s, e) --> <f_inject(e)>_s )
        - 'stat_ibs' -- stationary cooling, the injection function is not averaged
          over the IBS (i.e. f_inject(s, e) is used as is)
        - 'stat_mimic' -- stationary cooling, the injection function is not averaged
          over the IBS (i.e. f_inject(s, e) is used as is), but the adiabatic losses are
          mimicked by increasing the radiative losses
        - 'leak_apex' -- stationary cooling with leakage, the same meaning as 
            with stat_apex
        - 'leak_ibs' -- stationary cooling with leakage, the same meaning as 
            with stat_ibs
        - 'leak_mimic' -- stationary cooling with leakage, the same meaning as 
            with stat_mimic but the leakage time is mimicked
        - 'adv' -- stationary cooling with advection, the injection function is not
          averaged over the IBS (i.e. f_inject(s, e) is used as is); the solution
          of  the advection equation is obtained with the finite-difference method. 
    r_SP : float
        Binary separation [cm].
    ss : np.array of shape (Ns,)
        An array of positions along the IBS in units of r_SP.
    rs : np.array of shape (Ns,)
        An array of distances from the pulsar to the IBS in units of r_SP.
    thetas : np.array of shape (Ns,)
        An array of angles between the line of centers and the direction
        from the pulsar to the IBS in radians.
    edot_func : callable of signature edot_func(s, e, *tot_loss_args)
        A total energy loss function dE/dt [eV/s] for a single electron.
    f_inject_func : callable of signature f_inject_func(s, e, *f_args)
        An injection function dNdot/dsde [1/s/cm/eV].
    tot_loss_args : tuple
        Optional arguments for edot_func.
    f_args : tuple
        Optional arguments for f_inject_func.
    vel_func : callable, of signature vel_func(s, *v_args),  optional
        A bulk motion velocity along the IBS [cm/s]. If cooling != 'adv',
          vel_func is not used. The default is None.
    v_args : tuple, optional
        Optional argumetns for vel_func. The default is None.
    emin : float, optional
        Energy grid minimum. The default is 1e9.
    emax : float, optional
        Energy grid maximum. The default is 5.1e14.
    t_func : callable of signature t_func(s, e, *t_args). 
        A leakage time function T [s] for a single electron.
        The default is None.
    t_args : tuple, optional
        Optional arguments for t_func. The default is None.
    eta_flow_func : callable, of signature eta_flow_func(s, *eta_flow_args),
      optional
        Eta_ad for stat_mimic. The default is None.
    eta_flow_args : tuple, optional
        Optional arguments for eta_flow_func. The default is None.

    Returns
    -------
    dNe_de_IBS : np.ndarray of shape (Ns, Ne)
        The electron spectra dNedot/deds on the grid s_ and on a grid of energies
        between emin and emax.
    e_vals : np.ndarray of shape (Ne,)
        The electron energy grid on which the solution is provided.

    """
    if cooling not in ('no', 'stat_apex', 'stat_ibs', 'stat_mimic', 'leak_apex', 'leak_ibs',
                       'leak_mimic', 'adv'):
        print('cooling should be one of these options:')
        print('no', 'stat_apex', 'stat_ibs','stat_mimic', 'leak_apex', 'leak_ibs',
                           'leak_mimic', 'adv')
        print('setting cooling = \'no\' ')
        cooling = 'no'
    # e_vals = loggrid(emin, emax, n_dec)
    if cooling == 'no':
        # For each s, --> injected distribution
        e_vals = np.logspace(np.log10(emin), np.log10(emax), 977)
        smesh, emesh = np.meshgrid(ss*r_SP, e_vals, indexing = 'ij')
        dNe_de_IBS = f_inject_func(smesh, emesh, *f_args)
        
    if cooling in ('stat_apex', 'stat_ibs', 'stat_mimic'):
        # For each s, --> stationary distribution
        e_vals = np.logspace(np.log10(emin), np.log10(emax), 979)
        smesh, emesh = np.meshgrid(ss*r_SP, e_vals, indexing = 'ij')
        if cooling == 'stat_mimic':
            (B0, Topt, Ropt, r_SE, eta_fl, eta_sy, eta_ic, ss, rs, thetas, r_SP) = tot_loss_args
            eta_fl_new = eta_flow_func(smesh, *eta_flow_args)
            tot_loss_args = (B0, Topt, Ropt, r_SE, eta_fl_new * eta_fl,
                                 eta_sy, eta_ic, ss, rs, thetas, r_SP)    
        f_inj_se = f_inject_func(smesh, emesh, *f_args)
        edots_se = edot_func(smesh, emesh, *tot_loss_args)
        dNe_de_IBS = np.zeros((ss.size, e_vals.size))
                
        for i_s in range(ss.size):
            if cooling == 'stat_apex':
                f_inj_av = trapezoid(f_inj_se, ss, axis=0) / np.max(ss)
                dNe_de_IBS[i_s, :] = stat_distr(e_vals, f_inj_av, edots_se[0, :])
            if cooling in ('stat_ibs', 'stat_mimic'):
                dNe_de_IBS[i_s, :] = stat_distr(e_vals, f_inj_se[i_s, :], edots_se[i_s, :])

    if cooling in ('leak_apex', 'leak_ibs', 'leak_mimic'):
        e_vals = np.logspace(np.log10(emin), np.log10(emax), 987)
        smesh, emesh = np.meshgrid(ss*r_SP, e_vals, indexing = 'ij')
        f_inj_se = f_inject_func(smesh, emesh, *f_args)
        edots_se = edot_func(smesh, emesh, *tot_loss_args)
        dNe_de_IBS = np.zeros((ss.size, e_vals.size))
        ts_leak = t_func(smesh, emesh, *t_args)

        for i_s in range(ss.size):
            if cooling == 'leak_ibs':
                # dNe_de_IBS[i_s, :] = Stat_distr_with_leak_ivp(e  = e_vals,
                    # q_func, edot_func, T_func, q_args, edot_args, T_args)
                dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], Ts = ts_leak[i_s, :],
                    mode = 'analyt')
            if cooling == 'leak_apex':
                f_inj_av = trapezoid(f_inj_se, ss, axis=0) / np.max(ss)
                dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_av, Edots = edots_se[0, :], Ts = ts_leak[0, :])
            if cooling == 'leak_mimic':
                dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], 
                    Ts = ts_leak[i_s, :], mode = 'analyt') 
        
    if cooling == 'adv':
        dNe_de_IBS, e_vals = evolved_e_advection(s_ = ss*r_SP, edot_func=edot_func,
                    f_inject_func=f_inject_func, tot_loss_args=tot_loss_args,
                      f_args=f_args, vel_func=vel_func, v_args=v_args,
                      emin_grid=emin, emax_grid=emax)
            
    return dNe_de_IBS, e_vals
    
class ElectronsOnIBS: #!!!
    """
    Electron injection and cooling on the intrabinary shock (IBS).

    This class builds and evolves the electron distribution on a single IBS
    snapshot (given by an :class:`IBS` instance) using several cooling models.
    It provides injection profiles in energy and along the shock, radiative and
    adiabatic loss rates, optional leakage/advection mimics, and solvers for
    stationary/leaky/advection cases. Energies are in eV, lengths in cm, and
    times in seconds. The supplied ``ibs`` must already be initialized with a
    valid ``winds`` (and thus an ``orbit``, so it should be ibs:IBS, not
    ibs:IBS_norm). 

    Parameters
    ----------
    Bp_apex : float
        Pulsar magnetic field at the apex [G].
    ibs : IBS
        IBS geometry at a chosen epoch (must have ``ibs.winds`` with an
        attached ``orbit``). Used for arc-length grid, angles, gamma,
        and distances needed by losses. 
    cooling : {'no', 'stat_apex', 'stat_ibs', 'stat_mimic',
               'leak_apex', 'leak_ibs', 'leak_mimic', 'adv'} or None, optional
        Cooling/evolution mode. If not in the set above, it falls back to
        ``'no'`` at runtime. See *Notes* for meanings. Default is None.
    to_inject_e : {'pl', 'ecpl', 'secpl'}, optional
        Energy part of the injection law: power law, exponential cutoff PL,
        or super-exponential cutoff PL. Default 'ecpl'.
    to_inject_theta : {'2d', '3d'}, optional
        Angular weighting along the curve: uniform ('2d') or ∝ sinθ ('3d').
        Default '3d'. (You can also restrict to |θ| < ``where_cut_theta``.)
    ecut : float, optional
        Cutoff energy for 'ecpl'/'secpl' [eV]. Default 1e12.
    p_e : float, optional
        Injection spectral index. Default 2.0.
    norm_e : float, optional
        Injection normalization (particles/s). The angular–energy–integrated
        rate in the **forward hemisphere of one horn** equals ``norm_e/4``.
        Default 1e37.
    beta_e : float, optional
        Super-exponential index for 'secpl'. Default 1.
    Bs_apex : float, optional
        Stellar magnetic field at the apex [G]. Default 0.
    eta_a : float or None, optional
        Adiabatic-time scale factor. If None, set internally to 1e20 which
        effectively **disables** the adiabatic term in losses. Default None.
    eta_syn, eta_ic : float, optional
        Multipliers for synchrotron and inverse-Compton losses. Defaults 1.0.
    emin, emax : float, optional
        Injection energy bounds used for optional cutting [eV]. Defaults
        1e9 and 5.1e14.
    emin_grid, emax_grid : float, optional
        Energy range for the solver grids [eV]. Defaults 1e8 and 5.1e14.
    to_cut_e : bool, optional
        If True, set injection to zero outside [``emin``, ``emax``]. Default True.
    to_cut_theta : bool, optional
        If True, inject only for |θ| < ``where_cut_theta``. Default False.
    where_cut_theta : float, optional
        Angular cut (radians) if ``to_cut_theta`` is True. Default π/2.

    Attributes
    ----------
    ibs : IBS
        The supplied IBS snapshot; used for s-grid, θ(s), r(s), and γ(s).
    orbit : Orbit
        Attached via ``ibs.winds.orbit`` (validated at init).
    r_sp : float
        Star–pulsar separation at the IBS epoch [cm].
    Bp_apex, Bs_apex, B_apex : float
        Magnetic fields at the apex [G]; ``B_apex = Bp_apex + Bs_apex``.
    eta_a, eta_syn, eta_ic : float
        Coefficients controlling adiabatic, synchrotron, and IC losses.
        Note that is cooling==`stat_mimic`, ``eta_a`` is scaled along the IBS.
    u_g_apex : float
        Stellar photon energy density at the apex [erg/cm^3].
    _b, _u : ndarray, shape (Ns,)
        Dimensionless B(s)/B_apex and u(s)/u_apex along the IBS points.
    _b_mid, _u_mid : ndarray, shape (Ns-1,)
        Same as above at segment midpoints.
    dNe_deds_IBS : ndarray or None, shape (Ns, Ne)
        Electron distribution per (s, E) after :meth:`calculate` [1/s/cm/eV].
    e_vals : ndarray or None, shape (Ne,)
        Energy grid corresponding to ``dNe_deds_IBS`` [eV].
    e2dNe_deds_IBS : ndarray or None, shape (Ns, Ne)
        Convenience SED array ``E^2 dN/(dE ds)``.
    dNe_deds_mid : ndarray or None, shape (Ns-1, Ne)
        Distribution at segment midpoints.
    dNe_ds_mid : ndarray or None, shape (Ns-1,)
        Number per unit s at midpoints, integrated over energy [1/s/cm].
    dNe_de_mid : ndarray or None, shape (Ns-1, Ne)
        Per-segment spectra, i.e. ``(dN/dEds) * ds`` [1/s/eV].

    Methods
    -------
    calculate(to_return=False)
        Build the electron distribution on the IBS according to ``cooling``;
        mirrors one horn to the full two-horn array. Optionally returns
        both the distribution and its energy grid.
    f_inject(s_, e_, spat_coord='s')
        Injection law in (s, E). Supports 's' or 'theta' as the spatial density.
    edot(s_, e_)
        Total electron energy loss rate dE/dt at (s, E) including synchrotron,
        IC, and optional adiabatic term.
    vel(s)
        Bulk flow speed along the IBS from the γ(s) profile.
    b_and_u_s(s_)
        Dimensionless B(s)/B_apex and u(s)/u_apex at given arc-length(s).
    u_g_density(r_from_s)
        Stellar photon energy density at a distance from the star.
    t_leakage(s, e)
        Leakage time T(s, E) used to mimic advection solutions.
    eta_flow_mimic(s)
        Position-dependent adiabatic factor used in 'stat_mimic' and tests.
    analyt_adv_Ntot_self : property
        Total number of electrons along the **upper** horn (integrated over E).
    analyt_adv_Ntot(s_1d, f_inj_integrated)
        Integral form for N_tot(s) given ∫ f_inject(s,E) dE.
    peek(ax=None, to_label=True, show_many=True, **kwargs)
        Quick-look plots of E-SEDs, s·N(s), and apex cooling time.

    Notes
    -----
    **Cooling modes**
      - ``'no'`` : no cooling; distribution equals the injected one.
      - ``'stat_apex'`` : stationary; uses an IBS-averaged injection and apex losses.
      - ``'stat_ibs'`` : stationary; uses local injection and local losses.
      - ``'stat_mimic'`` : stationary; like 'stat_ibs' but scales adiabatic term
        via :meth:`eta_flow_mimic` to emulate advection. 
      - ``'leak_apex'`` / ``'leak_ibs'`` / ``'leak_mimic'`` : stationary with
        leakage term T(s,E) from :meth:`t_leakage`; 'apex' uses averaged injection,
        others use local. 'mimic' uses the same leakage form as 'ibs' but is
        intended to approximate advective escape.
      - ``'adv'`` : finite-difference solution of the advection–cooling transport
        equation along s (uses ``evolved_e_advection``).

    Adiabatic loss handling
        If ``eta_a`` is very large (≳1e10) the adiabatic loss term is effectively
        disabled in :func:`total_loss`. Setting ``eta_a=None`` at init enforces
        this behavior. 

    """
    def __init__(self, Bp_apex, ibs: IBS, cooling=None, to_inject_e = 'ecpl',
                 to_inject_theta = '3d', 
                 ecut = 1.e12, p_e = 2., norm_e = 1.e37, beta_e=1,
                 Bs_apex=0., eta_a = None,
                 eta_syn = 1., eta_ic = 1.,
                 emin = 1e9, emax = 5.1e14,
                 emin_grid=1e8, emax_grid=5.1e14,
                 to_cut_e = True, 
                 to_cut_theta =  False, 
                 where_cut_theta = pi/2):
        """
        We should provide the already initialized class ibs:IBS here with 
        winds:Winds and orbit:Orbit in it. But the logic is a little off here:
            Probably we should fix 
        it sometime to set magn and photon fileds and r_sp in a more general 
        way (maybe define a class like: `B & u properties`, idk).
        
        But since currently ibs initialized WITH winds, we actually know the 
        time since periastron we are at: it's self.ibs.t_forbeta
        """
        
        self.ibs = ibs # must contain ibs.winds 
        self.cooling = cooling
        self.allowed_coolings = ('no', 
                                 'stat_apex', 'stat_ibs', 'stat_mimic',
                                  'leak_apex', 'leak_ibs', 'leak_mimic',
                                 'adv')
        
        self.Bp_apex = Bp_apex # pulsar magn field in apex
        self.Bs_apex = Bs_apex # opt star magn field in apex
        self.B_apex = Bp_apex + Bs_apex # total magn field in apex
        # self.u_g_apex = u_g_apex # photon energy density in apex
        
        if eta_a  is None:
            self.eta_a = 1e20
        else:
            self.eta_a = eta_a # to scale adiabatic time

        self.eta_syn = eta_syn # to scale synchrotron losses
        self.eta_ic = eta_ic # to scale inverse compton losses
        
        self.to_inject_e = to_inject_e # injection function along energies
        self.to_inject_theta = to_inject_theta # injection function along theta
        self.p_e = p_e  # injection function spectral index
        self.ecut = ecut  # injection function cutoff energy
        self.beta_e = beta_e # index for super-exponential cutoff PL
        self.norm_e = norm_e # injection function normalization
        self.emin = emin  # minimum energy for the injection function
        self.emax = emax  # maximum energy for the injection function
        self.emin_grid = emin_grid  # minimum energy for e-energy grid
        self.emax_grid = emax_grid  # maximum energy for e-energy grid
        
        
        self.to_cut_e = to_cut_e # whether to leave only the part emin < e < emax
        self.to_cut_theta = to_cut_theta # whether to inject only at theta < where_cut_theta
        self.where_cut_theta = where_cut_theta # see above
        
        
        self._check_and_set_ibs() #checks if there's right ibs and sets r_sp
        self._set_b_and_u_ibs() # calculates dimensionless b_ and u_ from values in apex on entire ibs
        
        
        
        
    def _check_and_set_ibs(self):
        """
        Checks if ibs contains ibs.winds and ibs.winds.orbit
        """
        try:
            self.orbit = self.ibs.winds.orbit
            self.r_sp = self.ibs.winds.orbit.r(self.ibs.t_forbeta)
        except:
            raise ValueError(""""Your ibs:IBS should contain ibs.winds""")

    def b_and_u_s(self, s_):
        """
        Calculates dimensionless magnetic field and photon energy density
        on the IBS at the position s_ (in cm) from the apex, in units
        of the values in the apex. Explicitly uses that both magnetic
        fields from the star and from the pulsar scale as 1/r, while the
        photon energy density scales as 1/r^2.

        Parameters
        ----------
        s_ : np.ndarray
            The position(s) along the IBS from the apex
            in cm.

        Returns
        -------
        b_on_shock : np.ndarray
            The magnetic field on the IBS at the position s_,
            in units of the magnetic field in the apex.
        u_g_on_shock : np.ndarray
            The photon energy density on the IBS at the position s_,
            in units of the photon energy density in the apex.

        """
        # f_ =  self.ibs.f_
        r_to_p = self.ibs.s_interp(s_ = s_, what = 'r')/self.r_sp # dimless
        r_to_s = self.ibs.s_interp(s_ = s_, what = 'r1')/self.r_sp # dimless
        r_sa = (1. - self.ibs.ibs_n.x_apex)
        B_on_shock = (self.Bp_apex * self.ibs.ibs_n.x_apex / r_to_p + 
                      self.Bs_apex * r_sa / r_to_s)
        u_g_on_shock = r_sa**2 / r_to_s**2
        return B_on_shock / (self.B_apex), u_g_on_shock    
    
    def u_g_density(self, r_from_s):      
        """
        The energy density of the optical star at the distance r_from_s.

        Parameters
        ----------
        r_from_s : np.ndarray
            Distance from the point on the IBS to the star [cm].

        Returns
        -------
        u_dens : np.ndarray
            The energy density of the optical star at the distance r_from_s.

        """
        factor = 2. * (1. - (1. - (self.ibs.winds.Ropt / r_from_s)**2)**0.5 )
        u_dens = SIGMA_BOLTZ * self.ibs.winds.Topt**4 / C_LIGHT * factor
        return u_dens
    
    def _set_b_and_u_ibs(self):
        """
        Sets the dimensionless magnetic field and photon energy density
        on the IBS at all points (on the grid self.ibs.s) in the attributes
        self._b and self._u, respectively. Also sets the photon energy density
        in the apex in the attribute self.u_g_apex. Also sets the dimensionless
        magnetic field and photon energy density on the IBS at the midpoints
        of the grid self.ibs.s in the attributes self._b_mid and self._u_mid,
        respectively.
        """
        b_onibs, u_onibs = ElectronsOnIBS.b_and_u_s(self, s_ = self.ibs.s)
        b_mid, u_mid = ElectronsOnIBS.b_and_u_s(self, s_ = self.ibs.s_mid)
        
        self.u_g_apex = ElectronsOnIBS.u_g_density(self,
                        r_from_s = self.r_sp - self.ibs.x_apex)
        self._b = b_onibs
        self._u = u_onibs
        self._b_mid = b_mid
        self._u_mid = u_mid
        

    
    def vel(self, s): # s --- in real units
        """
        Velocity of a bulk motion along the shock.    
        Calculated from the bulk motion Lorentz factor.

        Parameters
        ----------
        s : np.ndarray
            The arclength from the apex to the point of interest [cm].
        Returns
        -------
        v : np.ndarray
        The velocity [cm/s].
    
        """
        _gammas = self.ibs.gma(s)
        return C_LIGHT * (beta_from_g(_gammas) + 1e-5)


    def edot(self, s_, e_): 
        """
        The total energy loss function dE/dt [eV/s] for a single electron
        at the position s_ (in cm) from the apex and with energy e_ (in eV).

        Parameters
        ----------
        s_ : np.ndarray with the shape matching e_.
            The position(s) along the IBS from the apex [cm].
        e_ : np.ndarray with the shape matching s_.
            The energy(ies) of electrons [eV].

        Returns
        -------
        np.ndarray with the shape matching s_ and e_.
            The total energy loss function dE/dt [eV/s] for a single electron
            at the position s_ from the apex and with energy e_.

        """
        r_sa_cm = (self.r_sp - self.ibs.x_apex)
        _b_s, _u_s = ElectronsOnIBS.b_and_u_s(self, s_ = s_)
        return total_loss(ee = e_, 
                          B = _b_s * self.B_apex, 
                          Topt = self.ibs.winds.Topt * _u_s**0.25,
                          Ropt=self.ibs.winds.Ropt,
                          dist = r_sa_cm,
                          eta_flow = self.eta_a, 
                          eta_syn = self.eta_syn,
                          eta_IC = self.eta_ic * _u_s)

    def f_inject(self, s_, e_, spat_coord='s'): 
        """
        An injection function at the position s_ (in cm)
        from the apex and with energy e_ (in eV).
        Essentially it is (d N_injected / dt ) / de / d spatial_coord
        where spatial_coord is eitehr theta or s.
        The function is normalized to self.norm_e: the total number
        of injected particles per second in the forward hemisphere
        in one horn of the IBS is self.norm_e / 4.

        Parameters
        ----------
        s_ : np.ndarray strictly of the shape (Ns, Ne)
            An arclength to the points on IBS from the apex [cm].
        e_ : np.ndarray strictly of the shape (Ns, Ne)
            Energies of electrons [eV].
        spat_coord : string, optional
            The spatial coordinate in which the injection function
            is expressed. It can be either 's' or 'theta'.
            The default is 's'.

        Raises
        ------
        ValueError
            If to_inject_theta or to_inject_e or spat_coord
            are not recognized or if the shapes of s_ and e_ do not match.

        Returns
        -------
        np.ndarray of the shape (Ns, Ne)
            The injection function at the position s_ from the apex
            and with energy e_.

        """
        if s_.shape != e_.shape:
            raise ValueError('in f_inject, `s`-shape should be == `e`-shape')
        thetas_here = self.ibs.s_interp(s_  = s_, what = 'theta')
        
        # print(thetas_here)
        if self.to_inject_theta == '2d':
            integral_over_theta_quarter = pi/2.
            thetas_part = np.ones(thetas_here.shape) # uniform along theta
        elif self.to_inject_theta == '3d':
            integral_over_theta_quarter = 1.
            thetas_part = np.abs(sin(thetas_here)) # \propto sin(th) how it should be in 3d
        else:
            raise ValueError("I don't know this to_inject_theta. It should be 2d, 3d.")
            
        if self.to_cut_theta:
            thetas_part[np.abs(thetas_here) >= self.where_cut_theta] = 0.
           
        if self.to_inject_e == 'ecpl':
            e_part = ecpl(e_, ind=self.p_e, ecut=self.ecut, norm=1.)
        elif self.to_inject_e == 'pl':
            e_part = pl(e_, ind=self.p_e, norm=1)
        elif self.to_inject_e == 'secpl':
            e_part = secpl(e_, ind=self.p_e, ecut=self.ecut, norm=1.,
                          beta_e = self.beta_e)
        else:
            raise ValueError("I don't know this to_inject_e. It should be pl or ecpl or secpl.")
                    
        if self.to_cut_e:
            mask = (e_ < self.emin) | (e_ > self.emax)
            e_part = np.where(mask, 0.0, e_part)
        
        integral_over_e = trapz_loglog(e_part[0, :], e_[0, :])  
        result = thetas_part * e_part 
    
        # now normalize the number of injected particles. I do it like this:
        # I ensure that the total number per second: N(s) = \int n(s, E) dE 
        # of electrons in  ONE horm in the forward hemisphere = 1/4 from norm:
        # \int_0^{pi/2} N(s(theta)) d theta = norm / 4
              
        # print(integral_over_e)
        overall_coef = self.norm_e / 4. / integral_over_e / integral_over_theta_quarter 
        result = result * overall_coef
        ### what we have here is dNdot / de d theta. If we want 
        ### dNdot / de / ds, we do additional:
        if spat_coord == 's':
            ds_dthetas_here =  self.ibs.s_interp(s_  = s_, what = 'ds_dtheta')
            return result / ds_dthetas_here
        elif spat_coord == 'theta':
            return result
        else:
            raise ValueError('no such spatial coordinate option in f_inject')


    # def analyt_adv_Ntot_tomimic(self, s, e): 
    #     # s and e -- 2d meshgrid arrays, but the function returns 1d-array 
    #     _ga = self.ibs.gamma_max - 1
    #     # f_ = f_inj_func(s, e, *f_inj_args)
    #     f_ = ElectronsOnIBS.f_inject(self, s_=s, e_=e)
    #     f_integrated = trapezoid(f_, e, axis=1)
    #     x__ = _ga*s[:, 0] / self.ibs.s_max_g / self.r_sp
    #     return f_integrated * self.r_sp / C_LIGHT * self.s_max_g/_ga * (2 * x__ + x__**2)**0.5
    @property
    def analyt_adv_Ntot_self(self):
        """
        Analytically calculated the total number of electrons
        on the IBS horn, i.e. integrated over energies. If 
        f1(s) = \int f_inject(s, e) de, then
        Ntot(s) = \int_0^s f1(s') / v(s') ds', where v(s) is the bulk
        motion velocity along the IBS.

        Returns
        -------
        np.ndarray of shape (Ns,)
            The total number of electrons in a point
              on the IBS.

        """
        s_1d_dim = np.linspace(0, np.max(self.ibs.s[self._up] ) * 1.02, 241)
        e_vals = loggrid(self.emin_grid, self.emax_grid, 201)
        ss_, ee_ = np.meshgrid(s_1d_dim, e_vals, indexing='ij')
        f_se = ElectronsOnIBS.f_inject(self, ss_, ee_)
        f_s = trapz_loglog(f_se, e_vals, axis=1)
        v_ = C_LIGHT * (beta_from_g(self.ibs.gma(s_1d_dim)) + 1e-5)
        res_ = cumulative_trapezoid(f_s/v_, s_1d_dim, initial=0)
        spl_ = interp1d(s_1d_dim, res_)
        return spl_(self.ibs.s[self._up])
        
    
    def analyt_adv_Ntot(self, s_1d, f_inj_integrated):
        """
        The total number of electrons on the IBS at
        the arclength s_1d (in cm) from the apex,
        given the injection function integrated over energies
        f_inj_integrated (in 1/s/cm), i.e.
        f_inj_integrated(s) = \int f_inject(s, e) de.
        If f1(s) = \int f_inject(s, e) de, then
        Ntot(s) = \int_0^s f1(s') / v(s') ds', where v(s) is the bulk
        motion velocity along the IBS.

        Parameters
        ----------
        s_1d : float | np.ndarray (i think?..)
            Arclength from the apex to the point of interest [cm].
        f_inj_integrated : np.ndarray of shape (Ns,)
            The e-integrated injection function at the points s_1d [1/s/cm].

        Returns
        -------
        res : np.ndarray of shape (Ns,)
            N_tot(s_1d).

        """
        ### s [cm] and f_inj_integrated --- 1-dimensional
        gammas = self.ibs.gma(s = s_1d)
        vs_ = beta_from_g(gammas) * C_LIGHT
        res = cumulative_trapezoid(f_inj_integrated / vs_, s_1d, initial=0)
        return res
    
    def t_leakage(self, s, e):
        """
        The leakage time function T [s] for a single electron
        as to simulate the advection equation. Defined as
        Ntot(s) / v(s) / (dNtot/ds), where Ntot(s) is the total number
        of electrons on the IBS horn at the arclength s from the apex,
        and v(s) is the bulk motion velocity along the IBS.

        Parameters
        ----------
        s : np.ndarray with the shape matching e
            Arclength from the apex to the point of interest [cm].
        e : np.ndarray with the shape matching s
            Electron energies [eV].

        Returns
        -------
        np.ndarray with the shape matching s and e
            T_leak_mimic(s, e) [s].

        """
        f_ = ElectronsOnIBS.f_inject(self, s_=s, e_=e)
        f_integrated = trapz_loglog(f_, e, axis=1)
        _s_1d = s[:, 0] # in cm
        Ntot = ElectronsOnIBS.analyt_adv_Ntot(self, s_1d = _s_1d,
                                              f_inj_integrated = f_integrated)
        gammas = self.ibs.gma(s = _s_1d)
        betas = beta_from_g(gammas)
        res = 1. /C_LIGHT * Ntot / np.gradient(Ntot, _s_1d, edge_order=2)/betas
        return res[:,None] * s / (1e-17 + s)
    
    def eta_flow_mimic(self, s):
        """eta_flow(s) for testing. Defined so that eta_flow(s)*dorb/c_light =
        = time of the bulk flow from apex to s = s.
        s in [cm], Gamma is a terminal lorentz-factor, s_max_g [dimless] is s at 
        which Gamma is reached, dorb is an
        orbital separation"""
        _ga = self.ibs.gamma_max - 1.
        s_max_g_dimless = self.ibs.ibs_n.s_max_g 
        _x = _ga * s / s_max_g_dimless / self.r_sp
        res_ = s_max_g_dimless / _ga * ( (1. + _x)**2 - 1.)**0.5
        floor_eta_a_ = 1e-2
        res_[res_ < floor_eta_a_] = floor_eta_a_
        return res_
    
    def calculate(self, to_return=False):        ### s [cm], e[eV]

        """
        Calculates the electron spectrum on the IBS.
        Sets the attributes: 
            self._f_inj (f_inj calculated at s_mesh x e_mesh),
            self._edots (edot calculated at s_mesh x e_mesh),
            self.dNe_deds_IBS (dNe/dsde calculated at s_mesh x e_mesh),
            self.e_vals (energy grid),
            self.dNe_deds_mid (same as dNe_deds_IBS but at midpoints of s-grid),
            self.dNe_ds_mid (tot number of e at midpoints, 1/s/cm),
            self.dNe_de_mid (e-spec for every segmens at midpoints, 1/s/eV).

        Parameters
        ----------
        to_return : bool, optional
            Whether to return dNe_deds_IBS_2horns and e_vals.
              The default is False.

        Raises
        ------
        ValueError
            If `cooling` is not recognized.

        Returns
        -------
        dNe_deds_IBS_2horns : np.ndarray of shape (Ns, Ne)
            The electron spectra dNedot/deds on the grid self.ibs.s 
            and on a grid of energies evals between emin_grid and emax_grid,
            [1/s/cm/eV].
        e_vals : TYPE
            The grid of energies [eV].

        """
        ### we use the symmetry of ibs: calculate dNe_de only for 1 horn

        s_1d_dim = self.ibs.s[self.ibs.n : 2*self.ibs.n] 
        
        if self.cooling not in self.allowed_coolings:
            print('cooling should be one of these options:')
            print(self.allowed_coolings)
            print('setting cooling = \'no\'... ')
            self.cooling = 'no'
        # ndec_e = 101 if self.cooling in ('leak_apex', 'leak_ibs', 'leak_mimic') else 301
        
        if self.cooling == 'no':
            e_vals = loggrid(self.emin_grid, self.emax_grid, 301)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            # For each s, --> injected distributions_dimless
            dNe_deds_IBS = ElectronsOnIBS.f_inject(self, smesh, emesh)
            
        elif self.cooling in ('stat_apex', 'stat_ibs', 'stat_mimic'):
            e_vals = loggrid(self.emin_grid, self.emax_grid, 303)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            # For each s, --> stationary distribution
            if self.cooling == 'stat_mimic':
                eta_fl_new = ElectronsOnIBS.eta_flow_mimic(self, smesh)
                self.eta_a = self.eta_a * eta_fl_new
                
            f_inj_se = ElectronsOnIBS.f_inject(self, smesh, emesh)
            edots_se = ElectronsOnIBS.edot(self, smesh, emesh)
            dNe_deds_IBS = np.zeros((s_1d_dim.size, e_vals.size))
                    
            for i_s in range(s_1d_dim.size):
                if self.cooling == 'stat_apex':
                    f_inj_av = trapezoid(f_inj_se, s_1d_dim, axis=0) / np.max(s_1d_dim)
                    dNe_deds_IBS[i_s, :] = stat_distr(e_vals, f_inj_av, edots_se[0, :])
                if self.cooling in ('stat_ibs', 'stat_mimic'):
                    dNe_deds_IBS[i_s, :] = stat_distr(e_vals, f_inj_se[i_s, :], edots_se[i_s, :])

        elif self.cooling in ('leak_apex', 'leak_ibs', 'leak_mimic'):
            e_vals = loggrid(self.emin_grid, self.emax_grid, 107)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            if self.cooling == 'stat_mimic':
                eta_fl_new = ElectronsOnIBS.eta_flow_mimic(self, smesh)
                self.eta_a = eta_fl_new
    
            f_inj_se = ElectronsOnIBS.f_inject(self, smesh, emesh)
            edots_se = ElectronsOnIBS.edot(self, smesh, emesh)
            dNe_deds_IBS = np.zeros((s_1d_dim.size, e_vals.size))
            ts_leak = ElectronsOnIBS.t_leakage(self, smesh, emesh)
            # f_inj_integr = trapezoid(f_inj_se, e_vals, axis=1)
            # Ntot_s = analyt_adv_Ntot_tomimic(smesh, emesh, f_inj_func = f_inject_func,
            #             f_inj_args = f_args, dorb=r_SP, s_max_g=4, Gamma=Gamma)

            for i_s in range(s_1d_dim.size):
                if self.cooling == 'leak_ibs':
                    dNe_deds_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], Ts = ts_leak[i_s, :],
                        mode = 'analyt')
                if self.cooling == 'leak_apex':
                    f_inj_av = trapezoid(f_inj_se, s_1d_dim, axis=0) / np.max(s_1d_dim)
                    dNe_deds_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_av, Edots = edots_se[0, :], Ts = ts_leak[0, :])
                if self.cooling == 'leak_mimic':
                    dNe_deds_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], 
                        Ts = ts_leak[i_s, :], mode = 'analyt') 
            
        elif self.cooling == 'adv':
            edot_func = lambda s_, e_: ElectronsOnIBS.edot(self, s_, e_)
            f_inject_func = lambda s_, e_: ElectronsOnIBS.f_inject(self, s_, e_)
            vel_func = lambda s_: ElectronsOnIBS.vel(self, s=s_)
            
            dNe_deds_IBS, e_vals = evolved_e_advection(s_ = s_1d_dim,
                edot_func=edot_func, f_inject_func=f_inject_func, 
                tot_loss_args=(), f_args=(), vel_func=vel_func, v_args=(),
                emin_grid=self.emin_grid, emax_grid=self.emax_grid)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
        else:
            raise ValueError('I don\'t know this `cooling`!')

        # we fill the 2nd horn with the values 
        # from the 1st horn
        self._f_inj = ElectronsOnIBS.f_inject(self, smesh, emesh)
        self._edot = ElectronsOnIBS.edot(self, smesh, emesh)
        dNe_deds_IBS_2horns = np.zeros((self.ibs.s.size, e_vals.size ))
        dNe_deds_IBS_2horns[:self.ibs.n, :] = -dNe_deds_IBS[::-1, :] # in reverse order from s=smax to ~0
        dNe_deds_IBS_2horns[self.ibs.n : 2*self.ibs.n, :] = dNe_deds_IBS        
        
        self.dNe_deds_IBS = dNe_deds_IBS_2horns
        self.e_vals = e_vals
        e_sed = np.array([ dNe_deds_IBS_2horns[i_s, :] * e_vals**2
                          for i_s in range(dNe_deds_IBS_2horns.shape[0])])
        self.e2dNe_deds_IBS = e_sed
        dNe_deds_m = dNe_deds_IBS_2horns[:self.ibs.s.size-1, :]
        dNe_deds_p = dNe_deds_IBS_2horns[1:, :]
        dNe_deds_mid = 0.5 * (dNe_deds_m + dNe_deds_p)
        self.dNe_deds_mid = dNe_deds_mid
        dNe_ds_mid = trapz_loglog(dNe_deds_mid, e_vals, axis=1) # shape=(s_mid.size)
        dNe_de_mid = dNe_deds_mid * self.ibs.ds[:, None] # basically, trapezoid rule; shape=(s_mid.size, e_vals.size)
        self.dNe_ds_mid = dNe_ds_mid
        self.dNe_de_mid = np.abs(dNe_de_mid)
        
        if to_return:
            return dNe_deds_IBS_2horns, e_vals
        
    @property
    def _up(self):
        """
        Aux function for indexes on the upper IBS horn:
            e.g. self.dNe_deds_IBS[self._up, :] gives you the dNe/dsde on
            the upper horn (incl zero).
        """
        return np.where(self.ibs.theta >= 0)[0]
    
    @property
    def _low(self):
        """
        Aux function for indexes on the lower IBS horn:
            e.g. self.dNe_deds_IBS[self._low, :] gives you the dNe/dsde on
            the lower horn (without zero).
        """
        return np.where(self.ibs.theta < 0)[0]

    @property
    def _up_mid(self):
        """
        Aux function for indexes on the upper IBS horn midpoints:
            e.g. self.dNe_deds_mid[self._up_mid, :] gives you the dNe/dsde on
            the upper horn midpoints (incl zero).
        """
        return np.where(self.ibs.theta_mid >= 0)[0]
    
    @property
    def _low_mid(self):
        """
        Aux function for indexes on the lower IBS horn midpoints:
            e.g. self.dNe_deds_mid[self._low_mid, :] gives you the dNe/dsde on
            the lower horn midpoints (without zero).
        """
        return np.where(self.ibs.theta_mid < 0)[0]

        
        

    def peek(self, ax=None, 
             to_label=True,
            show_many = True,
            **kwargs):
        """
        Quick look at the results of calculation.
        Plots the electron SED at the apex-averaged (over s) and
        at several positions on the IBS, and the total number of electrons
        on the IBS horn as a function of arclength from the apex.
        Also plots the cooling time at the apex as a function of energy.

        Parameters
        ----------
        ax : ax object of pyplot, optional
            Axis to draw on. Either None (then they are
              created), or axes with at least 1 row and 3 columns.
                The default is None.
        to_label : bool, optional
            Whether to show legends on axes 0 and 1. 
            The default is True.
        show_many : bool, optional
            Whether to show e-SEDs for several points at the IBS
            in addition to the IBS-averages e-SED.
              The default is True.
        **kwargs : 
            additional arguments, used simultaneously 
              on axes 0 and 1.

        Raises
        ------
        ValueError
            If the electron spectrum is not calculated yet.

        """
    
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(8, 6))    

        if self.dNe_deds_IBS is None or self.e_vals is None:
            raise ValueError("You should call `calculate()` first to set dNe_deds_IBS and e_vals")
        
        
        Ne_tot_s = trapz_loglog(self.dNe_deds_IBS, self.e_vals, axis=1)
        e_sed_averageS = trapezoid(self.e2dNe_deds_IBS[self.ibs.n+1 : 2*self.ibs.n-1, :],
                self.ibs.s[self.ibs.n+1 : 2*self.ibs.n-1], axis=0) / np.max(self.ibs.s)


        ax[0].plot(self.e_vals, e_sed_averageS, label=f'{self.cooling}', **kwargs)
        ax[1].plot(self.ibs.s, self.ibs.s*Ne_tot_s,  label=f'{self.cooling}', **kwargs)


        if show_many:
            _n = self.ibs.n
            for i_s in (
                        int(_n * (1+1/7)),
                        int(_n*(1+1/2)),
                        int(_n*(1+6/7))
                    ):
                label_s = fr"{self.cooling}, $s = {(self.ibs.s[i_s] / self.ibs.s_max) :.2f} s_\mathrm{{max}}$"
   
                ax[0].plot(self.e_vals, self.e2dNe_deds_IBS[i_s, :], alpha=0.3,
                           label=label_s, **kwargs)

            
        if to_label:
            ax[0].legend()
            ax[1].legend()
        
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel(r'$E$ [eV]')
        ax[0].set_ylabel(r'$E^2 dN/dEds$ [eV]')
        ax[0].set_title(r'$dN/dEds$')

        ax[0].set_ylim(np.nanmax(self.e2dNe_deds_IBS) * 1e-3, 
                       np.nanmax(self.e2dNe_deds_IBS) * 2)

        ax[1].set_xlabel(r'$s$')
        # ax[1].set_ylabel(r'$N(s)$')
        ax[1].set_title(r'$s ~dN/ds$')    
        
        smesh, emesh = np.meshgrid(self.ibs.s[self.ibs.n:2*self.ibs.n],
                                   self.e_vals, indexing='ij')
        edot_ = ElectronsOnIBS.edot(self, smesh, emesh)
        edot_avg = trapezoid(edot_, smesh[:, 0], axis=0) / trapezoid(np.ones(smesh[:, 0].size), smesh[:, 0], axis=0)
        ax[2].plot(self.e_vals, -self.e_vals / edot_avg)
        ax[2].set_xlabel('e, eV')
        ax[2].set_title('t cool [s] VS e apex')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        
        
class NonstatElectronEvol: #!!!
    """
    Time-dependent (non-stationary) one-zone electron evolution at the IBS apex.

    This class evolves the electron energy distribution N(E, t) in a single
    emission zone located at the intrabinary-shock (IBS) apex, using losses and
    injection that depend on the binary phase via a supplied :class:`Winds`
    model. It provides helpers to compute the instantaneous loss rate at the
    apex, the injection spectrum, a stationary comparator, and the fully
    time-evolved spectrum on a log-energy grid. Energies are in eV and times in
    seconds. 

    Parameters
    ----------
    winds : Winds
        Wind/geometry model used to obtain apex distance, fields, and stellar
        parameters as functions of time. 
    t_start : float
        Start time of the evolution window [s].
    t_stop : float
        Stop time of the evolution window [s].
    n_t : int, optional
        Nominal number of time samples (currently not used in the solver).
        Default is 105. 
    to_inject_e : {'ecpl', 'pl'}, optional
        Injection law in energy: exponential cut-off power law ('ecpl') or
        pure power law ('pl'). Default 'ecpl'. 
    to_inject_theta : {'2d','3d'}, optional
        Angular option placeholder (not used by the one-zone model). Default '3d'.
    ecut : float, optional
        Cut-off energy for 'ecpl' [eV]. Default 1e12.
    p_e : float, optional
        Injection spectral index. Default 2.0.
    norm_e : float, optional
        Injection normalization [1/s]. Default 1e37. The code renormalizes the
        spectrum so that the integrated rate in the forward hemisphere equals
        ``norm_e/2``. 
    eta_a : float or None, optional
        Adiabatic-timescale factor; if ``None`` it is set to a very large value
        (≈1e20), effectively disabling the adiabatic term. Default 1.0. 
    eta_syn, eta_ic : float, optional
        Multipliers for synchrotron and inverse-Compton losses. Defaults 1.0.
    emin, emax : float, optional
        Injection energy bounds [eV]. Defaults 1e9 and 5.1e14.
    to_cut_e : bool, optional
        If True, injection is zeroed outside ``[emin, emax]``. Default True. 
    emin_grid, emax_grid : float, optional
        Energy range for the evolution grid [eV]. Defaults 3e8 and 6e14.
    coef_down : float, optional
        Energy-grid parameter defining e_min_grid=emin_feid/coef_down for 
        the solver. Default 10. 
    n_dec_e : int, optional
        Points per decade for the log-energy grid. Default 35. 
    init_distr : {'stat','zero', tuple (e, n0)}, optional
        Initial condition for N(E,t) at ``t_start``.
        Can be a tuple of two arrays (e_grid, n_0(e_grid)).
        Default 'stat'. 
    eps_small, eps_big : float, optional
        Relative tolerances controlling the adaptive stepping heuristics.
        Defaults 1e-3 and 3e-3. 
    adaptive_dt : bool, optional
        Enable adaptive time-stepping. Default False. 
    dt_min, dt_max : float, optional
        Minimum/maximum allowed time steps [s]. Defaults 0.01*DAY and 5*DAY.
    dt_first : float or None, optional
        First time step [s]; if None, chosen automatically.

    Attributes
    ----------
    t_start, t_stop : float
        Evolution time window [s].
    winds : Winds
        The supplied winds model used for apex geometry/fields.
    eta_a, eta_syn, eta_ic : float
        Loss scaling parameters for adiabatic, synchrotron, and IC terms.
    emin, emax : float
        Injection band [eV]; applied if ``to_cut_e`` is True.
    emin_grid, emax_grid : float
        Energy range of the solver grid [eV].
    n_dec_e, coef_down : int, float
        Grid control parameters.
    ts : ndarray, shape (nt,)
        Time grid returned by the solver [s]. Set by :meth:`calculate`. 
    e_edg : ndarray
        Log-energy grid produced for the spectrum (treated as bin edges). Set by
        :meth:`calculate`. 
    e_c : ndarray, shape (ne,)
        Energy grid corresponding to the spectra [eV]. Set by :meth:`calculate`.
    dNe_des : ndarray, shape (nt, ne)
        Time-dependent spectra :math:`\\mathrm{d}N/\\mathrm{d}E` [1/eV]. Set by
        :meth:`calculate`. 
    edots_avg, q_avg : ndarray, shape (nt, ne)
        Time-averaged loss rates and injection used by the solver. Set by
        :meth:`calculate`.
    dn_de_spl, dnstat_de_spl : scipy.interpolate.interp1d
        Interpolants that return :math:`\\mathrm{d}N/\\mathrm{d}E` at arbitrary
        times for the non-stationary and stationary solutions, respectively.
        Set by :meth:`calculate`. 
    nstat : ndarray, shape (nt, ne)
        Stationary comparator spectra at the same time grid. 

    Methods
    -------
    edot_apex(e_, t_)
        Total loss rate :math:`\\dot E(E,t)` at the apex (synchrotron + IC +
        adiabatic). 
    f_inject(e_, t_)
        Injection spectrum :math:`Q(E,t)` (ECPL or PL) with optional energy cuts
        and internal renormalization. 
    stat_distr_at_time(t_, e_=None)
        Stationary spectrum at a given time using instantaneous losses and
        injection. 
    calculate(to_return=False)
        Run the non-stationary solver and populate grids/spectra. Optionally
        return ``ts, e_bins, dNe_des``. 
    dn_de(t), dnstat_de(t)
        Interpolate non-stationary / stationary spectra at time(s) ``t``.
    n_tot(t, emin=None, emax=None), nstat_tot(t, emin=None, emax=None)
        Integrate spectra over an energy band to get total N(t). 

    Notes
    -----
    * **Loss model.** The apex loss rate combines synchrotron, isotropic IC, and
      an adiabatic term (with timescale ``t_ad = eta_a * d / c``), using
      stellar parameters and apex distance from ``winds``. Signs follow the
      convention that losses are negative. 
    * **Adiabatic disabling.** Passing ``eta_a=None`` sets ``eta_a≈1e20`` inside
      the class, effectively removing the adiabatic term. 
    * **Grids.** The solver builds a log-energy grid over ``[emin_grid, emax_grid]``
      with ``n_dec_e`` points per decade; time sampling may be adaptive between
      ``dt_min`` and ``dt_max``. 
    
    Experimental. Currently cannot be used for the spec/LC calculation.
    My God I'm tired of writing the docs...
    All classes docstrings are ChatGPT-generated btw.
    """

    def __init__(self, winds, t_start, t_stop, n_t=105, 
                     
    to_inject_e = 'ecpl',   # el_ev
    to_inject_theta = '3d', ecut = 1.e12, p_e = 2., norm_e = 1.e37,
    eta_a = 1.,
    eta_syn = 1., eta_ic = 1.,
    emin = 1e9, emax = 5.1e14, to_cut_e = True, 
    emin_grid=3e8, emax_grid=6e14, coef_down=10,
     n_dec_e=35, 

     init_distr='stat', 
     eps_small = 1e-3, eps_big = 3e-3,
     adaptive_dt = False,
     dt_min = 1e-2 * DAY, dt_max = 5 * DAY, dt_first=None
    ):
       
        self.t_start = t_start
        self.t_stop = t_stop
        self.n_t = n_t # not used???
        self.winds = winds

        if eta_a  is None:
            self.eta_a = 1e20
        else:
            self.eta_a = eta_a # to scale adiabatic time

        self.eta_syn = eta_syn # to scale synchrotron losses
        self.eta_ic = eta_ic # to scale inverse compton losses
        
        self.to_inject_e = to_inject_e # injection function along energies
        self.to_inject_theta = to_inject_theta # injection function along theta
        self.p_e = p_e  # injection function spectral index
        self.ecut = ecut  # injection function cutoff energy
        self.norm_e = norm_e # injection function normalization
        self.emin = emin  # minimum energy for the injection function
        self.emax = emax  # maximum energy for the injection function
    
        self.to_cut_e = to_cut_e # whether to leave only the part emin < e < emax
  
        self.emin_grid = emin_grid
        self.emax_grid = emax_grid
        self.coef_down = coef_down
        self.n_dec_e = n_dec_e
        self.init_distr = init_distr
        
        # self.n_t_avg = n_t_avg
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.adaptive_dt = adaptive_dt
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_first = dt_first

        
        # self._set_grids()
        
            
    # def _set_grids(self):
    #     self.e_grid = loggrid(x1 = self.emin_grid, 
    #                           x2 = self.emax_grid, 
    #                           n_dec = int(self.n_dec_e)
    #                           )
        # self.t_grid = np.linspace(self.t_start, self.t_stop, self.n_t)
    
    def edot_apex(self, e_, t_): 
        """
        Energy loss rate in the emission zone [eV/s].

        Parameters
        ----------
        e_ : np.ndarray
            e-energy [eV].
        t_ : float
            time [s].

        Returns
        -------
        np.ndarray
            Edot(e_, t).

        """
        r_sa = self.winds.dist_se_1d(t_)
        B_p_apex, B_s_apex = self.winds.magn_fields_apex(t_)
        return total_loss(ee = e_, 
                          B = B_p_apex + B_s_apex, 
                          Topt = self.winds.Topt,
                          Ropt=self.winds.Ropt,
                          dist = r_sa, 
                          eta_flow = self.eta_a, 
                          eta_syn = self.eta_syn,
                          eta_IC = self.eta_ic)

    def f_inject(self, e_, t_): 
        """
        Injection function at the emission zone dNe/dtde [1/s/eV].

        Parameters
        ----------
        e_ : np.ndarray
            e-energy [eV].
        t_ : float
            time [s]. Currently not used, but can be.

        Raises
        ------
        ValueError
            If to_inject_e is not recognized.

        Returns
        -------
        np.ndarray
            f_inject(e_, t_) [1/s/eV].

        """

        if self.to_inject_e == 'ecpl':
            e_part = ecpl(e_, ind=self.p_e, ecut=self.ecut, norm=1.)
        elif self.to_inject_e == 'pl':
            e_part = pl(e_, ind=self.p_e, norm=1)
        else:
            raise ValueError("I don't know this to_inject_theta. It should be pl or ecpl.")
            
            
        ### if we assume that the `emission zone` is the forward hemisphere,
        ### then the NUMBER of injected e don't change with time. 
        ### We can include the t-dependence if we fugure out how to do so
        ### in a meaningful way.
        ### The density dNtot/ds changes, of course.
        
        result = e_part * self.norm_e 
        
        if self.to_cut_e:
            mask = (e_ < self.emin) | (e_ > self.emax)
            result = np.where(mask, 0.0, result)
        
        # now normalize the number of injected particles. I do it like this:
        # I ensure that the total number per second: N(s) = \int n(s, E) dE 
        # of electrons is half the Normalization
        N_total = trapz_loglog(result, e_)
        overall_coef = self.norm_e / 2 / N_total
        return result * overall_coef
    
    def stat_distr_at_time(self, t_, e_=None):
        """
        Stat e-spectrum at time t_, dNe/de [1/eV].

        Parameters
        ----------
        t_ : float
            time [s].
        e_ : np.ndarray
            e-energy [eV].

        Returns
        -------
        np.ndarray
            dNe/de(e_) |_{at t=t_}.

        """
        if e_ is None:
            e_ = self.e_grid
        return stat_distr(Es = e_,  
                    Qs = NonstatElectronEvol.f_inject(self, e_, t_),
                    Edots = NonstatElectronEvol.edot_apex(self, e_, t_)
                             )

    def calculate(self, to_return=False):
        """
        Calculates the non-stationary evolution of the electron spectrum
        at the emission zone. Sets the attributes:
        self.e_edg --- energy bin edges for dNe_des. I think, if ne is the 
        number of energy bins, then e_edg has the shape (ne+1,)
        self.dNe_des (shape (nt, ne)) -- the electron spectrum dNe/de at times ts
        self.ts (shape (nt,)) -- the time grid
        self.edots_avg (shape (nt, ne)) -- edot averaged over the time steps
        self.q_avg = (shape (nt, ne)) -- injection function averaged over the time steps
        self.dn_de_spl -- interp1d object for dNe/de at arbitrary time
        self.e_c = e_bins -- the energy grid for dNe_des
        self.nstat (shape (nt, ne)) -- the stationary spectrum at times ts
        self.dnstat_de_spl -- interp1d object for nstat at arbitrary time
        
        Parameters
        ----------
        to_return : bool, optional
            Whether to return ts, e_bins, dNe_des.
              The default is False.

        Returns
        -------
        ts : np.array of shape (nt,)
            The time grid [s].
        e_bins : np.array of shape (ne,)
            The energy grid [eV].
        dNe_des : np.array of shape (nt, ne)
            The electron spectrum dNe/de at times ts [1/eV].

        """
        _edot_func = lambda e_, t_: NonstatElectronEvol.edot_apex(self, e_, t_)
        _q_func = lambda e_, t_:  NonstatElectronEvol.f_inject(self, e_, t_)
        
        
        ts, e_bins, dNe_des, edots_avg, q_avg = \
              evolved_e_nonstat_1zone(t_start=self.t_start,
                                       t_stop = self.t_stop,  
                            q_func=_q_func,
                              edot_func= _edot_func, 
                          init=self.init_distr,
                          e_grid=None,
                          emin=self.emin_grid,
                            emax=self.emax_grid,
                              coef_down=1., 
                              ne_dec=self.n_dec_e,
                          eps_big=self.eps_big,
                            eps_small=self.eps_small,
                              dt_min=self.dt_min,
                          dt_max=self.dt_max,
                            dt_first=self.dt_first,
                              adaptive_dt=self.adaptive_dt,)
        self.e_edg = loggrid(x1 = self.emin_grid, 
                                  x2 = self.emax_grid, 
                                  n_dec = int(self.n_dec_e))
        self.dNe_des = dNe_des
        self.ts = ts
        self.edots_avg = edots_avg
        self.q_avg = q_avg
        self.dn_de_spl = interp1d(self.ts, self.dNe_des, axis=0)
        self.e_c = e_bins
        
        nstat = []
        for t_ in ts:
            nstat.append(NonstatElectronEvol.stat_distr_at_time(self, t_=t_,
                                                                e_=e_bins))
        nstat=np.array(nstat)
        self.nstat=nstat
        self.dnstat_de_spl = interp1d(self.ts, self.nstat, axis=0)
        
        
        if to_return:
            return ts, e_bins, dNe_des
        
    def dn_de(self, t):
        """
        E-spectrum at time t, dNe/de [1/eV].

        Parameters
        ----------
        t : np.ndarray
            Time [s].

        Raises
        ------
        ValueError
            If the electron spectrum is not calculated yet.

        Returns
        -------
        np.ndarray
            dNe/de |_{at t}.

        """
        if self.dNe_des is None:
            raise ValueError('you should calculate() first')
        return self.dn_de_spl(t)
    
    def dnstat_de(self, t):
        """
        Stationary E-spectrum at time t, dNe/de [1/eV].

        Parameters
        ----------
        t : np.ndarray
            Time [s].

        Raises
        ------
        ValueError
            If the electron spectrum is not calculated yet.

        Returns
        -------
        np.ndarray
            dNe_stat/de|_{at t}.

        """
        if self.dNe_des is None:
            raise ValueError('you should calculate() first')
        return self.dnstat_de_spl(t)
    
    
    def n_tot(self, t, emin=None, emax=None):
        """
        The total number of electrons at time t, N_tot,
        between emin and emax.

        Parameters
        ----------
        t : np.ndarray
            Times [s].
        emin : float, optional
            Min energy, [eV]. The default is None.
        emax : float, optional
            Max energy, [eV]. The default is None.
        Returns
        -------
        np.ndarray
            Ntot(t).

        """
        if emin is None:
            emin = self.emin
        if emax is None:
            emax = self.emax
        mask = np.logical_and(self.e_c >= emin, self.e_c <= emax)
        return trapz_loglog(self.dn_de(t)[:, mask], self.e_c[mask], axis=1)

    def nstat_tot(self, t, emin=None, emax=None):
        """
        The total number of electrons at time t, N_tot,
        between emin and emax, for a stationary spectrum.

        Parameters
        ----------
        t : np.ndarray
            Times [s].
        emin : float, optional
            Min energy, [eV]. The default is None.
        emax : float, optional
            Max energy, [eV]. The default is None.
        Returns
        -------
        np.ndarray
            N_stat_tot(t).

        """
        if emin is None:
            emin = self.emin
        if emax is None:
            emax = self.emax
        mask = np.logical_and(self.e_c >= emin, self.e_c <= emax)
        return trapz_loglog(self.dnstat_de(t)[:, mask], self.e_c[mask], axis=1)

    
    
    
    
    
# if __name__ == '__main__':
    # from ibsen.orbit import Orbit
    # from ibsen.winds import Winds
    # from ibsen.ibs import IBS
    # import time

    # sys_name = 'psrb' 
    # orb = Orbit(sys_name = sys_name, n=1003)
    # winds_full = Winds(orbit=orb, sys_name = sys_name, alpha=-10/180*pi, incl=23*pi/180,
    #             f_d=165, f_p=0.1, delta=0.015, np_disk=3, rad_prof='pl', r_trunk=None,
    #             height_exp=0.25,
    #                 ns_field_model = 'linear', ns_field_surf = 0.2, ns_r_scale = 1e13,
    #                 opt_field_model = 'linear', opt_field_surf = 0, opt_r_scale = 1e13,)
    # # winds_full.peek(showtime=(-100*DAY, 100*DAY))


    # el = NonstatElectronEvol(winds=winds_full, t_start=-100*DAY, t_stop=100*DAY,
    #                         emin=1e9, emax=1e13, emin_grid=1e8, emax_grid=1e13,
    #                         p_e=1.7, init_distr='zero', eps_big=5e-3, eps_small=3e-3,
    #                         n_dec_e=201, dt_min=0.01*DAY, dt_first=None,
    #                         adaptive_dt=True, eta_a=1e20)
    # start = time.time()
    # ts, es, ns = el.calculate(to_return=True)
    # # e_ = es[1, :]
    # e_ = el.e_c
    # print(time.time() - start)

    # plt.subplot(1, 3, 1)
    # print(el.ts.shape)
    # print(el.dNe_des.shape)
    # print(type(el.ts))
    # print(type(el.dNe_des))
    # plt.plot(e_, el.dn_de(-70*DAY)*e_**2, color='r', label='-60 days')
    # plt.plot(e_, el.stat_distr_at_time(-70*DAY, e_)*e_**2, ls='--', color='r')
    
    # plt.plot(e_, el.dn_de(-15*DAY)*e_**2, color='g', label='-15 days')
    # plt.plot(e_, el.stat_distr_at_time(-15*DAY, e_)*e_**2, ls='--', color='g')
    
    
    # plt.plot(e_, el.dn_de(0)*e_**2, color='k', label='0 days')
    # plt.plot(e_, el.stat_distr_at_time(0, e_)*e_**2, ls='--', color='k')

    # plt.plot(e_, el.dn_de(15*DAY)*e_**2, color='m', label='20 days')
    # plt.plot(e_, el.stat_distr_at_time(15*DAY, e_)*e_**2, ls='--', color='m')

    # plt.plot(e_, el.dn_de(70*DAY)*e_**2, color='b', label='70 days')
    # plt.plot(e_, el.stat_distr_at_time(70*DAY, e_)*e_**2, ls='--', color='b')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('e, eV')
    # plt.ylim(1e45, 5e49)
    # plt.legend(fontsize=10)
    # plt.title('e-SED')


    # plt.subplot(1, 3, 2)
    # Ntot = el.n_tot(ts)
    # Ntot_stat = el.nstat_tot(ts)
    # plt.plot(ts/DAY, Ntot, label='nonstat')
    # plt.plot(ts/DAY, Ntot_stat, label='stat')
    
    # plt.xlabel('t, days')
    # plt.title('total Ne in [emin, emax]')
    # plt.legend()
    
    # plt.subplot(1, 3, 3)
    # dt = ts[1:] - ts[:-1]
    # plt.plot(ts[1:]/DAY, dt/DAY)
    # plt.title('time step (d)')
    # plt.xlabel('t, days')
    # plt.yscale('log')



if __name__ == "__main__":
    from ibsen.orbit import Orbit

    DAY = 86400.
    AU = 1.5e13
    
    sys_name = 'psrb' 
    orb = Orbit(sys_name = sys_name, n=1003)
    from ibsen.winds import Winds
    winds = Winds(orbit=orb, sys_name = sys_name, alpha=-10/180*pi, incl=23*pi/180,
              f_d=165, f_p=0.1, delta=0.02, np_disk=3, rad_prof='pl', r_trunk=None,
             height_exp=0.25)     
    

    t = 0 * DAY
    Nibs = 51
    ibs = IBS(winds=winds,
              gamma_max=1.8,
              s_max=3.,
              s_max_g=3.,
              n=Nibs,
              t_to_calculate_beta_eff=t) # the same IBS as before
    # ibs1.peek(show_winds=True, to_label=False, showtime=(-100*DAY, 100*DAY),
    #          ibs_color='doppler')
    ### leak_mimic
    # for emin_grid in (1e8, 3e8, 1e9):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for cooling in ('adv', 'leak_mimic', 'stat_ibs', 'stat_mimic'):
        emin_grid=1e8
        eta_a = 6 if cooling in ('stat_ibs', 'stat_mimic') else 1e20
        els = ElectronsOnIBS(Bp_apex=1, ibs=ibs, cooling=cooling, eta_a = eta_a,
                         to_inject_e = 'ecpl', emin=1e9, emax=1e13, emin_grid=emin_grid,
                         emax_grid=1e14,
                         p_e=1.8, ecut=5e12) 
        
        els.calculate(to_return=False)
        # print('el_ev calculated')
        els.peek(ax=ax, show_many=False)
        # plt.plot(ibs.s_mid, els.dNe_de_mid[:, 10])
        # print(els.dNe_deds_mid.shape)
        # print(ibs.s_mid.shape)
        # ax[1].plot(ibs.s[els._up], ibs.s[els._up]*els.analyt_adv_Ntot_self, label=cooling+' an')
        # print(els._up_mid)
        # plt.loglog(els.e_vals, els._f_inj[10, :])
        # plt.loglog(els.e_vals, els._f_inj[Nibs-1, :], ls='--')
        # plt.loglog(els.e_vals, els._f_inj[1, :], ls=':')
        # cond_ = np.where(
        #     np.logical_and(els.ibs.theta[els._up]>0, els.ibs.theta[els._up]<=pi/2)
        #     )
        # Ntot1 = trapezoid(els._f_inj, els.e_vals, axis=1)
        # Ntot2 = trapezoid(Ntot1[cond_], els.ibs.s[cond_])
        # print('1 = ', Ntot2*4)
        # print('1 / 2 = ', Ntot2*4/els.norm_e)
        
        
        # plt.plot(els.ibs.s[els._up], trapz_loglog(els._f_inj, els.e_vals, axis=1))
        
        # Ntot1 = 2*trapezoid(els.dNe_deds_mid[els._up_mid, :], ibs.s_mid[els._up_mid], axis=0)
        # Ntot2 = trapz_loglog(Ntot1, els.e_vals)
        plt.legend()
        # print(Ntot2)
        # print(np.sum(trapz_loglog(np.abs(els.dNe_de_mid), els.e_vals, axis=1)))
        # plt.show()
    
    # plt.plot(els.ibs.s, els._b)
    # plt.plot(els.ibs.s, els._u)
    