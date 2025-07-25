import numpy as np
# from matplotlib import pyplot as plt
from numpy import pi, sin, cos, exp
import astropy.units as u
from astropy import constants as const
from scipy.integrate import trapezoid, cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
# import time
# from joblib import Parallel, delayed
from ibsen.utils import  beta_from_g, loggrid, t_avg_func

from ibsen.transport_solvers.transport_on_ibs_solvers import solve_for_n, nonstat_characteristic_solver
# import xarray as xr
# from pathlib import Path

# from .orbit import Orbit
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
    '''Synchrotron losses, dE/dt. Bis in G, electron energy -- eV '''
    # return -4e5 * B**2 * (ee/1e10)**2 #eV/s ???
    return -2.5e5 * B**2 * (ee/1e10)**2 #eV/s ???

# def Gma(s, sm, G_term):
    # return 1 + (G_term - 1) * s / sm

def gfunc(x, a = -0.362, b = 0.826, alpha = 0.682, beta = 1.281): #from 1310.7971
    return (1 +  (a * x**alpha) / (1 + b * x**beta) )**(-1)
    
def Gani(u_, c = 6.13):
    cani = c
    return cani * u_ * np.log(1 + 2.16 * u_ / cani) / (1 + cani * u_ / 0.822)

def Giso(u_, c = 4.62):
    ciso = c
    return ciso * u_ * np.log(1 + 0.722 * u_ / ciso) / (1 + ciso * u_ / 0.822)

def Giso_full(x):
    return Giso(x, 5.68) * gfunc(x)

def ic_loss(Ee, Topt, Ropt, dist): # Ee in eV !!!!!!
    kappa = (Ropt / 2 / dist)**2
    T_me = (K_BOLTZ * Topt / MC2E)
    Ee_me = Ee  / ERG_TO_EV / MC2E
    coef = 2 * R_ELECTRON**2 * M_E**3 * C_LIGHT**4 * kappa * T_me**2 / pi / HBAR**3 # 1/s
    Edot = coef * MC2E * ERG_TO_EV * Giso_full(4 * T_me * Ee_me) #[eV/s]
    return -Edot #[eV/s]

    
def t_adiab(dist, eta_flow):
    return eta_flow * dist / C_LIGHT

def ecpl(E, ind, ecut, norm):
    return norm * E**(-ind) * exp(-E / ecut)

def pl(E, ind, norm):
    return norm * E**(-ind)


def total_loss(ee, B, Topt, Ropt, dist, eta_flow, eta_syn, eta_IC):
    # if isinstance(eta_flow, np.ndarray)
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
        calculates the stationary spectrum with leakage: the solution of
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
        
        # Recover n = u / f
        u_numeric = sol.y[0][::-1]        # Flip back to ascending order
        n_numeric = u_numeric / Edots
        return n_numeric
    if mode == 'analyt':
        inv_Tf = 1 / (Ts * Edots)
        # First, compute ∫_∞^{e} (1 / (T * f)) de
        inner_int = cumulative_trapezoid(inv_Tf[::-1], Es[::-1], initial=0)[::-1]  # Shape (N,) 
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
              tot_loss_args, f_args, vel_func, v_args, emin = 1e9, 
              emax = 5.1e14):  
        # s -- in [cm] !!!
        
        # we calculate it on e-grid emin / extend_d < e <  emax * extend_u
        # and hope that zero boundary conditions will be not important
        
        # extend_u = 10; extend_d = 10; 
        extend_u = 2; extend_d = 10; 
        Ns, Ne = 601, 603 
        # Ns, Ne = 201, 203 
        Ne_real = int( Ne * np.log10(extend_u * emax / extend_d / emin) / np.log10(emax / emin) )
        e_vals_sample = np.logspace(np.log10(emin / extend_d), np.log10(emax * extend_u), Ne_real)
        
        # now we'll look where the f_inject is not negligible and only there
        # will we solve the equation 
        ssmesh, emesh = np.meshgrid(s_, e_vals_sample, indexing='ij')
        f_sample = f_inject_func(ssmesh, emesh, *f_args)
        f_sample_sed = f_sample[1, :] * e_vals_sample**2
        e_where_good = np.where(f_sample_sed > np.max(f_sample_sed) / 1e6 )
        s_vals = np.linspace(0, np.max(s_), Ns)
        e_vals = e_vals_sample[e_where_good]
        dNe_de = solve_for_n(v_func = vel_func, edot_func = edot_func,
                            f_func = f_inject_func,
                            v_args = v_args, 
                            edot_args = tot_loss_args,
                            f_args = f_args,
                            s_grid = s_vals, e_grid = e_vals, 
                            method = 'FDM_cons', bound = 'neun')
        # #### Only leave the part of the solution between emin < e < emax 
        ind_int = np.logical_and(e_vals <= emax, e_vals >= emin)
        e_vals = e_vals[ind_int]
        dNe_de = dNe_de[:, ind_int]
        
        #### and evaluate the values on the IBS grid previously obtained:dNe_de_IBS, e_vals
        interp_x = interp1d(s_vals, dNe_de, axis=0, kind='linear', fill_value='extrapolate')
        dNe_de_IBS = interp_x(s_)
        dNe_de_IBS[dNe_de_IBS <= 0] = np.min(dNe_de_IBS[dNe_de_IBS>0]) / 3.14
        
        return dNe_de_IBS, e_vals
    
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
        If e_grid is not set, this is used as the maximum energy. The default is 1e9.
    emax : float, optional
        If e_grid is not set, this is used as the minimum energy OF INTEREST. The default is 5.1e14.
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
    dt_min : floor, optional
        If adaptive_dt=True, this is treated as the floor for a time step dt.
        Should be < dt_max (not forced).
          If adaptive_dt=False, this is the constant time step.
            The default is 1e-2*DAY.
    dt_max : floor, optional
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
        DESCRIPTION.

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
    if cooling not in ('no', 'stat_apex', 'stat_ibs', 'stat_mimic', 'leak_apex', 'leak_ibs',
                       'leak_mimic', 'adv'):
        print('cooling should be one of these options:')
        print('no', 'stat_apex', 'stat_ibs','stat_mimic', 'leak_apex', 'leak_ibs',
                           'leak_mimic', 'adv')
        print('setting cooling = \'no\' ')
        cooling = 'no'
        
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
            # if eta_flow_func == None:
            #     eta_flow_func = eta_flow_mimic
            eta_fl_new = eta_flow_func(smesh, *eta_flow_args)
            tot_loss_args = (B0, Topt, Ropt, r_SE, eta_fl_new * eta_fl,
                                 eta_sy, eta_ic, ss, rs, thetas, r_SP)    
        f_inj_se = f_inject_func(smesh, emesh, *f_args)
        edots_se = edot_func(smesh, emesh, *tot_loss_args)
        dNe_de_IBS = np.zeros((ss.size, e_vals.size))
                
            # edots_se = edot_func(smesh, emesh, *tot_loss_args_new)
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
        # f_inj_integr = trapezoid(f_inj_se, e_vals, axis=1)
        # Ntot_s = analyt_adv_Ntot_tomimic(smesh, emesh, f_inj_func = f_inject_func,
        #             f_inj_args = f_args, dorb=r_SP, s_max_g=4, Gamma=Gamma)

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
        dNe_de_IBS, e_vals = evolved_e_advection(r_SP, ss, edot_func,
                    f_inject_func, tot_loss_args, f_args, vel_func, v_args)
            
    return dNe_de_IBS, e_vals
    
class ElectronsOnIBS: #!!!
    """
    A class representing the electrons on the IBS.
    
    Attributes
    ----------
    Bp_apex : float
        pulsar magn field in apex.
    u_g_apex : float
        photon energy density in apex.
    ibs : IBS
        must contain ibs.winds. Should be dimentionless (not rescaled) and in 
        natural coordinates (not rotated) !!!
    to_inject_e : str, optional
        injection function along energies. Either 'pl' or 'ecpl'.
        The default is 'ecpl'.
    to_inject_theta : str, optional
        injection function along theta. Either '2d' or '3d'. 
        The default is '3d'.
    ecut : float, optional
        injection function cutoff energy. The default is 10^12 eV.
    p_e : float, optional
        injection function spectral index. The default is 2..
    norm_e : float, optional
        injection function normalization. The default is 1..
    Bs_apex : float, optional
        opt star magn field in apex. The default is 0..
    eta_a : float, optional
        to scale adiabatic time. The default is 1..
    eta_syn : float, optional
        to scale synchrotron losses. The default is 1..
    eta_ic : float, optional
        to scale inverse compton losses. The default is 1.
    emin : float, optional
        minimum energy for the injection function. The default is 1e9.
    emax : float, optional
        maximum energy for the injection function. The default is 5.1e14.
    to_cut_e : bool, optional
        whether to leave only the part emin < e < emax. The default is True.
    to_cut_theta : bool, optional
        whether to inject only at theta < where_cut_theta. The default is False.
    where_cut_theta : float, optional
        see above. The default is pi/2.
    
"""
    def __init__(self, Bp_apex, ibs: IBS, cooling=None, to_inject_e = 'ecpl',
                 to_inject_theta = '3d', ecut = 1.e12, p_e = 2., norm_e = 1.e37,
                 Bs_apex=0., eta_a = None,
                 eta_syn = 1., eta_ic = 1.,
                 emin = 1e9, emax = 5.1e14, to_cut_e = True, 
                 to_cut_theta =  False, 
                 where_cut_theta = pi/2):
        """
        We should provide the already initialized class ibs:IBS here with 
        winds:Winds and orbit:Orbit in it. So the ibs
        initialized without an winds:Winds will not do. Probably we should fix 
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
        self.norm_e = norm_e # injection function normalization
        self.emin = emin  # minimum energy for the injection function
        self.emax = emax  # maximum energy for the injection function
        
        self.to_cut_e = to_cut_e # whether to leave only the part emin < e < emax
        self.to_cut_theta = to_cut_theta # whether to inject only at theta < where_cut_theta
        self.where_cut_theta = where_cut_theta # see above
        
        
        self._check_and_set_ibs() #checks if there's right ibs and sets r_sp
        self._set_b_and_u_ibs() # calculates dimensionless b_ and u_ from values in apex on entire ibs
        
        self.dNe_de_IBS = None
        self.e_vals = None
        
        
        
        
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
        r_to_p = self.ibs.s_interp(s_ = s_ / self.r_sp, what = 'r') # dimless
        r_to_s = self.ibs.s_interp(s_ = s_ / self.r_sp, what = 'r1') # dimless
        r_sa = (1. - self.ibs.x_apex)
        B_on_shock = (self.Bp_apex * self.ibs.x_apex / r_to_p + 
                      self.Bs_apex * r_sa / r_to_s)
        u_g_on_shock = r_sa**2 / r_to_s**2
        return B_on_shock / (self.B_apex), u_g_on_shock
    
    def u_g_density(self, r_from_s):      
        factor = 2. * (1. - (1. - (self.ibs.winds.Ropt / r_from_s)**2)**0.5 )
        u_dens = SIGMA_BOLTZ * self.ibs.winds.Topt**4 / C_LIGHT * factor
        return u_dens
    
    def _set_b_and_u_ibs(self):
        b_onibs, u_onibs = ElectronsOnIBS.b_and_u_s(self, s_ = self.ibs.s * self.r_sp)
        self.u_g_apex = ElectronsOnIBS.u_g_density(self,
                        r_from_s = (1. - self.ibs.x_apex) * self.r_sp)
        self._b = b_onibs
        self._u = u_onibs

    
    def vel(self, s): # s --- in real units
        """
        Velocity of a bulk motion along the shock.    
    
        Parameters
        ----------
        s : np.ndarray
            The arclength from the apex to the point of interest [cm].
        Gamma : float
            Terminal (max) lorentz-factor.
        smax_g : TYPE
            The dimentionless arclength along the shock at which Gamma is reached.
        dorb : float
            Orbiral separation [cm]. Needed for setting the scale (since smax_g is 
                                            dimentionless)
    
        Returns
        -------
        The velocity [cm/s].
    
        """
        # smax_g_cm = smax_g * dorb
        gammas = self.ibs.gma(s / self.r_sp)
        return self.ibs.vel_from_g(gammas)


    def edot(self, s_, e_): 
        # r_to_p = self.ibs.s_interp(s_ = s_ / self.r_sp, what = 'r') # dimless
        # r_to_s = self.ibs.s_interp(s_ = s_ / self.r_sp, what = 'r1') # dimless
        r_sa = (1. - self.ibs.x_apex)
        # B_on_shock = (self.Bp_apex * self.ibs.x_apex / r_to_p + 
        #               self.Bs_apex * r_sa / r_to_s)
        # eta_ic_on_shock = self.eta_ic * r_sa**2 / r_to_s**2
        _b_s, _u_s = ElectronsOnIBS.b_and_u_s(self, s_ = s_)
        return total_loss(ee = e_, 
                          B = _b_s * self.B_apex, 
                          Topt = self.ibs.winds.Topt,
                          Ropt=self.ibs.winds.Ropt,
                          dist = self.r_sp * r_sa, 
                          eta_flow = self.eta_a, 
                          eta_syn = self.eta_syn,
                          eta_IC = self.eta_ic * _u_s)

    def f_inject(self, s_, e_): 
        if s_.shape != e_.shape:
            raise ValueError('in f_inject, `s`-shape should be == `e`-shape')
        thetas_here = self.ibs.s_interp(s_  = s_ / self.r_sp, what = 'theta')
        if self.to_inject_theta == '2d':
            thetas_part = np.zeros(thetas_here.shape) + 1. # uniform along theta
        elif self.to_inject_theta == '3d':
            thetas_part = sin(thetas_here) # \propto sin(th) how it should be in 3d
            # thetas_part = s_ / np.max(s_)
        else:
            raise ValueError("I don't know this to_inject_theta. It should be 2d, 3d.")
            
        if self.to_cut_theta:
            thetas_part[np.abs(thetas_here) >= self.where_cut_theta] = 0.
           
        if self.to_inject_e == 'ecpl':
            e_part = ecpl(e_, ind=self.p_e, ecut=self.ecut, norm=1.)
        elif self.to_inject_e == 'pl':
            e_part = pl(e_, ind=self.p_e, norm=1)
        else:
            raise ValueError("I don't know this to_inject_theta. It should be pl or ecpl.")
            
        result = thetas_part * e_part * self.norm_e
        
        if self.to_cut_e:
            mask = (e_ < self.emin) | (e_ > self.emax)
            result = np.where(mask, 0.0, result)
        # return sin(thetas_here)
        
        # now normalize the number of injected particles. I do it like this:
        # I ensure that the total number per second: N(s) = \int n(s, E) dE 
        # of electrons in  ONE horm in the forward hemisphere = 1/4 from norm:
        # 2 pi \int_0^{pi/2} N(s(theta)) d theta = norm / 2
        N_total = trapezoid(result, e_, axis = 1)
        thetas_forward = thetas_here[:, 0][thetas_here[:, 0] < pi/2]
        # print(N_total.shape)
        # print(thetas_forward.shape)
        N_total_forward = N_total[thetas_here[:, 0] < pi/2]
        integral_forward = trapezoid(N_total_forward, thetas_forward)
        #if integral_forward != 0:
        overall_coef = self.norm_e / 8 / pi / integral_forward
        #else:
            # print('а какого хера')
            # print(self.ibs.t_forbeta / DAY)
        return result * overall_coef
    
    # def analyt_adv_Ntot_tomimic(self, s, e): 
    #     # s and e -- 2d meshgrid arrays, but the function returns 1d-array 
    #     _ga = self.ibs.gamma_max - 1
    #     # f_ = f_inj_func(s, e, *f_inj_args)
    #     f_ = ElectronsOnIBS.f_inject(self, s_=s, e_=e)
    #     f_integrated = trapezoid(f_, e, axis=1)
    #     x__ = _ga*s[:, 0] / self.ibs.s_max_g / self.r_sp
    #     return f_integrated * self.r_sp / C_LIGHT * self.s_max_g/_ga * (2 * x__ + x__**2)**0.5
    
    def analyt_adv_Ntot(self, s_1d, f_inj_integrated):
        ### s [dimless] and f_inj_integrated --- 1-dimensional
        gammas = self.ibs.gma(s = s_1d)
        betas = beta_from_g(gammas)
        res = cumulative_trapezoid(f_inj_integrated / betas, s_1d, initial=0) / C_LIGHT
        return res
    
    def t_leakage(self, s,e):
        f_ = ElectronsOnIBS.f_inject(self, s_=s, e_=e)
        f_integrated = trapezoid(f_, e, axis=1)
        _s_1d = s[:, 0] # in cm
        Ntot = ElectronsOnIBS.analyt_adv_Ntot(self, s_1d = _s_1d / self.r_sp,
                                              f_inj_integrated = f_integrated)
        gammas = self.ibs.gma(s = _s_1d / self.r_sp)
        betas = beta_from_g(gammas)
        res = 1. /C_LIGHT * Ntot / np.gradient(Ntot, _s_1d, edge_order=2)/betas
        return res[:,None] * s / (1e-17 + s)
    
    def eta_flow_mimic(self, s):
        """eta_flow(s) for testing. Defined so that eta_flow(s)*dorb/c_light =
        = time of the buulk flow from apex to s = s.
        s in [cm], Gamma is a terminal lorentz-factor, s_max_g [dimless] is s at 
        which Gamma is reached, dorb is an
        orbital separation"""
        _ga = self.ibs.gamma_max - 1.
        _x = _ga * s / self.ibs.s_max_g / self.r_sp
        return self.ibs.s_max_g / _ga * ( (1. + _x)**2 - 1.)**0.5
    
    def calculate(self, to_set_onto_ibs=True, to_return=False):
        ### we use the symmetry of ibs: calculate dNe_de only for 1 horn
        if self.ibs.one_horn:
            s_1d_dim = self.ibs.s * self.r_sp
        if not self.ibs.one_horn:
            s_1d_dim = self.ibs.s[self.ibs.n : 2*self.ibs.n] * self.r_sp
        
        if self.cooling not in self.allowed_coolings:
            print('cooling should be one of these options:')
            print(self.allowed_coolings)
            print('setting cooling = \'no\'... ')
            self.cooling = 'no'
            
        if self.cooling == 'no':
            # For each s, --> injected distribution
            e_vals = np.logspace(np.log10(self.emin), np.log10(self.emax), 977)
            s2d, e2d = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            dNe_de_IBS = ElectronsOnIBS.f_inject(self, s2d, e2d)
            
        if self.cooling in ('stat_apex', 'stat_ibs', 'stat_mimic'):
            # For each s, --> stationary distribution
            e_vals = np.logspace(np.log10(self.emin), np.log10(self.emax), 979)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            if self.cooling == 'stat_mimic':
                eta_fl_new = ElectronsOnIBS.eta_flow_mimic(self, smesh)
                self.eta_a = eta_fl_new
    
            f_inj_se = ElectronsOnIBS.f_inject(self, smesh, emesh)
            edots_se = ElectronsOnIBS.edot(self, smesh, emesh)
            dNe_de_IBS = np.zeros((s_1d_dim.size, e_vals.size))
                    
            for i_s in range(s_1d_dim.size):
                if self.cooling == 'stat_apex':
                    f_inj_av = trapezoid(f_inj_se, s_1d_dim, axis=0) / np.max(s_1d_dim)
                    dNe_de_IBS[i_s, :] = stat_distr(e_vals, f_inj_av, edots_se[0, :])
                if self.cooling in ('stat_ibs', 'stat_mimic'):
                    dNe_de_IBS[i_s, :] = stat_distr(e_vals, f_inj_se[i_s, :], edots_se[i_s, :])

        if self.cooling in ('leak_apex', 'leak_ibs', 'leak_mimic'):
            e_vals = np.logspace(np.log10(self.emin), np.log10(self.emax), 381)
            smesh, emesh = np.meshgrid(s_1d_dim, e_vals, indexing = 'ij')
            if self.cooling == 'stat_mimic':
                eta_fl_new = ElectronsOnIBS.eta_flow_mimic(self, smesh)
                self.eta_a = eta_fl_new
    
            f_inj_se = ElectronsOnIBS.f_inject(self, smesh, emesh)
            edots_se = ElectronsOnIBS.edot(self, smesh, emesh)
            dNe_de_IBS = np.zeros((s_1d_dim.size, e_vals.size))
            ts_leak = ElectronsOnIBS.t_leakage(self, smesh, emesh)
            # f_inj_integr = trapezoid(f_inj_se, e_vals, axis=1)
            # Ntot_s = analyt_adv_Ntot_tomimic(smesh, emesh, f_inj_func = f_inject_func,
            #             f_inj_args = f_args, dorb=r_SP, s_max_g=4, Gamma=Gamma)

            for i_s in range(s_1d_dim.size):
                if self.cooling == 'leak_ibs':
                    # dNe_de_IBS[i_s, :] = Stat_distr_with_leak_ivp(e  = e_vals,
                        # q_func, edot_func, T_func, q_args, edot_args, T_args)
                    dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], Ts = ts_leak[i_s, :],
                        mode = 'analyt')
                if self.cooling == 'leak_apex':
                    f_inj_av = trapezoid(f_inj_se, s_1d_dim, axis=0) / np.max(s_1d_dim)
                    dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_av, Edots = edots_se[0, :], Ts = ts_leak[0, :])
                if self.cooling == 'leak_mimic':
                    dNe_de_IBS[i_s, :] = stat_distr_with_leak(Es = e_vals,
                        Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], 
                        Ts = ts_leak[i_s, :], mode = 'analyt') 
            
        if self.cooling == 'adv':
            edot_func = lambda s_, e_: ElectronsOnIBS.edot(self, s_, e_)
            f_inject_func = lambda s_, e_: ElectronsOnIBS.f_inject(self, s_, e_)
            vel_func = lambda s_: ElectronsOnIBS.vel(self, s=s_)
            
            dNe_de_IBS, e_vals = evolved_e_advection(s_ = s_1d_dim,
                edot_func=edot_func, f_inject_func=f_inject_func, 
                tot_loss_args=(), f_args=(), vel_func=vel_func, v_args=())
            
        # if to_lor_trans, we transfer the spectrum to the co-moving
        # reference frames for each s on ibs
        # if to_lor_trans:
            # dNe_de_IBS_transfered = np.zeros(dNe_de_IBS.shape)
            # for i_s in range(s_1d_dim.size):
                # dNe_de_IBS_transfered[i_s, :] = lor_trans_ug_e_spec(E_lab = e_vals, dN_dE_lab, gamma)
                
        # if there were 2 horns in ibs, we fill the 2nd horn with the values 
        # from the 1st horn
        if not self.ibs.one_horn:
            dNe_de_IBS_2horns = np.zeros(((self.ibs.s).size, e_vals.size ))
            dNe_de_IBS_2horns[:self.ibs.n, :] = dNe_de_IBS[::-1, :] # in reverse order from s=smax to ~0
            dNe_de_IBS_2horns[self.ibs.n : 2*self.ibs.n, :] = dNe_de_IBS
            dNe_de_IBS = dNe_de_IBS_2horns
            
        
        if to_set_onto_ibs:
            self.dNe_de_IBS = dNe_de_IBS
            self.e_vals = e_vals
            e_sed = np.array([ dNe_de_IBS[i_s, :] * e_vals**2 for i_s in range(dNe_de_IBS.shape[0])])
            self.e_sed = e_sed
            # print('dNe_de_IBS and e_vals are set')
        if to_return:
            return dNe_de_IBS, e_vals
        

    def peek(self, ax=None, 
             to_label=True,
            show_many = True,
            **kwargs):
    
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(8, 6))    

        if self.dNe_de_IBS is None or self.e_vals is None:
            raise ValueError("You should call `calculate()` first to set dNe_de_IBS and e_vals")
        
        
        Ne_tot_s = trapezoid(self.dNe_de_IBS, self.e_vals, axis=1)
        e_sed_averageS = trapezoid(self.e_sed[self.ibs.n+1 : 2*self.ibs.n-1, :],
                self.ibs.s[self.ibs.n+1 : 2*self.ibs.n-1], axis=0) / np.max(self.ibs.s)


        ax[0].plot(self.e_vals, e_sed_averageS, label=f'{self.cooling}', **kwargs)
        ax[1].plot(self.ibs.s, Ne_tot_s,  label=f'{self.cooling}', **kwargs)


        if show_many:
            _n = self.ibs.n
            for i_s in (
                        int(_n * (1+1/7)),
                        int(_n*(1+1/2)),
                        int(_n*(1+6/7))
                    ):
                label_s = fr"{self.cooling}, $s = {(self.ibs.s[i_s] / self.ibs.s_max) :.2f} s_\mathrm{{max}}$"
   
                ax[0].plot(self.e_vals, self.e_sed[i_s, :], alpha=0.3,
                           label=label_s, **kwargs)

            
        if to_label:
            ax[0].legend()
            ax[1].legend()
        
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel(r'$E$ [eV]')
        ax[0].set_ylabel(r'$E^2 dN/dE$ [eV]')
        ax[0].set_title(r'$dN/dE$')

        ax[0].set_ylim(np.nanmax(self.e_sed) * 1e-3, np.nanmax(self.e_sed) * 2)

        ax[1].set_xlabel(r'$s$')
        ax[1].set_ylabel(r'$N(s)$')
        ax[1].set_title(r'$N(s)$')    
        
        
        edot_apex = ElectronsOnIBS.edot(self, 0, self.e_vals)
        ax[2].plot(self.e_vals, -self.e_vals / edot_apex)
        ax[2].set_xlabel('e, eV')
        ax[2].set_title('t cooling [s] VS e')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        
        
class NonstatElectronEvol: #!!!
    """
    A class representing the non-stationary evolution of ultrarelativistic
    electrons in one-zone model (this zone = apex of the IBS).
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
        self.n_t = n_t
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
        
        self.dNe_de_IBS = None
        self.e_vals = None
        
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

        if self.to_inject_e == 'ecpl':
            e_part = ecpl(e_, ind=self.p_e, ecut=self.ecut, norm=1.)
        elif self.to_inject_e == 'pl':
            e_part = pl(e_, ind=self.p_e, norm=1)
        else:
            raise ValueError("I don't know this to_inject_theta. It should be pl or ecpl.")
            
            
        ### if we assume that the `emission zone` is the forward hemisphere,
        ### then the NUMBER of e don't change with time. The density changes,
        ### of course
        
        result = e_part * self.norm_e 
        
        if self.to_cut_e:
            mask = (e_ < self.emin) | (e_ > self.emax)
            result = np.where(mask, 0.0, result)
        
        # now normalize the number of injected particles. I do it like this:
        # I ensure that the total number per second: N(s) = \int n(s, E) dE 
        # of electrons is half the Normalization
        N_total = trapezoid(result, e_)
        overall_coef = self.norm_e / 2 / N_total
        return result * overall_coef
    
    def stat_distr_at_time(self, t_, e_=None):
        if e_ is None:
            e_ = self.e_grid
        return stat_distr(Es = e_,  
                    Qs = NonstatElectronEvol.f_inject(self, e_, t_),
                    Edots = NonstatElectronEvol.edot_apex(self, e_, t_)
                             )
    
    

    def calculate(self, to_return=False):
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
        if self.dNe_des is None:
            raise ValueError('you should calculate() first')
        return self.dn_de_spl(t)
    
    def dnstat_de(self, t):
        if self.dNe_des is None:
            raise ValueError('you should calculate() first')
        return self.dnstat_de_spl(t)
    
    
    def n_tot(self, t, emin=None, emax=None):
        if emin is None:
            emin = self.emin
        if emax is None:
            emax = self.emax
        mask = np.logical_and(self.e_c >= emin, self.e_c <= emax)
        return trapezoid(self.dn_de(t)[:, mask], self.e_c[mask], axis=1)

    def nstat_tot(self, t, emin=None, emax=None):
        if emin is None:
            emin = self.emin
        if emax is None:
            emax = self.emax
        mask = np.logical_and(self.e_c >= emin, self.e_c <= emax)
        return trapezoid(self.dnstat_de(t)[:, mask], self.e_c[mask], axis=1)
    
    
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ibsen.orbit import Orbit
    from ibsen.winds import Winds
    from ibsen.ibs import IBS
    import time

    sys_name = 'psrb' 
    orb = Orbit(sys_name = sys_name, n=1003)
    winds_full = Winds(orbit=orb, sys_name = sys_name, alpha=-10/180*pi, incl=23*pi/180,
                f_d=165, f_p=0.1, delta=0.015, np_disk=3, rad_prof='pl', r_trunk=None,
                height_exp=0.25,
                    ns_field_model = 'linear', ns_field_surf = 0.2, ns_r_scale = 1e13,
                    opt_field_model = 'linear', opt_field_surf = 0, opt_r_scale = 1e13,)
    # winds_full.peek(showtime=(-100*DAY, 100*DAY))


    el = NonstatElectronEvol(winds=winds_full, t_start=-100*DAY, t_stop=100*DAY,
                            emin=1e9, emax=1e13, emin_grid=1e8, emax_grid=1e13,
                            p_e=1.7, init_distr='zero', eps_big=5e-3, eps_small=3e-3,
                            n_dec_e=201, dt_min=0.01*DAY, dt_first=None,
                            adaptive_dt=True, eta_a=1e20)
    start = time.time()
    ts, es, ns = el.calculate(to_return=True)
    # e_ = es[1, :]
    e_ = el.e_c
    print(time.time() - start)

    plt.subplot(1, 3, 1)
    print(el.ts.shape)
    print(el.dNe_des.shape)
    print(type(el.ts))
    print(type(el.dNe_des))
    plt.plot(e_, el.dn_de(-70*DAY)*e_**2, color='r', label='-60 days')
    plt.plot(e_, el.stat_distr_at_time(-70*DAY, e_)*e_**2, ls='--', color='r')
    
    plt.plot(e_, el.dn_de(-15*DAY)*e_**2, color='g', label='-15 days')
    plt.plot(e_, el.stat_distr_at_time(-15*DAY, e_)*e_**2, ls='--', color='g')
    
    
    plt.plot(e_, el.dn_de(0)*e_**2, color='k', label='0 days')
    plt.plot(e_, el.stat_distr_at_time(0, e_)*e_**2, ls='--', color='k')

    plt.plot(e_, el.dn_de(15*DAY)*e_**2, color='m', label='20 days')
    plt.plot(e_, el.stat_distr_at_time(15*DAY, e_)*e_**2, ls='--', color='m')

    plt.plot(e_, el.dn_de(70*DAY)*e_**2, color='b', label='70 days')
    plt.plot(e_, el.stat_distr_at_time(70*DAY, e_)*e_**2, ls='--', color='b')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('e, eV')
    plt.ylim(1e45, 5e49)
    plt.legend(fontsize=10)
    plt.title('e-SED')


    plt.subplot(1, 3, 2)
    Ntot = el.n_tot(ts)
    Ntot_stat = el.nstat_tot(ts)
    plt.plot(ts/DAY, Ntot, label='nonstat')
    plt.plot(ts/DAY, Ntot_stat, label='stat')
    
    plt.xlabel('t, days')
    plt.title('total Ne in [emin, emax]')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    dt = ts[1:] - ts[:-1]
    plt.plot(ts[1:]/DAY, dt/DAY)
    plt.title('time step (d)')
    plt.xlabel('t, days')
    plt.yscale('log')
