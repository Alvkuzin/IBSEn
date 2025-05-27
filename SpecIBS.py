import numpy as np
import naima
from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
# from scipy.optimize import brentq, root, fsolve, least_squares, minimize
# from scipy.optimize import curve_fit
from pathlib import Path
from numpy import pi, sin, cos, tan, exp
from joblib import Parallel, delayed
# import multiprocessing
from scipy.interpolate import splev, splrep, interp1d, RegularGridInterpolator
import time
import ElEv
import TransportShock
import Orbit as Obt
import xarray as xr
import Absorbtion
from ShapeIBS import approx_IBS


# start = time.time()
G = 6.67e-8
c_light = 3e10
sigma_b = 5.67e-5
h_planck_red = 1.05e-27
m_e = 9.109e-28
MC2E = m_e * c_light**2

Rsun = 7e10
AU = 1.5e13
DAY = 86400.
# Mopt = 24
# Ropt = 10 * Rsun
# Mx = 1.4
# GM = G * (Mopt + Mx) * 2e33
# P = 1236.724526
# Torb = P * DAY
# a = (Torb**2 * GM / 4 / pi**2)**(1/3)
# e = 0.87
# e = 0
orb_p_psrb = Obt.Get_PSRB_params()

# r_periastron = a * (1 - e)
# D_system = 2.4e3 * 206265 * AU
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

def interplg(x, xdata, ydata, axis=0):
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    spl_ = interp1d(np.log10(xdata), np.log10(ydata), axis=axis)
    return 10**( spl_( np.log10(x) ) )

# very general: just calculating v from Gamma
def beta_from_g(g_vel):
    if isinstance(g_vel, np.ndarray):
        res = np.zeros_like(g_vel)
        cond = (g_vel > 1.0) 
        res[cond] = ((g_vel[cond]-1.0) * (g_vel[cond]+1.0))**0.5 / g_vel[cond]
    else:
        if g_vel > 1.0:
            res =  ((g_vel-1.0) * (g_vel+1.0))**0.5 / g_vel
        else:
            res = 0.0
    return res 

def int_an(g0, sm):
    return sm / (g0 - 1) * (g0**2 - 1)**0.5 

# very general: just calculating v from Gamma
def vel_from_g(g_vel):
    return c_light * beta_from_g(g_vel) 

def Gma(s, sm_g, G_term):
    return 1 + (G_term - 1) * s / sm_g

# just defining ecponential cutoff power law
def ecpl(e, ind, ecut):
    return e**(-ind) * exp(-e / ecut)

def lin_appr(arr1, arr2, x1, x2, xn):
    return (arr1*(xn-x2) + arr2*(x1-xn))/(x1-x2)


# def Theta_inf(beta):
#     to_solve1 = lambda tinf: tinf - tan(tinf) - pi / (1. - beta)
#     th_inf = brentq(to_solve1, pi/2 + 1e-5, pi - 1e-5)
#     return th_inf

# def Theta1_CRW(theta, beta):
#     if theta == 0:
#         return 0
#     else:
#         th1_inf = pi - Theta_inf(beta) 
#         to_solve2 = lambda t1: t1 / tan(t1) - 1. - beta * (theta / tan(theta) - 1)
#         th1 = brentq(to_solve2, 1e-5, th1_inf)
#         return th1
    
def d_boost(Gammas, angle_beta_obs):
    beta_vel = vel_from_g(Gammas)/c_light
    return 1 / Gammas / (1 - beta_vel * cos(angle_beta_obs))


def B_and_u_test(Bx, Bopt, r_SE, r_PE, Topt, Ropt):      # just for testing
    L_spindown, sigma_magn = 8e35 * (Bx / 3e11)**2, 1e-2
    B_puls = (L_spindown * sigma_magn / c_light / r_PE**2)**0.5
    B_star = Bopt * (Ropt / r_SE)
    factor = 2 * (1 - (1 - (Ropt / r_SE)**2)**0.5 )
    u_dens = sigma_b * Topt**4 / c_light * factor
    return B_puls, B_star, u_dens

def LorTrans_B_iso(B_iso, gamma):
    bx, by, bz = B_iso / 3**0.5, B_iso / 3**0.5, B_iso / 3**0.5
    bx_comov = bx
    by_comov, bz_comov = by * gamma, bz * gamma
    return (bx_comov**2 + by_comov**2 + bz_comov**2)**0.5

def LorTrans_ug_iso(ug_iso, gamma): # Relativistic jets... eq. (2.57)
    # delta_doppl = d_boost(gamma, ang_beta_obs)
    return ug_iso * gamma**2 * (3 + beta_from_g(gamma)) / 3.

def transform_to_comoving(E_lab, dN_dE_lab, gamma, E_comov=None, n_mu=101):
    """
    Returns (E_comov, dN_dE_comov), the angle-averaged spectrum in the cloud frame.

    Steps:
      1. Build an interpolator for the lab spectrum (zero outside input range).
      2. Define a grid of cosines mu' in [-1,1].
      3. For each E' in E_comov, compute the Doppler-shifted lab energies
         E = Γ (E' + β p' c mu'), then sample the lab spectrum there,
         weight by the Jacobian J = 1/[Γ(1+β mu')], and integrate over μ'.
         Currently assumes that all particles are ultra-relativistic.
         
    Parameters
    ----------
    E_lab : np.ndarray
        1D array of lab-frame energies (must be sorted ascending).
    dN_dE_lab : np.ndarray
        1D array of dN/dE in lab frame, same shape as E_lab.
    gamma : float
        bulk Lorentz factor Γ of the cloud.
    E_comov : np.ndarray, optional
        optional 1D array of desired comoving energies; if None, will use a 
        grid spanning from min(E_lab) * Gamma * (1-beta) to 
        max(E_lab) * Gamma * (1+beta). The default is None.
    n_mu : int, optional
        number of μ' samples for angle-average (must be odd for symmetry).
        The default is 101.

    Returns
    -------
    E_comov : np.ndarray
        1D array of comoving energies.
    dN_dE_comov : ndarray
        1D array of angle-averaged dN'/dE' in comoving frame.

    """
    # derived boost quantities
    beta_v = beta_from_g(gamma)

    # if user did not supply E_comov, take same dynamic range scaled down by Γ
    if E_comov is None:
        Emi = E_lab.min()
        Ema = E_lab.max()
        Emi_co = Emi * gamma * (1.0 - beta_v)
        Ema_co = Ema * gamma * (1.0 + beta_v)
        needed_len = int(len(E_lab) * np.log10(Ema_co / Emi_co) / 
                         np.log10(Ema / Emi))
        E_comov = np.logspace(np.log10(Emi_co), np.log10(Ema_co), needed_len)

    # set up lab-spectrum interpolator, zero outside
    lab_interp = interp1d(
        E_lab, dN_dE_lab,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # cosθ' grid for angle-averaging
    u_even = np.linspace(0, 1, int(n_mu * gamma))
    mu_prime = np.tanh(gamma * (2*u_even**2 - 1)) # denser grid near mu'=1

    # prepare output array
    dN_dE_comov = np.zeros_like(E_comov)

    # now loop (vectorized over mu')
    # Here we assume E_lab and E_comov are kinetic energies >> rest mass, so p'c≈E'.
    # If you need exact, include rest-mass term.

    # for each E', compute E_lab grid for all mu'
    # shape will be (n_E', n_mu)
    Ep = E_comov[:,None]
    E_shift = gamma * Ep * (1 + beta_v * mu_prime[None,:])

    # Jacobian factor J = 1 / [Γ (1 + β μ')]
    # J = d^3 p' / d^3 p = (E/E')^2 * dE/dE' * dOmega / dOmega'
    J = 1.0 / (gamma * (1.0 + beta_v * mu_prime))[None,:]

    # sample lab spectrum at each shifted energy
    F_lab_at_E = lab_interp(E_shift)

    # integrand = J * F_lab
    integrand = J * F_lab_at_E

    # integrate over μ' and multiply by 2π. Then divide by 4π for averaging
    dN_dE_comov = 2.0 * np.pi * trapezoid(integrand, mu_prime, axis=1) / 4.0 / np.pi #!!!

    return E_comov, dN_dE_comov

# -----------------------------------------------------------------------------
# now we define the real functions for solving the advection equation
# vel_func(s, ...) dn/ds + d(edot(s, e, ...) * n) / ds = f_inject(s, e, ...)
# I'm trying to pass the value s -- in cm, the value e -- in eV,
# everything else in CGS if not stated otherwise in comments
# -----------------------------------------------------------------------------

def vel_func(s, Gamma, smax_g, dorb): # s --- in real units, smax - dimentionless
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
    smax_g_cm = smax_g * dorb
    gammas = Gma(s, smax_g_cm, Gamma)
    return vel_from_g(gammas)

def edot(s, e, B, Topt, Ropt, dist, eta_flow, eta_syn, eta_IC, s_table, r_table, th_table, dorb): 
    s_t = s_table * dorb
    r_t = r_table * dorb
    th_spl = interp1d(x = s_t, y = th_table, fill_value='extrapolate')
    r_spl = interp1d(x = s_t, y = r_t, fill_value='extrapolate')
    th_interp = th_spl(s)
    r_toP = r_spl(s)
    r_toS = (dorb**2 + r_toP**2 - 2 * dorb * r_toP * cos(th_interp))**0.5
    B_on_shock = B * np.min(r_toP) / r_toP #TODO: now it's assumed the field is only from the pulsar... should fix this in future
    eta_IC_on_shock = eta_IC * (np.min(r_toS) / r_toS)**2 # can i do that??????
    return ElEv.total_loss(ee = e, B = B_on_shock, Topt = Topt, Ropt=Ropt,
                           dist = dist, eta_flow = eta_flow, eta_syn = eta_syn,
                           eta_IC = eta_IC_on_shock)

def f_inject(s, e, dorb, s_table, th_table, p, ecut, Norm, emin, emax): 
    s_arr = np.asanyarray(s)
    e_arr = np.asanyarray(e)
    
    thetas_here = np.interp(s_arr, s_table * dorb, th_table)
    thetas_part = sin(thetas_here)
    e_part = ecpl(e_arr, p, ecut)
    result = thetas_part * e_part * Norm
    mask = (e_arr < emin) | (e_arr > emax)
    result = np.where(mask, 0.0, result)
    return result

# -----------------------------------------------------------------------------
# Now, we calculate a spectrum from an IBS in all honesty!
# (1) We initialize the IBS and pre-compute the patamerers like B and u in 
# N_shock points (N_shock \approx 20)
# (2) Using TransportShock.solve_for_n we calculate the electrons 
# spectrum dN/de (s, e). We do it on a fine s_grid (s_grid.size \approx 1000)
# and then evaluate it at the N_grid points of interest on IBS
# (3) Iterating over IBS grids, we calculate dN_ph/dE_ph (s, E_ph):
# (3.1) Lorentz-boots B, u, n(E) to the co-moving frame 
# (3.2) and calculate SED with naima
# (4) Integrate SED(s, E) over s with a weights of delta_doppl^3
# -----------------------------------------------------------------------------
    
def SED_from_IBS(E, B_apex, u_g_apex, Topt, Ropt, r_SP, E0_e, Ecut_e, Ampl_e, p_e,
                 beta_IBS, Gamma, s_max_em, s_max_g, N_shock, bopt2bpuls_ap, 
                 phi, delta_power=3, orb = 'psrb',
                 lorentz_boost = True, simple = False, eta_a = 1e20,
                 cooling='no', abs_photoel = False, abs_gg = False, Nh_tbabs = 0.8,
                 nu_los_ggabs = 2.3, t_ggabs = 10 * DAY, syn_only = False):
    
    # phi is an angle between LoS and shock front X axis
    r_SE = r_SP / (1 + beta_IBS**0.5)
    
    # -------------------------------------------------------------------------
    # (1) initialize shock (don't forget that it's just one horn, not both)
    # -------------------------------------------------------------------------

    xs, ys, thetas, rs, ss, th1s, r2opt, th_tan_up, th_inf, xe = approx_IBS(b = np.abs(-np.log10(beta_IBS)),
                    Na = N_shock, s_max = s_max_em, full_output=True)
    # th_tan_up = np.arctan(np.gradient(ys, xs, edge_order=2))
    th_tan_low = - th_tan_up
    Gammas = Gma(ss, s_max_g, Gamma)
    r_ = bopt2bpuls_ap / (1 + bopt2bpuls_ap)
    b_ = (1 - r_) * xe / rs + r_ * (1 - xe) / r2opt 
    u_ = (1-xe)**2 / r2opt**2 
    n_ = (1 - cos(thetas)) / 2
    
    # -------------------------------------------------------------------------
    # (2) calculate dN_e/de (s, e)
    # -------------------------------------------------------------------------
    
    emin, emax = 1e9, 5.1e14
    ecut = Ecut_e * 1e12
    # (B, Topt, dist, eta_flow, eta_syn, eta_IC, ss_tab, rs_tab, thetas_tab, r_SP)
    tot_loss_args = (B_apex, Topt, Ropt, r_SE, eta_a, 1, 1, ss, rs, thetas, r_SP)
    f_args = (r_SP, ss, thetas, p_e, ecut, Ampl_e, emin, emax)
    # print('start 2')
    # if cooling  s_adv:
        # v_input = None
        # v_args_input = None        

    if cooling == 'adv':
        v_input = vel_func
        v_args_input = (Gamma, s_max_g, r_SP)
        eta_args = None
    else:
        eta_args = (s_max_g, Gamma, r_SP)
        v_input = None
        v_args_input = None
        
    dNe_de_IBS, e_vals = ElEv.evolved_e(cooling=cooling, r_SP=r_SP, ss=ss,
    rs=rs, thetas=thetas, edot_func=edot, f_inject_func=f_inject,
    tot_loss_args=tot_loss_args, f_args=f_args, vel_func=v_input,
    v_args = v_args_input, eta_flow_args=eta_args)

    
    # -------------------------------------------------------------------------
    # (3) for each segment of IBS, we calulate a spectrum and put it into
    # the 2-dimentional array SED_s_E. The E_ph is always the same and it is 
    # E but extended: min(E) / max(delta) < E_ph < max(E) * max(delta)
    # if simple=True, then the simple apex-spectrum rescaling and integration
    # is performed, without calculating a spec for each s-segment
    # -------------------------------------------------------------------------
        
    d_max =  d_boost(Gamma, 0)
    # print('d max = ', d_max)
    # print('start 3')
    E_ext = np.logspace(np.log10(np.min(E)/d_max/1.1), np.log10(np.max(E)*d_max*1.1),
    int(2 * E.size * np.log10(d_max**2 * 1.21 * np.max(E) / np.min(E)) / np.log10(np.max(E) / np.min(E)) ))
    SED_s_E = np.zeros((ss.size, E_ext.size))
    if simple:
        # average e-spectrum along the shock. It will be used as the e-spectrum
        # in case s_adv = False
        dNe_de_1d = trapezoid(dNe_de_IBS, ss, axis=0) / trapezoid(np.zeros(ss.size)+1, ss)
        ok = np.where(dNe_de_1d > 0)
        e_spec_for_naima = naima.models.TableModel(e_vals[ok]*u.eV, (dNe_de_1d[ok])/u.eV )
        
        # calculating a spectrum in the apex once and then insert it in every
        # s-segment
        E_dim = E_ext * u.eV

        Sync = Synchrotron(e_spec_for_naima, B = B_apex * u.G, nEed=173)
        sed_synchr = Sync.sed(E_dim, distance = 2.4 * u.kpc)
        SED_sy_apex = sed_synchr / sed_unit
        
        if not syn_only:
            seed_ph = ['star', Topt * u.K, u_g_apex * u.erg / u.cm**3]
            IC = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=71)
            sed_IC = IC.sed(E_dim, distance = 2.4 * u.kpc)
            # and putting a total dimentionLESS spec into SED_s_E
            SED_ic_apex = sed_IC / sed_unit
        
    for i_ibs in range(0, ss.size):
        # rescaling B and u_g to the point on an IBS. I do it even in case
        # simple = True, so that I can retrieve the values later in addition 
        # to the SED 
        B_here = B_apex * b_[i_ibs]
        u_g_here = u_g_apex * u_[i_ibs]
        # Calculating B, u_g, and electron spectrum in the frame comoving
        # along the shock with a bulk Lorentz factor of Gammas[i_ibs]
        if lorentz_boost:
            Gamma_here = Gammas[i_ibs]
            B_comov = LorTrans_B_iso(B_here, Gamma_here)
            u_g_comov = LorTrans_ug_iso(u_g_here, Gamma_here)
            if not simple:
                e_vals_comov, dN_de_comov = transform_to_comoving(e_vals, dNe_de_IBS[i_ibs, :], Gamma_here,
                                        n_mu=101)
        else:
            B_comov = B_here
            u_g_comov = u_g_here
            e_vals_comov, dN_de_comov = e_vals, dNe_de_IBS[i_ibs, :]
        
        if not simple:
            # Preparing e_spec so in can be fed to Naima
            if np.max(dN_de_comov) == 0:
                continue
            ok = np.where(dN_de_comov > 0)
            e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
            E_dim = E_ext * u.eV

            # calculating an actual spectrum
            Sync = Synchrotron(e_spec_for_naima, B = B_comov * u.G)
            sed_syncr = Sync.sed(E_dim, distance = 2.4 * u.kpc)
            sed_tot = sed_syncr 

            if not syn_only:
                seed_ph = ['star', Topt * u.K, u_g_comov * u.erg / u.cm**3]
                IC = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=101)
                sed_IC = IC.sed(E_dim, distance = 2.4 * u.kpc)
                sed_tot += sed_IC
            # and putting a total dimentionLESS spec into SED_s_E
            SED_s_E[i_ibs, :] = sed_tot / sed_unit
        if simple:
            # and putting a total dimentionLESS apex spec into SED_s_E
            sy_here = SED_sy_apex * (B_comov / B_apex)**2 * n_[i_ibs] 
            sed_tot = sy_here
            SED_s_E[i_ibs, :] = sy_here  

            if not syn_only:
                ic_here = SED_ic_apex * (u_g_comov / u_g_apex) * n_[i_ibs]
                sed_tot += ic_here
                
            SED_s_E[i_ibs, :] = sed_tot
            
    
    # -------------------------------------------------------------------------
    # (4) finally, we integrate over IBS the value
    # delta_doppl^delta_power * SED(E_ph / delta_doppl)
    # -------------------------------------------------------------------------

    # It's maybe not the best idea to use RGI here, seems like it sometimes
    # interpolates too rough. But I haven't figured out a way to use interp1d 
    RGI = RegularGridInterpolator((ss, E_ext), SED_s_E, bounds_error=False,
    fill_value=0., method = 'linear')
    
    ang_up = pi - phi + th_tan_up # shape (ss.size, )
    ang_low = pi - phi + th_tan_low # shape (ss.size, )
    
    deltas_up = d_boost(Gammas, ang_up)
    deltas_low = d_boost(Gammas, ang_low)
    
    E_new_up = (E[None, :] / deltas_up[:, None])
    E_new_low = (E[None, :] / deltas_low[:, None])
    pts_up = np.column_stack([
        np.repeat(ss, E.size),  # shape (M*P_sel,)
        E_new_up.ravel()        # shape (M*P_sel,)
    ])
    pts_low = np.column_stack([
        np.repeat(ss, E.size), # shape (M*P_sel,)
        E_new_low.ravel()      # shape (M*P_sel,)
    ])
    
    vals_up = RGI(pts_up) # shape (M*P_sel,)
    vals_low = RGI(pts_low)               
    
    I_interp_up = vals_up.reshape(ss.size, E.size)    # → (M, P_sel)
    I_interp_low = vals_low.reshape(ss.size, E.size)    # → (M, P_sel)
    # if simple:
    #     div = trapezoid(np.zeros(ss.size)+1, ss)
    # if not simple:
    div = trapezoid(np.zeros(ss.size)+1, ss)
    SED_E_up = trapezoid(I_interp_up * deltas_up[:, None]**delta_power, ss, axis=0) /  div
    SED_E_low = trapezoid(I_interp_low * deltas_low[:, None]**delta_power, ss, axis=0) /  div
    SED_E = SED_E_up + SED_E_low
    if abs_photoel:
        SED_E = SED_E * Absorbtion.abs_photoel(E * 1.6e-12, Nh = Nh_tbabs)
    if abs_gg:
        SED_E = SED_E * Absorbtion.abs_gg_tab(E * 1.6e-12,
                            nu_los = nu_los_ggabs, t = t_ggabs, Teff=Topt)
        
    return SED_E

def dummy_LC(t, Bx, Bopt, Topt, E0_e, Ecut_e, Ampl_e, p_e,
                 beta_IBS, Gamma, s_max_em, s_max_g, orb, delta_power=3, nu_los = 2.4,
                 lorentz_boost = True, simple = False, eta_a=1e20,
                 cooling='no', abs_photoel = False, abs_gg = False, Nh_tbabs = 0.8,
                 nu_los_ggabs = 2.3, t_ggabs = 10 * DAY):
    if orb == 'psrb':    
        T_orb, exc, Mtot, Ropt = [orb_p_psrb[key] for key in ('T', 'e', 'M', 'Ropt')]
    elif orb == 'circ':    
        T_orb, exc, Mtot, Ropt = [orb_p_psrb[key] for key in ('T', 'e', 'M', 'Ropt')]
        exc = 0
    else:
        T_orb, exc, Mtot, Ropt = [orb[key] for key in ('T', 'e', 'M', 'Ropt')]
        
    r_SP = Obt.Radius(t, Torb=T_orb, e=exc, Mtot=Mtot)
    r_SE = r_SP / (1 + beta_IBS**0.5)
    phi = Obt.True_an(t, Torb=T_orb, e=exc)
    phi_LoS_ShockX = pi - (nu_los - phi)
    Es = np.logspace(np.log10(300), 4, 59) # 0.3-10 keV only
    Bp, Bo, u = B_and_u_test(Bx, Bopt, r_SE, r_PE = r_SP - r_SE, Topt = Topt, Ropt=Ropt)
    Bp = 1.
    sed = SED_from_IBS(E = Es, B_apex = Bp, u_g_apex = u, Topt = Topt, Ropt=Ropt,
        r_SP = r_SP, E0_e = E0_e, Ecut_e = Ecut_e, Ampl_e = Ampl_e, p_e = p_e,
        beta_IBS = beta_IBS, Gamma = Gamma, s_max_em = s_max_em, s_max_g = s_max_g, N_shock = 10, 
        bopt2bpuls_ap = Bo/Bp, phi = phi_LoS_ShockX, delta_power=3,
        lorentz_boost = lorentz_boost, simple = simple, eta_a=eta_a,
        cooling=cooling, abs_photoel=abs_photoel, abs_gg = abs_gg, Nh_tbabs =Nh_tbabs,
        nu_los_ggabs = nu_los, t_ggabs = t, syn_only=True)
    flux = trapezoid(sed/Es, Es)
    return flux
    
if __name__=='__main__':
    # t = 15 * DAY
    # nu_true = Orb.True_an(t)
    # LoS_to_orb = 2.3
    # LoS_to_shock = - (pi - (LoS_to_orb - nu_true))
    pe = 1.7
    # r_SP = Orb.Radius(t)
    beta_eff = 0.1
    # r_SE = r_SP / (1 + beta_eff**0.5)
    Topt = 3.3e4
    Bx = 5e11
    Bopt = 0
    Gamma = 2.0
    
    # phi = pi/2
    # Bp_apex, Bs_apex, uapex = B_and_u_test(Bx = Bx, Bopt = Bopt, 
    #                                        r_SE = r_SE, r_PE = (r_SP - r_SE),
    #                             T_opt = Topt)
    # B_field =  Bp_apex+Bs_apex
    smax = 3
    # Nibs = 3
    # Es = np.logspace(1, 14, 2003)
    # cond_syn = (Es < 1e17)
    # E0_e = 1; Ecut_e = 5; Ampl_e = 1e25
    # eta_a = 1
    # sed = SED_from_IBS(E = Es, B_apex = B_field,
    #                       u_g_apex = uapex, Topt = Topt, 
    #                    r_SP = r_SP, E0_e = E0_e, Ecut_e = Ecut_e, Ampl_e = Ampl_e,
    #                    p_e = pe,  beta_IBS = beta_eff, Gamma = Gamma, s_max = smax,
    #                    N_shock = Nibs, bopt2bpuls_ap = Bs_apex/Bp_apex, phi = LoS_to_shock,
    #                    s_adv = False, lorentz_boost=True, simple = False, eta_a = eta_a)
    
    # # sed = sed[Es>1e6]
    # # Es = Es[Es>1e6]
    # norm = trapezoid(sed[cond_syn]/Es[cond_syn]**2, Es[cond_syn])
    # # norm = np.max(sed[cond_syn])
    # # norm = 1
    # plt.plot(Es, sed/norm, label = 'new')
    
    # sed_s, sed_sy, sed_ic = SED_old_PSRB(E = Es, B_puls = Bp_apex, B_star = Bs_apex,
    #                     e_density = uapex, r_SE = r_SE, r_SP = r_SP, E0_e=E0_e,
    #                     Ecut_e = Ecut_e, Ampl_e = Ampl_e, p_e = pe, T_opt = Topt,
    #                     LoS_to_IBS = LoS_to_shock, if_boost = True,
    #                     Gamma = Gamma, delta_power=3, 
    #                      if_MF_boost=False, beta_ecpl=1, cooling=True, eta_flow=eta_a,
    #                      eta_syn=1, eta_IC=1)
    
    # # sed_s, sed_sy, sed_ic = SED_old_PSRB(Es, Bp_apex, Bs_apex,
    # #                 uapex, r_SE, r_SP, E0_e, Ecut_e,
    # #                      Ampl_e, pe, Topt, LoS_to_IBS = LoS_to_shock, 
    # #                      if_boost = True, Gamma = Gamma, delta_power=3, 
    # #                      if_MF_boost=False, beta_ecpl=1, cooling=True,  eta_flow=1e20,
    # #                      eta_syn=1, eta_IC=1)
    # sed_s, sed_sy, sed_ic = [arr / sed_unit for arr in (sed_s, sed_sy, sed_ic)]
    # # sed = sed[Es>1e6]
    # # Es = Es[Es>1e6]
    # norm_s = trapezoid(sed_s[cond_syn]/Es[cond_syn]**2, Es[cond_syn])
    # # norm_s = np.max(sed_s[cond_syn])
    # # norm = 1
    # plt.plot(Es, sed_s/norm_s, label = 'old')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    P = orb_p_psrb['P']
    tplot = np.linspace(-0.15*P, P*0.11, 61) * DAY
    f = np.zeros(tplot.size)
    f_sim = np.zeros(tplot.size)
    start = time.time()
    # for i in range(tplot.size):
    # def func_honest(i):
    #     f_ = dummy_LC(tplot[i], Bx, Bopt, Topt, E0_e=1, Ecut_e=5, Ampl_e=1, p_e=pe,
    #                     beta=beta, Gamma=Gamma, s_max=3)   
    #     return f_
    # f= Parallel(n_jobs=10)(delayed(func_honest)(i) for i in range(0, len(tplot)))
    # res=np.array(f)
    # plt.plot(tplot/DAY, f/np.max(f), label = 'full calculation')
    # print('honest method took ', time.time() - start)
    # for adv, eta in zip ([False, False], [1e20, 1]):
    # for adv, eta in zip ([False, False], [1e20, 1]):
    eta_an = int_an(Gamma, smax)
    # for eta in (1e20, 100, 10, eta_an, 1, 0.1, 0.01, 1e-20):
    # for eta in (1e20,  eta_an, 1, 1e-20):     
    for eta in ( 6, ):     

        start = time.time()
        # for i in range(tplot.size):
        cool = 'stat_ibs'
        if eta == 6:
            cool = 'stat_mimic'
        def func_simple(i):
            f_sim_ = dummy_LC(tplot[i], Bx, Bopt, Topt, E0_e=1, Ecut_e=1, Ampl_e=1, p_e=pe,
                            beta_IBS=beta_eff, Gamma=Gamma, s_max_em='bow',
                            s_max_g=4., simple=False, orb='circ',
                            eta_a = eta, lorentz_boost=True, cooling='no')   
            return f_sim_
        f_sim = Parallel(n_jobs=10)(delayed(func_simple)(i) for i in range(0, len(tplot)))
        f_sim=np.array(f_sim)
        # label = 'simple calculation'
        label = f'no adv, eta = {eta}'
        plt.plot(tplot/DAY, f_sim/(f_sim[np.argmin(np.abs(tplot))]), label = label, ls='--')
        print('full method took ', time.time() - start)
        
    # f_sim = np.zeros(tplot.size)    
    # def func_simple(i):
    # # for i in range(tplot.size):
    #     # print(i)
    #     f_sim_ = dummy_LC(tplot[i], Bx, Bopt, Topt, E0_e=1, Ecut_e=5, Ampl_e=1, p_e=pe,
    #                     beta_IBS=beta_eff, Gamma=Gamma, s_max_g=3., s_max_em = 'bow',
    #                     simple=False, cooling='adv',
    #                     eta_a = 1e20)   
    #     # f_sim[i] = f_sim_
    #     return f_sim_
    # f_sim = Parallel(n_jobs=10)(delayed(func_simple)(i) for i in range(0, len(tplot)))
    # f_sim=np.array(f_sim)
    # # label = 'simple calculation'
    # label = 'advection'
    # plt.plot(tplot/DAY, f_sim//(f_sim[np.argmin(np.abs(tplot))]), label = label, color='k', ls='--')
    # print('full method took ', time.time() - start)
    
    # plt.legend()
    # plt.show()
    
    # start = time.time()
    # tplot = np.linspace(-200, 100, 100) * DAY
    # # tplot = 10 * DAY
    # nu_los = 2.3
    # Teff = 3.1e4
    # for tpl_ in tplot:
    #     E = np.logspace(9, 13, 500)*1.6e-12 # erg
    #     tau = Absorbtion.abs_gg_tab(E, 2.33, tpl_, 3.1e4)
    #     # plt.scatter(tplot / DAY, tau, s=1)
    # plt.scatter(E/m_e, tau, s=1)
    # print(time.time() - start)
    # plt.xscale('log')







# spec_energies = np.logspace(8, 14.5, 1000)
# ECPL = My_e_evol.Evolved_ECPL(spec_energies=spec_energies, logNorm = 25,
#                     E0_e = 1, Ecut_e=5, p_e=pe, beta=1,
#                     B= B_field, Topt=Topt, dist=r_SE, eta_flow=0.1,
#                     eta_syn=1, eta_IC=1)
# Sync = Synchrotron(ECPL, B = B_field * u.G)
# # psi_scatter = LoS_to_orb - Mean_motion()
# # seed_ph = ['star', T_opt * u.K, e_density * u.erg / u.cm**3,
# #            psi_scatter * u.rad]
# seed_ph = ['star', Topt * u.K, uapex * u.erg / u.cm**3] #!!!

# IC = InverseCompton(ECPL, seed_photon_fields = [seed_ph])
# sed_syncr = Sync.sed(Es*u.eV, distance = 2.4 * u.kpc)
# sed_IC = IC.sed(Es*u.eV, distance = 2.4 * u.kpc)
# SED_splSY = splrep(x = Es, y = sed_syncr/sed_unit)
# SED_splIC = splrep(x = Es, y = sed_IC/sed_unit)
# SED_boostSY = SED_boosted(nus = Es,
#         SED_spline = SED_splSY, beta = beta, Gamma = Gamma,
#         s_max = 4, phi = phi, bopt2bpuls_ap = Bs_apex/Bp_apex,
#         delta_power = 3, if_MF_boost = False,
#         what_to_boost = 'Syn')
# SED_boostIC = SED_boosted(nus = Es,
#         SED_spline = SED_splIC, beta = beta, Gamma = Gamma,
#         s_max = 4, phi = phi, bopt2bpuls_ap = Bs_apex/Bp_apex,
#         delta_power = 3, if_MF_boost = False,
#         what_to_boost = 'IC')
# SED_boost = SED_boostSY + SED_boostIC
# # norm1 = trapezoid(SED_boost[cond_syn]/Es[cond_syn]**2, Es[cond_syn])
# norm1 = np.max(SED_boost[cond_syn])
# # norm1 = 1
# plt.plot(Es, SED_boost/norm1, ls='--')
# plt.xscale('log')
# plt.yscale('log')

    
'''
def F_test(nu, ind):
    return nu**(-ind)
    
        
fig1, ax1 = plt.subplots()

s_max = 4
diff = []
diff_anal = []
# Gamma_term = 1.1
tplot = np.linspace(0, P, 1500) * DAY
# tplot = np.linspace(-150, 170, 150) * DAY
phi_offset = True_an(60*DAY)
print(phi_offset)
phi_orb = True_an(tplot)
phi_obs = -(pi - (phi_offset - phi_orb))
# phi_obs = np.linspace(0, 2*pi, 100)
nu_prime = np.logspace(1, 5, 1000)
f_prime = F_test(nu_prime, 1.5)
f_spl = splrep(x = nu_prime, y = f_prime)
nu = np.logspace(2.5, 3.51, 50)

### ax_inset = inset_axes(ax1, width="35%", height="25%", loc="upper left")
# ax_inset = fig1.add_axes([0.18, 0.5, 0.3, 0.3])
x_zoom_start, x_zoom_end = -50, 150
# y_zoom_start, y_zoom_end = 0.42, 2.27
y_zoom_start, y_zoom_end = 0.6, 3.2

rect = Rectangle(
    (x_zoom_start, y_zoom_start),  # Bottom-left corner
    x_zoom_end - x_zoom_start,    # Width
    y_zoom_end - y_zoom_start,    # Height
    linewidth=1.5,
    edgecolor="black",
    linestyle="--",
    facecolor='none',
    # alpha=1
)
# ax1.add_patch(rect)
colors = plt.cm.seismic(np.linspace(0, 1, 6))
# plt.rc('axes', prop_cycle=cycler('color', colors))
i_c = 0
for Gamma_term in (1,  1.03,  1.1, 1.2, 1.3):
# for Gamma_term in (1, 1.03, 1.1, 1.2, 1.4, 1.8):
# for Gamma_term in (1, 2, 3, 4, 5):
    color = colors[i_c]
    i_c += 1  
    flux = np.zeros(phi_obs.size)
    # color = [1-random()**0.7, random()**1.5, random()**0.7]

    def to_parall(iphi):
        beta=0.1
        phi = phi_obs[iphi]
        ftot = SED_boosted(nu, f_spl, beta, Gamma_term, s_max, phi, 0)
        return trapezoid(ftot/nu, nu)
        # return ftot
    
    res = Parallel(n_jobs=8)(delayed(to_parall)(i) for i in range(0, len(flux)))
    flux = np.array(res)
        
    diff.append(np.max(flux) / np.min(flux))
    beta_term = (1 - 1/Gamma_term**2)**0.5
    diff_anal.append(((1 + beta_term) / (1 - beta_term))**2)    
    radii = Radius(tplot)
    flux = flux / (radii**2)
    # ax1.plot(phi_obs/2/pi, flux, label = r'$\Gamma = %s$'%(Gamma_term), color=color)
    # ax1.plot(phi_obs/2/pi+1, flux, color=color)    
    ax1.plot(tplot/DAY, flux/flux[-1], color=color, label = r'$\Gamma = %s$'%(Gamma_term))
    ax1.plot(tplot/DAY-P, flux/flux[-1], color=color)
    ax1.plot(tplot/DAY-2*P, flux/flux[-1], color=color, )
    ax1.plot(tplot/DAY+P, flux/flux[-1], color=color)
    ax1.set_xlim(-1.1*P, 1.1*P)
    # ax_inset.plot(tplot/DAY, flux/flux[-1], color=color,)
    # ax_inset.plot(tplot/DAY-P, flux/flux[-1], color=color)
    # ax_inset.set_xlim(x_zoom_start, x_zoom_end)
    # ax_inset.set_ylim(y_zoom_start, y_zoom_end)
    # ax_inset.tick_params(direction = 'in', which = 'both')
    # mark_inset(ax1, ax_inset, loc1=2, loc2=4, color="none", linestyle="--")
'''
    


# ax1.legend(loc='upper right')
# ax1.set_xlabel('T, days', fontsize=14)
# ax1.set_ylabel('F/F(periastron)', fontsize=14)
# ax1.axvline(x=0, color='k', alpha=0.2)
# ax1.axvline(x=P, color='k', alpha=0.2)
# ax1.axvline(x=-P, color='k', alpha=0.2)
# # ax_inset.axvline(x=0, color='k', alpha=0.2)
# plt.show()


# print(diff)
# print(diff_anal)
# print(time.time() - start)

    
# ax1.scatter(0, 0, color='r')
# ax1.scatter(1, 0, color='b')
# ax.scatter(0, 0, color='r')
# ax.scatter(0, 1, color='b')
# ax1.legend()
# ax.legend()

# fig1.show()
# fig.show()

# colors = plt.cm.seismic(np.linspace(0, 1, 3))
# plt.rc('axes', prop_cycle=cycler('color', colors))
# colors = {'0.01': 'blue', '0.3': 'orange'}
# fig, ax = plt.subplots()
# for b in (0.01, 0.3):
#     xs, ys, thetas, rs, ss = approx_IBS(-np.log10(b), 300, 4., full_output=False)
#     ax.plot(xs, ys,
#                 label = fr'$\beta = {b}$', color=colors[str(b)])
#     ax.plot(xs, -ys, color=colors[str(b)])
    
#     xs, ys, thetas, rs, ss = approx_IBS(-np.log10(b), 100, 'bow', full_output=False)
#     ax.plot(xs, ys,
#                 label = fr'bow, $\beta = {b}$', color=colors[str(b)], lw=4)
#     ax.plot(xs, -ys, color=colors[str(b)], lw=4)
    
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
# ax.axhline(y=0, color='k', ls='--')
# ax.scatter(0, 0, color='r')
# ax.scatter(1, 0, color='b')
# ax.legend() 
# plt.show()
#     # ax.set_title('b = %s' % b)