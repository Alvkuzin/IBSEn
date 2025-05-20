import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sin, cos, exp
import naima
from scipy.special import expn
import astropy.units as u
# from scipy import integrate
from scipy.integrate import trapezoid, quad, cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator
from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import time
from joblib import Parallel, delayed

# from cycler import cycler
# from TransportShock import Denys_solver
from scipy.optimize import curve_fit
import TransportShock
import xarray as xr
from pathlib import Path

G = 6.67e-8
c_light = 3e10
sigma_b = 5.67e-5
h_planck_red = 1.05e-27
m_e = 9.11e-28
re = 2.81794e-13 #cm ; classical electron radius
erg_to_eV = 6.24E11 # 1 erg = 6.24E11 eV
k_boltz = 1.38e-16
# print(m_e * c_light**2 * erg_to_eV)

Rsun = 7e10
AU = 1.5e13
DAY = 86400.
Mopt = 24.
Ropt = 10 * Rsun
Mx = 1.4
GM = G * (Mopt + Mx) * 2e33
P = 1236.724526
Torb = P * DAY
a = (Torb**2 * GM / 4 / pi**2)**(1/3)
e = 0.87

# We don't want to read the file each time functions are called, so we read 
# it once
### --------------------- For shock front shape --------------------------- ###
file_sh = Path(Path.cwd(), 'TabData', "Shocks4.nc")
ds_sh = xr.load_dataset(file_sh)
# ### --------------------- For spectra with adv ---------------------------- ###
# path_adv = Path(Path.cwd(), 'TabData', "E_spec_adv.nc")
# ds_adv = xr.load_dataset(path_adv)

def interplg(x, xdata, ydata, axis=0):
    # asc = np.argsort(xdata)
    # xdata, ydata = xdata[asc], ydata[asc] 
    spl_ = interp1d(np.log10(xdata), np.log10(ydata), axis=axis)
    return 10**( spl_( np.log10(x) ) )

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

def ecpl(e, ind, ecut):
    return e**(-ind) * exp(-e / ecut)

# very general: just calculating v from Gamma
def vel_from_g(g_vel):
    return c_light * beta_from_g(g_vel) 

def Gma(s, sm, G_term):
    return 1 + (G_term - 1) * s / sm


def gfunc(x, a = -0.362, b = 0.826, alpha = 0.682, beta = 1.281): #from 1310.7971
    return (1 +  (a * x**alpha) / (1 + b * x**beta) )**(-1)
    
def Gani(u, c = 6.13):
    cani = c
    return cani * u * np.log(1 + 2.16 * u / cani) / (1 + cani * u / 0.822)

def Giso(u, c = 4.62):
    ciso = c
    return ciso * u * np.log(1 + 0.722 * u / ciso) / (1 + ciso * u / 0.822)

def Giso_full(x):
    return Giso(x, 5.68) * gfunc(x)


r_periastron = a * (1 - e)
# D_system = 2.4e3 * 206265 * AU
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

Lsun=3.9E33#erg/s
Lstar = 5.8E4*Lsun #erg/s

# emax = 5e14
# emin = 6e8

def ICLoss(Ee, Topt, dist): # Ee in eV !!!!!!
    kappa = (Ropt / 2 / dist)**2
    T_me = (k_boltz * Topt / m_e / c_light**2)
    Ee_me = Ee  / erg_to_eV / m_e / c_light**2
    coef = 2 * re**2 * m_e**3 * c_light**4 * kappa * T_me**2 / np.pi / h_planck_red**3 # 1/s
    Edot = coef * m_e * c_light**2 * erg_to_eV * Giso_full(4 * T_me * Ee_me) #[eV/s]
    return -Edot #[eV/s]

def SyncLoss(ee, B):
    '''Synchrotron losses, dE/dt. Bis in G, electron energy -- eV '''
    # return -4e5 * B**2 * (ee/1e10)**2 #eV/s ???
    return -2.5e5 * B**2 * (ee/1e10)**2 #eV/s ???

    
def t_adiab(dist, eta_flow):
    return eta_flow * dist / c_light

def ecpl_test(e, ind, ecut):
    return e**(-ind) * exp(-e / ecut)

def total_loss(ee, B, Topt, dist, eta_flow, eta_syn, eta_IC):
    # if isinstance(eta_flow, np.ndarray)
    eta_flow_max = np.max(np.asarray(eta_flow))
    if eta_flow_max < 1e10:
        return eta_syn * SyncLoss(ee, B) + eta_IC * ICLoss(ee, Topt, dist) - ee / t_adiab(dist, eta_flow)
    else:
        return eta_syn * SyncLoss(ee, B) + eta_IC * ICLoss(ee, Topt, dist)
    

def approx_IBS_test(b, Na, s_max, full_output = False):
    # b = | log10 (beta_eff) |
    # first, find the shape in given b by interpolation
    intpl = ds_sh.interp(abs_logbeta=b)
    # and get its x, y, theta, r, s, theta1, r1 as np.arrays
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = (intpl.x, intpl.y, intpl.theta, intpl.r, intpl.s, intpl.theta1, intpl.r1, )
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [np.array(arr) for arr in (xs_, ys_, ts_, rs_, ss_, t1s_, r1s_)]
    
    ok = np.where(ss_ < s_max)
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [arr[ok] for arr in (xs_, ys_, ts_, rs_, ss_, t1s_, r1s_)]

    intx, ints, intth, intr, intth1, intr1= (interp1d(ys_, xs_), interp1d(ys_, ss_),
            interp1d(ys_, ts_), interp1d(ys_, rs_), interp1d(ys_, t1s_), 
            interp1d(ys_, r1s_))    
    yplot = np.linspace(np.min(ys_)*1.001, np.max(ys_)*0.999, Na)
    xp, tp, rp, sp, t1p, r1p = (intx(yplot), intth(yplot), intr(yplot), ints(yplot),
            intth1(yplot), intr1(yplot))
    yp = yplot
    if full_output:
        return xp, yp, tp, rp, sp, t1p, r1p, intpl.theta_inf.item(), intpl.r_apex.item()
    if not full_output:
        return xp, yp, tp, rp, sp
    
def B_and_u_test(Bx, Bopt, r_SE, r_PE, T_opt):      # just for testing
    L_spindown, sigma_magn = 8e35 * (Bx / 3e11)**2, 1e-2
    B_puls = (L_spindown * sigma_magn / c_light / r_PE**2)**0.5
    B_star = Bopt * (Ropt / r_SE)
    factor = 2 * (1 - (1 - (Ropt / r_SE)**2)**0.5 )
    u_dens = sigma_b * T_opt**4 / c_light * factor
    return B_puls, B_star, u_dens

def Stat_distr(Es, Qs, Edots):
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


def Stat_distr_with_leak(Es, Qs, Edots, Ts, mode = 'ivp'):
    """  
        calculates the stationary spectrum with leakage: the solution of
        d(n * Edot)/dE + n/T(E) = Q(E), which is
            
        n(E) = 1/|Edot| \int_E ^\infty Q(E') K(E, E') dE', where
        K(E, E') = exp( \int_E^E' dE''/Edot(E'') / T(E'') )

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


def vel_func_test(s, Gamma, smax, dorb): # s --- in real units, smax - dimentionless
    """v(s) for testing, same as in SpecIBS. s in [cm], Gamma is a terminal
    lorentz-factor, smax [dimless] is s at which Gamma is reached, dorb is an
    orbital separation"""
    smax_cm = smax * dorb
    gammas = Gma(s, smax_cm, Gamma)
    return vel_from_g(gammas)

def eta_flow_mimic(s, s_max_g, Gamma, dorb):
    """eta_flow(s) for testing. Defined so that eta_flow(s)*dorb/c_light =
    = time of the buulk flow from apex to s = s.
    s in [cm], Gamma is a terminal lorentz-factor, s_max_g [dimless] is s at 
    which Gamma is reached, dorb is an
    orbital separation"""
    ga = Gamma - 1.
    return s_max_g/ga * ((1. + ga*s/s_max_g/dorb)**2 - 1.)**0.5
    

def edot_test(s, e, B, Topt, dist, eta_flow, eta_syn, eta_IC, s_table,
              r_table, th_table, dorb): # s_table and r_table both dimentionless
    s_t = s_table * dorb
    r_t = r_table * dorb
    th_interp = np.interp(s, s_t, th_table)
    r_toP = np.interp(s, s_t, r_t)
    r_toS = (dorb**2 + r_toP**2 - 2 * dorb * r_toP * cos(th_interp))**0.5
    B_on_shock = B * np.min(r_t) / r_toP #TODO: now it's assumed the field is only from the pulsar... should fix this in future
    eta_IC_on_shock = eta_IC * (dist / r_toS)**2 # can i do that??????
    return total_loss(e, B_on_shock, Topt, dist, eta_flow,
                                eta_syn, eta_IC_on_shock)

def f_inject_test(s, e, dorb, s_table, th_table, p, ecut, Norm, emin, emax): 
    s_arr = np.asanyarray(s)
    e_arr = np.asanyarray(e)
    
    thetas_here = np.interp(s_arr, s_table * dorb, th_table)
    thetas_part = sin(thetas_here)
    e_part = ecpl(e_arr, p, ecut)
    result = thetas_part * e_part * Norm
    mask = (e_arr < emin) | (e_arr > emax)
    result = np.where(mask, 0.0, result)
    return result
    # return e_part

def flow_time_to_s(s, s_max_g, Gamma, r_SP):
    ga = Gamma - 1
    return r_SP/c_light * s_max_g/ga * ((1 + ga*s/s_max_g/r_SP)**2 - 1)**0.5

def analyt_adv_Ntot(s, f_inj_integrated, Gamma, r_SP, s_max_g):
    ga = Gamma - 1
    gammas = (1 + ga*s/s_max_g / r_SP)
    betas = beta_from_g(gammas)
    res = np.zeros(f_inj_integrated.size)
    res = cumulative_trapezoid(f_inj_integrated / betas, s, initial=0) / c_light
    return res
    # return f_inj_integrated * r_SP/c_light * s_max_g/ga * ((1 + ga*s/s_max_g)**2  -1)**0.5

def analyt_adv_Ntot_tomimic(s,e, f_inj_func, f_inj_args, dorb, s_max_g, Gamma): 
    # s and e -- meshgrids, but returns 1d-array 
    ga = Gamma - 1
    f_ = f_inj_func(s, e, *f_inj_args)
    # print('mimic f_', f_.shape)
    f_integrated = trapezoid(f_, e, axis=1)
    # print('mimic f_int', f_integrated.shape)
    # s_t = s_table * dorb
    # q_s = sin(th_table)
    x__ = ga*s[:, 0]/s_max_g/dorb
    return f_integrated * r_SP/c_light * s_max_g/ga * (2 * x__ + x__**2)**0.5

def t_leak_test(s,e, f_inj_func, f_inj_args, dorb, s_max_g, Gamma):
    ga = Gamma - 1
    f_ = f_inj_func(s, e, *f_inj_args)
    f_integrated = trapezoid(f_, e, axis=1)
    Ntot = analyt_adv_Ntot(s = s[:, 0], f_inj_integrated = f_integrated, Gamma=Gamma,
                           r_SP=dorb, s_max_g = s_max_g)
    # print(Ntot.shape)
    # print(s.shape)
    # print(s[:, 0].shape)
    ga = Gamma-1
    gs = 1 + ga * s[:, 0]/s_max_g / r_SP
    res = 1/c_light * Ntot / np.gradient(Ntot, s[:, 0], edge_order=2)/beta_from_g(gs)
    res2d = np.zeros(ss.shape)
    # return res[:, None] + np.max(res) / 1e16
    return res[:,None] * s / (1e-17 + s)
    # return dorb/c_light * s / np.max(s)
    # return np.interp(s, s_t, res)
    # s_t = s_table * dorb
    # q_s = sin(th_table)
    # return f_integrated * r_SP/c_light * s_max_g/ga * ((1 + ga*s/s_max_g/dorb)**2  -1)**0.5

# def t_leak_test(s, e, eta, s_table, r1_table, th_table, dorb, s_max_g, Gamma):
#     s_t = s_table * dorb
#     # r1_t = r1_table * dorb
#     # th = np.interp(s, s_t, th_table)
#     q_s = sin(th_table)+1e-3
#     # r_toS = np.interp(s, s_t, r1_t)
#     # return eta * r_toS / c_light
#     # return eta * dorb / c_light * s / np.max(s)
#     # return  s *1e44
#     # return eta * flow_time_to_s(s = s, s_max_g = s_max_g, Gamma = Gamma, r_SP = dorb)
#     ga = Gamma-1
#     gs = 1 + ga * s_table/s_max_g
#     # denom = np.gradient(q_s * (gs**2-1)**0.5, s_table, edge_order=2)
#     # numer = dorb/c_light * q_s * gs
#     # res = numer / denom
#     Ntot = analyt_adv_Ntot(s = s_table, f_inj_integrated = q_s, Gamma=Gamma,
#                            r_SP=dorb, s_max_g = s_max_g)
#     res = dorb/c_light * Ntot / np.gradient(Ntot, s_table, edge_order=2)/beta_from_g(gs)
#     return np.interp(s, s_t, res)
    
def Analyt_stat(energies, Q_func, B, Topt, dist, eta_flow, eta_syn, eta_IC, mode = 'add to Edot'): # LESHA. Blum & Gould, and common sense
    if mode == 'add to Edot':
        return Stat_distr(Es = energies, Qs = Q_func(energies), 
                Edots = total_loss(energies, B, Topt, dist,  eta_flow, eta_syn, eta_IC))
        integral_here = np.zeros(energies.size)+1
        energies_back = energies[::-1]
        Q_func_array = Q_func(energies_back)
        integral_here = -cumulative_trapezoid(Q_func_array, energies_back, initial = 0)[::-1]
        return 1 / np.abs(total_loss(energies, B, Topt, dist,  eta_flow, eta_syn, eta_IC)) * integral_here
    # if mode == 'add to eq':
    #     edot = lambda epr: total_loss(epr, B, Topt, dist, eta_flow, eta_syn, eta_IC) + epr / t_adiab(dist, eta_flow)
    #     int_in_exp = lambda xi, e: quad(func = lambda epr: 1 / t_adiab(dist, eta_flow) / edot(epr),
    #                                  a = e, b = xi)[0]
    #     under_int = lambda xi, e: np.exp(int_in_exp(xi, e)) * Q_func(xi)
    #     res = np.empty(energies.size)
    #     for i in range(energies.size):
    #         res[i] = quad(func = under_int, a = energies[i], b = emax,
    #                       args=energies[i])[0] / np.abs(edot(energies[i]))
    #     return res
    
def ECPL(e_, logNorm, E0_e, Ecut_e, p_e, beta, normalize='overall', emin=1e9,
         emax=5.1e14):
    if normalize == 'overall':
        result = 10**logNorm * (e_ / (E0_e*1e12))**(-p_e) * np.exp(-( (e_ / (Ecut_e*1e12) ) )**beta)
        mask = (e_ < emin) | (e_ > emax)
        result = np.where(mask, 0.0, result)
        return result    
    # if normalize == 'rate':
    #     e_int = np.logspace(np.log10(6e8), np.log10(5e13), 1000)
    #     res = lambda ee: (ee / (E0_e*1e12))**(-p_e) * np.exp(-( (ee / (Ecut_e*1e12) ) )**beta)
    #     norm = 10**logNorm / trapezoid(res(e_int), e_int)
    #     return norm * res(e_)
    
# def tab_adv(r_SP, beta_eff, Gamma, Bap):
#     # Linear interpolation across parameters, no interpolation over x/y
#     res_interp = ds_adv["res"].interp(r_SP=r_SP, beta_eff=beta_eff, Gamma=Gamma,
#                                   B_apex=Bap)
#     return res_interp
    
def evolved_e(cooling, r_SP, ss, rs, thetas, s_adv, edot_func, f_inject_func,
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
        e_vals = np.logspace(np.log10(emin), np.log10(emax), 977)
        smesh, emesh = np.meshgrid(ss*r_SP, e_vals, indexing = 'ij')
        if cooling == 'stat_mimic':
            (B0, Topt, r_SE, eta_fl, eta_sy, eta_ic, ss, rs, thetas, r_SP) = tot_loss_args
            if eta_flow_func == None:
                eta_flow_func = eta_flow_mimic
            eta_fl_new = eta_flow_func(smesh, *eta_flow_args)
            tot_loss_args = (B0, Topt, r_SE, eta_fl_new * eta_fl,
                                 eta_sy, eta_ic, ss, rs, thetas, r_SP)    
        f_inj_se = f_inject_func(smesh, emesh, *f_args)
        edots_se = edot_func(smesh, emesh, *tot_loss_args)
        dNe_de_IBS = np.zeros((ss.size, e_vals.size))
                
            # edots_se = edot_func(smesh, emesh, *tot_loss_args_new)
        for i_s in range(ss.size):
            if cooling == 'stat_apex':
                f_inj_av = trapezoid(f_inj_se, ss, axis=0) / np.max(ss)
                dNe_de_IBS[i_s, :] = Stat_distr(e_vals, f_inj_av, edots_se[0, :])
            if cooling in ('stat_ibs', 'stat_mimic'):
                dNe_de_IBS[i_s, :] = Stat_distr(e_vals, f_inj_se[i_s, :], edots_se[i_s, :])

    if cooling in ('leak_apex', 'leak_ibs', 'leak_mimic'):
        e_vals = np.logspace(np.log10(emin), np.log10(emax), 977)
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
                dNe_de_IBS[i_s, :] = Stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], Ts = ts_leak[i_s, :],
                    mode = 'analyt')
            if cooling == 'leak_apex':
                f_inj_av = trapezoid(f_inj_se, ss, axis=0) / np.max(ss)
                dNe_de_IBS[i_s, :] = Stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_av, Edots = edots_se[0, :], Ts = ts_leak[0, :])
            if cooling == 'leak_mimic':
                dNe_de_IBS[i_s, :] = Stat_distr_with_leak(Es = e_vals,
                    Qs = f_inj_se[i_s, :], Edots = edots_se[i_s, :], 
                    Ts = ts_leak[i_s, :], mode = 'analyt') 
        
        
    if cooling == 'adv':
        # we calculate it on e-grid emin / extend_d < e <  emax * extend_u
        # and hope that zero boundary conditions will be not that important
        
        extend_u = 10; extend_d = 10; 
        # extend_u = 3; extend_d = 3; 
        Ns, Ne = 601, 603
        # Ns, Ne = 201, 203 
        
        Ne_real = int( Ne * np.log10(extend_u * emax / extend_d / emin) / np.log10(emax / emin) )
        
        e_vals = np.logspace(np.log10(emin / extend_d), np.log10(emax * extend_u), Ne_real)
        s_vals = np.linspace(0, np.max(ss), Ns) * r_SP
        dNe_de = TransportShock.solve_for_n(v_func = vel_func, edot_func = edot_func,
                            f_func = f_inject_func,
                            v_args = v_args, 
                            edot_args = tot_loss_args,
                            f_args = f_args,
                            s_grid = s_vals, e_grid = e_vals, 
                            method = 'FDM_cons', bound = 'dir')
        # #### Now we only leave the part of the solution between emin < e < emax: #!!!
        ind_int = np.logical_and(e_vals <= emax, e_vals >= emin)
        e_vals = e_vals[ind_int]
        dNe_de = dNe_de[:, ind_int]
        
        #### and evaluate the values on the IBS grid previously obtained:
        interp_x = interp1d(s_vals, dNe_de, axis=0, kind='linear', fill_value='extrapolate')
        dNe_de_IBS = interp_x(ss * r_SP)
        dNe_de_IBS[dNe_de_IBS <= 0] = np.min(dNe_de_IBS[dNe_de_IBS>0]) / 3.14
            
    return dNe_de_IBS, e_vals

def Evolved_ECPL(spec_energies, logNorm, E0_e, Ecut_e, p_e, beta_ecpl, B, Topt, dist, eta_flow, eta_syn, eta_IC, mode='add to Edot',
                 normalize = 'overall', to_evolve = True):
    # if p_e == 2:
    #     first = spec_energies**(1-p_e)*(E0_e*1e12)**(p_e) * expn(2, spec_energies/(Ecut_e*1e12))        
    #     integral_here = 10**logNorm * first
    #     naima_evolved = 1 / np.abs(total_loss(spec_energies, B, Topt, dist,  eta_flow, eta_syn, eta_IC)) * integral_here
    # else:
    Q_func = lambda e_: ECPL(e_, logNorm, E0_e, Ecut_e, p_e, beta_ecpl, normalize)
    if to_evolve:
        naima_evolved = Analyt_stat(energies = spec_energies, Q_func = Q_func,
                                 B=B, Topt=Topt, dist=dist, eta_flow=eta_flow,
                                 eta_syn=eta_syn, eta_IC=eta_IC, mode=mode)
    if not to_evolve:
        naima_evolved = Q_func(spec_energies)
    if __name__=='__main__':
        return naima_evolved
    else:
        ok = np.where(naima_evolved > 0)
        electrons_spectrum = naima.models.TableModel( spec_energies[ok]*u.eV,
                                            (naima_evolved[ok])/u.eV )
        return electrons_spectrum

    
if __name__=='__main__':
    Topt = 3e4
    p_e = 1.7
    Es = np.logspace(8, 15, 1000)
    Ecut_e = 5 # TeV
    smax= 4

    r_SP = 3e13
    # B0 = 1.7
    Gamma = 1.6
    beta_eff = 1e-2
    start = time.time()
    r_SE = r_SP / (1 + beta_eff**0.5)
    xs, ys, thetas, rs, ss, th1s, r2opt, th_inf, xe = approx_IBS_test(-np.log10(beta_eff), 57, smax, True)
    #dorb, s_table, th_table, p, ecut, Norm, emin, emax
    # inject_ecpl = f_inject_test(1, Es, r_SP, ss, thetas, p_e, ecut = Ecut_e*1e12, Norm = 1,
    #                             emin=1e9, emax=5.1e14)*Es**2
    # where = np.logical_and(Es > 1e9, Es < 5.1e14)
    # Q_func = lambda e_: f_inject_test(1, e_, p_e, Ecut_e*1e12, 1)
    # Q_func_1 = lambda e_: f_inject_test(1, e_, p_e, Ecut_e*1e12/3, 1)
    colors = {0.1: 'r', 1: 'g', 10: 'b'}
    fig, ax = plt.subplots(1, 2)
    for B0 in (1e-1, 1, 10):
        color = colors[B0]
        start = time.time()
        r_SE = r_SP / (1 + beta_eff**0.5)
        
        ####### ---------------- advection -----------------------------#######
        tot_loss_args = (B0, Topt, r_SE, 1e49, 1, 1, ss, rs, thetas, r_SP)
        f_args = (r_SP, ss, thetas, p_e, Ecut_e*1e12, 1, 1e9, 5.1e14)
        v_input = vel_func_test
        v_args_input = (Gamma, smax, r_SP)
        N2d, ee = evolved_e('adv', r_SP, ss, rs, thetas, s_adv=True, edot_func=edot_test,
                              f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
                              vel_func=v_input, v_args=v_args_input)
        
        sed_avg = trapezoid(N2d, ss, axis=0) / np.max(ss) * ee**2
        Ntot = trapezoid(N2d, ee, axis=1)
        ax[0].plot(ee, sed_avg, label = B0, color=color)
        # ax[0].plot(ee, N2d[10, :], label = f'adv {B0} a', ls='-', color=color)
        # ax[0].plot(ee, N2d[20, :], label = f'adv {B0} b', ls='-', color=color)
        ax[1].plot(ss, Ntot, label = B0, color=color)
        
        e_ = np.logspace(9, np.log10(5.1e14), 1001)
        ss2d, ee2d = np.meshgrid(ss*r_SP, e_, indexing='ij')
        finj2d = f_inject_test(ss2d, ee2d, *f_args)
        f_inj_integrated = trapezoid(finj2d, e_, axis=1)
        Ntot_an = analyt_adv_Ntot(s = ss*r_SP, f_inj_integrated=f_inj_integrated,
                        Gamma=Gamma, r_SP=r_SP, s_max_g=smax)
        ax[1].plot(ss, Ntot_an,  ls='-.', label = B0, color='k')
        print('adv', time.time() - start)
        
        # ####### ---------------- w leakage -----------------------------#######
        start = time.time()
        tot_loss_args = (B0, Topt, r_SE, 1e49, 1, 1, ss, rs, thetas, r_SP)
        f_args =  (r_SP, ss, thetas, p_e, Ecut_e*1e12, 1, 1e9, 5.1e14)
        #   est(s,e, f_inj_func, f_inj_args, dorb, s_max_g, Gamma):
        t_args = (f_inject_test, f_args, r_SP, smax, Gamma)
        N2d, ee = evolved_e('leak_mimic', r_SP, ss, rs, thetas, s_adv=True, edot_func=edot_test,
                              f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
                              vel_func=None, v_args=None, t_args=t_args, t_func=t_leak_test)
        
        sed_avg = trapezoid(N2d[2:, :], ss[2:], axis=0) / (ss[-1]-ss[1]) * ee**2
        # for is_ in range(ss.size):
        #     ax[0].plot(ee, N2d[is_, :], label = is_)
        Ntot = trapezoid(N2d, ee, axis=1)
        ax[0].plot(ee, sed_avg, label = f'leak {B0}', ls='--', color=color)
        # ax[0].plot(ee, N2d[10, :], label = f'leak {B0} a', ls='--', color=color)
        # ax[0].plot(ee, N2d[20, :], label = f'leak {B0} b', ls='--', color=color)
        
        ax[1].plot(ss, Ntot, label = f'leak {B0}', ls = '--', color=color)
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        
        print('leak', time.time() - start)
        
        
        # ####### ---------------- simple evol  -------------- #######
        # start = time.time()
        # eta_mimic = eta_flow_mimic(s = ss*r_SP, s_max_g = smax, Gamma=Gamma, dorb=r_SP)
        # # eta_flow_args = (smax, Gamma, r_SP)
        # # print(eta_mimic)  eta_mimic[:, None]
        # tot_loss_args = (B0, Topt, r_SE, eta_mimic[:, None], 1, 1, ss, rs, thetas, r_SP)
        # f_args =  (r_SP, ss, thetas, p_e, Ecut_e*1e12, 1, 1e9, 5.1e14)
        # #   est(s,e, f_inj_func, f_inj_args, dorb, s_max_g, Gamma):
        # # t_args = (f_inject_test, f_args, r_SP, smax, Gamma)
        # N2d, ee = evolved_e('stat_ibs', r_SP, ss, rs, thetas, s_adv=True, edot_func=edot_test,
        #                       f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
        #                       vel_func=None, v_args=None, t_args=None, t_func=None,
        #                       eta_flow_func = None, 
        #                       eta_flow_args = None)
        
        # sed_avg = trapezoid(N2d, ss, axis=0) / np.max(ss) * ee**2
        # Ntot = trapezoid(N2d, ee, axis=1)
        # ax[0].plot(ee, sed_avg, label = f'simple {B0}', ls='-', color=color)
        # ax[1].plot(ss, Ntot, label = f'simple {B0}', ls = '-', color=color)
        # print('stat sim', time.time() - start)

        ####### ---------------- simple evol w mimicking -------------- #######
        # start = time.time()
        # eta_mimic = eta_flow_mimic(s = ss*r_SP, s_max_g = smax, Gamma=Gamma, dorb=r_SP)
        # eta_flow_args = (smax, Gamma, r_SP)
        # # print(eta_mimic)  eta_mimic[:, None]
        # tot_loss_args = (B0, Topt, r_SE, None, 1, 1, ss, rs, thetas, r_SP)
        # f_args =  (r_SP, ss, thetas, p_e, Ecut_e*1e12, 1, 1e9, 5.1e14)
        # #   est(s,e, f_inj_func, f_inj_args, dorb, s_max_g, Gamma):
        # # t_args = (f_inject_test, f_args, r_SP, smax, Gamma)
        # N2d, ee = evolved_e('stat_mimic', r_SP, ss, rs, thetas, s_adv=True, edot_func=edot_test,
        #                       f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
        #                       vel_func=None, v_args=None, t_args=None, t_func=None,
                               
        #                       eta_flow_args = eta_flow_args)
        
        # sed_avg = trapezoid(N2d, ss, axis=0) / np.max(ss) * ee**2
        # Ntot = trapezoid(N2d, ee, axis=1)
        # ax[0].plot(ee, sed_avg, label = f'mimic {B0}', ls=':', color='k')
        # ax[1].plot(ss, Ntot, label = f'mimic {B0}', ls = ':', color='k')
        # print('mimic sim', time.time() - start)       
        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        
        
        for col in (0, 1):
            ax[col].legend()
      
            
            
     
    # print(time.time() - start)
    
    
    # tabulate e-specs on a grid of B, r_SP, beta_eff, Gamma
    # we now will fix the injected spec to p = 1.7, Ecut = 5 TeV with 
    # undefined but same normalization
    # r_SPs = np.linspace(1.5e13, 1e14, 7)
    # beta_effs = np.logspace(-4.5, -0.9, 21)
    # Gammas = np.linspace(1.05, 2, 9)
    # Bapexs = np.logspace(-3, 3, 41)
    # Ne_towrite, Ns_towrite = 201, 27
    # E_towrite = np.logspace(9, np.log10(5.1e14), Ne_towrite)
    # s_dim_towrite = np.linspace(0, smax, Ns_towrite)
    # sswr, EEwr = np.meshgrid(s_dim_towrite, E_towrite, indexing='ij')
    # spec_all = np.zeros((len(r_SPs), len(beta_effs), len(Gammas), len(Bapexs), len(s_dim_towrite), len(E_towrite)))
    
    # for ir, r_SP in enumerate(r_SPs):
    #     for ibet, beta_eff in enumerate(beta_effs):
    #         r_SE = r_SP / (1 + beta_eff**0.5)
    #         for ig, Gamma in enumerate(Gammas):
    #             print(ir, ibet, ig)
    #             def func_par(ib):
    #                 Bap = Bapexs[ib]
    #             # for ib, Bap in enumerate(Bapexs):
    #                 # print(ir, ibet, ig, ib)
    #                 xs, ys, thetas, rs, ss, th1s, r2opt, th_inf, xe = approx_IBS_test(-np.log10(beta_eff), Ns_towrite, smax, True)
    #                 # print(ss.size)
    #                 tot_loss_args = (Bap, Topt, r_SE, 1e49, 1, 1, ss, rs, thetas, r_SP)
    #                 f_args = (p_e, Ecut_e*1e12, 1)
    #                 v_input = vel_func_test
    #                 v_args_input = (Gamma, smax, r_SP)
    #                 N2d, ee = evolved_e(True, r_SP, ss, rs, thetas, s_adv=True, edot_func=edot_test,
    #                                       f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
    #                                       vel_func=v_input, v_args=v_args_input)
    #                 # print(N2d.shape)
    #                 # print(E_towrite)
    #                 # print(ee)
    #                 res = interplg(E_towrite, ee, N2d, axis=1)
    #                 res = res.astype(np.float32)
    #                 return res
    #                 # spec_all[ir, ibet, ig, ib] = res
    #                 # res = np.sin(EEwr*sswr)  * (r_SP * beta_eff * Gamma * Bap)
    #                 # res = res.astype(np.float32)
    #                 # spec_all[ir, ibet, ig, ib] = res
    #             f= Parallel(n_jobs=16)(delayed(func_par)(ib) for ib in range(0, len(Bapexs)))
    #             res=np.array(f)
    #             spec_all[ir, ibet, ig] = res
                    
    # spec_all = spec_all.astype(np.float32)
    # # Wrap in xarray
    # ds = xr.Dataset(
    #     {
    #         "res": (["r_SP", "beta_eff", "Gamma", "B_apex", "E", "s_dimless"], spec_all)
    #     },
    #     coords={
    #         "r_SP": r_SPs,
    #         "beta_eff": beta_effs,
    #         "Gamma": Gammas,
    #         "B_apex": Bapexs,
    #         "x": s_dim_towrite,
    #         "y": E_towrite
    #     }
    # )

    # # Save to NetCDF
    # path = Path(Path.cwd(), 'TabData', "E_spec_adv.nc")
    # ds.to_netcdf(path)
    
    
    
    
    
    
    ## some e-distributions
    # B0 = 1.3
    # dist = 1.8e13

    # eta_a = 1
    # beta_eff = 0.1
    # # r_SE = dist / (1 + beta_eff**0.5)
    # simple_ecpl = ECPL(Es, 1, 1, 5, p_e, 1)*Es**2
    # naima_ecpl = ExponentialCutoffPowerLaw.eval(e=Es*u.eV, amplitude = 1*u.Unit('1 / (cm2 s TeV)'), e_0 = 1*u.eV, alpha = p_e,
    #             e_cutoff = Ecut_e*u.TeV, beta=1)*Es**2

    
    # Gamma = 1.3
    # # colors = ['r', 'g', ]
    # for dist in (3e12, 1e13, 3e13, 1e14):
    #     r_SE = dist / (1 + beta_eff**0.5)
    #     Bp, Bs, u = B_and_u_test(Bx = 3e12, Bopt = 0, r_SE = r_SE,
    #                              r_PE = dist - r_SE, T_opt = Topt)
    #     B0 = Bp+Bs
    #     xs, ys, thetas, rs, ss, th1s, r2opt, th_inf, xe = approx_IBS_test(-np.log10(beta_eff), 121, 3, True)
    #     tot_loss_args = (B0, Topt, r_SE, 1e49, 1, 1, ss, rs, thetas, dist)
    #     f_args = (p_e, Ecut_e*1e12, 1)
    #     v_input = vel_func_test
    #     v_args_input = (Gamma, 3, dist)
    #     N2d, ee = evolved_e(True, dist, ss, rs, thetas, s_adv=True, edot_func=edot_test,
    #                           f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args,
    #                           vel_func=v_input, v_args=v_args_input)
    #     sed1d = trapezoid(N2d, ss, axis=0) / trapezoid(np.zeros(ss.size)+1, ss) * ee**2
    #     plt.plot(ee, sed1d, label = f'r_SP = {dist/1.5e13 :.2e} au' )
        
    #     Nsim = Analyt_stat(energies = ee, Q_func=Q_func, B=B0, Topt=Topt,
    #                      dist=dist, eta_flow=10, eta_syn=1, eta_IC=1) * ee**2
    #     plt.plot(ee, Nsim, label = f'eff r_SP = {dist/1.5e13 :.2e} au', ls='--')
        

    # plt.xlabel('E electrons, eV')
    # plt.ylabel(r'$E^2 \frac{dN_e}{dE}$')
    
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    
    
    # # different eta_a
    # for eta_a in (1e-5, 1e-1, 1, 10, 100):
    #     ev_sed = Analyt_stat(energies = Es, Q_func=Q_func, B=B0, Topt=Topt,
    #                          dist=dist, eta_flow=eta_a, eta_syn=1, eta_IC=1)*Es**2
    #     norm = np.max(ev_sed)
    #     norm = 1
    #     plt.plot(Es, ev_sed/norm, label = rf'$\eta_{{ad}} = {eta_a}$')
    # ev_sed_inf = Analyt_stat(energies = Es, Q_func=Q_func, B=B0, Topt=Topt,
    #                      dist=dist, eta_flow=1e49, eta_syn=1, eta_IC=1)*Es**2
    # norm = np.max(ev_sed)
    # norm = 1
    # plt.plot(Es, ev_sed/norm, label = r'$\eta_{ad} = \infty$', color='b')
    # norm = np.max(Q_func(Es)*Es**2)
    # norm = 1
    # plt.plot(Es, Q_func(Es)*Es**2/norm, c='k', ls='--', label = 'inj Ecut = 5 TeV')
    # norm = np.max(Q_func_1(Es)*Es**2)
    # norm = 1
    # plt.plot(Es, Q_func_1(Es)*Es**2/norm, c='r', ls='--', label = 'inj Ecut = 0.33 * 5 TeV')

    
    # # plt.ylim(1e-5, 1.5)
    # plt.xlabel('E electrons, eV')
    # plt.ylabel(r'$E^2 \frac{dN_e}{dE}$')
    
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    
    
    ### compare different methods of evolving
    # plt.plot(Es, simple_ecpl/np.max(simple_ecpl[where]), alpha=0.2,ls='--', label = 'simple')
    # plt.plot(Es, naima_ecpl/np.max(naima_ecpl[where]),alpha=0.2,ls='--', label = 'naima')
    # plt.plot(Es, inject_ecpl/np.max(inject_ecpl[where]),alpha=0.2,ls='--', label = 'inject')
    
    # ev_ecpl = Evolved_ECPL(Es, 1, 1, Ecut_e, p_e, 1, B0, Topt, r_SE, eta_a, 1, 1)*Es**2
    # an_ev_ecpl = Analyt_stat(Es, Q_func = lambda e_: ECPL(e_, 1, 1, Ecut_e, p_e, 1),
    #                     B=B0, Topt=Topt, dist=r_SE, eta_flow=eta_a, eta_syn=1, eta_IC=1)*Es**2
    # an_ev_inj = Analyt_stat(Es, Q_func = lambda e_: f_inject_test(1, e_, p_e, Ecut_e*1e12, 1),
    #                     B=B0, Topt=Topt, dist=r_SE, eta_flow=eta_a, eta_syn=1, eta_IC=1)*Es**2
    
    # plt.plot(Es, ev_ecpl/np.max(ev_ecpl[where]), label = 'ev ecpl')
    # plt.plot(Es, an_ev_ecpl/np.max(an_ev_ecpl[where]), label = 'an ev ecpl')
    # plt.plot(Es, an_ev_inj/np.max(an_ev_inj[where]), label = 'an ev inj')
    
    
    # xs, ys, thetas, rs, ss, th1s, r2opt, th_inf, xe = approx_IBS_test(-np.log10(beta_eff), 21, 3, True)
    # tot_loss_args = (B0, Topt, r_SE, eta_a, 1, 1, ss, rs, thetas, dist)
    # f_args = (p_e, Ecut_e*1e12, 1)
    # N2d, ee = evolved_e(True, dist, ss, rs, thetas, s_adv=False, edot_func=edot_test,
    #                       f_inject_func=f_inject_test, tot_loss_args=tot_loss_args, f_args=f_args)
    # sed1d = trapezoid(N2d, ss, axis=0) / trapezoid(np.zeros(ss.size)+1, ss)*ee**2
    # whereSed = np.logical_and(ee > 1e9, ee < 5.1e14)
    # plt.plot(ee, sed1d/np.max(sed1d[whereSed]), label = 'sed 1d')
    
    
    # plt.ylim(1e-3, 1.5)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    
    
    
    
    """
    rows = 2
    cols = 2
    fig, ax = plt.subplots(nrows = rows, ncols = cols)
    # # start_time = time.time()
    spec_energies = np.logspace(8, np.log10(emax)-1, 301)
    for row in range(rows):
        for col in range(cols):
            ax_ = ax[row, col]
            if row == 0:
                B0 = 0.1
            if row == 1:
                B0 = 1
            if col == 0:
                dist = 1.5e13
            if col == 1:
                dist = 5e13
    # B0 = 0.5
            Topt = 30000
    # dist = 3e13
            p_e = 2
    # def rhs_for_solveIvp(t, e):
    #     return total_loss(e[0], B0, Topt, dist)
    # def event(t, e):
    #     return e[0] - 2.525e8
    # event.direction = -1
    # event.terminal=True
    # res = solve_ivp(rhs_for_solveIvp, [0, 1e10], y0=(emax,), events=event, 
    #                     atol=1e-15, rtol=1e-10, dense_output=True)
    # t_final = float(res.t_events[0][0])
    # print('Total considered evolution time LESHA: ', t_final)
    # t_plot = np.logspace(-3, np.log10(t_final), 1000)
    # all_en_lesha = res.sol(t_plot)[0]
    # plt.errorbar(t_plot, all_en_lesha, fmt='k-', label='$E_e$ lesha')
    # print('Lesha method takes ---%s--- sec'%(time.time() - start_time))
    # i = 0
            start = time.time()
            # plt.style.use('seaborn')
            # colors = plt.cm.plasma(np.linspace(0, 1, 6))
            # fig, ax  = plt.subplots()
            # colors = plt.cm.seismic(np.linspace(0, 1, 6))
            # plt.rc('axes', prop_cycle=cycler('color', colors))
            res_init = ECPL(spec_energies, 2, E0_e=emin/1e12, Ecut_e=5, p_e=p_e, beta=1, 
                            normalize='rate')
            sed_init = res_init*spec_energies**2
            # plt.plot(spec_energies, sed_init/trapezoid(res_init, spec_energies),
            #                                            color='k', label='Initial spectrum',
            #                                            ls='--')
            def tofit(e, eta):
                res_spec = Evolved_ECPL(e, logNorm = np.log10(3.e32),
                                        E0_e=emin/1e12, Ecut_e=5, p_e=p_e, beta=1,
                                        B=B0, Topt=Topt, dist=dist, eta_IC=1, eta_flow=eta,
                                        eta_syn=1)
                # sed2 = res_spec*e**2 / trapezoid(res_spec, e)
                sed2 = res_spec*e**2# / np.max(res_spec*e**2)
                
                return sed2
            
            for tflow in (1e-10, 1e-1, 1, 10, 100, 1e10):
                sed2 = tofit(spec_energies, tflow)
                ax_.plot(spec_energies, sed2,
                         label = r'$\eta = %s$'%(tflow,))
            # for t_evol in (1e2, 3e2, 1e3, 1e4, 1e5):
            # # for t_evol in (3e4,):  
            #     if t_evol <= 1e4:
            #         parall = True
            #     else:
            #         parall = True
            #     eD, sedD = Denys_solver(edot_func = total_loss, Q_func = ECPL,
            #                               t_evol = t_evol, edot_args = (B0, Topt, dist, 1e10, 1, 1),
            #                               Q_args = (np.log10(3.e32), emin/1e12, 5, p_e, 1,  'rate'),
            #                               step_shortest_cool_time=1e-3,
            #                               emin=emin, emax=emax/10,
                                          
            #                               test_energies = spec_energies, parall = parall)
            #     # sed_norm = sedD/trapezoid(sedD/eD**2, eD)
            #     # sed_norm = sedD/np.max(sedD)
            #     sed_norm = sedD
            #     ax_.plot(eD, sed_norm,
            #              label = rf'$t = {t_evol} \rightarrow \eta_{{eff}} = {t_evol*c_light/dist}$')
            # spec_e_stat = np.logspace(np.log10(emin), np.log10(emax), 1000)
            # res_spec1 = Evolved_ECPL(spec_e_stat, logNorm = np.log10(3.e32),
            #                             E0_e=emin/1e12, Ecut_e=5, p_e=p_e, beta=1,
            #                             B=B0, Topt=Topt, dist=dist, eta_IC=1, eta_flow=1,
            #                             eta_syn=1, normalize = 'rate')
            # # norm1 = trapezoid(res_spec1, spec_e_stat)
            # # norm1 = np.max(res_spec1)
            # norm1 = 1
            # sed1 = res_spec1 / norm1 * spec_e_stat**2
            # ax_.plot(spec_e_stat, sed1, ls = '--', color='r', label = r'evolved, $\eta = 1$',
            #          alpha = 0.3)
            # res_spec2 = Evolved_ECPL(spec_e_stat, logNorm = np.log10(3.e32),
            #                             E0_e=emin/1e12, Ecut_e=5, p_e=p_e, beta=1,
            #                             B=B0, Topt=Topt, dist=dist, eta_IC=1, eta_flow=1e10,
            #                             eta_syn=1, normalize = 'rate')
            # # norm2 = trapezoid(res_spec2, spec_e_stat)
            # # norm2 = np.max(res_spec2)
            # norm2 = 1
            # sed2 = res_spec2 / norm2 * spec_e_stat**2

            # ax_.plot(spec_e_stat, sed2, ls = '--', color='k', label = r'evolved, $\eta = \infty$',
            #          alpha = 0.3)
            
            
        # eta, deta = curve_fit(f = tofit, xdata = eD, ydata = sed_norm, p0 = (1,),
        #                       bounds = (0, 100))
        # eta, deta = eta[0], deta[0][0]
        # plt.plot(eD, tofit(eD, eta), ls='--', label = f'fit eta = {eta:.3g} pm {deta:.3g}')
        
        
        
    # print('2nd way is ', time.time() - start)
    # # plt.plot(spec_energies, (res_spec*spec_energies**2 - sed1)/sed1)
            ax_.set_xscale('log')
            ax_.set_yscale('log')
            if row == rows - 1:
                ax_.set_xlabel(r'E, eV')
            if col == 0:
                ax_.set_ylabel(r'$E^2 dN/dE$')
            # ax_.set_ylim(np.max(sed1)/1e4, np.max(sed1)*3)
            ax_.set_xlim(1e8/1.2, emax/10 * 1.2)
            ax_.set_title(rf'$B = {B0}, d (10^{{13}}) = {dist/1e13}$')
            # plt.xlim(7e7, 5e13)
            # plt.ylim(1e3, 1e9)
            ax_.legend()
            
    plt.show()
    """
    
