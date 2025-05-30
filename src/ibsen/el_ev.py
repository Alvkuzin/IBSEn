import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sin, cos, exp
import astropy.units as u
from astropy import constants as const
from scipy.integrate import trapezoid, quad, cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator
import time
# from joblib import Parallel, delayed

from ibsen.transport_solvers.transport_on_ibs_solvers import solve_for_n
import xarray as xr
from pathlib import Path

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

def SyncLoss(ee, B):
    '''Synchrotron losses, dE/dt. Bis in G, electron energy -- eV '''
    # return -4e5 * B**2 * (ee/1e10)**2 #eV/s ???
    return -2.5e5 * B**2 * (ee/1e10)**2 #eV/s ???

def Gma(s, sm, G_term):
    return 1 + (G_term - 1) * s / sm

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

def ICLoss(Ee, Topt, Ropt, dist): # Ee in eV !!!!!!
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
        return eta_syn * SyncLoss(ee, B) + eta_IC * ICLoss(ee, Topt, Ropt, dist) - ee / t_adiab(dist, eta_flow)
    else:
        return eta_syn * SyncLoss(ee, B) + eta_IC * ICLoss(ee, Topt, Ropt, dist)
    
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
    
    
def evolved_e_advection(r_SP, ss, edot_func, f_inject_func,
              tot_loss_args, f_args, vel_func, v_args, emin = 1e9, 
              emax = 5.1e14):    
        # we calculate it on e-grid emin / extend_d < e <  emax * extend_u
        # and hope that zero boundary conditions will be not important
        
        # extend_u = 10; extend_d = 10; 
        extend_u = 3; extend_d = 10; 
        Ns, Ne = 901, 903
        # Ns, Ne = 201, 203 
        Ne_real = int( Ne * np.log10(extend_u * emax / extend_d / emin) / np.log10(emax / emin) )
        e_vals_sample = np.logspace(np.log10(emin / extend_d), np.log10(emax * extend_u), Ne_real)
        
        # now we'll look where the f_inject is not negligible and only there
        # will we solve the equation 
        ssmesh, emesh = np.meshgrid(ss * r_SP, e_vals_sample, indexing='ij')
        f_sample = f_inject_func(ssmesh, emesh, *f_args)
        f_sample_sed = f_sample[int(ss.size / 2.)] * e_vals_sample**2
        e_where_good = np.where(f_sample_sed > np.max(f_sample_sed) / 1e6 )
        s_vals = np.linspace(0, np.max(ss), Ns) * r_SP
        e_vals = e_vals_sample[e_where_good]
        dNe_de = solve_for_n(v_func = vel_func, edot_func = edot_func,
                            f_func = f_inject_func,
                            v_args = v_args, 
                            edot_args = tot_loss_args,
                            f_args = f_args,
                            s_grid = s_vals, e_grid = e_vals, 
                            method = 'FDM_cons', bound = 'dir')
        # #### Only leave the part of the solution between emin < e < emax #!!!
        ind_int = np.logical_and(e_vals <= emax, e_vals >= emin)
        e_vals = e_vals[ind_int]
        dNe_de = dNe_de[:, ind_int]
        
        #### and evaluate the values on the IBS grid previously obtained:dNe_de_IBS, e_vals
        interp_x = interp1d(s_vals, dNe_de, axis=0, kind='linear', fill_value='extrapolate')
        dNe_de_IBS = interp_x(ss * r_SP)
        dNe_de_IBS[dNe_de_IBS <= 0] = np.min(dNe_de_IBS[dNe_de_IBS>0]) / 3.14
        
        return dNe_de_IBS, e_vals
    
class ElectronsOnIBS:
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
    def __init__(self, Bp_apex, u_g_apex, ibs: IBS,  to_inject_e = 'ecpl',
                 to_inject_theta = '3d', ecut = 1.e12, p_e = 2., norm_e = 1.,
                 Bs_apex=0., eta_a = 1.,
                 eta_syn = 1., eta_ic = 1.,
                 emin = 1e9, emax = 5.1e14, to_cut_e = True, to_cut_theta =  False,
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
        
        self.Bp_apex = Bp_apex # pulsar magn field in apex
        self.Bs_apex = Bs_apex # opt star magn field in apex
        self.B_apex = Bp_apex + Bs_apex # total magn field in apex
        self.u_g_apex = u_g_apex # photon energy density in apex
        
        # self.r_sp = r_sp # orbital separation (for rescaling ibs)
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
        
        
        
        
    def _check_and_set_ibs(self):
        """
        Checks if ibs contains ibs.winds and ibs.winds.orbit
        """
        try:
            self.orbit = self.ibs.winds.orbit
            self.r_sp = self.ibs.winds.orbit.r(self.ibs.t_forbeta)
        except:
            raise ValueError(""""Your ibs:IBS should contain ibs.winds""")

    
    def vel_func(self, s): # s --- in real units
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


    def edot(self, s, e): 
        # s_t = s_table * dorb
        # r_t = r_table * dorb
        # th_spl = interp1d(x = s_t, y = th_table, fill_value='extrapolate')
        # r_spl = interp1d(x = s_t, y = r_t, fill_value='extrapolate')
        # th_interp = th_spl(s)
        # r_toP = r_spl(s)
        # r_sp = self.ibs.winds.orbit.r()
        # th_interp = self.ibs.s_interp(s_ = s / r_sp, what = 'theta')
        r_to_p = self.ibs.s_interp(s_ = s / self.r_sp, what = 'r') # dimless
        r_to_s = self.ibs.s_interp(s_ = s / self.r_sp, what = 'r1') # dimless
        r_sa = (1. - self.x_apex)
        # r_toS = (dorb**2 + r_toP**2 - 2 * dorb * r_toP * cos(th_interp))**0.5
        # B_on_shock = B * np.min(r_toP) / r_toP #TODO: now it's assumed the field is only from the pulsar... should fix this in future
        B_on_shock = (self.Bp_apex * self.x_apex / r_to_p + 
                      self.Bs_apex * r_sa / r_to_s)
        eta_ic_on_shock = self.eta_ic * r_sa**2 / r_to_s**2
        # eta_IC_on_shock = eta_IC * (np.min(r_toS) / r_toS)**2 # can i do that??????
        return total_loss(ee = e, 
                          B = B_on_shock, 
                          Topt = self.ibs.winds.Topt,
                          Ropt=self.ibs.winds.Ropt,
                          dist = self.r_sp * r_sa, 
                          eta_flow = self.eta_a, 
                          eta_syn = self.eta_syn,
                          eta_IC = eta_ic_on_shock)

    def f_inject(self, s_, e_): 
        # s_arr = np.asanyarray(s)
        # e_arr = np.asanyarray(e)
        thetas_here = self.ibs.s_interp(s_  = s_ / self.r_sp, what = 'theta')
        # thetas_here = np.interp(s_arr, s_table * dorb, th_table)
        if self.to_inject_theta == '2d':
            thetas_part = np.zeros(thetas_here.shape) + 1. # uniform along theta
        elif self.to_inject_theta == '3d':
            thetas_part = sin(thetas_here) # \propto sin(th) how it should be in 3d
            # thetas_part = s_ / np.max(s_)
        else:
            raise ValueError("I don't know this to_inject_theta. It should be 2d, 3d.")
            
        if self.to_cut_theta:
            thetas_part[thetas_here >= self.where_cut_theta] = 0.
            
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
        return result
    
    
    
    
    

