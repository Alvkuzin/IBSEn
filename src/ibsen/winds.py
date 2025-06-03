import numpy as np
from numpy import pi#, sin, cos
from scipy.optimize import brentq

from astropy import constants as const
# from astropy import units as u


from ibsen.get_obs_data import get_parameters
from .utils import rotated_vector, mydot, mycross, n_from_v, absv
from ibsen.orbit import Orbit


G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

def unpack_star(sys_name=None, Topt=None, Ropt=None, Mopt=None):
    """
    Unpack star parameters with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        Topt, Ropt, Mopt: float or None
            Explicit values that override defaults.

    Returns:
        Tuple of Topt, Ropt, Mopt
    """
    # Step 1: Determine the source of defaults
    if isinstance(sys_name, str):
        known_types = ['psrb', 'rb', 'bw']
        if sys_name not in known_types:
            raise ValueError(f"Unknown orbit type: {sys_name}")
        defaults = get_parameters(sys_name)
    elif isinstance(sys_name, dict):
        defaults = sys_name
    else:
        defaults = {}

    # Step 2: Build final values, giving priority to explicit arguments
    Topt_final = Topt if Topt is not None else defaults.get('Topt')
    Ropt_final = Ropt if Ropt is not None else defaults.get('Ropt')
    Mopt_final = Mopt if Mopt is not None else defaults.get('Mopt')

    result = [Topt_final, Ropt_final, Mopt_final]
    return tuple(result)


class Winds:
    def __init__(self, orbit: Orbit, 
                 sys_name=None, 
                 Ropt=None, Topt=None, Mopt=None, 
                 M_ns = 1.4*M_SOLAR, f_p = 0.1, 
                 alpha=0, incl=30./180.*pi,   
                 f_d=10., np_disk = 3., delta=0.01, 
                 rad_prof = 'pl', r_trunk = None):
        Topt_, Ropt_, Mopt_ = unpack_star(sys_name=sys_name, Topt=Topt, Ropt=Ropt, Mopt=Mopt)
        self.orbit = orbit
        self.Topt = Topt_
        self.Ropt = Ropt_
        self.Mopt = Mopt_
        self.m_ns = M_ns
        self.alpha = alpha
        self.incl = incl
        self.f_d = f_d
        self.f_p = f_p
        
        self.np_disk = np_disk
        self.delta = delta
        self.rad_prof = rad_prof
        if r_trunk is None:
            self.r_trunk = 5 * Ropt_
        else:
            self.r_trunk = r_trunk
        
        
    @property
    def n_disk(self):
        return rotated_vector(alpha = self.alpha, incl = self.incl)


    def Dist_to_disk(self, rvec):
        return mydot(rvec, self.n_disk)
    
    @property
    def times_of_disk_passage(self):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        Dist_to_disk_time = lambda t: mydot(self.orbit.vector_sp(t), self.n_disk)
        t1 = brentq(Dist_to_disk_time, -self.orbit.T/2., 0)
        t2 = brentq(Dist_to_disk_time, 0, self.orbit.T/2.)
        return t1, t2

    @property
    def vectors_of_disk_passage(self):
        t1, t2 = self.times_of_disk_passage
        vec1 = self.orbit.vector_sp(t1)
        vec2 = self.orbit.vector_sp(t2)
        return vec1, vec2


    def vec_r_to_dp(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        radius = self.orbit.vector_sp(t)
        ndisk = self.n_disk
        return mydot(radius, ndisk) * ndisk
    
    def vec_r_in_dp(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        radius = self.orbit.vector_sp(t)
        # ndisk = N_disk(alpha, incl)
        # d_to_disk = mydot(radius, ndisk)
        return radius - Winds.r_to_dp(self, t)
    

    def n_DiskMatter(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        n_indisk = n_from_v(Winds.vec_r_in_sp(self, t))
        ndisk = self.n_disk
        return mycross(ndisk, n_indisk)
    
    def pulsar_wind_pressure(self, r_from_p):
        return self.f_p * (self.Ropt / r_from_p)**2
    
    def polar_wind_pressure(self, r_from_s):
        return (self.Ropt / r_from_s)**2
    
    def disk_height(self, r):
        return self.delta * r * (r / self.Ropt)**0.25
    
    def decr_disk_pressure(self, vec_r_from_s):
        ndisk = self.n_disk
        r_fromdisk = mydot(vec_r_from_s, ndisk) * ndisk
        r_indisk = vec_r_from_s - r_fromdisk
        r_to_d = absv(r_fromdisk) 
        r_indisk = vec_r_from_s - r_fromdisk
        r_in_d = absv(r_indisk)
        z0 = Winds.disk_height(self, r_in_d)
        vert = np.exp(-r_to_d**2 / 2 / z0**2)
        if self.rad_prof == 'pl':
            rad = (self.Ropt / r_in_d)**self.np_disk
        if self.rad_prof == 'broken_pl':
            if r_in_d < self.r_trunk:
                rad = (self.Ropt / r_in_d)**self.np_disk
            if r_in_d >= self.r_trunk:
                rad = (self.Ropt / self.r_trunk)**self.np_disk * (self.r_trunk / r_in_d)**2
        return self.f_d * rad * vert 
    
    def _dist_se_1d_notvectorized(self, t):   
        r_sp_vec = self.orbit.vector_sp(t)
        nwind = n_from_v(r_sp_vec) # unit vector from S to P
        r_sp = absv(r_sp_vec)
        pres_w = lambda r_se: Winds.polar_wind_pressure(self, r_from_s = r_se)
        pres_d = lambda r_se: Winds.decr_disk_pressure(self, vec_r_from_s = nwind * r_se)
        pres_p = lambda r_se: Winds.pulsar_wind_pressure(self, r_from_p = np.abs(r_sp - r_se))
        to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
        rse = brentq(to_solve, self.Ropt, r_sp*(1-1e-6))
        ### ---------------- test if the solution is good -------------------------
        p_ref = pres_p(rse)
        max_rel_err = np.max(to_solve(rse) / p_ref)
        if max_rel_err > 1e-3:
            print('t = %s error is huge: %s'%(t/DAY, max_rel_err))  
        return rse
    
    def dist_se_1d(self, t):   
        if isinstance(t, np.ndarray):
            res = np.array( [Winds._dist_se_1d_notvectorized(self, t_) for t_ in t] )
        else:
            res = Winds._dist_se_1d_notvectorized(self, t)
        return res
    
    def beta_eff(self, t):
        r_sp = self.orbit.r(t)
        r_se = Winds.dist_se_1d(self, t)
        r_pe = r_sp - r_se
        return (r_pe / r_se)**2
        