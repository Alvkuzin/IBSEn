import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq

from astropy import constants as const
# from astropy import units as u
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params
from ibsen.utils import rotated_vector, mydot, mycross, n_from_v, absv, enhanche_jump
from ibsen.orbit import Orbit
import matplotlib.pyplot as plt


G = float(const.G.cgs.value)
C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)


R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

winds_docstring =     f"""
    All information about the Be-star and the pulsar.

    This class encapsulates the outflows from the massive star (polar wind
    and equatorial decretion disk) and the relativistic pulsar wind. It provides
    geometry (disk orientation and crossings), pressure fields, stand-off distance
    solutions, effective momentum ratio at the apex, magnetic fields at the apex,
    and stellar photon energy density. All distances/fields are in cgs unless
    stated otherwise. Angles are in radians. Time ``t`` is measured relative to
    periastron (``t = 0``). 

    Parameters
    ----------
    orbit : Orbit
        The binary orbit object providing positions/vectors as functions of time.
    sys_name : {known_names} or None, optional
        If str, load default stellar parameters via ``get_parameters(sys_name)``.
         Explicit keyword arguments
        below override any defaults. If None, all stellar parameters must be
        given explicitly or be in a dict `sys_params`.
    sys_params : dict or None, optional
        If dict, load stellar parameters from this dictionary. Explicit keywords
        below override any defaults. If None, all stellar parameters must be
        given explicitly or be in `sys_name`.
    allow_missing : bool, optional
        Fill the missing parameters (not explicitly provided, no keyword 
        recognized, and not found in `sys_params`) with None. Default False
    Ropt : float, optional
        Optical star radius [cm].
    Topt : float, optional
        Optical star effective temperature [K].
    Mopt : float, optional
        Optical star mass [g].
    M_ns : float, optional
        Neutron star mass [g]. Default is ``1.4*M_SOLAR``.
    f_p : float, optional
        Dimensionless pulsar-wind pressure normalization (∝ 1/r²). Default 0.1.
    alpha : float, optional
        Disk plane position angle: rotation about the +Z axis (see Notes). Default 0.
    incl : float, optional
        Disk axis inclination from +Z (0 means disk in the orbital plane). Default ``30 deg``.
    f_d : float, optional
        Dimensionless decretion-disk pressure normalization. Default 0.
    t_forwinds : float, optional
        Time on which to calculate the winds structure. Default None.
    p_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk pressure. Default [1,].
    h_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk height. Default [1,].
    p_enh_times : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of times for pressure enhancement. Default [0,].
    h_enh_times : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of times for pressure enhancement. Default [0,].
    np_disk : float, optional
        Radial power-law index for the disk pressure in-plane, ``P∝r^{{-np_disk}}``.
        Default 3.
    delta : float, optional
        Disk opening parameter at the stellar surface, ``z0/r`` at ``r=Ropt``. Default 0.01.
    height_exp : float, optional
        Exponent in the disk scale-height law, ``z0/r ∝ (r/Ropt)^{{height_exp}}``. Default 0.5.
    rad_prof : 'pl', 'bkpl', optional
        Radial profile: ``'pl'`` = single power law; ``'bkpl'`` = broken power law
        that transitions to ``∝ r^{{-2}}`` beyond ``r_trunk``. Default 'pl'.
    r_trunk : float or None, optional
        Truncation radius [cm] used when ``rad_prof='bkpl'``. If None, set to ``5*Ropt``.
    ns_b_model : 'linear', 'dipole', 'from_L_sigma', optional
        Model for the pulsar magnetic field with radius (see ``ns_field``). Default 'linear'.
    ns_b_ref : float or None, optional
        Pulsar surface field parameter ``B_surf`` [G] for 'linear'/'dipole' models.
    ns_r_ref : float or None, optional
        Scale radius ``r_ref`` [cm] for 'linear'/'dipole' models.
    ns_L_spindown : float or None, optional
        Pulsar spin-down luminosity ``L`` [erg/s] for 'from_L_sigma'.
    ns_sigma_magn : float or None, optional
        Magnetization ``σ`` for 'from_L_sigma'.
    opt_b_model : 'linear', 'dipole', optional
        Model for the optical star magnetic field with radius (see ``opt_field``).
        Default 'linear'.
    opt_b_ref : float or None, optional
        Stellar surface field parameter ``B_surf`` [G] for 'linear'/'dipole'.
    opt_r_ref : float or None, optional
        Scale radius ``r_ref`` [cm] for 'linear'/'dipole'.

    Attributes
    ----------
    orbit : Orbit
        The supplied orbit object (used for geometry and timing).
    Topt, Ropt, Mopt : float
        Stellar temperature [K], radius [cm], and mass [g] used by the model.
    M_ns : float
        Neutron star mass [g].
    alpha, incl : float
        Disk orientation angles (radians).
    f_d, f_p : float
        Disk and pulsar-wind pressure normalizations (dimensionless).
    np_disk, delta, height_exp : float
        Disk radial index, base opening, and opening exponent.
    rad_prof : 'pl', 'bkpl'
        Choice of disk radial profile.
    r_trunk : float
        Disk truncation radius [cm] if using a broken power law.
    ns_b_model, ns_b_ref, ns_r_ref, ns_L_spindown, ns_sigma_magn
        Parameters controlling the pulsar magnetic-field model.
    opt_b_model, opt_b_ref, opt_r_ref
        Parameters controlling the stellar magnetic-field model.

    Methods
    -------
    n_disk
        Unit normal vector to the disk plane (property).
    times_of_disk_passage
        Two times within one period when the pulsar crosses the disk plane (property).
    vectors_of_disk_passage
        Star→pulsar vectors at those crossing times (property).
    Dist_to_disk(rvec)
        Distance from a point to the disk plane (absolute value returned). 
    vec_r_to_dp(t), vec_r_in_dp(t)
        Decomposition of the star→pulsar vector into components normal to / in the disk plane.
    pulsar_wind_pressure(r_from_p), polar_wind_pressure(r_from_s), decr_disk_pressure(vec_r_from_s)
        Dimensionless pressure laws for the pulsar wind, stellar polar wind, and disk. 
    dist_se_1d(t)
        Distance from the star to the stand-off (shock apex) point along the star–pulsar line.
    beta_eff(t)
        Effective momentum-flux ratio at the apex, related to ``r_pe/r_sp``.
    magn_fields_apex(t)
        Pulsar and stellar magnetic fields at the apex [G].
    u_g_density_apex(t)
        Stellar photon energy density at the apex [erg/cm³].
    ns_field(...), opt_field(...), u_g_density(...)
        Static-style utility functions for fields and radiation energy density. 
    peek(ax=None, showtime=None, plot_rs=True)
        Quick-look plot of orbit, disk plane, and pressure contours.

    Notes
    -----
    * Disk orientation (1): the vector normal to the disk plane is obtained by
    inclining the unit vector in the +Z direction by ``incl`` along the +X axis,
    then rotating it by ``alpha`` about the +Z axis. See ``rotated_vector()``.
    * Disk orientation (2): the disk normal is ``rotated_vector(alpha, incl)`` in the
      orbit frame; ``incl=0`` places the disk in the orbital plane. When the disk
      is exactly in the orbital plane, the root-finding for disk-crossing times
      can be ill-conditioned. 
    * ``dist_se_1d`` solves for ``r_se`` where disk+polar pressures equal the
      pulsar-wind pressure along the line connecting the star and pulsar; the
      apex star→pulsar distance is then ``r_pe = r_sp - r_se``. 
    * The neutron star mass M_ns is currently unused.
    * If the time t_forwinds is provided, the disk pressure and height are 
      multiplied by p_enh or h_enh at the time t_forwinds. 

    """

class Winds:
    __doc__ = winds_docstring
    def __init__(self, orbit: Orbit, # orb:Orbit --- the orbit to use 
                 sys_name=None, # the system name; str,  or None
                 sys_params=None, # system parameters dict
                 allow_missing=False,
                 Ropt=None,
                 Topt=None,
                 Mopt=None, 
                 M_ns = 1.4*M_SOLAR,
                 f_p = 0.1, # dimentionless pulsar wind pressure strength
                 alpha = 0, # Be-star rotation axis position angle: 2d angle between two planes:
                            # the perpendicular to xy-plane and containing the Omega_Be vector
                            # and the X-0-Z plane.
                 incl = 30./180.*pi,   # Be-star rotation axis position angle: 
                                     # the angle between the Omega_Be vector and
                                     # z-axis.
                 f_d = 0.,  # dimentionless decretion disk pressure strength
                 np_disk = 3., # exponent for the pressure r-dependence for the disk
                 delta = 0.01, # z0/r for the disk at the star surface
                 height_exp = 0.5, # exponent for z0/r \propto r^height_exp
                 rad_prof = 'pl', # rad profile for r-dependence of disk pressure
                 r_trunk = None, # if rad_prof == 'bkpl', it's used as a truncation radius
                                  # beyond which P \propto r^-2
                 ### for modeling jumps in pressure/height of the disk with time
                 t_forwinds = None,
                 p_enh = [1, ],
                 p_enh_times = [0, ],
                 h_enh = [1, ],
                 h_enh_times = [0, ],
                
                 ### for magn fields of NS and opt star
                 ns_b_model = 'linear', # model of ns magn field
                 ns_b_ref = None, # B- parameter for ns field scaling 
                 ns_r_ref = None, # r-scale for NS field
                 ns_L_spindown = None, # NS spin-down lum 
                 ns_sigma_magn = None, # NS PWE magnetization
                 opt_b_model = 'linear', # model of the Be magn field
                 opt_b_ref = None, # B- parameter for Be field scaling 
                 opt_r_ref = None, # r-scale for Be magn field
                 ):
        Topt_, Ropt_, Mopt_ = unpack_params(('Topt', 'Ropt', 'Mopt'),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names, get_defaults_func=get_parameters,
                               Topt=Topt, Ropt=Ropt, Mopt=Mopt,
                               allow_missing=allow_missing)
        self.orbit = orbit
        self.Topt = Topt_
        self.Ropt = Ropt_
        self.Mopt = Mopt_
        self.M_ns = M_ns
        
        self.alpha = alpha
        self.incl = incl
        self.initiate_disk(f_d, delta, t_forwinds, p_enh,
                           p_enh_times, h_enh, h_enh_times) # sets self.delta and self.f_d 
        self.f_p = f_p
        self.np_disk = np_disk
        self.height_exp = height_exp
        self.rad_prof = rad_prof
        if r_trunk is None:
            self.r_trunk = 5 * Ropt_
        else:
            self.r_trunk = r_trunk
            
        self.ns_b_model = ns_b_model
        self.ns_b_ref = ns_b_ref
        self.ns_r_ref = ns_r_ref
        self.ns_L_spindown = ns_L_spindown
        self.ns_sigma_magn = ns_sigma_magn
        
        self.opt_b_model = opt_b_model
        self.opt_b_ref = opt_b_ref
        self.opt_r_ref = opt_r_ref
        self.unit_star = n_from_v(rotated_vector(alpha=alpha, incl=incl)  )
        
    
    def initiate_disk(self, f_d, delta, t_forwinds, p_enh, p_enh_times,
                      h_enh, h_enh_times):
        """ sets self.f_d and self.delta at the moment t_forwinds"""
        self.t_forwinds = t_forwinds
        self.p_enh = p_enh
        self.h_enh = h_enh
        self.p_enh_times = p_enh_times
        self.h_enh_times = h_enh_times
        if t_forwinds is None:
            self.f_d = f_d
            self.delta = delta
        else:
            t1_, t2_ = self.times_of_disk_passage
            # print('winds ', h_enh_times)
            
            f_d_mult = enhanche_jump(t = t_forwinds, t1_disk=t1_, t2_disk=t2_,
                                     times_enh=p_enh_times, param_to_enh=p_enh)
            h_mult = enhanche_jump(t = t_forwinds, t1_disk=t1_, t2_disk=t2_,
                                     times_enh=h_enh_times, param_to_enh=h_enh)
            self.f_d = f_d * f_d_mult
            self.delta = delta * h_mult
        
    
    def ns_field(r_to_p, model='linear', B_ref = None, r_ref = None,
                 L_spindown = None, sigma_magn = None):   
        """
        The magnetic field of the NS [G] at the distance r_to_p.
        
        Parameters
        ----------
        r_to_p : TYPE
            DESCRIPTION.
        model : str, optional
            How to calculate the magnetic field from the NS. Options:
            -  'linear': B = B_ref * (r_ref / r_to_p). You should provide 
            B_ref and r_ref.
            - 'dipole': B = B_ref * (r_ref / r_to_p)^3. You should provide 
            B_ref and r_ref.
            - 'from_L_sigma': according to Kennel & Coroniti 1984a,b:
                B = sqrt(L_spindown * sigma_magn / c / r_to_p^2). 
                You should provide L_spindown and sigma_magn.
            
            The default is 'linear'.
            
        B_ref : float, optional
            The field [G] at the distance r_ref from NS. Fefault is None.
        r_ref : float, optional
            The scale radius for model = 'linear' or model = 'dipole'.
            Default is None.
        L_spindown : float, optional
            The spin-down luminosity of the NS [erg / s]. The default is None.
        sigma_magn : float, optional
            Magnetization of the piulsar wind. The default is None.

        Returns
        -------
        The magnetic field of the NS [G] at the distance r_to_p.

        """    
        model_opts = ['linear', 'dipole', 'from_L_sigma']

        if model not in (model_opts):
            raise ValueError('the NS field model should be one of:',
                             model_opts)
        if model == 'linear':
            B_puls = B_ref * (r_ref / r_to_p)
        if model == 'dipole':
            B_puls = B_ref * (r_ref / r_to_p)**3
        
        if model == 'from_L_sigma':        
            B_puls = (L_spindown * sigma_magn / C_LIGHT / r_to_p**2)**0.5
        
        return B_puls
    
    def opt_field(r_to_s, model = 'linear', r_ref=None, B_ref=None):
        """
        The magnetic field of the opt. star [G] at the distance r_to_s.
        
        Parameters
        ----------
        r_to_a : float
            The distance from the star [cm] to the point.
        model : str, optional
            How to calculate the magnetic field from the star. Options:
            -  'linear': B = B_ref * (r_ref / r_to_p). You should provide 
            B_ref and r_ref.
            - 'dipole': B = B_ref * (r_ref / r_to_p)^3. You should provide 
            B_ref and r_ref.
            The default is 'linear'.
            
        B_ref : float, optional
            The field at the star surface [G]. Default is None.
        r_ref : float, optional
            The scale radius for model = 'linear' or model='dipole'.
            Default is None.

        Returns
        -------
        The magnetic field of the NS [G] at the distance r_to_p.

        """    
        model_opts = ['linear', 'dipole']

        if model not in (model_opts):
            raise ValueError('the opt star field model should be one of:',
                             model_opts)
        if model == 'linear':
            B_opt = B_ref * (r_ref / r_to_s)
        if model == 'dipole':
            B_opt = B_ref * (r_ref / r_to_s)**3
        
        return B_opt
    
    def u_g_density(r_from_s, r_star, T_star):      
        """
        Optical start photon field energy density at the distance r_from_s from the star.

        Parameters
        ----------
        r_from_s : float | np.ndarray
            Distance from the star [cm].
        r_star : float | np.ndarray
            Star radius [cm].
        T_star : float | np.ndarray
            Effective temperature of the star [K].

        Returns
        -------
        u_dens : float | np.ndarray
            Energy density of the star photon field at the
              distance r_from_s [erg / cm^3].

        """
        factor = 2. * (1. - (1. - (r_star / r_from_s)**2)**0.5 ) # checked!
        u_dens = SIGMA_BOLTZ * T_star**4 / C_LIGHT * factor # checked!
        return u_dens
    
    @property
    def n_disk(self):
        """
        Vector normal to the disk plane or the Be-star equatorial plane.

        Returns
        -------
        np.array of shape (3,)
        The unit vector normal to the disk plane.

        """
        return rotated_vector(alpha = self.alpha, incl = self.incl)


    def Dist_to_disk(self, rvec):
        """
        Calculate the distance from the point with radius vector rvec to the disk plane.

        Parameters
        ----------
        rvec : np.array of shape (3,)
            The radius vector of the point.

        Returns
        -------
        float
            The distance from the point to the disk plane.

        """
        return np.abs(mydot(rvec, self.n_disk))
    
    @property
    def times_of_disk_passage(self):
        """
        Times of the pulsar passage through the disk plane.
        Two solution of the equation vec{r(t)} \dot vec{n_disk} = 0.
        If the disk is in the orbital plane (incl=0), unpredictable results.

        Returns
        -------
        t1 : float
            Negative solution [s]. 
        t2 : TYPE
            Positive solution [s].

        """
        Dist_to_disk_time = lambda t: mydot(self.orbit.vector_sp(t), self.n_disk)
        t1 = brentq(Dist_to_disk_time, -self.orbit.T/2., 0)
        t2 = brentq(Dist_to_disk_time, 0, self.orbit.T/2.)
        return t1, t2

    @property
    def vectors_of_disk_passage(self):
        """
        Vectors from the star to the pulsar at the times of disk passage.

        Returns
        -------
        vec1 : np.array of shape (3,)
            The vector at the first disk passage time.
        vec2 : np.array of shape (3,)
            The vector at the second disk passage time.

        """
        t1, t2 = self.times_of_disk_passage
        vec1 = self.orbit.vector_sp(t1)
        vec2 = self.orbit.vector_sp(t2)
        return vec1, vec2


    def vec_r_to_dp(self, t):
        """
        The normal component of the radius vector to the disk plane.
        The normal is not necessarily TO the disk, but it is a normal
        componens such that vec_r = vec_r_to_dp + vec_r_in_dp.
        Parameters
        ----------
        t : float
            Time relative to periastron [s].

        Returns
        -------
        np.array of shape (3,)
            The normal component of the radius vector to the disk plane.

        """
        radius = self.orbit.vector_sp(t)
        ndisk = self.n_disk
        return mydot(radius, ndisk) * ndisk
    
    def vec_r_in_dp(self, t):
        """
        The component of the radius vector in the disk plane, such that
        vec_r = vec_r_to_dp + vec_r_in_dp.
        Parameters
        ----------
        t : float
            Time relative to periastron [s].

        Returns
        -------
        np.array of shape (3,)
            The component of the radius vector in the disk plane.

        """
        radius = self.orbit.vector_sp(t)
        return radius - Winds.vec_r_to_dp(self, t)
    

    def n_DiskMatter(self, t):
        """
        The vector of the Keplerian disk matter velocity at the point of the pulsar location.

        Parameters
        ----------
        t : float
            Time relative to periastron [s].

        Returns
        -------
        np.array of shape (3,)
            The unit vector of the Keplerian disk matter velocity at the point of the pulsar location.

        """
        n_indisk = n_from_v(Winds.vec_r_in_sp(self, t))
        ndisk = self.n_disk
        return mycross(ndisk, n_indisk)
    
    def pulsar_wind_pressure(self, r_from_p):
        """
        Dimensionless pulsar wind pressure at the distance r_from_p from the pulsar.
        Assuming isotropic pulsar wind and constant velocity.
        Parameters
        ----------
        r_from_p : float
            Distance from the pulsar [cm].

        Returns
        -------
        float
            P_pulsar(r_from_p), dimensionless.

        """
        return self.f_p * (self.Ropt / r_from_p)**2
    
    def polar_wind_pressure(self, r_from_s):
        """
        Dimensionless Be-star polar wind pressure at the distance r_from_s from the star.
        Assuming isotropic polar wind and constant velocity.
        The coefficient is set so that at r=Ropt, P_w = 1.
        Parameters
        ----------
        r_from_s : float
            Distance from the star [cm].

        Returns
        -------
        float
            P_w(r_from_s), dimensionless.

        """
        return (self.Ropt / r_from_s)**2
    
    def disk_height(self, r):
        """
        The height of the disk at the distance r from the star.

        Parameters
        ----------
        r :float
            The equatorial distance from the star [cm].

        Returns
        -------
        float
            z0(r) [cm].

        """
        return self.delta * r * (r / self.Ropt)**self.height_exp
    
    def decr_disk_pressure(self, vec_r_from_s):
        """
        Dimensionless decretion disk pressure at the point
          with radius vector vec_r_from_s.

        Parameters
        ----------
        vec_r_from_s : np.ndarray of shape (3,)
            Vector from the star to the point.

        Returns
        -------
        float
            P_d(vec_r_from_s), dimensionless.

        """
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
        if self.rad_prof == 'bkpl':
            if r_in_d < self.r_trunk:
                rad = (self.Ropt / r_in_d)**self.np_disk
            if r_in_d >= self.r_trunk:
                rad = (self.Ropt / self.r_trunk)**self.np_disk * (self.r_trunk
                                                                  / r_in_d)**2
        return self.f_d * rad * vert 
    
    def _dist_se_1d_nonvec(self, t):  
        """
        Distance from the star to the wind-wind interaction region
        defined as the point where the disk+polar wind pressure
        equals the pulsar wind pressure.

        Parameters
        ----------
        t : float
            Time relative to periastron [s].

        Raises
        ------
        ValueError
            The solution of the disk-wind balance equation is not accurate enough.

        Returns
        -------
        rse : float
            Distance from the star to the wind-wind interaction region [cm].

        """
        r_sp_vec = self.orbit.vector_sp(t)
        nwind = n_from_v(r_sp_vec) # unit vector from S to P
        r_sp = absv(r_sp_vec)
        pres_w = lambda r_se: Winds.polar_wind_pressure(self, r_from_s = r_se)
        pres_d = lambda r_se: Winds.decr_disk_pressure(self, vec_r_from_s = nwind * r_se)
        pres_p = lambda r_se: Winds.pulsar_wind_pressure(self, r_from_p = np.abs(r_sp - r_se))
        to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
        rse = brentq(to_solve, self.Ropt, r_sp*(1-1e-8))
        ### ---------------- test if the solution is good -------------------------
        p_ref = pres_p(rse)
        max_rel_err = np.max(to_solve(rse) / p_ref)
        if max_rel_err > 1e-3:
            raise ValueError(f'Huge relative error of {max_rel_err} in the solution of the '
                          'disk-wind balance equation at {t/DAY}    days. ')        # r_fromdisk = Winds.vec_r_to_dp(self, t=0)  # t is irrelevant here

        return rse
    
    def dist_se_1d(self, t): 
        """
        Distance from the star to the wind-wind interaction region
        defined as the point where the disk+polar wind pressure
        equals the pulsar wind pressure.

        Parameters
        ----------
        t : float | np.ndarray
            Time relative to periastron [s].

        Raises
        ------
        ValueError
            The solution of the disk-wind balance equation is not accurate enough.

        Returns
        -------
        rse : float | np.ndarray
            Distance from the star to the wind-wind interaction region [cm].

        """
        t_ = np.asarray(t)
        if t_.ndim == 0:
            return float(Winds._dist_se_1d_nonvec(self, float(t_)))
        
        return np.array( [
            Winds._dist_se_1d_nonvec(self, t_now) for t_now in t_
            ] )
    
    def beta_eff(self, t):
        """
        The effective momentum flux ratio of the pulsar wind to the
        external medium (polar wind + decretion disk) at the apex point.
        Defined so that r_pe/r_sp = beta_eff^0.5 / (1 + beta_eff^0.5),

        Parameters
        ----------
        t : float | np.ndarray
            Time relative to periastron [s].

        Returns
        -------
        TYPE
            Effective winds momentum ratio at the apex point.

        """
        r_sp = self.orbit.r(t)
        r_se = Winds.dist_se_1d(self, t)
        r_pe = r_sp - r_se
        return (r_pe / r_se)**2
    
    def magn_fields_apex(self, t):
        """
        NS and optical star magnetic fields at the apex point.

        Parameters
        ----------
        t : float | np.ndarray
            Time relative to periastron [s].

        Returns
        -------
        B_ns_apex : float | np.ndarray
            The magnetic field of the NS at the apex point [G].
        B_opt_apex : float | np.ndarray
            The magnetic field of the optical star at the apex point [G].

        """
        r_se = Winds.dist_se_1d(self, t)
        r_pe = self.orbit.r(t) - r_se
        B_ns_apex = Winds.ns_field(r_to_p =r_pe, model=self.ns_b_model,
                              B_ref = self.ns_b_ref,
                              r_ref = self.ns_r_ref,
                              L_spindown = self.ns_L_spindown,
                              sigma_magn =self.ns_sigma_magn)
        B_opt_apex = Winds.opt_field(r_to_s = r_se,
                                     model = self.opt_b_model,
                                     r_ref=self.opt_r_ref,
                                     B_ref=self.opt_b_ref)
        
        return B_ns_apex, B_opt_apex
    
    def u_g_density_apex(self, t): 
        """
        The optical star photon field energy density at the apex point.

        Parameters
        ----------
        t : float | np.ndarray
            Time relative to periastron [s].

        Returns
        -------
        float | np.ndarray
          The optical star photon field energy density at the apex point.

        """
        r_se = Winds.dist_se_1d(self, t)
        return Winds.u_g_density(r_from_s = r_se,
                                 r_star = self.Ropt,
                                 T_star = self.Topt)
    
    def peek(self, ax=None,
             showtime = None,
             plot_rs = True,):
        """
        Quick look at the orbit, disk plane, and pressures.

        Parameters
        ----------
        ax : axes object, optional
            Axes to draw the winds on. If plot_rs is True, it should be 
            an array of at least two axes. If plot_rs is False, it 
            should be at least a single axis. If None, new figure and axes
            will be created. The default is None.
        showtime : tuple of (tmin, tmax) or None, optional
            See Orbit.peek(showtime=...). The default is None.
        plot_rs : bool, optional
            Whether to plot r_pe/se/sp(t) on an additional axis.
              The default is True.

        Returns
        -------
        None.

        """
        if ax is None:
            if plot_rs:
                fig, ax = plt.subplots(nrows=1, ncols=2)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1)

        if plot_rs:  ax0 = ax[0]
        else: ax0 = ax

        if showtime is None:
            showtime = [-self.orbit.T/2, self.orbit.T/2]
            
        _t = np.linspace(showtime[0], showtime[1], 307)
 
            
        show_cond  = np.logical_and(self.orbit.ttab > showtime[0], 
                                    self.orbit.ttab < showtime[1])
        
        ############# ------ disk passage related stuff ------- ###############

        t1, t2 = self.times_of_disk_passage
        # print('disk equator passage times [days]:')
        # print(t1/DAY, t2/DAY)
        vec_disk1, vec_disk2 = self.vectors_of_disk_passage

        
        orb_x, orb_y = self.orbit.xtab[show_cond], self.orbit.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
                ]))
        
        coord_scale = np.max(np.array([
            np.min(orb_x), np.max(orb_x), np.min(orb_y), np.max(orb_y),
            np.max( (orb_x**2 + orb_y**2)**0.5 )
            ]))
        ################### ------ drawing the orbit again ------- ############
        ax0.plot(orb_x, orb_y)                                                    
        ax0.scatter(0, 0, c='r')                                                  
        ax0.plot([np.min(orb_x),
                    np.max(orb_x)], [0, 0],
                    color='k', ls='--')      
        ax0.plot([0, coord_scale*cos(self.orbit.nu_los)],                            
                [0, coord_scale*sin(self.orbit.nu_los)], color='g', ls='--')        
        xx1, yy1, zz1 = vec_disk1                                                 
        xx2, yy2, zz2 = vec_disk2                                                 
        ax0.plot([xx1, xx2], [yy1, yy2], color='orange', ls='--', lw=2)    

        x_forp = np.linspace(np.min(orb_x)*3, np.max(orb_x)*4, 301)
        y_forp = np.linspace(np.min(orb_y)*2, np.max(orb_y)*2, 201)
        
        ###### ------------ tabulating pressure values ------------------ #####

        XX, YY = np.meshgrid(x_forp, y_forp, indexing='ij')
        disk_ps = np.zeros((x_forp.size, y_forp.size))
        for ix in range(x_forp.size):
            for iy in range(y_forp.size):
                vec_from_s_ = np.array([x_forp[ix], y_forp[iy], 0])
                r_ = (x_forp[ix]**2 + y_forp[iy]**2)**0.5
                disk_ps[ix, iy] = (Winds.decr_disk_pressure(self, vec_from_s_) 
                                +
                                Winds.polar_wind_pressure(self, r_)
                                )
        disk_ps = np.log10(disk_ps)

        # P_norm = (disk_ps - np.min(disk_ps)) / (np.max(disk_ps) - np.min(disk_ps))

        from matplotlib.colors import ListedColormap

        ########### some magic for displaying the winds, never mind ###########
        orange_rgba = np.array([1.0, 0.5, 0.0, 1.0]) 
        n_levels = 20
        colors = np.tile(orange_rgba[:3], (n_levels, 1))  
        alphas = np.linspace(0, 1, n_levels)           
        colors = np.column_stack((colors, alphas))     
        custom_cmap = ListedColormap(colors)
        disk_ps[(XX**2 + YY**2)**0.5 < self.Ropt] = np.nan
        disk_ps[disk_ps < np.nanmax(disk_ps)-4.5] = np.nan
        # norm = Normalize(vmin=np.min(disk_ps), vmax=np.max(disk_ps))
        ax0.contourf(XX, YY, disk_ps, levels=n_levels, cmap=custom_cmap)          

        ax0.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.orbit.r_periastr) )
        ax0.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        #######################################################################    
        
        if plot_rs:
            dists_se = Winds.dist_se_1d(self, _t)
            rs = self.orbit.r(_t)
            ax[1].plot(_t/DAY, dists_se, label='se', ls='--')
            ax[1].plot(_t/DAY, rs, label = 'sp', ls='-')
            ax[1].plot(_t/DAY, rs-dists_se, label='pe', ls=':')
            ax[1].axvline(x=t1/DAY, color='k', alpha=0.3)
            ax[1].axvline(x=t2/DAY, color='k', alpha=0.3)
            ax[1].axvline(x=self.orbit.t_los/DAY, color='g', ls='--', alpha=0.3)
            
            ax[0].set_title('overview')
            ax[1].set_title(r'$r_\mathrm{SP}, r_\mathrm{SE}, r_\mathrm{PE}$')
            ax[1].legend()
        # plt.show()
            
    # def peek_3d(self, #ax=None,
    #          showtime = None,
    #          # plot_rs = True,
    #          ):
    #     # if ax is None:
    #     #     if plot_rs:
    #     #         fig, ax = plt.subplots(nrows=1, ncols=2)
    #     #     else:
    #     #         fig, ax = plt.subplots(nrows=1, ncols=1)

    #     # if plot_rs:  ax0 = ax[0]
    #     # else: ax0 = ax
    #     fig = plt.figure(figsize=(8,8))
    #     ax0 = fig.add_subplot(111, projection="3d")

    #     if showtime is None:
    #         showtime = [-self.orbit.T/2, self.orbit.T/2]
            
    #     # _t = np.linspace(showtime[0], showtime[1], 307)
 
            
    #     show_cond  = np.logical_and(self.orbit.ttab > showtime[0], 
    #                                 self.orbit.ttab < showtime[1])
        
    #     ############# ------ disk passage related stuff ------- ###############

    #     t1, t2 = self.times_of_disk_passage
    #     # print('disk equator passage times [days]:')
    #     # print(t1/DAY, t2/DAY)
    #     vec_disk1, vec_disk2 = self.vectors_of_disk_passage

        
    #     orb_x, orb_y = self.orbit.xtab[show_cond], self.orbit.ytab[show_cond]
    #     x_scale = np.max(np.array([
    #         np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
    #         ]))
    #     y_scale = np.max(np.array([
    #             np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
    #             ]))
        
    #     coord_scale = np.max(np.array([
    #         np.min(orb_x), np.max(orb_x), np.min(orb_y), np.max(orb_y),
    #         np.max( (orb_x**2 + orb_y**2)**0.5 )
    #         ]))
        
    #     ################### --- 1. Star as a sphere --- #######################
    #     R_star = 1.0
    #     u = np.linspace(0, 2*np.pi, 40)
    #     v = np.linspace(0, np.pi, 20)
    #     xs = R_star * np.outer(np.cos(u), np.sin(v))
    #     ys = R_star * np.outer(np.sin(u), np.sin(v))
    #     zs = R_star * np.outer(np.ones_like(u), np.cos(v))
    #     ax0.plot_surface(xs, ys, zs, color="yellow", alpha=0.6, linewidth=0)
        
    #     ################### ------ drawing the orbit again ------- ################
        
        
    #     ax0.plot(orb_x, orb_y, 0*orb_x, "b-")                                                    
    #     # ax0.scatter(0, 0, c='r')                                                  
    #     ax0.plot([np.min(orb_x),
    #                 np.max(orb_x)], [0, 0], [0, 0],
    #                 color='k', ls='--')      # line of symmetry os the orbit
    #     ax0.plot([0, coord_scale*cos(self.orbit.nu_los)],                            
    #             [0, coord_scale*sin(self.orbit.nu_los)],  [0, 0],
                
    #             color='g', 
    #             ls='--') # line from S to P        
    #     xx1, yy1, zz1 = vec_disk1                                                 
    #     xx2, yy2, zz2 = vec_disk2                                                 
    #     ax0.plot([xx1, xx2], [yy1, yy2], [zz1,zz2], color='orange', ls='--', lw=2)    # line of the disk plane
    #     ### tabulating pressure in the line plane perp to Be symmetry axis ####
        
    #     # axis of symmetry
    #     u_be = rotated_vector(alpha=self.alpha, incl=self.incl)   
    #     u_be = n_from_v(u_be)   # normalize
        
    #     # Build perpendicular basis (v,w)
    #     # Pick any vector not parallel to u
    #     tmp = np.zeros(3)
    #     tmp[np.argmin(np.abs(u_be))] = 1.0
    #     tmp = n_from_v(tmp)
    #     v = mycross(u_be, tmp); v = n_from_v(v)
    #     w = mycross(u_be, v); w = n_from_v(w)


    #     h_forp = np.linspace(np.min(orb_x)*3, np.max(orb_x)*3, 251) # height over the disk
    #     r_forp = np.linspace(np.min(orb_x)*3, np.max(orb_x)*3, 81)  # "radius from axis"
        
    #     # print("dot(u,v)=", mydot(u_be, v), " dot(u,w)=", mydot(u_be, w), " dot(v,w)=", mydot(v,w))
    #     # print("||u||,||v||,||w|| =", np.linalg.norm(u_be), np.linalg.norm(v), np.linalg.norm(w))
    #     # # Visual debug: plot the axis line
    #     # ax0.plot([0, u_be[0]*np.max(h_forp)], [0, u_be[1]*np.max(h_forp)], [0, u_be[2]*np.max(h_forp)], 'r-')

    #     # XX, YY = np.meshgrid(x_forp, y_forp, indexing="ij")
    #     disk_ps = np.zeros((h_forp.size, r_forp.size))
    #     for ir in range(r_forp.size):
    #         for ih in range(h_forp.size):
    #             vec_from_s_ = u_be * h_forp[ih] + v * r_forp[ir]
    #             r_ = absv(vec_from_s_)
    #             disk_ps[ih, ir] = (Winds.decr_disk_pressure(self, vec_from_s_) 
    #                             +
    #                             Winds.polar_wind_pressure(self, r_)
    #                             )
    #     disk_ps = np.log10(disk_ps)
        
    #     # Symmetrize by rotating YY around axis u
    #     n_phi = 60
    #     phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        
    #     # Build 3D coordinates
    #     Hp, Rp, Phip = np.meshgrid(h_forp, r_forp, phi, indexing="ij")
    #     # Coordinates in 3D: r = h*u + r*cos(phi)*v + r*sin(phi)*w
    #     coords = (
    #         Hp[..., None] * u_be[None, None, None, :] +
    #         Rp[..., None] * np.cos(Phip)[..., None] * v[None, None, None, :] +
    #         Rp[..., None] * np.sin(Phip)[..., None] * w[None, None, None, :]
    #     )

    #     X3, Y3, Z3 = coords[..., 0], coords[..., 1], coords[..., 2]
        
    #     # X3, Y3, Z3 = coords[...,0], coords[...,1], coords[...,2]
        
    #     # Repeat pressure along phi
    #     disk_ps3 = np.repeat(disk_ps[:,:,None], n_phi, axis=2)
        
    #     # Normalize pressure to [0,1]
    #     mask = ~np.isnan(disk_ps3)
    #     flatP = disk_ps3[mask].ravel()
    #     flatP = (flatP - flatP.min()) / (flatP.max()-flatP.min())


    #     # Choose a few iso levels near the top of logP
    #     top = np.nanmax(disk_ps3)
    #     levels = [#top - 0.5,
    #               top - 1.0, 
    #               top - 1.5
    #               ]   # tweak to taste
        
    #     plot_isosurface_parametric(ax0,
    #                                h_forp, r_forp, phi,
    #                                u_be, v, w,
    #                                disk_ps3,
    #                                levels=levels,
    #                                color=(1.0, 0.5, 0.0),
    #                                alpha=0.30,
    #                                step_size=2)   # step_size>1 speeds up extraction
        
    #     ax0.set_box_aspect((1,1,1))
    #     ax0.set_xlabel("x"); ax0.set_ylabel("y"); ax0.set_zlabel("z")
    #     # N = 20000
    #     # prob = flatP / flatP.sum()
    #     # idx = np.random.choice(flatP.size, size=N, p=prob)
        
    #     # xs = X3[mask].ravel()[idx]
    #     # ys = Y3[mask].ravel()[idx]
    #     # zs = Z3[mask].ravel()[idx]
    #     # alphas = flatP[idx]
        
    #     # # Per-point transparency: build RGBA manually (alpha ∝ normalized logP)
    #     # rgba = np.ones((N, 4))
    #     # rgba[:, 0:3] = np.array([1.0, 0.5, 0.0])   # orange RGB
    #     # rgba[:, 3] = alphas                        # per-point alpha
        
    #     # ax0.scatter(xs, ys, zs, s=3, c=rgba, linewidths=0)
    #     # from skimage.measure import marching_cubes
    #     # verts, faces, _, _ = marching_cubes(disk_ps3, level=-3)

    #     # # verts[:, (z,y,x)] → fractional indices, so map with interpolation
    #     # verts_x = np.interp(verts[:,2], np.arange(X3.shape[0]), X3[:,0,0])
    #     # verts_y = np.interp(verts[:,1], np.arange(X3.shape[1]), Y3[0,:,0])
    #     # verts_z = np.interp(verts[:,0], np.arange(X3.shape[2]), Z3[0,0,:])
        
    #     # ax0.plot_trisurf(verts_x, verts_y, faces, verts_z,
    #     #         color="orange", alpha=0.3, linewidth=0)
    #     ax0.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.orbit.r_periastr) )
    #     ax0.set_ylim(-1.2*y_scale, 1.2*y_scale) 

