# ibsen/winds.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq, least_squares, root
from astropy import constants as const
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params
from ibsen.utils import rotated_vector, mydot, mycross, n_from_v, absv, \
    enhanche_jump, angles_from_vec, vector_angle, orthonormal_basis_perp, rotate_vec1_around_vec2
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
        Time on which to calculate the winds structure. Default 0.
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
    ns_b_apex : float or None, optional
        The pulsar-originating field [G] in the apex at time t_forwinds. If 
        None (default), not used.
    ns_r_ref : float or None, optional
        Scale radius ``r_ref`` [cm] for 'linear'/'dipole' models.
    ns_L_spindown : float or None, optional
        Pulsar spin-down luminosity ``L`` [erg/s] for 'from_L_sigma'.
    ns_sigma_magn : float or None, optional
        Magnetization ``σ`` for 'from_L_sigma'.
    opt_b_model : 'linear', 'dipole', optional
        Model for the optical star magnetic field with radius (see ``opt_field``).
        Default 'linear'.
    opt_b_apex : float or None, optional
        The optical star-originating field [G] in the apex at time t_forwinds. If 
        None (default), not used.
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
    t1_pass, t2_pass : float
        Times of the pulsar passage through the disk plane
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
    ns_field_initialized(r, t), opt_field_initialized(r, t) 
        NS/opt star magnetic fields at distance r from NS/star. Argument t 
        specifies at what moment the apex field(s) were specified during 
        initialization.
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
                 f_w = 1.0, #  dimentionless Be star polar wind pressure strength
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
                 v_polar_wind = 3e8, # (constant) radial velocity of polar wind
                 ### for modeling jumps in pressure/height of the disk with time
                 t_forwinds = 0,
                 p_enh = [1, ],
                 p_enh_times = [0, ],
                 h_enh = [1, ],
                 h_enh_times = [0, ],
                
                 ### for magn fields of NS and opt star
                 ns_b_model = 'linear', # model of ns magn field
                 ns_b_apex = None, 
                 ns_b_ref = 0, # B- parameter for ns field scaling 
                 ns_r_ref = None, # r-scale for NS field
                 ns_L_spindown = 0, # NS spin-down lum 
                 ns_sigma_magn = 0, # NS PWE magnetization
                 
                 opt_b_model = 'linear', # model of the Be magn field
                 opt_b_apex = None,
                 opt_b_ref = 0, # B- parameter for Be field scaling 
                 opt_r_ref = None, # r-scale for Be magn field
                 ):
        Topt_, Ropt_, Mopt_ = unpack_params(('Topt', 'Ropt', 'Mopt'),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names, get_defaults_func=get_parameters,
                               Topt=Topt, Ropt=Ropt, Mopt=Mopt,
                               allow_missing=allow_missing)
        self.orbit = orbit
        self.sys_name = sys_name
        self.Topt = Topt_
        self.Ropt = Ropt_
        self.Mopt = Mopt_
        self.M_ns = M_ns
        
        self.alpha = alpha
        self.incl = incl
        self.initiate_disk(f_d, delta, t_forwinds, p_enh,
                           p_enh_times, h_enh, h_enh_times) # sets self.delta and self.f_d 
        self.f_p = f_p
        self.f_w = f_w
        self.np_disk = np_disk
        self.height_exp = height_exp
        self.rad_prof = rad_prof
        if r_trunk is None:
            self.r_trunk = 5 * Ropt_
        else:
            self.r_trunk = r_trunk
            
        self.v_polar_wind = v_polar_wind
        self.ns_b_model = ns_b_model
        self.ns_b_apex = ns_b_apex
        self.ns_b_ref = ns_b_ref 
        self.ns_r_ref = ns_r_ref  if ns_r_ref is not None else Ropt_
        self.ns_L_spindown = ns_L_spindown
        self.ns_sigma_magn = ns_sigma_magn
        
        self.opt_b_model = opt_b_model
        self.opt_b_apex = opt_b_apex
        self.opt_b_ref = opt_b_ref 
        self.opt_r_ref = opt_r_ref  if opt_r_ref is not None else Ropt_
        self.unit_star = n_from_v(rotated_vector(alpha=alpha, incl=incl)  )
        
    
    def initiate_disk(self, f_d, delta, t_forwinds, p_enh, p_enh_times,
                      h_enh, h_enh_times):
        """ sets self.f_d and self.delta at the moment t_forwinds"""
        self.t_forwinds = t_forwinds
        self.p_enh = p_enh
        self.h_enh = h_enh
        self.p_enh_times = p_enh_times
        self.h_enh_times = h_enh_times
        t1_, t2_ = self.times_of_disk_passage
        self.t1_pass = t1_
        self.t2_pass = t2_
        
        if t_forwinds is None:
            self.f_d = f_d
            self.delta = delta
        else:
            f_d_mult = enhanche_jump(t = t_forwinds, t1_disk=t1_, t2_disk=t2_,
                                     times_enh=p_enh_times, param_to_enh=p_enh)
            h_mult = enhanche_jump(t = t_forwinds, t1_disk=t1_, t2_disk=t2_,
                                     times_enh=h_enh_times, param_to_enh=h_enh)
            self.f_d = f_d * f_d_mult
            self.delta = delta * h_mult
        
    
    def ns_field(self, r_to_p, model='linear', B_apex=None, t_b_ns = None,
                 B_ref = None, r_ref = None,
                 L_spindown = None, sigma_magn = None, orientation=None):   
        """
        The magnetic field of the NS [G] at the distance r_to_p. If B_apex 
        is provided, the field is calculated relative to the IBS apex 
        (point where P_disk + P_polar = P_pulsar) according to `model`. In 
        this case, t_b_ns should be provided.
        
        If B_apex is not
        provided, the B_ref, r_ref, ... are used.
        
        
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
            
        B_apex : float, optional
            The field [G] at the distance r_apex from NS. If None, calculates
            the field from B_ref, r_ref, ...
        t_b_ns : floar, optional
            The time [s] at which to calculate r_apex.
        B_ref : float, optional
            The field [G] at the distance r_ref from NS. Default is None.
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
        if B_apex is not None:    
            if t_b_ns is None:
                # t_b_ns = 
                raise ValueError("""To calculate NS field from apex 
                                 value, provide t_b_ns.""")
            # r_se = self.dist_se_1d(t_b_ns)
            # r_pe = self.orbit.r(t_b_ns) - r_se
            r_pe = self.dist_pe(t_b_ns, orientation)
            if model in ('linear', 'from_L_sigma'):
                b_puls = B_apex * r_pe / r_to_p
            else:
                b_puls = B_apex * r_pe**3 / r_to_p**3
        else:
            
            if model == 'linear':
                b_puls = B_ref * (r_ref / r_to_p)
            if model == 'dipole':
                b_puls = B_ref * (r_ref / r_to_p)**3
            
            if model == 'from_L_sigma':        
                b_puls = (L_spindown * sigma_magn / C_LIGHT / r_to_p**2)**0.5
        
        return b_puls
    
    def ns_field_initialized(self, r_to_p, t, orientation=None):
        """
        Helper that collects all parameters passed to Winds class and returns
        the NS field at distance r_to_sp, supposing that if any B_apex was
        provided, it was an apex at time t.

        Parameters
        ----------
        r_to_p : float | np.ndarray
            The distance [cm] from the neutron star (pulsar) to the point of
            interest.
        t : float 
            The time at which, it will be assumed, B_apex is passed 
            (generally, different from t_forwinds).

        Returns
        -------
        float | np.ndarray
            The ns-originating magnetic field.

        """
        return self.ns_field(r_to_p, model=self.ns_b_model,
                              B_apex=self.ns_b_apex, t_b_ns = t,
                     B_ref = self.ns_b_ref, r_ref = self.ns_r_ref,
                     L_spindown = self.ns_L_spindown,
                     sigma_magn = self.ns_sigma_magn,
                     orientation=orientation)
    
    def opt_field(self, r_to_s, model = 'linear', B_apex=None, 
                  t_b_opt = None,
                  r_ref=None, B_ref=None, orientation=None):
        """
        The magnetic field of the opt. star [G] at the distance r_to_s.
        If B_apex 
        is provided, the field is calculated relative to the IBS apex 
        (point where P_disk + P_polar = P_pulsar) according to `model`. In 
        this case, t_b_opt should be provided.
        
        If B_apex is not
        provided, the B_ref, r_ref, ... are used.
        
        Parameters
        ----------
        r_to_s : float
            The distance from the star [cm] to the point.
        B_apex : float, optional
            The field [G] at the distance r_apex from NS. If None, calculates
            the field from B_ref, r_ref, ...
        t_b_opt : float, optional
            The time [s] at which to calculate r_apex.
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
        if B_apex is not None:    
            if t_b_opt is None:
                raise ValueError("""To calculate Opt field from apex 
                                 value, `Winds` class should be initialized
                                 with t_forwinds value.""")
            # r_se = self.dist_se_1d(t_b_opt)
            r_pe = self.dist_pe(t_b_opt, orientation)
            r_se = self.orbit.r(t_b_opt) - r_pe
            if model == 'linear':
                b_opt = B_apex * r_se / r_to_s
            else:
                b_opt = B_apex * r_se**3 / r_to_s**3
        else:
            if model == 'linear':
                b_opt = B_ref * (r_ref / r_to_s)
            if model == 'dipole':
                b_opt = B_ref * (r_ref / r_to_s)**3
        
        return b_opt
    
    def opt_field_initialized(self, r_to_s, t, orientation=None):
        """
        Helper that collects all parameters passed to Winds class and returns
        the optical star field at distance r_to_s, supposing that if any B_apex was
        provided, it was an apex at time t.

        Parameters
        ----------
        r_to_p : float | np.ndarray
            The distance [cm] from the optical star to the point of
            interest.
        t : float 
            The time at which, it will be assumed, B_apex is passed 
            (generally, different from t_forwinds).

        Returns
        -------
        float | np.ndarray
            The otical star-originating magnetic field.

        """
        return self.opt_field(r_to_s, model=self.opt_b_model,
                              B_apex=self.opt_b_apex, t_b_opt = t,
                     B_ref = self.opt_b_ref, r_ref = self.opt_r_ref,
                     orientation=orientation)
    
    def u_g_density(self, r_from_s, r_star, T_star):      
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
        # factor = 2. * (1. - (1. - (r_star / r_from_s)**2)**0.5 ) # checked!
        factor = (r_star / r_from_s)**2 # checked!
        
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
        return (self.Ropt / r_from_s)**2 * self.f_w
    
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
        vec_r_from_s : np.ndarray of shape (N1, ..., Nm, 3,)
            Vector from the star to the point.

        Returns
        -------
        np.ndarray of shape (N1, ..., Nm)
            P_d(vec_r_from_s), dimensionless.

        """
        ndisk = self.n_disk # (3,)
        r_fromdisk = mydot(vec_r_from_s, ndisk)[..., None] * ndisk # (shape0, 3)
        r_indisk = vec_r_from_s - r_fromdisk # (shaape0, 3)
        r_to_d = absv(r_fromdisk) # (shape0)
        r_in_d = absv(r_indisk) # (shape0)
        z0 = self.disk_height(r_in_d) # (shape0)
        vert = np.exp(-r_to_d**2 / 2 / z0**2) # (shape0)
        if self.rad_prof == 'pl':
            rad = (self.Ropt / r_in_d)**self.np_disk  # (shape0)
        if self.rad_prof == 'bkpl':
            rad = np.where(r_in_d < self.r_trunk,
                    (self.Ropt / r_in_d)**self.np_disk,
                    (self.Ropt / self.r_trunk)**self.np_disk * (self.r_trunk / r_in_d)**2
                           )

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
        pres_w = lambda r_se: self.polar_wind_pressure(r_from_s = r_se)
        pres_d = lambda r_se: self.decr_disk_pressure(vec_r_from_s = nwind * r_se)
        pres_p = lambda r_se: self.pulsar_wind_pressure(r_from_p = np.abs(r_sp - r_se))
        to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
        rse = brentq(to_solve, self.Ropt, r_sp*(1-1e-8))
        ### ---------------- test if the solution is good -------------------------
        p_ref = pres_p(rse)
        max_rel_err = np.max(to_solve(rse) / p_ref)
        if max_rel_err > 1e-3:
            raise ValueError(f'Huge relative error of {max_rel_err} in the solution of the '
                          'disk-wind balance equation at {t/DAY}    days. ')       

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
            return float(self._dist_se_1d_nonvec(float(t_)))
        
        return np.array( [
            self._dist_se_1d_nonvec(t_now) for t_now in t_
            ] )
    
    def _vecs(self, t, vec, which):
        """
        If `which` == 'se', treats `vec` as s-e-vector; if 
        `which` == 'pe', treats `vec` as a p-e-vector;
        returns vec_pe, vec_se.
        """
        vec_sp = self.orbit.vector_sp(t)
        if which == 'pe':
            vec_pe = vec
            vec_se = vec_pe + vec_sp
        elif which == 'se':
            vec_se = vec
            vec_pe = vec_se - vec_sp
        return vec_pe, vec_se
    
    def u_polar_w(self, t, vec, which='pe'):
        """
        Vector of the polar wind  relative to pulsar at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        v_orbital_pulsar =self.orbit.vector_v(t)
        vec_pe, vec_se = self._vecs(t, vec, which)
        norm_se = n_from_v(vec_se)
        v_wind = norm_se * self.v_polar_wind
        relative_v_wind = v_wind - v_orbital_pulsar[..., :]
        return relative_v_wind
    
    def vec_polar_w(self, t, vec, which='pe'):
        """
        Vector of the polar wind pressure projection onto a vector pe
        at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        vec_pe, vec_se = self._vecs(t, vec, which)
        n_pe = n_from_v(vec_pe)
        relative_v_wind = self.u_polar_w(t, vec, which)
        n_v_w = n_from_v(relative_v_wind)
        # p_w = self.polar_wind_pressure(absv(vec_se)) * mydot(n_v_w, norm_se) * n_v_w
        # p_w = self.polar_wind_pressure(absv(vec_se)) * n_v_w
        p_w = self.polar_wind_pressure(absv(vec_se)) * mydot(n_v_w, n_pe) * n_v_w
        # p_w = self.polar_wind_pressure(absv(vec_se)) * n_v_w    
        return p_w
    
    def u_disk_w(self, t, vec, which='pe'):
        """
        Vector of the disk wind  (relative to pulsar?..) at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        v_orbital_pulsar =self.orbit.vector_v(t)
        vec_sp = self.orbit.vector_sp(t)
        vec_pe, vec_se = self._vecs(t, vec, which)
        vec_se = vec_pe + vec_sp
        norm_se = n_from_v(vec_se)
        vec_r_in_disc = vec_se - mydot(vec_se, self.n_disk) * self.n_disk
        v_disc_absv = np.sqrt(self.orbit.GM / absv(vec_r_in_disc))
        keplerian_direction = n_from_v(mycross(self.n_disk, norm_se))
        v_disc = v_disc_absv * keplerian_direction
        relative_v_disc = v_disc - v_orbital_pulsar
        return relative_v_disc
    
    def vec_disk_w(self, t, vec, which='pe'):
        """
        Vector of the decretion disk pressure projection onto a vector pe
        at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        vec_sp = self.orbit.vector_sp(t)
        vec_pe, vec_se = self._vecs(t, vec, which)
        vec_se = vec_pe + vec_sp
        n_pe = n_from_v(vec_pe)
        relative_v_disc = self.u_disk_w(t, vec, which)
        n_v_d = n_from_v(relative_v_disc)
        # p_d = self.decr_disk_pressure(vec_se) * mydot(n_v_d, norm_se) * n_v_d
        # p_d = self.decr_disk_pressure(vec_se) * n_v_d
        p_d = self.decr_disk_pressure(vec_se) * mydot(n_v_d, n_pe) * n_v_d
        # p_d = self.decr_disk_pressure(vec_se) * n_v_d
        return p_d
    
    def vec_pulsar_p(self, t, vec, which='pe'):
        """
        Vector of the pulsar pressure projection onto a vector pe
        at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        vec_pe, vec_se = self._vecs(t, vec, which)
        # p_p = self.pulsar_wind_pressure(absv(vec_pe)) * norm_se
        p_p = self.pulsar_wind_pressure(absv(vec_pe)) * n_from_v(vec_pe)
        return p_p
    
    def make_vec_(self, r, theta, phi):
        """
        Vector vec_se is defined as such: its length is r_se, while theta
        and phi are the positional angles relative to the orbit coordinate
        system.
        """
        return r * rotated_vector(phi, theta)

    def _vec_pe_3d_novec(self, t, eps=1e-3, orientation='flow'):
        """
        
        """
        if orientation not in ('flow', 'projection', 'direction', 'full'):
            raise ValueError("Invalid keyword `orientation` in Winds.vec_pe_3d.")
        vec_sp = self.orbit.vector_sp(t)
        rsp = absv(vec_sp)
            
        r_se_1d = self.dist_se_1d(t) # zero approximation
        r_pe_1d = rsp - r_se_1d
        r_pe_wind = self.f_p**0.5 / (self.f_p**0.5+1.) * rsp
        
        def zero_approx_resid(q_red, r_to_use=None):
            theta, phi = q_red
            _n = rotated_vector(alpha=phi, incl=theta)
            if r_to_use is None:
                vec_pe_zero_approx = r_pe_1d * _n
            else:
                vec_pe_zero_approx = r_to_use * _n
            return (self.vec_disk_w(t, vec_pe_zero_approx) + 
                    self.vec_polar_w(t, vec_pe_zero_approx) - 
                    self.vec_pulsar_p(t, vec_pe_zero_approx) )

        q0_prev = [rsp, pi/2., self.orbit.true_an(t)+pi]

        count = 0
        def d_nn1_dth(n1, theta, phi):
            """
            d (n \dot n1) / d theta, where n = rotated_vector(phi, theta)
            """
            return mydot(n1, rotated_vector(phi, theta+pi/2.))
        
        def d_nn1_dph(n1, theta, phi):
            """
            d (n \dot n1) / d phi, where n = rotated_vector(phi, theta)
            """
            return mydot(n1, rotated_vector(phi+pi/2., theta))
        
        def jac_zero_approx(q_red, r_to_use=None):
            theta, phi = q_red
            _n = rotated_vector(alpha=phi, incl=theta)
            if r_to_use is None:
                vec_pe_zero_approx = r_pe_1d * _n
            else:
                vec_pe_zero_approx = r_to_use * _n
            vec_se = vec_pe_zero_approx + vec_sp
            vec_pw = self.vec_disk_w(t, vec_pe_zero_approx) 
            vec_pd = self.vec_polar_w(t, vec_pe_zero_approx) 
            vec_pp = self.vec_pulsar_p(t, vec_pe_zero_approx) 
            n_w = n_from_v(vec_pw)
            n_d = n_from_v(vec_pd)
            # n_p = n_from_v(vec_pp)
            pw = self.polar_wind_pressure(absv(vec_se))
            pd = self.decr_disk_pressure(vec_se)
            pp = self.pulsar_wind_pressure(absv(vec_pp))
            dnp_dtheta = rotated_vector(phi, theta+pi/2.)
            dnp_dphi = rotated_vector(phi+pi/2., theta)
            
            df_dth = pw * d_nn1_dth(n_w, theta, phi)*n_w + pd * d_nn1_dth(n_d, theta, phi)*n_d - pp * cos(phi)*dnp_dtheta
            df_dph = pw * d_nn1_dph(n_w, theta, phi)*n_w + pd * d_nn1_dph(n_d, theta, phi)*n_d - pp * cos(phi)*dnp_dphi
            return np.array([
                            [df_dth[0], df_dph[0]],
                            [df_dth[1], df_dph[1]],
                            [df_dth[2], df_dph[2]]
                            ])
        
            
        
        """
        Get initial direction estimation as a direction opposite of the 
        vector:
            P_w \vec{n_u_w} + P_d \vec{n_u_d} (as if 'external' pressure). 
        """

        q0_sp = np.array([r_pe_1d, pi/2., self.orbit.true_an(t)+pi])
        pw = self.polar_wind_pressure(r_se_1d)
        pd = self.decr_disk_pressure(vec_sp*r_pe_1d/rsp)
        totp = pw+pd
        vec_pw_zero = (pw+1e-3*totp) * n_from_v(self.u_polar_w(t, vec_sp*r_pe_1d/rsp, 'se'))
        vec_pd_zero = (pd+1e-3*totp) * n_from_v(self.u_disk_w(t, vec_sp*r_pe_1d/rsp, 'se'))
        
        tot_ext_at_p = -vec_pw_zero - vec_pd_zero
        # print('d / w = ', absv(vec_pd_zero) / absv(vec_pw_zero))
        # print('ang(d, w)', np.rad2deg(vector_angle(vec_pw_zero, vec_pd_zero)))
        # print('ang(result, w)', np.rad2deg(vector_angle(vec_pw_zero, vec_pw_zero)))
        
        init_direction = n_from_v(tot_ext_at_p)
        v_phi, v_theta = angles_from_vec(n_from_v(tot_ext_at_p))
        v_phi = (v_phi + 2. * pi) % (2. * pi)
        q0_ext_p = np.array([r_pe_1d, v_theta, v_phi])
        e1, e2, _ = orthonormal_basis_perp(init_direction)
        def from_loc_to_glob(theta, phi):
            """
            Treat theta, phi like local angles around init_direction; translate
            them into a real direction: theta_glob, phi_glob; 
            returns a normalized vector in that direction.
            """
            norm_1 = rotate_vec1_around_vec2(init_direction, e1, theta)
            norm_2 = rotate_vec1_around_vec2(norm_1, init_direction, phi)
            return n_from_v(norm_2)

        q0_full = [r_pe_1d, 0., 0.]
        if orientation == 'flow':
            _r, _theta, _phi = q0_full
            vec_pe_solution = from_loc_to_glob(_theta, _phi) * _r
            return vec_pe_solution
    
        

        def full_resid(q):
            
            r_pe, theta, phi = q
            
            # vec_pe = self.make_vec_(r_pe, theta, phi)
            vec_pe = r_pe * from_loc_to_glob(theta, phi)
            vec_p_w = self.vec_disk_w(t, vec_pe) 
            vec_p_d = self.vec_polar_w(t, vec_pe)
            vec_p_p = self.vec_pulsar_p(t, vec_pe)
            return (vec_p_d + vec_p_w - vec_p_p)
        
        def full_resid_scalar(r, direction):
            vec_pe = r * direction
            return (mydot(self.vec_disk_w(t, vec_pe), direction) + 
                    mydot(self.vec_polar_w(t, vec_pe), direction) -
                    self.pulsar_wind_pressure(r) )

        
        if orientation == 'projection':
            sol = root(full_resid_scalar, x0=r_pe_1d, args=(init_direction,),
                      tol=eps,)
            _r = sol.x[0]
            
            return _r * init_direction
            

        
        while (np.max( np.abs( (np.array(q0_prev) - np.array(q0_full) ) / np.array(q0_full) )) > eps) and (count < 10):
            rmin = 1e-3 * rsp
            # rmax = 0.49999 * rsp
            rmax = r_pe_wind*1.2

            
            lb_dir = [0., 0.]
            rb_dir = [pi/2., 2.*pi]
            lb = [rmin, 0., 0.]
            rb = [rmax, pi/2., 2.*pi]

            
            q0_prev = q0_full
            if orientation == 'direction':
                sol_direction = least_squares(zero_approx_resid, 
                                        x0=[q0_full[1], q0_full[2]],
                                          bounds=(lb_dir, rb_dir), args=(q0_full[0], ),
                                          ftol=eps, 
                                          # method='trf', 
                                          method='dogbox', 
                                          jac=jac_zero_approx,
                                          # jac='3-point',
                                          )

                q0_full = [q0_full[0], sol_direction.x[0], sol_direction.x[1]]         
            if orientation == 'full':
                sol_full = least_squares(fun=full_resid, x0=q0_full, bounds=(lb, rb),
                                         jac='3-point', 
                                         method='dogbox',
                                         # tr_solver='exact',
                                         tr_solver='lsmr',
                                         ftol=eps, xtol=1e-3,
                                         x_scale=(r_pe_1d, 0.1, 0.3),
                                         # method='trf'
                                         )
                q0_full = sol_full.x
            q0_prev = q0_full
            count += 1
            if count > 9:
                print('a!')

        _r, _theta, _phi = q0_full
        vec_pe_solution = from_loc_to_glob(_theta, _phi) * _r
        return vec_pe_solution        

    def vec_pe_3d(self, t, eps=1e-3, orientation='flow'): 
        t_ = np.asarray(t)
        if t_.ndim == 0:
            return self._vec_pe_3d_novec(float(t), eps, orientation)
        
        return np.array( [
            self._vec_pe_3d_novec(t_now, eps, orientation) for t_now in t_
            ] )
    
    def dist_pe(self, t, orientation=None):
        r_sp = self.orbit.r(t)
        if orientation is None:
            r_se = self.dist_se_1d(t)
            r_pe = r_sp - r_se
        else:
            vec_pe = self.vec_pe_3d(t, 1e-3, orientation)
            r_pe = absv(vec_pe)
        return r_pe
        
    def beta_eff(self, t, orientation=None):
        """
        The effective momentum flux ratio of the pulsar wind to the
        external medium (polar wind + decretion disk) at the apex point.
        Defined so that r_pe/r_sp = beta_eff^0.5 / (1 + beta_eff^0.5),

        Parameters
        ----------
        t : float | np.ndarray
            Time relative to periastron [s].
        orientation : str or None, optional
            How to calculate the orientation of the IBS at a given time. 
            If None (default), supposes the symmetry axis is the S-P line.
            Else, see method `vec_pe_3d`.

        Returns
        -------
        TYPE
            Effective winds momentum ratio at the apex point.

        """
        r_sp = self.orbit.r(t)
        r_pe = self.dist_pe(t, orientation)
        return (r_pe / (r_sp - r_pe))**2
    
    def magn_fields_apex(self, t, orientation=None):
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

        r_pe = self.dist_pe(t, orientation)
        r_se = self.orbit.r(t) - r_pe
        _b_ns_apex = self.ns_field(r_to_p = r_pe, model=self.ns_b_model,
                              B_ref = self.ns_b_ref, B_apex=self.ns_b_apex,
                              t_b_ns=self.t_forwinds,
                              r_ref = self.ns_r_ref,
                              L_spindown = self.ns_L_spindown,
                              sigma_magn =self.ns_sigma_magn)

        _b_opt_apex = self.opt_field(r_to_s = r_se,
                                     model = self.opt_b_model,
                                     B_apex=self.opt_b_apex,
                                     t_b_opt=self.t_forwinds,
                                     r_ref=self.opt_r_ref,
                                     B_ref=self.opt_b_ref)

        
        return _b_ns_apex, _b_opt_apex
    
    
    def u_g_density_apex(self, t, orientation=None): 
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
        r_pe = self.dist_pe(t, orientation)
        r_se = self.orbit.r(t) - r_pe
        return self.u_g_density( r_from_s = r_se,
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
                fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
            else:
                fig, ax0 = plt.subplots(nrows=1, ncols=1)
        else:
            if plot_rs:
                ax0, ax1 = ax
            else:
                ax0 = ax

        if showtime is None:
            showtime = [-self.orbit.T/2, self.orbit.T/2]
            
        _t = np.linspace(showtime[0], showtime[1], 307)
 
            
        show_cond  = np.logical_and(self.orbit.ttab > showtime[0], 
                                    self.orbit.ttab < showtime[1])
        
        ############# ------ disk passage related stuff ------- ###############

        t1, t2 = self.times_of_disk_passage
        vec_disk1, vec_disk2 = self.vectors_of_disk_passage
        _r_scale = absv(vec_disk1)
        
        orb_x, orb_y = self.orbit.xtab[show_cond], self.orbit.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x)), 1.5*_r_scale,
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y)), 1.5*_r_scale, 
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
        
        Nx = 301
        Ny = 201
        x_forp = np.linspace(np.min(orb_x)*3, np.max(orb_x)*4, Nx)
        y_forp = np.linspace(np.min(orb_y)*2, np.max(orb_y)*2, Ny)
        
        ###### ------------ tabulating pressure values ------------------ #####

        XX, YY = np.meshgrid(x_forp, y_forp, indexing='ij')
        disk_ps = np.zeros((x_forp.size, y_forp.size))

        vec_from_s_ = np.empty((Nx, Ny, 3))
        vec_from_s_[:, :, 0] = XX
        vec_from_s_[:, :, 1] = YY
        vec_from_s_[:, :, 2] = XX * 0

        disk_ps = self.decr_disk_pressure(vec_from_s_) + self.polar_wind_pressure(absv(vec_from_s_))
        disk_ps = np.log10(disk_ps) 

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
        ax0.contourf(XX, YY, disk_ps, levels=n_levels, cmap=custom_cmap)          
        
        # ax0.contour(XX, YY, pres_diff,  levels=[0], colors='k')

        ax0.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.orbit.r_periastr) )
        ax0.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        #######################################################################    
        
        if plot_rs:
            dists_se = Winds.dist_se_1d(self, _t)
            rs = self.orbit.r(_t)
            ax1.plot(_t/DAY, dists_se, label='se', ls='--')
            ax1.plot(_t/DAY, rs, label = 'sp', ls='-')
            ax1.plot(_t/DAY, rs-dists_se, label='pe', ls=':')
            ax1.axvline(x=t1/DAY, color='k', alpha=0.3)
            ax1.axvline(x=t2/DAY, color='k', alpha=0.3)
            ax1.axvline(x=self.orbit.t_los/DAY, color='g', ls='--', alpha=0.3)
            
            ax0.set_title('overview')
            ax1.set_title(r'$r_\mathrm{SP}, r_\mathrm{SE}, r_\mathrm{PE}$')
            ax1.legend()
