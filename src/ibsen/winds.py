# ibsen/winds.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from astropy import constants as const
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params
from ibsen.utils import rotated_vector, mydot, mycross, n_from_v, absv, \
    enhanche_jump, angles_from_vec, vector_angle, orthonormal_basis_perp, rotate_vec1_around_vec2
from ibsen.orbit import Orbit


G = float(const.G.cgs.value)
C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

pulsar_docstring =     """
    Defines dependencies of the pulsar magnetic field and winds on
    distance.
    
    Parameters
    ----------
    
    f_p : float, optional
        Dimensionless pulsar-wind pressure normalization (~ 1/r^2). Default 0.1.
    r_p_ref : float, opional
        The refererence distance for the pulsar wind pressure: 
            P = f_p (r / r_p_ref)**2. Default 10 R_solar.
    b_model : 'linear', 'dipole', 'from_L_sigma', optional
        Model for the pulsar magnetic field with radius (see ``ns_field``). Default 'linear'.
    b_ref : float or None, optional
        Pulsar surface field parameter ``B_surf`` [G] for 'linear'/'dipole' models.
        Default 0
    r_b_ref : float or None, optional
        Scale radius ``r_ref`` [cm] for 'linear'/'dipole' models. Default 1e13
    L_spindown : float or None, optional
        Pulsar spin-down luminosity ``L`` [erg/s] for 'from_L_sigma'. Default None.
    sigma_magn : float or None, optional
        Magnetization ``σ`` for 'from_L_sigma'. Default None
        
    Methods
    ----------
    
     b(r_to_p):
         Magnetic field at the distance of t_to_p.
    wind_pressure(r_to_p):
        Pulsar wind pressure at the distance of t_to_p.
    """

class Pulsar: # !!!
    __doc__ = pulsar_docstring
    def __init__(self, #*args, **kwargs,
                 f_p = 0.1,
                 r_p_ref = 10. * R_SOLAR, # r-scale for the pulsar wind pressure

                 b_model = 'linear', # model of ns magn field
                 # ns_b_apex = None, 
                 b_ref = 0, # B- parameter for ns field scaling 
                 r_b_ref = 10. * R_SOLAR, # r-scale for NS field
                 L_spindown = None, # NS spin-down lum 
                 sigma_magn = None, # NS PWE magnetization
                 ):
        self.f_p = f_p
        self.r_p_ref = r_p_ref
        
        self.b_model = b_model
        # self.ns_b_apex = ns_b_apex
        self.b_ref = b_ref 
        self.r_b_ref = r_b_ref 
        self.L_spindown = L_spindown
        self.sigma_magn = sigma_magn
        # super().__init__(*args, **kwargs)
        
    def b(self, r_to_p):   
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

        if self.b_model not in model_opts:
            raise ValueError('the NS field model should be one of:',
                             model_opts)
        if self.b_model == 'from_L_sigma':        
            if self.L_spindown is None or self.sigma_magn is None:
                raise ValueError("""For this pulsar magnetic field model, you should
                                 \nprovide `L_spindown` and `sigma_magn`.""")
            b_puls = (self.L_spindown * self.sigma_magn / C_LIGHT / r_to_p**2)**0.5
                    
        elif self.b_model == 'linear':
            b_puls = self.b_ref * (self.r_b_ref / r_to_p)
        else:
            b_puls = self.b_ref * (self.r_b_ref / r_to_p)**3

        return b_puls
    
    def wind_pressure(self, r_from_p):
        """
        Dimensionless pulsar wind pressure at the distance r_from_p from the pulsar.r_ref
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
        return self.f_p * (self.r_p_ref / r_from_p)**2
    
star_docstring = """
    Stores the information about the optical star properties, including
    the physical properties, outflows pressures and velocities, 
    magnetic and photon fields.
    
    Parameters
    ----------

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
    alpha_disk_deg : float, optional
        Disk plane position angle: rotation about the +Z axis (see Notes). Default 0.
    incl_disk_deg : float, optional
        Disk axis inclination from +Z (0 means disk in the orbital plane). Default ``30 deg``.
    f_d : float, optional
        Dimensionless decretion-disk pressure normalization. Default 0.
    p_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk pressure. Default [1,].
    h_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk height. Default [1,].
    p_enh_true_an : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of true anomalies for pressure enhancement. Default [0,].
    h_enh_true_an : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of true anomalies for pressure enhancement. Default [0,].
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

    b_model : 'linear', 'dipole', optional
        Model for the optical star magnetic field with radius.
        Default 'linear'.
    b_ref : float or None, optional
        Stellar surface field parameter ``B_surf`` [G] for 'linear'/'dipole'.
    r_b_ref : float or None, optional
        Scale radius ``r_ref`` [cm] for 'linear'/'dipole'.

    Attributes
    ----------
    Topt, Ropt, Mopt : float
        Stellar temperature [K], radius [cm], and mass [g] used by the model.
    alpha_disk, incl_disk : float
        Disk orientation angles (radians).
    f_d : float
        Disk and pulsar-wind pressure normalizations (dimensionless).
    np_disk, delta, height_exp : float
        Disk radial index, base opening, and opening exponent.
    rad_prof : 'pl', 'bkpl'
        Choice of disk radial profile.
    r_trunk : float
        Disk truncation radius [cm] if using a broken power law.
    t1_pass, t2_pass : float
        Times of the pulsar passage through the disk plane
    b_model, b_ref, r_b_ref
        Parameters controlling the star magnetic field model.
    n_disk
        Unit normal vector to the disk plane (property).
        
    Methods
    -------
    
    b(r_to_s):
        The star magnetic field as a function of a distance to the star.
    u_g_density(r):
        The star photon field energy density as a function of a distance 
        to the star.
    times_of_disk_passage
        Two times within one period when the pulsar crosses the disk plane (property).
    vectors_of_disk_passage
        Star→pulsar vectors at those crossing times (property).
    disk_height(r_in_d):
        Disk height as a function of an equatorial distance from the star.
    decr_disk_pressure(vec_r_from_s, true_an):
        Decretion disk pressure.
    polar_wind_pressure(r_from_s):
        Polar wind pressure.
    u_disk_w_lab_frame(vec_from_s):
        The disk velocity vector.
    u_polar_w_lab_frame(vec_from_s):
        The polar wind velocity.
                """

class OpticalStar: #!!!
    __doc__ = star_docstring
    def __init__(self,    
            sys_name=None, # the system name; str,  or None
            sys_params=None, # system parameters dict
            allow_missing=False,
            Ropt=None,
            Topt=None,
            Mopt=None, 
            f_w = 1.0, #  dimentionless Be star polar wind pressure strength
            
            alpha_disk_deg = 0., # Be-star rotation axis position angle: 2d angle between two planes:
                       # the perpendicular to xy-plane and containing the Omega_Be vector
                       # and the X-0-Z plane.
            incl_disk_deg = 30.,   # Be-star rotation axis position angle: 
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
            
            p_enh = [1, ],
            p_enh_true_an = [0, ],
            h_enh = [1, ],
            h_enh_true_an = [0, ],
            
            b_model = 'linear', # model of the Be magn field
            # opt_b_apex = None,
            b_ref = 0, # B- parameter for Be field scaling 
            r_b_ref = None, # r-scale for Be magn field):
                ):
        
        Topt_, Ropt_, Mopt_ = unpack_params(('Topt', 'Ropt', 'Mopt'),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names, get_defaults_func=get_parameters,
                               Topt=Topt, Ropt=Ropt, Mopt=Mopt,
                               allow_missing=allow_missing)
        self.sys_name = sys_name
        self.Topt = Topt_
        self.Ropt = Ropt_
        self.Mopt = Mopt_
        self.GMopt = G * Mopt_
        self.alpha_disk = np.deg2rad(alpha_disk_deg)
        self.incl_disk = np.deg2rad(incl_disk_deg)
        self.f_d = f_d
        self.f_w = f_w
        self.v_polar_wind = v_polar_wind
        self.delta = delta
        self.height_exp = height_exp
        self.np_disk = np_disk
        self.rad_prof = rad_prof
        self.r_trunk = r_trunk
        self.n_disk = n_from_v(rotated_vector(alpha=self.alpha_disk, incl=self.incl_disk)  )
        
        self.p_enh = p_enh
        self.p_enh_true_an = p_enh_true_an
        self.h_enh = h_enh
        self.h_enh_true_an = h_enh_true_an
        
        
        self.b_model = b_model
        self.b_ref = b_ref 
        self.r_b_ref = r_b_ref if r_b_ref is not None else Ropt_
                
    def b(self, r_to_s):
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

        if self.b_model not in (model_opts):
            raise ValueError('the opt star field model should be one of:',
                             model_opts)

        if self.b_model == 'linear':
            b_opt = self.b_ref * (self.r_b_ref / r_to_s)
        if self.b_model == 'dipole':
            b_opt = self.b_ref * (self.r_b_ref / r_to_s)**3
        
        return b_opt
    
    def u_g_density(self, r_from_s):      
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
        factor = (self.Ropt / r_from_s)**2 # checked!
        
        u_dens = SIGMA_BOLTZ * self.Topt**4 / C_LIGHT * factor # checked!
        return u_dens
        
    def _delta_eff(self, true_an=0.):
        h_mult = enhanche_jump(t = true_an, # enhanceent along true anomaly
                               t1_disk=-pi/2. + self.alpha_disk, 
                               t2_disk=pi/2. + self.alpha_disk,
                               times_enh=self.h_enh_true_an, 
                               param_to_enh=self.h_enh)
        return self.delta * h_mult
    
    def _f_d_eff(self, true_an=0.):
        p_mult = enhanche_jump(t = true_an, # enhanceent along true anomaly
                               t1_disk=-pi/2. + self.alpha_disk, 
                               t2_disk=pi/2. + self.alpha_disk,
                               times_enh=self.p_enh_true_an, 
                               param_to_enh=self.p_enh)
        return self.f_d * p_mult
    
    def disk_height(self, r, true_an=0.):
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
        return self._delta_eff(true_an=true_an) * r * (r / self.Ropt)**self.height_exp

    
    def decr_disk_pressure(self, vec_r_from_s, true_an=0.):
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
        r_indisk = vec_r_from_s - r_fromdisk # (shape0, 3)
        r_to_d = absv(r_fromdisk) # (shape0)
        r_in_d = absv(r_indisk) # (shape0)
        z0 = self.disk_height(r_in_d, true_an=true_an) # (shape0)
        vert = np.exp(-r_to_d**2 / 2 / z0**2) # (shape0)
        if self.rad_prof == 'pl':
            rad = (self.Ropt / r_in_d)**self.np_disk  # (shape0)
        if self.rad_prof == 'bkpl':
            rad = np.where(r_in_d < self.r_trunk,
                    (self.Ropt / r_in_d)**self.np_disk,
                    (self.Ropt / self.r_trunk)**self.np_disk * (self.r_trunk / r_in_d)**2
                           )
        return self._f_d_eff(true_an=true_an) * rad * vert 


    def polar_wind_pressure(self, r_from_s):
        """
        Dimensionless Be-star polar wind pressure at the distance r_from_s from the star.
        Assuming isotropic polar wind and constant velocity.
        The coefficient is set so that at r=Ropt, P_w = f_w.
        Parameters
        ----------
        r_from_s : np.ndarray
            Distance from the star [cm].

        Returns
        -------
        np.ndarray
            P_w(r_from_s), dimensionless.

        """
        return self.f_w *(self.Ropt / r_from_s)**2 
    
    def u_disk_w_lab_frame(self, vec_se):
        """
        Vector of the disk matter velocity in a lab frame. 
        
        `vec_se` is a vector from the optical star.
        """
        norm_se = n_from_v(vec_se)
        vec_r_in_disc = vec_se - mydot(vec_se, self.n_disk) * self.n_disk
        v_disc_absv = np.sqrt(self.GMopt / absv(vec_r_in_disc))
        keplerian_direction = n_from_v(mycross(self.n_disk, norm_se))
        v_disc = v_disc_absv * keplerian_direction
        return v_disc
    

    def u_polar_w_lab_frame(self, vec_se):
        """
        Vector of the polar stellar wind in a lab frame.
        
        `vec_pe` is a vector from the star.
        """
        norm_se = n_from_v(vec_se)
        v_wind = norm_se * self.v_polar_wind
        relative_v_wind = v_wind 
        return relative_v_wind
    

winds_docstring = """
    Describes the pulsar and star's outflows collisions, the position of a 
    stand-off point as a function of time, and magnetic/photon fields in this point.
    
    Parameters
    ----------
    orbit : Orbit
        The binary orbit object providing positions/vectors as functions of time.
    star : OpticalStar
        The optical star object.
    pulsar : Pulsar
        The pulsar object.
    hyst : bool, optional
        Whether to include the hysteresis-like calculation. Default False
    t_precalculate : np.ndarray, optional
        On what time to precalculate r_sp in case of hysteresis. If None,
        chosen as to cover the -T/4 < t < T/4 part of the orbit. Default None
    k_time : float, optional
        Decay time constant for hysteresis calculation. Default 1.0
    alpha_interaction float, optional
        Coupling constant for hysteresis calculation. Default 1.0


    Attributes
    ----------
    
    orbit, star, pulsar
        Inputs
    t1_pass, t2_pass
        Times of passage through the equator of the disk.
    hyst, k_time, alpha_interaction, t_precalculate 
        For hysteresis calculation.
    times_of_disk_passage
        Two times within one period when the pulsar crosses the disk plane (property).
    vectors_of_disk_passage
        Star→pulsar vectors at those crossing times (property).
        
    Methods
    -------
    
    Dist_to_disk(rvec)
        Distance from a point to the disk plane (absolute value returned). 
    dist_se_1d(t)
        Distance from the star to the stand-off (shock apex) point along the star–pulsar line.
    u_polar_w_pulsar_frame(t, vec, which), u_disk_w_pulsar_frame(t, vec, which):
        Disk/polar wind matter velocity vectors relative to the pulsar motion.
    vec_polar_w(t, vec, which), vec_disk_w(t, vec, which), vec_pulsar_p(t, vec, which):
        Vectors of pressures defined as P * dot(n, vec{v}/v)* vec{v}/v 
    vec_pe_3d(t):
        A vector from pulsar to the stand-off point calculated in the 'flow'
        paradigm.    
    dist_pe(t):
        P-E distance. Unifies all prescriptions of PE-line orientation.
    beta_eff(t)
        Effective momentum-flux ratio at the apex, related to ``r_pe/r_sp``.
    dbeta_dn(t):
        Derivative d beta / dt.
    magn_fields_apex(t)
        Pulsar and stellar magnetic fields at the apex [G].
    u_g_density_apex(t)
        Stellar photon energy density at the apex [erg/cm3].
    """

class Winds: # !!!
    __doc__ = winds_docstring
    def __init__(self, 
                 orbit: Orbit, 
                 star: OpticalStar,
                 pulsar: Pulsar,
                 
                 hyst=False,
                 t_precalculate = None,
                 k_time = 1.0,
                 alpha_interaction = 1.0,
                 ):
        self.orbit = orbit
        self.star = star
        self.pulsar = pulsar
        t1_, t2_ = self.times_of_disk_passage
        self.t1_pass = t1_
        self.t2_pass = t2_
        self.sys_name = star.sys_name
        
        self.hyst = hyst
        if t_precalculate is not None:
            self.t_precalculate = t_precalculate
        else:
            self.t_precalculate = np.linspace(-self.orbit.T/4., self.orbit.T/4., 501)
        self.k_time = k_time
        self.alpha_interaction = alpha_interaction
        
        if self.hyst:
            self._precalculate_dist_pe_hyst()
    
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
        Dist_to_disk_time = lambda t: mydot(self.orbit.vector_sp(t), self.star.n_disk)
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
        pres_w = lambda r_se: self.star.polar_wind_pressure(r_from_s = r_se)
        pres_d = lambda r_se: self.star.decr_disk_pressure(vec_r_from_s = nwind * r_se, true_an=self.orbit.true_an(t))
        pres_p = lambda r_se: self.pulsar.wind_pressure(r_from_p = np.abs(r_sp - r_se))
        to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
        rse = brentq(to_solve, self.star.Ropt, r_sp*(1-1e-8))
        ### ---------------- test if the solution is good -------------------------
        p_ref = pres_p(rse)
        max_rel_err = np.max(to_solve(rse) / p_ref)
        if max_rel_err > 1e-3:
            raise ValueError(f'Huge relative error of {max_rel_err} in the solution of the '
                          'disk-wind balance equation at {t/DAY}    days. ')       

        return rse
    
    def _precalculate_dist_pe_hyst(self):
        _t_ev = self.t_precalculate
        if _t_ev is None:
            _t_ev = self.orbit.t_from_true_an(nu = np.linspace(-0.75*pi, 0.75*pi))
        else:
            _t_ev = np.asarray(_t_ev)
        def _rhs(t, f_add, k_time, eps1, return_rpe=False):
            r_sp = self.orbit.r(t)
            t_kepl = self.orbit.kepl_period(t)
            r_sp_vec = self.orbit.vector_sp(t)
            _nu = self.orbit.true_an(t)
            nwind = n_from_v(r_sp_vec) # unit vector from S to P
            pres_p = lambda r_se: self.pulsar.wind_pressure(r_from_p = np.abs(r_sp - r_se))
            pres_ext = lambda r_se: (self.star.polar_wind_pressure(r_from_s = r_se) + 
                                     self.star.decr_disk_pressure(vec_r_from_s = nwind * r_se, true_an=_nu) )
            to_solve = lambda r_se: pres_ext(r_se) + f_add - pres_p(r_se)
            # rse = brentq(to_solve, self.star.Ropt, r_sp*(1-1e-8), rtol=1e-7)
            try:
                rse = brentq(to_solve, r_sp*0.01, r_sp*(1-1e-6), rtol=1e-7)
            except:
                print(to_solve(r_sp*0.5), to_solve(r_sp*0.8), to_solve(r_sp*(1-1e-6)))
                print(f_add)
                print(t / DAY)
            rse = max(self.star.Ropt, rse)
            char_outer_p = pres_ext(r_se = r_sp)
            rpe = np.abs(r_sp - rse)
            if return_rpe:
                return rpe
            if f_add <= 0.0:
                return 0.
            return (
                    (-(f_add - 1e-5*char_outer_p) / k_time +
                    eps1 * pres_ext(rse) * (rpe / self.orbit.r_periastr)**(-3)
                    )  / (t_kepl)
                    )
        
        t_i = _t_ev.min()
        t_f = _t_ev.max()


        sol = solve_ivp(fun = _rhs, t_span = [t_i, t_f], y0 = (1e-17,), dense_output=True,
                        args=(self.k_time, self.alpha_interaction, False), atol=1e-10, rtol=1e-6, )
        f_adds = sol.sol(_t_ev)[0]
        rpes_new = np.empty(f_adds.shape)
        for (i, f), t in zip(enumerate(f_adds), _t_ev):
            rpes_new[i] = _rhs(t=t, f_add=f, k_time=self.k_time,
                               eps1=self.alpha_interaction, return_rpe=True)
        self.rpes_hyst = rpes_new
        

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
        if not self.hyst:
            if t_.ndim == 0:
                return float(self._dist_se_1d_nonvec(float(t_)))
            
            return np.array( [
                self._dist_se_1d_nonvec(t_now) for t_now in t_
                ] )
        if self.hyst:
            _spl = interp1d(x = self.t_precalculate, 
                            y=self.orbit.r(self.t_precalculate) - self.rpes_hyst,
                            bounds_error=True)
            return _spl(t)
    
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
    

    def u_polar_w_pulsar_frame(self, t, vec, which='pe'):
        """
        Vector of the polar wind  relative to pulsar at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        v_orbital_pulsar =self.orbit.vector_v(t)
        vec_pe, vec_se = self._vecs(t, vec, which)
        v_wind = self.star.u_polar_w_lab_frame(vec_se = vec_se)
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
        relative_v_wind = self.u_polar_w_pulsar_frame(t=t, vec=vec, which=which)
        n_v_w = n_from_v(relative_v_wind)
        p_w = self.polar_wind_pressure(absv(vec_se)) * mydot(n_v_w, n_pe) * n_v_w
        return p_w
    
    def u_disk_w_pulsar_frame(self, t, vec, which='pe'):
        """
        Vector of the disk wind  (relative to pulsar?..) at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        v_orbital_pulsar =self.orbit.vector_v(t)
        vec_pe, vec_se = self._vecs(t, vec, which)
        v_disc = self.star.u_disk_w_lab_frame(vec_se=vec_se)
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
        relative_v_disc = self.u_disk_w_pulsar_frame(t=t, vec=vec, which=which)
        n_v_d = n_from_v(relative_v_disc)
        p_d = (self.star.decr_disk_pressure(vec_se, true_an=self.orbit.true_an(t)) 
               * mydot(n_v_d, n_pe) * n_v_d )
        return p_d
    
    def vec_pulsar_p(self, t, vec, which='pe'):
        """
        Vector of the pulsar pressure projection onto a vector pe
        at the point in space
        defined by `vec`, which is treated either as pe- or se-vector depending
        on the  `which`={'se', 'pe'}.
        """
        vec_pe, vec_se = self._vecs(t, vec, which)
        p_p = self.pulsar.wind_pressure(r_from_p=absv(vec_pe)) * n_from_v(vec_pe)
        return p_p
    
    
    def make_vec_(self, r, theta, phi):
        """
        Vector vec_se is defined as such: its length is r_se, while theta
        and phi are the positional angles relative to the orbit coordinate
        system.
        """
        return r * rotated_vector(phi, theta)
    
    
    def _vec_pe_3d_novec(self, t, eps=1e-3):
        """
        A vector from the pulsar to the emission zone calculated in 3d. 
        
        t should be float [s]
        """
        vec_sp = self.orbit.vector_sp(t)
        _nu = self.orbit.true_an(t)
        rsp = absv(vec_sp)
            
        r_se_1d = self.dist_se_1d(t) # zero approximation
        r_pe_1d = rsp - r_se_1d

        pw = self.star.polar_wind_pressure(r_se_1d)
        effective_se = vec_sp*r_se_1d/rsp 
        
        pd = self.star.decr_disk_pressure(effective_se, true_an=_nu)
        totp = pw+pd
        vec_pw_zero = (pw+1e-3*totp) * n_from_v(self.u_polar_w_pulsar_frame(t, effective_se, 'se'))
        vec_pd_zero = (pd+1e-3*totp) * n_from_v(self.u_disk_w_pulsar_frame(t, effective_se, 'se'))

        tot_ext_at_p = -vec_pw_zero - vec_pd_zero

        init_direction = n_from_v(tot_ext_at_p)

        return r_pe_1d * init_direction
    

    def vec_pe_3d(self, t, eps=1e-3): 
        """
        A vector from the pulsar to the emission zone calculated in 3d. 
        
        t is a float or a 1d np.ndarray [s].
        """
        t_ = np.asarray(t)
        if t_.ndim == 0:
            return self._vec_pe_3d_novec(float(t), eps)
        
        return np.array( [
            self._vec_pe_3d_novec(t_now, eps) for t_now in t_
            ] )
    
    def dist_pe(self, t, orientation=None, return_se=False):
        """
        An absolute value of the distance from the pulsar to the emission zone. 
        
        t should be float [s]
        """
        r_sp = self.orbit.r(t)
        if orientation is None:
            r_se = self.dist_se_1d(t)
            r_pe = r_sp - r_se
        else:
            vec_pe = self.vec_pe_3d(t, 1e-3)
            vec_sp = self.orbit.vector_sp(t)
            r_pe = absv(vec_pe)
            r_se = absv(vec_sp + vec_pe)
        if not return_se:
            return r_pe
        return r_pe, r_se
        
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
    
    def dbeta_dn(self, t, eps=1e-4):
        """
        calculates T_orb d(beta_eff)/dt
        """
        _nu = self.orbit.true_an(t)
        t_i, t_f = self.orbit.t_from_true_an(_nu - pi*eps), self.orbit.t_from_true_an(_nu + pi*eps)
        return  (self.beta_eff(t_f) - self.beta_eff(t_i) ) / 2. / (t_f - t_i) * self.orbit.T
    
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

        r_pe, r_se = self.dist_pe(t, orientation, return_se=True)
        _b_puls_apex = self.pulsar.b(r_to_p=r_pe)
        _b_opt_apex = self.star.b(r_to_s = r_se)
        return _b_puls_apex, _b_opt_apex
    
    
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
        r_pe, r_se = self.dist_pe(t, orientation, return_se=True)
        return self.star.u_g_density( r_from_s = r_se)
    
    def peek(self, ax=None,
             showtime = None,
             plot_rs = True,
             t_forwinds=0.):
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
            import matplotlib.pyplot as plt
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
        _nu = self.orbit.true_an(t_forwinds)
        disk_ps = (self.star.decr_disk_pressure(vec_from_s_, true_an=_nu) + 
                   self.star.polar_wind_pressure(absv(vec_from_s_)))
        disk_ps = np.log10(disk_ps) 

        from matplotlib.colors import ListedColormap

        ########### some magic for displaying the winds, never mind ###########
        orange_rgba = np.array([1.0, 0.5, 0.0, 1.0]) 
        n_levels = 20
        colors = np.tile(orange_rgba[:3], (n_levels, 1))  
        alphas = np.linspace(0, 1, n_levels)           
        colors = np.column_stack((colors, alphas))     
        custom_cmap = ListedColormap(colors)
        disk_ps[(XX**2 + YY**2)**0.5 < self.star.Ropt] = np.nan
        disk_ps[disk_ps < np.nanmax(disk_ps)-4.5] = np.nan
        ax0.contourf(XX, YY, disk_ps, levels=n_levels, cmap=custom_cmap)          
        
        # ax0.contour(XX, YY, pres_diff,  levels=[0], colors='k')

        ax0.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.orbit.r_periastr) )
        ax0.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        #######################################################################    
        
        if plot_rs:
            dists_se = self.dist_se_1d( _t)
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
