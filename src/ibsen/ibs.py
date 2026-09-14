# ibsen/ibs.py
import numpy as np
from numpy import pi, sin, cos
from scipy.interpolate import interp1d
from ibsen.winds import Winds
from ibsen.ibs_norm import IBS_norm, IBS_norm3D
from ibsen.utils import plot_with_gradient, \
 lor_trans_ug_iso, lor_trans_b_iso, lor_trans_Teff_iso, rotated_vector, absv, \
     plot_surface_quads, vector_angle, n_from_v, vec_between, trapz_loglog, doppler_delta,\
         project_point_for_gg
from ibsen.absorption.absorption import gg_analyt, gg_tab
from ibsen.get_obs_data import known_names

from astropy import constants as const

C_LIGHT = 2.998e10
DAY = 86400
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)
M_E = float(const.m_e.cgs.value)

def gamma_test(g0, par, coef):
    return 1. + (g0-1.) / (1. + coef * np.abs(par)**0.5) 

PEEK_KEYS = ('doppler', 'scattering', 'scattering_comoving',
                  'gg_tau', 'gg_abs', 'b_ns', 'b_ns_comov', 
                  'b_s', 'b_s_comov', 'b', 'b_comov', 
                  'ug', 'ug_comov', )

ibs_docstring = f"""
    Planar intrabinary shock (IBS) in physical cgs units at one orbital epoch.

    This adapter builds a **dimensionless** shock via :class:`IBS_norm` using the
    effective momentum-flux ratio `\\beta` from a supplied :class:`Winds`
    model at time ``t_to_calculate_beta_eff``, rotates it to align the symmetry
    axis with the instantaneous star–pulsar line, rescales all length-like
    quantities by the current separation :math:`r_{{\\rm sp}}(t)`, and then shifts
    the curve to the pulsar’s instantaneous position in the orbital plane.
    Unitless/angle-like properties are delegated to the underlying normalized
    object.

    Reference frames
    ----------------
    The normalized shock is defined relative to the pulsar and is rotated to
    the instantaneous star→pulsar direction.  After scaling by ``r_sp``, its
    Cartesian points are translated to the pulsar's barycentric position.
    Thus ``x``, ``y``, ``r_vec`` and their midpoint forms are barycentric
    coordinates.  ``vec_pIBS``/``r`` are pulsar→IBS vectors/distances, while
    ``vec_sIBS``/``r1`` are star→IBS vectors/distances.  Stellar radiation and
    magnetic-field quantities use the latter, never barycentric distances.

    Parameters
    ----------
    winds : Winds
        Wind environment tied to an :class:`Orbit`; used to compute
        :math:`\\beta_\\mathrm{{eff}}(t)`, separation ``r_sp(t)``, line of sight,
        and rotation. **Required** for construction.
    t_to_calculate_beta_eff : float
        Time (s) relative to periastron at which to evaluate
        :math:`\\beta_\\mathrm{{eff}}(t)` and position/orientation. Stored as
        ``t_forbeta``.
    s_max : float, optional
        Arclength cutoff (dimensionless) passed to :class:`IBS_norm`. Default is 1.0.
    gamma_max : float, optional
        Maximum bulk Lorentz factor reached at ``s_max_g``. Default is 3.0.
    s_max_g : float, optional
        Arclength (dimensionless) at which ``gamma == gamma_max``; passed through
        to :class:`IBS_norm` and later rescaled to cm. Default is 4.0.
    n : int, optional
        Sampling points used to build the normalized IBS (per horn, before
        mirroring). Default is 31.
    include_incl_in_los : bool, optional
        If True, include ``orbit.incl_los`` in the line-of-sight vector;
        otherwise use its projection onto the orbital plane. Default False.
    abs_gg_filename : str or None, optional
        Name of the file with tabukated gg-opacity. If None (default),
        IBSEn searches for a file tabulated for winds.sys_name if it is in
        the known names: {known_names}. Can also be one of known names, then 
        searches for a file tabulated for a system with this name.
    

    Attributes
    ----------
    winds : Winds
        The supplied winds model.
    t_forbeta : float
        Time (s) relative to the periastron passage at which the IBS is evaluated.
    abs_gg_filename : str or None :
        Name of the file with tablated gg-opacities.
    beta : float or ndarray
        Effective momentum-flux ratio at ``t_forbeta`` from ``winds.beta_eff``.
    peek_keys : tuple
        List of keys recognized by peek(ibs_color=<key>).
    r_sp : float
        Star–pulsar separation at ``t_forbeta`` [cm]; used for rescaling and
        shifting.
    ibs_n : IBS_norm
        Underlying **normalized** IBS rotated by ``π + true_an(t_forbeta)``.
    Array organization
    ------------------
    Point arrays have shape ``(Ns,)`` and contain both reflected horns in one
    signed-arclength ordering: from ``s=-s_max`` through the apex to
    ``s=+s_max``.  Segment/midpoint arrays, identifiable by ``_mid`` or
    ``ds``, have shape ``(Ns-1,)``.  ``*_m`` and ``*_p`` are segment endpoint
    arrays.  The coordinate arrays below are barycentric.

    x, y : ndarray, shape (Ns,)
        Barycentric IBS coordinates in the orbital plane [cm].
    s : ndarray
        Signed arclength along the IBS [cm]; increases through the apex.
    s_max, s_max_g : float
        Arclength cut and the location of ``gamma_max`` [cm] (rescaled from
        the normalized values).
    r, r1 : ndarray
        Pulsar→IBS and star→IBS distances, respectively [cm].
    r_mid, r1_mid : ndarray
        Distances from pulsar→IBS_mid and star→IBS_mid, respectively [cm].
    x_apex : float
        Pulsar→apex distance [cm].
    ds_dtheta : ndarray
        :math:`\\mathrm{{d}}s/\\mathrm{{d}}\\theta` along the curve [cm].
    s_m, s_p, s_mid, ds : ndarray
        Midpoint/arclength helper arrays [cm].
    x_m, x_p, x_mid, dx : ndarray
        X-coordinate midpoint helpers [cm].
    y_m, y_p, y_mid, dy : ndarray
        Y-coordinate midpoint helpers [cm].
    g, g_mid : ndarray
        Bulk Lorentz factor along the rescaled IBS/IBS_mid (via `gma`) 
        
    b_ns, b_ns_mid, b_ns_comov, b_ns_mid_comov : ndarray
        NS-originating magnetic field on the IBS, IBS_mid, IBS (in co-moving
            reference frame), IBS_mid (in co-moving reference frame)
    b_s, b_s_mid, b_s_comov, b_s_mid_comov : ndarray
        Optical star-originating magnetic field on the IBS, IBS_mid, IBS (in co-moving
            reference frame), IBS_mid (in co-moving reference frame)
    b, b_mid, b_comov, b_mid_comov : ndarray
        Total magnetic field on the IBS, IBS_mid, IBS (in co-moving
            reference frame), IBS_mid (in co-moving reference frame)
    ug, ug_mid, ug_comov, ug_mid_comov : ndarray
        Photon field energy density on the IBS, IBS_mid, IBS (in co-moving
            reference frame), IBS_mid (in co-moving reference frame)
        
        

    Methods
    -------
    calculate()
        Top-level staged calculation: build normalized IBS and rescale/rotate.
    calculate_normalized_ibs()
        Compute ``beta``, ``r_sp``, LoS, and create the rotated :class:`IBS_norm`.
    rescale_to_position()
        Rescale all length-like attributes to cm and shift to the pulsar’s position.
    s_interp(s_, what)
        Interpolate an attribute (e.g. ``'x'``, ``'y'``) at arclength ``s_`` [cm].
    gma(s)
        Bulk Lorentz factor at arclength ``s`` [cm].
    peek(fig=None, ax=None, show_winds=False, ibs_color='k', to_label=True, showtime=None)
        Quick-look plot of the IBS (optionally color-coded by other stuff,
                                    see `peek_keys`)
        and, if requested, the winds/orbit context. 

    Notes
    -----
    * The line of sight is built from the orbit’s ``nu_los`` as
      ``[cos(nu_los), sin(nu_los), 0]`` and the normalized IBS is rotated by
      ``π + true_an(t_forbeta)`` before rescaling, aligning the symmetry axis
      with the star–pulsar line. 
    * Most non-length attributes/methods (e.g., ``dopl``, scattering angles)
      are accessed via delegation to ``ibs_n`` (``__getattr__``). 

    See Also
    --------
    IBS_norm : Dimensionless IBS geometry used internally.
    Winds : Wind and radiation fields; provides ``beta_eff`` and orbital geometry.
    Orbit : Keplerian orbit used by :class:`Winds`.

    """

class IBS: #!!!
    __doc__ = ibs_docstring
    def __init__(self, t_to_calculate_beta_eff, s_max=1.0, gamma_max=3.0, s_max_g=4.0, n=31, 
                 winds = None, coef_quench=0.0,
                 abs_gg_filename = None,
                 include_incl_in_los=False):
        """Initialize and calculate the planar shock at one orbital epoch."""
        self.t_forbeta = t_to_calculate_beta_eff
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.coef_quench = coef_quench
        self.n = n
        self.winds = winds
        self.ug_apex = winds.u_g_density_apex(t=t_to_calculate_beta_eff)
        b_ns_apex, b_s_apex = winds.magn_fields_apex(t=t_to_calculate_beta_eff)
        self.b_ns_apex, self.b_s_apex = b_ns_apex, b_s_apex
        self.b_apex = b_ns_apex + b_s_apex
        
        self.abs_gg_filename = abs_gg_filename
        self.include_incl_in_los = include_incl_in_los
        self.peek_keys = PEEK_KEYS
        
        self.calculate()
        
    def calculate(self):
        """Build normalized planar geometry and rescale it to the selected epoch."""
        self.calculate_normalized_ibs()
        self.rescale_to_position()
        # self.rescale_gamma()
    
    def calculate_normalized_ibs(self):
        """Create the rotated dimensionless planar IBS at ``t_forbeta``.

        The normalized origin is the pulsar; its geometry is rotated into the
        physical relative star→pulsar orientation before rescaling.
        """
        self.beta = self.winds.beta_eff(self.t_forbeta)
        self.r_sp = self.winds.orbit.r(self.t_forbeta)
        self.r_se = self.winds.dist_se_1d(self.t_forbeta)
        if not self.include_incl_in_los:
            unit_los_ = np.array([cos(self.winds.orbit.nu_los),
                             sin(self.winds.orbit.nu_los),
                             0])
        else:
            unit_los_ = rotated_vector(alpha=self.winds.orbit.nu_los, incl=self.winds.orbit.incl_los)
        self.unit_los = unit_los_
        _nu_tr = self.winds.orbit.true_an(self.t_forbeta)
        # par = self.winds.dbeta_dn(self.t_forbeta)
        self.ibs_n = IBS_norm(beta=self.beta, s_max=self.s_max,
                        gamma_max=self.gamma_max,#gamma_test(g0=self.gamma_max,  par=par, coef=self.coef_quench), 
                        s_max_g=self.s_max_g, n=self.n,
                    unit_los=unit_los_).rotate(pi + _nu_tr)
        
    def rescale_to_position(self):
        """
        Scale the rotated planar shock and place its points barycentrically.

        All normalized lengths are multiplied by ``|orbit.vector_sp|``.
        Pulsar-relative shock vectors are translated by ``orbit.vector_p`` to
        form barycentric point coordinates.  Star-relative and pulsar-relative
        displacement vectors are then derived by subtracting ``vector_s`` and
        ``vector_p`` respectively.
        """
        r_sp_vec = self.winds.orbit.vector_sp(self.t_forbeta)
        r_p_vec = self.winds.orbit.vector_p(self.t_forbeta)
        r_s_vec = self.winds.orbit.vector_s(self.t_forbeta)
        _r_sp = absv(r_sp_vec)
        self.vec_sp = r_sp_vec
        self.vec_p = r_p_vec
        self.vec_s = r_s_vec
        self.x_s, self.y_s = r_s_vec[:2]
        self.x_pulsar, self.y_pulsar = r_p_vec[:2]

        # Build the normalized P->IBS vectors from the rotated planar
        # coordinates.  ``IBS_norm.rotate`` rotates x/y but does not refresh
        # its cached r_vec, so using that cache here would lose the orbital
        # rotation away from true anomaly zero.
        norm_r_vec = np.column_stack((
            self.ibs_n.x,
            self.ibs_n.y,
            np.zeros_like(self.ibs_n.x),
        )) * _r_sp
        norm_r_vec_mid = 0.5 * (norm_r_vec[:-1] + norm_r_vec[1:])
        self.r_vec = norm_r_vec + r_p_vec[None, :]
        self.r_vec_mid = norm_r_vec_mid + r_p_vec[None, :]

        # These are physical displacement vectors, not barycentric points.
        self.vec_pIBS = self.r_vec - r_p_vec[None, :]
        self.vec_pIBS_mid = self.r_vec_mid - r_p_vec[None, :]
        self.vec_sIBS = self.r_vec - r_s_vec[None, :]
        self.vec_sIBS_mid = self.r_vec_mid - r_s_vec[None, :]
        self.r1_vec = self.vec_sIBS
        self.r1_vec_mid = self.vec_sIBS_mid

        self.x, self.y = self.r_vec[:, 0], self.r_vec[:, 1]
        self.x_mid, self.y_mid = self.r_vec_mid[:, 0], self.r_vec_mid[:, 1]
        self.x_m, self.x_p = self.x[:-1], self.x[1:]
        self.y_m, self.y_p = self.y[:-1], self.y[1:]
        
        for name in ("s", "s_max_g", "x_apex", "ds_dtheta",
                     "s_m", "s_p", "s_mid", "ds", "dx", "dy", "s_max_cm"):
            setattr(self, name, _r_sp * getattr(self.ibs_n, name))
   
        
        self.r = absv(self.vec_pIBS)
        self.r_mid = absv(self.vec_pIBS_mid)
        self.r1 = absv(self.vec_sIBS)
        self.r1_mid = absv(self.vec_sIBS_mid)
        self.r_m, self.r_p = self.r[:-1], self.r[1:]
        self.r1_m, self.r1_p = self.r1[:-1], self.r1[1:]

        vec_apex = r_s_vec + self.r_se / _r_sp * r_sp_vec
        self.x_apex_coord, self.y_apex_coord = vec_apex[:2]
        self.symm_ax = n_from_v(r_sp_vec)
        self.scatter_angle_apex = vector_angle(self.symm_ax, self.unit_los)
        self.dopl_apex_eff = doppler_delta(self.gamma_max,
                        vector_angle(self.unit_los, self.symm_ax))
        

        
    # def rescale_gamma(self):
    #     # r_pIBS = np.array([np.array([_x, _y, 0.]) for _x, _y in zip(self.x, self.y)])
    #     vec_sp = self.winds.orbit.vector_sp(self.t_forbeta)
    #     vec_sIBS = self.r_vec
    #     vec_pIBS = -(vec_sp[None, :] - vec_sIBS)
    #     # r_sIBS = r_pIBS + r_sp[None, :]
    #     pdisk = self.winds.star.decr_disk_pressure(vec_sIBS)
    #     ppolar = self.winds.star.polar_wind_pressure(absv(vec_sIBS))
    #     ppulsar = self.winds.pulsar.wind_pressure(absv(vec_pIBS))
    #     dispers_out = np.std((pdisk + ppolar) / ppulsar )
    #     self.ibs_n.gamma_max = gamma_test(self.gamma_max, dispers_out, self.coef_quench)
        
    
        
    
    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_ [cm].
 
        Parameters
        ----------
        s_ : np.ndarray
            The arclength along the upper horn of the IBS to find the value at.
            [cm].

        Returns
        -------
        The desired value of ibs.what in the coordinate s_. 

        """
        try:
            data = getattr(self, what)
        except AttributeError:
            raise ValueError(f"No such attribute '{what}' in IBS.")
        ##### here I set fill_value='extrapolate' instead of raising an error_1horn
        ##### or like filling with NaNs, cause the values at the ends of an
        ##### IBS sometimes behave weirdly, and we DO need these values. So
        ##### since this is the internal function that should not be used
        ##### by an external user, we put `extrapolate` and use it VERY
        ##### cautiously!!!
        interpolator = interp1d(self.s, data, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return interpolator(s_)
    
    def gma(self, s):
        """Bulk Lorentz factor at signed planar arclength ``s`` [cm]."""
        return self.ibs_n.gma(s / self.r_sp)
    
    @property
    def g(self):
        """Bulk Lorentz factor along the IBS."""
        return self.gma(s = self.s)
    
    @property
    def g_mid(self):
        """Bulk Lorentz factor along the IBS-mid"""
        return self.gma(s = self.s_mid)
    
    @property
    def dopl_star(self):
        """Doppler-factor at the IBS for the direction to the star"""
        return doppler_delta(self.g,
                vector_angle(self.unit_beta, n_from_v(self.vec_sIBS))
                )
    
    @property
    def dopl_star_mid(self):
        """Doppler-factor at the IBS-mid for the direction to the star"""
        return doppler_delta(self.g_mid,
                vector_angle(self.unit_beta_mid, n_from_v(self.vec_sIBS_mid))
                )
    
    
    def gg_abs(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in every point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (Ns, e_phot.size)
            The tabulated/analytic opacity is evaluated with each emission
            point expressed relative to the optical star.
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x[i]-self.x_s, y = self.y[i]-self.y_s,
                         R_star=self.winds.star.R_s, T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for i in range(self.x.size)])
        else:
            gg_res = gg_tab(E=e_phot, x=self.x-self.x_s[None], y=self.y-self.y_s[None],
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    
    def gg_abs_mid(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in every mid point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorption/absorp_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorption for a `sys_name` provided for `winds` arg.
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (Ns-1, e_phot.size)
            Midpoint emission positions are expressed relative to the star.
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x_mid[i]-self.x_s, y = self.y_mid[i]-self.y_s,
                         R_star=self.winds.star.R_s, T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for i in range(self.x_mid.size)])
        else:
            gg_res = gg_tab(E=e_phot, x=self.x_mid-self.x_s[None], y=self.y_mid-self.y_s[None],
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    def gg_abs_apex(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in the IBS apex.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorption/absorp_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorption for a `sys_name` provided for `winds` arg.
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (e_phot.size, )
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')

        if analyt:
            gg_res = gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x_apex_coord - self.x_s,
                         y = self.y_apex_coord - self.y_s,
                         R_star=self.winds.star.R_s, T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
        else:
            gg_res = gg_tab(E=e_phot,
                            x=self.x_apex_coord - self.x_s,
                            y=self.y_apex_coord - self.y_s,
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    
    ###########################################################################
    @property
    def ug(self):
        """Photon field energy density on the IBS [erg/cm^3]."""
        return self.winds.star.u_g_density(r_from_s = self.r1)
    
    @property
    def ug_mid(self):
        """Photon field energy density on the IBS_mid [erg/cm^3]."""
        return self.winds.star.u_g_density(r_from_s = self.r1_mid)
    
    @property
    def ug_comov_iso(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3].
        Isotropc approximation."""
        return lor_trans_ug_iso(ug_iso = self.ug, gamma=self.g)
      
    @property
    def ug_mid_comov_iso(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3].
        Isotropc approximation."""          
        return lor_trans_ug_iso(ug_iso = self.ug_mid, gamma=self.g_mid)
      
    @property
    def ug_comov_ani(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3].
        Anisotropc approximation."""
        return self.ug / self.dopl_star**2
      
    @property
    def ug_mid_comov_ani(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3].
        Anisotropc approximation."""          
        return self.ug_mid / self.dopl_star_mid**2 
        
    ###########################################################################
    @property
    def b_pulsar(self):
        """Neutron star-originating magnetic field on the IBS [G]."""
        return self.winds.pulsar.b(r_to_p = self.r)
    
    @property
    def b_pulsar_mid(self):
        """Neutron star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.pulsar.b(r_to_p = self.r_mid)
    
    @property
    def b_pulsar_comov(self):
        """Neutron star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_pulsar, gamma=self.g)
    
    @property
    def b_pulsar_mid_comov(self):
        """Neutron star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_pulsar_mid, gamma=self.g_mid)
    
    
    ###########################################################################
    @property
    def b_s(self):
        """Optical star-originating magnetic field on the IBS [G]."""
        return self.winds.star.b(r_to_s = self.r1)
    
    @property
    def b_s_mid(self):
        """Optical star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.star.b(r_to_s = self.r1_mid,)
    
    @property
    def b_s_comov(self):
        """Optical star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_s, gamma=self.g)
    
    @property
    def b_s_mid_comov(self):
        """Optical star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_s_mid, gamma=self.g_mid)
    
    ###########################################################################
    @property
    def b(self):
        """Total magnetic field on the IBS [G]."""
        return self.b_pulsar + self.b_s
    
    @property
    def b_mid(self):
        """Total magnetic field on the IBS_mid [G]."""
        return self.b_pulsar_mid + self.b_s_mid
    
    @property
    def b_comov(self):
        """Total magnetic field on the IBS in the comoving frame [G]."""
        return self.b_pulsar_comov + self.b_s_comov
    
    @property
    def b_mid_comov(self):
        """Total magnetic field on the IBS_mid in the comoving frame [G]."""
        return self.b_pulsar_mid_comov + self.b_s_mid_comov
    
    ###########################################################################
    @property
    def T_s_eff(self):
        """Optical star effective temperature on the IBS [K]. 
        Simply the star temperature everywhere."""
        return self.winds.star.T_s * np.ones(self.r.size)
    
    @property
    def T_s_eff_mid(self):
        """Optical star effective temperature on the IBS_mid [K]. 
        Simply the star temperature everywhere."""
        return self.winds.star.T_s * np.ones(self.r_mid.size)

    @property
    def T_s_eff_comov_iso(self):
        """Optical star effective temperature on the IBS in the comoving frame [K].
        Isotropc approximation."""
        return lor_trans_Teff_iso(Teff_iso = self.T_s_eff, gamma=self.g)
    
    @property
    def T_s_eff_mid_comov_iso(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K].
        Isotropc approximation."""
        return lor_trans_Teff_iso(Teff_iso = self.T_s_eff_mid, gamma=self.g_mid)
    
    @property
    def T_s_eff_comov_ani(self):
        """Optical star effective temperature on the IBS in the comoving frame [K].
        Anisotropc approximation."""
        return self.T_s_eff / self.dopl_star
    
    @property
    def T_s_eff_mid_comov_ani(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K].
        Anisotropc approximation."""
        return self.T_s_eff_mid / self.dopl_star_mid

    
    peek_docs = f"""
    Quick look at the IBS in the orbital plane.

    Parameters
    ----------
    fig : fig object of pyplot, optional
         The default is None.
    ax : ax object of pyplot, optional
        DESCRIPTION. The default is None.
    show_winds : bool, optional
        Whether to show the winds (requires Winds to be provided).
          The default is False.
    ibs_color : str, optional
        Can be one of {PEEK_KEYS} 
        ; or any matplotlib color. 
        The default is 'k'.
    to_label : bool, optional
        Whether to put a label `beta=...` on a plot.
          The default is True.
    showtime : tuple of (tmin, tmax), optional
        For orbit displaying (see orbit.peek()). 
        The default is None.
    E_for_gg : float, optional
        At which energy [eV] to calculate the gamma-gamma absorption, if ibs_color
        is 'gg_tau' or 'gg_abs'. Default 1e12

    Raises
    ------
    ValueError
        If the ibs_color is not one of the recognizable options.

    Returns
    -------
    None.

    """
    def peek(self, fig=None, ax=None, show_winds=False,
             ibs_color='k', to_label=True,
             showtime=None, E_for_gg=1e12,
             special_contours_winds=None, kwargs_special_contours_winds={},
             min_param=None, max_param=None,
             label_colorbar=None,
             colorbar_kwargs={},
             plot_puls_pos=True,
             plot_line_to_puls=True):
        import matplotlib.colors as mcolors
        

        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))    

        if to_label:
            label = rf'$\beta = {self.beta}$'
        else:
            label = None



        if ibs_color in PEEK_KEYS:

            if ibs_color == 'doppler':
                color_param = self.dopl
            elif ibs_color == 'scattering':
                color_param = self.scattering_angle
            elif ibs_color == 'scattering_comoving':
                color_param = self.scattering_angle_comov
            elif ibs_color == 'gg_abs':
                color_param = IBS.gg_abs(self, e_phot=E_for_gg)
            elif ibs_color == 'gg_tau':
                color_param = IBS.gg_abs(self, e_phot=E_for_gg,
                                         what_return='tau')
            else:
                ### finds the attribute via getarray
                if not hasattr(self, ibs_color):
                    raise AttributeError(f"No attribute '{ibs_color}' in {type(self).__name__}")
                color_param = getattr(self, ibs_color)
            
            if label_colorbar is None:
                label_colorbar = ibs_color
            line_ = plot_with_gradient(fig=fig, ax=ax, xdata=self.x, ydata=self.y,
                            some_param=color_param, colorbar=to_label, lw=2, ls='-',
                            colorbar_label=label_colorbar, minimum=min_param, maximum=max_param,
                            colorbar_kwargs=colorbar_kwargs)

        elif mcolors.is_color_like(ibs_color):
            line_ = ax.plot(self.x, self.y, color=ibs_color, label = label)     
        else:
            raise ValueError(f"""ibs_colos should be either oe of
                             {PEEK_KEYS} or a matpotlib color.""")

        if show_winds:
            if not isinstance(self.winds, Winds):
                raise ValueError("You should provide winds:Winds to show the winds.")
            self.winds.peek(ax=ax, showtime=showtime,
                            plot_rs=False, special_contours=special_contours_winds,
                            kwargs_special_contours=kwargs_special_contours_winds)

        puls_vector = self.winds.orbit.vector_p(self.t_forbeta)
        star_vector = self.winds.orbit.vector_s(self.t_forbeta)
        _xp, _yp = puls_vector[0], puls_vector[1]
        _xs, _ys = star_vector[0], star_vector[1]
        if plot_puls_pos:
            ax.scatter(_xp, _yp, color='b') # pulsaR
        # ...and the star was alreay plotted in the winds.peek()
        if plot_line_to_puls:
            ax.plot([_xs, _xp], [_ys, _yp], color='k', alpha=0.3, ls='--')

        ##################################################################################
        if showtime is None:
            showtime = [-self.winds.orbit.T/2, self.winds.orbit.T/2]
        show_cond  = np.logical_and(self.winds.orbit.ttab > showtime[0], 
                                    self.winds.orbit.ttab < showtime[1])
        orb_x, orb_y = self.winds.orbit.xtab_p[show_cond], self.winds.orbit.ytab_p[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
            ]))
        
        ax.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.winds.orbit.r_periastr) )
        ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        
            
        # ax.legend()
        return line_ 
    peek.__doc__ = peek_docs

    
    def __getattr__(self, name):
        """Delegate normalized, dimensionless geometry attributes to ``ibs_n``."""
        ibs_n_ = self.__dict__.get("ibs_n", None)
        if ibs_n_ is None:
            raise AttributeError(name)
        return getattr(ibs_n_, name)
    
    
class IBS3D: #!!!
    """
    Axisymmetric three-dimensional intrabinary shock in physical cgs units.

    A normalized surface is rotated so that its axis is either the relative
    star→pulsar direction or a flow-defined direction from :class:`Winds`.
    Its points are then scaled by the relative separation and translated to
    the pulsar's barycentric position.  Hence ``r_vec`` is a barycentric point
    coordinate, whereas ``vec_pIBS`` and ``vec_sIBS`` are pulsar→IBS and
    star→IBS displacement vectors.  ``r`` is the pulsar distance and
    ``r1``/``r_from_s`` the stellar distance; stellar fields are evaluated
    only from the latter.

    Array organization
    ------------------
    Point arrays have shape ``(Nphi, Ns)``.  The first axis samples azimuth
    around the IBS symmetry axis and the second samples one horn from the
    apex outward.  Segment/midpoint arrays have shape ``(Nphi, Ns-1)``.
    Cartesian ``r_vec`` arrays append a final coordinate axis and therefore
    have shape ``(Nphi, Ns, 3)`` (or ``(Nphi, Ns-1, 3)`` at midpoints).
    ``*_m``/``*_p`` are lower/upper arclength segment endpoints.  Scalar
    apex quantities are single values at the selected epoch.
    
    Parameters
    ----------
    winds : Winds
        Wind environment tied to an :class:`Orbit`; used to compute
        :math:`\\beta_\\mathrm{{eff}}(t)`, separation ``r_sp(t)``, line of sight,
        and rotation. **Required** for construction.
    t_to_calculate_beta_eff : float
        Time (s) relative to periastron at which to evaluate
        :math:`\\beta_\\mathrm{{eff}}(t)` and position/orientation. Stored as
        ``t_forbeta``.
    s_max : float, optional
        Arclength cutoff (dimensionless) passed to :class:`IBS_norm`. Default is 1.0.
    gamma_max : float, optional
        Maximum bulk Lorentz factor reached at ``s_max_g``. Default is 3.0.
    s_max_g : float, optional
        Arclength (dimensionless) at which ``gamma == gamma_max``; passed through
        to :class:`IBS_norm` and later rescaled to cm. Default is 4.0.
    n : int, optional
        Sampling points used to build the normalized IBS (per horn, before
        mirroring). Default is 31.
    n_phi : int, optional
        The number of sampling points over azimuth, meaning, in the plane
            perpendicular to the line of symmetry. Default is 33.
    orientation : str or None, optional
        ``None`` aligns the symmetry axis with the relative star→pulsar line.
        ``'flow'`` obtains a pulsar→apex vector from :meth:`Winds.vec_pe_3d`
        and aligns the shock axis oppositely.  These are the consistently
        supported options in this class.  Although ``Winds`` also implements
        ``'flow_p'``, this class currently constructs its axis with the
        default ``'flow'`` vector; do not use ``'flow_p'`` here expecting a
        distinct axis.
        
    shield_star : float, optional
        Dimensionless multiplier for attenuation of the stellar photon field
        through integrated star-relative external pressure. Default 0.
    abs_gg_filename : str or None, optional
        Name of the file with tabukated gg-opacity. If None (default),
        IBSEn searches for a file tabulated for winds.sys_name if it is in
        the known names: {known_names}. Can also be one of known names, then 
        searches for a file tabulated for a system with this name.
    
    Important attributes
    --------------------
    ``x``, ``y``, ``z`` and ``r_vec`` are barycentric coordinates.  ``ug``,
    ``b_s``, ``disk_pressure`` and ``polar_pressure`` use star-relative
    geometry.  ``b_pulsar`` uses pulsar-relative geometry.  The normalized
    object and dimensionless angular quantities remain available through
    ``ibs_n`` and attribute delegation.
    """
    def __init__(self, t_to_calculate_beta_eff, s_max=1.0, gamma_max=3.0, s_max_g=4.0,
                 n=31, n_phi=33, orientation = None,
                 winds = None, coef_quench=0.0, shield_star = 0.0,
                 abs_gg_filename = None):
        """Initialize and calculate the axisymmetric 3D shock at one epoch."""
        self.t_forbeta = t_to_calculate_beta_eff
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.n = n
        self.n_phi = n_phi
        self.winds = winds
        self.orientation = orientation
        self.coef_quench = coef_quench
        self.shield_star = shield_star
        self.ug_apex = winds.u_g_density_apex(
            t=t_to_calculate_beta_eff, orientation=self.orientation
        )
        b_ns_apex, b_s_apex = winds.magn_fields_apex(
            t=t_to_calculate_beta_eff, orientation=self.orientation
        )
        self.b_ns_apex, self.b_s_apex = b_ns_apex, b_s_apex
        self.b_apex = b_ns_apex + b_s_apex
        
        self.abs_gg_filename = abs_gg_filename
        self.peek_keys = PEEK_KEYS
       
        self._calculate_normalized_ibs()
        self._rescale_to_position()
        # self.rescale_gamma()

    
    def _calculate_normalized_ibs(self):
        """Create the normalized 3D shock and align its symmetry axis.

        ``orientation=None`` follows the relative star→pulsar separation;
        otherwise the axis is inferred from the flow-defined pulsar→apex
        vector returned by :class:`Winds`.
        """
        vec_sp = self.winds.orbit.vector_sp(t = self.t_forbeta)
        self.r_sp = absv(vec_sp)
        if self.orientation is None:
            self.beta = self.winds.beta_eff(self.t_forbeta)
            self.r_se = self.winds.dist_se_1d(self.t_forbeta)
            self.r_pe = self.r_sp - self.r_se
            self.symm_ax = n_from_v(self.winds.orbit.vector_sp(self.t_forbeta))
        else:
            vec_pe_3d = self.winds.vec_pe_3d(self.t_forbeta) 
            self.r_pe = absv(vec_pe_3d)
            self.r_se = absv(vec_pe_3d + vec_sp)
            self.symm_ax = -n_from_v(vec_pe_3d)
            self.beta = self.winds.beta_eff(self.t_forbeta, orientation=self.orientation)
        unit_los_ = rotated_vector(self.winds.orbit.nu_los, self.winds.orbit.incl_los)
        self.unit_los = unit_los_

        par = self.winds.dbeta_dn(self.t_forbeta)
        self.ibs_n = IBS_norm3D(beta=self.beta, s_max=self.s_max,
            gamma_max=gamma_test(g0=self.gamma_max,  par=par, coef=self.coef_quench),
            s_max_g=self.s_max_g, n=self.n,
            n_phi=self.n_phi,
            unit_los=unit_los_).rotate_to_ax(new_axis=self.symm_ax)
        
    def _rescale_to_position(self):
        """Scale the 3D shock and translate its points to barycentric space.

        ``r_vec`` has shape ``(Nphi, Ns, 3)`` and is barycentric.  The
        star-relative and pulsar-relative displacement arrays are derived from
        it before computing fields and distances.
        """
        r_sp_vec = self.winds.orbit.vector_sp(self.t_forbeta)
        r_p_vec = self.winds.orbit.vector_p(self.t_forbeta)
        r_s_vec = self.winds.orbit.vector_s(self.t_forbeta)
        
        self.vec_sp = r_sp_vec
        self.vec_p = r_p_vec
        self.vec_s = r_s_vec
        
        _r_sp = absv(r_sp_vec)
        # ``ibs_n.r_vec`` is P->IBS.  Scale it and add the barycentric
        # pulsar position to obtain the physical, barycentric IBS points.
        # In contrast, ``ibs_n.r1_vec`` is S->IBS, so it must never receive
        # this barycentric translation.
        for suffix in ("", "_mid"):
            p_to_ibs = _r_sp * getattr(self.ibs_n, "r_vec" + suffix)
            r_vec = p_to_ibs + r_p_vec[None, None, :]
            setattr(self, "r_vec" + suffix, r_vec)

            # Derive displacement vectors from the physical endpoints.
            s_to_ibs = r_vec - r_s_vec[None, None, :]
            setattr(self, "vec_sIBS" + suffix, s_to_ibs)
            setattr(self, "r1_vec" + suffix, s_to_ibs)

            p_to_ibs = r_vec - r_p_vec[None, None, :]
            setattr(self, "vec_pIBS" + suffix, p_to_ibs)
        

        for name, i in zip(("x", "y", "z"), (0, 1, 2)):
            for suffix in ("", "_mid"):
                setattr(self, name+suffix, getattr(self, "r_vec"+suffix)[..., i])

            # Segment endpoints are barycentric points, rather than scaled
            # normalized coordinates.  The differences below remain physical
            # displacement vectors and are therefore translation-invariant.
            coord = getattr(self, name)
            setattr(self, name + "_m", coord[:, :-1])
            setattr(self, name + "_p", coord[:, 1:])

        ### rescale the stuff that's relative to the IBS
        for name in ("s", "s_max_g", "ds_dtheta",
                     "s_m", "s_p", "s_mid", "ds", "dx", "dy", "s_max_cm"):
            setattr(self, name, _r_sp * getattr(self.ibs_n, name))

        ### rescale and/or set distances from star and from pulsar
        self.r_from_s = absv(self.vec_sIBS)
        self.r_from_s_mid = absv(self.vec_sIBS_mid)
        self.r = absv(self.vec_pIBS)
        self.r_mid = absv(self.vec_pIBS_mid)
        # Preserve the normalized-class public naming, now in physical units.
        self.r1 = self.r_from_s
        self.r1_mid = self.r_from_s_mid
        self.x_apex = _r_sp * self.ibs_n.x_apex
        for name in ("r", "r1"):
            distance = getattr(self, name)
            setattr(self, name + "_m", distance[:, :-1])
            setattr(self, name + "_p", distance[:, 1:])

   
        vec_apex = self.winds.orbit.vector_p(self.t_forbeta) + (-self.symm_ax) * self.r_pe 
        self.x_apex_coord = vec_apex[0]
        self.y_apex_coord = vec_apex[1]
        vec_s_apex = (-r_s_vec) + vec_apex

        self.scatter_angle_apex = vector_angle(vec_s_apex, self.unit_los)
        self.dopl_apex_eff = doppler_delta(self.gamma_max,
                        vector_angle(self.unit_los, self.symm_ax))

        
        
    # def rescale_gamma(self):
    #     true_an = self.winds.orbit.true_an(self.t_forbeta)
    #     pdisk = self.winds.star.decr_disk_pressure(self.vec_sIBS,
    #                                                 true_an=true_an)
    #     ppolar = self.winds.star.polar_wind_pressure(self.r_from_s)
    #     ppulsar = self.winds.pulsar.wind_pressure(self.r)
    #     dispers_out = np.std((pdisk + ppolar) / ppulsar )
    #     self.ibs_n.gamma_max = gamma_test(self.gamma_max, dispers_out, self.coef_quench)
    #     # ``vec_beta`` is cached by the normalized class and is used by the
    #     # comoving scattering-angle properties, so refresh it after changing
    #     # the Lorentz-factor prescription.
    #     self.ibs_n._set_beta_vecs()
    #     self.dopl_apex_eff = doppler_delta(
    #         self.ibs_n.gamma_max,
    #         vector_angle(self.winds.orbit.unit_los, self.symm_ax),
    #     )
    
    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_ [cm] averaged over phi-angle. This means that only the `what` is
        averaged over phi for each 's'.
 
        Parameters
        ----------
        s_ : np.ndarray
            The arclength along the upper horn of the IBS to find the value at.
            [cm].

        Returns
        -------
        The desired value of ibs3d.what in the coordinate s_ at phi=0. 

        """
        try:
            data = getattr(self, what)
        except AttributeError:
            raise ValueError(f"No such attribute '{what}' in IBS.")
        ##### here I set fill_value='extrapolate' instead of raising an error_1horn
        ##### or like filling with NaNs, cause the values at the ends of an
        ##### IBS sometimes behave weirdly, and we DO need these values. So
        ##### since this is the internal function that should not be used
        ##### by an external user, we put `extrapolate` and use it VERY
        ##### cautiously!!!
        phi_avg_data = np.average(data, axis=0)
        interpolator = interp1d(self.s[0, :], phi_avg_data, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return interpolator(s_)
    
    def gma(self, s):
        """Bulk Lorentz factor at 3D-shock arclength ``s`` [cm]."""
        return self.ibs_n.gma(s / self.r_sp)
    
    @property
    def g(self):
        """Bulk Lorentz factor along the IBS."""
        return self.gma(s = self.s)
    
    @property
    def g_mid(self):
        """Bulk Lorentz factor along the IBS-mid"""
        return self.gma(s = self.s_mid)
    
    @property
    def dopl_star(self):
        """Doppler-factor at the IBS for the direction to the star"""
        return doppler_delta(self.g,
                vector_angle(self.unit_beta, n_from_v(self.vec_sIBS))
                )
    
    @property
    def dopl_star_mid(self):
        """Doppler-factor at the IBS-mid for the direction to the star"""
        return doppler_delta(self.g_mid,
                vector_angle(self.unit_beta_mid, n_from_v(self.vec_sIBS_mid))
                )
    
    def gg_abs(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in every point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB. Since the gg-absorption is tabulated 
        only in an orbital plane, we use this simplified approach: we take the 
        IBS arch for phi=0; project it onto an orbital plane; calculate the
        corresponding gg-abs coefs, and set these coefs to all IBS archs.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
        
        returns : np.ndarray of shape (n_phi, n, e_phot.size)
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res_1horn = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = _x, y = _y,
                         R_star=self.winds.star.R_s, T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for _x, _y in zip(self.vec_sIBS[0, :, 0], self.vec_sIBS[0, :, 1])])
            gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        else:
            proj_vecs = project_point_for_gg(self.vec_sIBS, self.unit_los)
            xproj, yproj = np.average(proj_vecs[..., 0], axis=0), np.average(proj_vecs[..., 1], axis=0)
            gg_res_1horn = gg_tab(E=e_phot, x=xproj, y=yproj, 
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
            gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        
        return gg_res
    
    
    def gg_abs_mid(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in every mid point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
        
        returns : np.ndarray of shape (n_phi, n-1, e_phot.size)
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res_1horn = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = _x, y = _y,
                         R_star=self.winds.star.R_s, T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for _x, _y in zip(self.vec_sIBS_mid[0, :, 0], self.vec_sIBS_mid[0, :, 1])])
            gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        else:
            proj_vecs_mid = project_point_for_gg(self.vec_sIBS_mid, self.unit_los)
            xproj, yproj = np.average(proj_vecs_mid[..., 0], axis=0), np.average(proj_vecs_mid[..., 1], axis=0)
            gg_res_1horn = gg_tab(E=e_phot, x=xproj, y=yproj, 
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
            gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        return gg_res
    
    def gg_abs_apex(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorption coefficient (as e^-tau) in the IBS apex.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorption/absorp_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorption for a `sys_name` provided for `winds` arg.
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (e_phot.size, )
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')

        if analyt:
            gg_res = gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x_apex_coord - self.vec_s[0],
                         y = self.y_apex_coord - self.vec_s[1],
                         R_star=self.winds.star.R_s,
                         T_star = self.winds.star.T_s,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
        else:
            gg_res = gg_tab(E=e_phot, 
                            x=self.x_apex_coord - self.vec_s[0],
                            y=self.y_apex_coord - self.vec_s[1],
                            orb=self.winds.orbit,
                            filename=filename, 
                            what_return=what_return)
        return gg_res
    
    @property
    def integrated_pressure(self):
        """
        An integral of external pressure over the line connecting the star's
        surface to the IBS.

        Creates a line from star to the point on IBS, calculates pressure on
        this line, then integrates it over the line.
        
        returns : integrated p, shape of (Nphi, N)
        """
        param = np.geomspace(1e-3, 1., 27)
        vecs_s_ibs = vec_between(vec_i = np.zeros((3,)), vec_f = self.vec_sIBS, param=param)
        true_an = self.winds.orbit.true_an(self.t_forbeta)
        ps = (self.winds.star.polar_wind_pressure(r_from_s=absv(vecs_s_ibs)) + 
             self.winds.star.decr_disk_pressure(vec_r_from_s=vecs_s_ibs,
                                                 true_an=true_an))
        ps[absv(vecs_s_ibs) < self.winds.star.R_s] = 1e-100
        integrated_p = trapz_loglog(ps, param, axis=0)
        return integrated_p
    
    @property
    def integrated_pressure_mid(self):
        """
        An integral of external pressure over the line connecting the star's
        surface to the mid-IBS.
        
        returns : integrated p, shape of (Nphi, N-1)
        """
        param = np.geomspace(1e-3, 1., 27)
        vecs_s_ibs = vec_between(vec_i = np.zeros((3,)), vec_f = self.vec_sIBS_mid, param=param)
        true_an = self.winds.orbit.true_an(self.t_forbeta)
        ps = (self.winds.star.polar_wind_pressure(r_from_s=absv(vecs_s_ibs)) + 
             self.winds.star.decr_disk_pressure(vec_r_from_s=vecs_s_ibs,
                                                 true_an=true_an))
        ps[absv(vecs_s_ibs) < self.winds.star.R_s] = 1e-100
        integrated_p = trapz_loglog(ps, param, axis=0)
        return integrated_p
    
    @property
    def soft_ph_abs(self):
        """Stellar-photon attenuation on point cells, shape ``(Nphi, Ns)``."""
        return np.exp(- self.shield_star * self.integrated_pressure)
    
    @property
    def soft_ph_abs_mid(self):
        """Stellar-photon attenuation on segment cells, shape ``(Nphi, Ns-1)``."""
        return np.exp(- self.shield_star * self.integrated_pressure_mid)
    
    ###########################################################################
    @property
    def ug(self):
        """Photon field energy density on the IBS [erg/cm^3]."""
        _u = self.winds.star.u_g_density(r_from_s = self.r_from_s)
        if self.shield_star == 0.:
            return _u
        return _u * self.soft_ph_abs
    
    @property
    def ug_mid(self):
        """Photon field energy density on the IBS_mid [erg/cm^3]."""
        _u = self.winds.star.u_g_density(r_from_s = self.r_from_s_mid)
        if self.shield_star == 0.:
            return _u
        return _u * self.soft_ph_abs_mid
    
    @property
    def ug_comov_iso(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3].
        Isotropc approximation."""
        return lor_trans_ug_iso(ug_iso = self.ug, gamma=self.g)
      
    @property
    def ug_mid_comov_iso(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3].
        Isotropc approximation."""          
        return lor_trans_ug_iso(ug_iso = self.ug_mid, gamma=self.g_mid)
      
    @property
    def ug_comov_ani(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3].
        Anisotropc approximation."""
        return self.ug / self.dopl_star**2
      
    @property
    def ug_mid_comov_ani(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3].
        Anisotropc approximation."""          
        return self.ug_mid / self.dopl_star_mid**2     
    
    ###########################################################################
    @property
    def b_pulsar(self):
        """Neutron star-originating magnetic field on the IBS [G]."""
        return self.winds.pulsar.b(r_to_p = self.r)
    
    @property
    def b_pulsar_mid(self):
        """Neutron star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.pulsar.b(r_to_p = self.r_mid)
    
    @property
    def b_pulsar_comov(self):
        """Neutron star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_pulsar, gamma=self.g)
    
    @property
    def b_pulsar_mid_comov(self):
        """Neutron star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_pulsar_mid, gamma=self.g_mid)
    
    
    ###########################################################################
    @property
    def b_s(self):
        """Optical star-originating magnetic field on the IBS [G]."""
        return self.winds.star.b(r_to_s = self.r_from_s)
    
    @property
    def b_s_mid(self):
        """Optical star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.star.b(r_to_s = self.r_from_s_mid)
    
    @property
    def b_s_comov(self):
        """Optical star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_s, gamma=self.g)
    
    @property
    def b_s_mid_comov(self):
        """Optical star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_s_mid, gamma=self.g_mid)
    
    ###########################################################################
    @property
    def b(self):
        """Total magnetic field on the IBS [G]."""
        return self.b_pulsar + self.b_s
    
    @property
    def b_mid(self):
        """Total magnetic field on the IBS_mid [G]."""
        return self.b_pulsar_mid + self.b_s_mid
    
    @property
    def b_comov(self):
        """Total magnetic field on the IBS in the comoving frame [G]."""
        return self.b_pulsar_comov + self.b_s_comov
    
    @property
    def b_mid_comov(self):
        """Total magnetic field on the IBS_mid in the comoving frame [G]."""
        return self.b_pulsar_mid_comov + self.b_s_mid_comov
    
    ###########################################################################
    @property
    def T_s_eff(self):
        """Optical star effective temperature on the IBS [K]. 
        Simply the star temperature everywhere."""
        return self.winds.star.T_s * np.ones(self.r.shape)
    
    @property
    def T_s_eff_mid(self):
        """Optical star effective temperature on the IBS_mid [K]. 
        Simply the star temperature everywhere."""
        return self.winds.star.T_s * np.ones(self.r_mid.shape)
    
    @property
    def T_s_eff_comov_iso(self):
        """Optical star effective temperature on the IBS in the comoving frame [K].
        Isotropc approximation."""
        return lor_trans_Teff_iso(Teff_iso = self.T_s_eff, gamma=self.g)
    
    @property
    def T_s_eff_mid_comov_iso(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K].
        Isotropc approximation."""
        return lor_trans_Teff_iso(Teff_iso = self.T_s_eff_mid, gamma=self.g_mid)
    
    @property
    def T_s_eff_comov_ani(self):
        """Optical star effective temperature on the IBS in the comoving frame [K].
        Anisotropc approximation."""
        return self.T_s_eff / self.dopl_star
    
    @property
    def T_s_eff_mid_comov_ani(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K].
        Anisotropc approximation."""
        return self.T_s_eff_mid / self.dopl_star_mid
    
    
    @property
    def disk_pressure(self):
        """Decretion disk pressure calculated on the IBS."""
        return self.winds.star.decr_disk_pressure(
            vec_r_from_s=self.vec_sIBS,
            true_an=self.winds.orbit.true_an(self.t_forbeta),
        )
    
    
    @property
    def polar_pressure(self):
        """Polar wind pressure calculated on the IBS."""
        return self.winds.star.polar_wind_pressure(r_from_s = self.r_from_s)
    
    @property
    def tot_external_pressure(self):
        """Total external pressure (P_disk + P_polar wind) calculated on the IBS."""
        return self.disk_pressure + self.polar_pressure
      
    
    peek_docs = f"""
    Quick look at the IBS in the orbital plane.

    Parameters
    ----------
    fig : fig object of pyplot, optional
         The default is None.
    ax : ax object of pyplot, optional
        DESCRIPTION. The default is None.
    show_winds : bool, optional
        Whether to show the winds (requires Winds to be provided).
          The default is False.
    ibs_color : str, optional
        Can be one of {PEEK_KEYS} 
        ; or any matplotlib color. 
        The default is 'k'.
    to_label : bool, optional
        Whether to put a label `beta=...` on a plot.
          The default is True.
    showtime : tuple of (tmin, tmax), optional
        For orbit displaying (see orbit.peek()). 
        The default is None.
    E_for_gg : float, optional
        At which energy [eV] to calculate the gamma-gamma absorption, if ibs_color
        is 'gg_tau' or 'gg_abs'. Default 1e12
    scale : str, optional
        Scale for displaying the color-coded parameter. If 'lin' or 'linear',
        (default), the parameter itself is displayed. If 'log', its log10 is
        shown.

    Raises
    ------
    ValueError
        If the ibs_color is not one of the recognizable options.

    Returns
    -------
    None.

    """
    def peek(self,  ax=None, fig=None, show_winds=False,
             ibs_color='k', to_label=True,
             edgecolor='k', linewidth=0.1,
             alpha=0.5, colorbar=True,
             showtime=None, E_for_gg=1e12,
             scale='linear',
             show_star=False):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")   
        if ibs_color == 'doppler':
            color_param = self.dopl
            bar_label = r'doppler factor $\delta$'
        elif ibs_color in ('scattering', 'scatter'):
            color_param = self.scattering_angle
            bar_label = r'scattering angle / $\pi$'
        elif ibs_color == 'scattering_comoving':
            color_param = self.scattering_angle_comov
            bar_label = r'comoving scattering angle / $\pi$'
        elif ibs_color == 'gg_abs':
            color_param = self.gg_abs(e_phot=E_for_gg)
            bar_label = r'$e^{-\tau} \gamma-\gamma $'
        elif ibs_color == 'gg_tau':
            color_param = self.gg_abs(e_phot=E_for_gg,
                                     what_return='tau')
            bar_label = r'$\tau \gamma-\gamma $'
        elif hasattr(self, ibs_color):
            color_param = getattr(self, ibs_color)
            bar_label = ibs_color
        elif mcolors.is_color_like(ibs_color):
            color_param = ibs_color
            bar_label = None
            colorbar = False
        else:
            raise ValueError(f"""ibs_color={ibs_color} is invalid; it should be
                             \neither one of the IBS3D class attributes, or a 
                             \nvalid matplotlib color.""")
        
        if scale in ('lin', 'linear'):
            pass
        elif scale == 'log':
            color_param = np.log10(color_param)
        else:
            raise ValueError("'scale' can be 'linear' or 'log'.")


        puls_vector = self.winds.orbit.vector_p(self.t_forbeta)
        star_vector = self.winds.orbit.vector_s(self.t_forbeta)
        _xp, _yp = puls_vector[0], puls_vector[1]
        _xs, _ys = star_vector[0], star_vector[1]
        ax.scatter(_xp, _yp, 0, color='b') # pulsaR
        if show_star:
            ax.scatter(_xs, _ys, 0, color='r') # optical star
        ax.scatter(0., 0., 0, color='k', marker='x') # barycenter

        ax.plot([_xs, _xp], [_ys, _yp], [0, 0], color='k', alpha=0.3, ls='--')
        vec_disk1, vec_disk2 = self.winds.vectors_of_disk_passage
        vec_disk1 = star_vector + vec_disk1
        vec_disk2 = star_vector + vec_disk2
        xx1, yy1, zz1 = vec_disk1                                                 
        xx2, yy2, zz2 = vec_disk2                                                 
        ax.plot([xx1, xx2], [yy1, yy2], [zz1, zz2], color='orange', ls='--', lw=2)    

        ##################################################################################
        if showtime is None:
            showtime = [-self.winds.orbit.T/2, self.winds.orbit.T/2]
        show_cond  = np.logical_and(self.winds.orbit.ttab > showtime[0], 
                                    self.winds.orbit.ttab < showtime[1])
        orb_x, orb_y = self.winds.orbit.xtab_p[show_cond], self.winds.orbit.ytab_p[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
            ]))
        ax.plot(orb_x, orb_y, 0*orb_x)
        if show_star:
            orb_x_s, orb_y_s = self.winds.orbit.xtab_s[show_cond], self.winds.orbit.ytab_s[show_cond]
            ax.plot(orb_x_s, orb_y_s, 0*orb_x_s)
        ax.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.winds.orbit.r_periastr) )
        ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        ax.set_zlim(-1.2*x_scale, 1.2*x_scale) 
        
        plot_surface_quads(ax=ax, coords=self.r_vec, param=color_param, linewidth=linewidth,
                           edgecolor=edgecolor, colorbar=colorbar, phi_close=True,
                           cbar_label=bar_label, alpha=alpha,
                           )
        # ax.scatter(xstar_, ystar_, zstar_, color='b')
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        # ax.set_zlim(-2, 2)
        xlos_, ylos_, zlos_ = self.unit_los * x_scale
        ax.quiver(_xs, _ys, star_vector[2], xlos_, ylos_, zlos_,
                  arrow_length_ratio=0.12, linewidth=2, color='g')
        # ax.legend()
        # return line_ 
        return ax

    
    def __getattr__(self, name):
        """Delegate normalized, dimensionless geometry attributes to ``ibs_n``."""
        ibs_n_ = self.__dict__.get("ibs_n", None)
        if ibs_n_ is None:
            raise AttributeError(name)
        return getattr(ibs_n_, name)
