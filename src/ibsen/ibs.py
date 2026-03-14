# ibsen/ibs.py
import numpy as np
from numpy import pi, sin, cos
from scipy.interpolate import interp1d
from ibsen.winds import Winds
from ibsen.ibs_norm import IBS_norm, IBS_norm3D
from ibsen.utils import plot_with_gradient, \
 lor_trans_ug_iso, lor_trans_b_iso, lor_trans_Teff_iso, rotated_vector, absv, \
     plot_surface_quads, vector_angle
from ibsen.absorbtion.absorbtion import gg_analyt, gg_tab
from ibsen.get_obs_data import known_names

from astropy import constants as const

C_LIGHT = 2.998e10
DAY = 86400
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)
M_E = float(const.m_e.cgs.value)

PEEK_KEYS = ('doppler', 'scattering', 'scattering_comoving',
                  'gg_tau', 'gg_abs', 'b_ns', 'b_ns_comov', 
                  'b_opt', 'b_opt_comov', 'b', 'b_comov', 
                  'ug', 'ug_comov', )

ibs_docstring = f"""
    Intrabinary shock (IBS) in physical (cgs) units at a given orbital epoch.

    This adapter builds a **dimensionless** shock via :class:`IBS_norm` using the
    effective momentum-flux ratio `\\beta` from a supplied :class:`Winds`
    model at time ``t_to_calculate_beta_eff``, rotates it to align the symmetry
    axis with the instantaneous star–pulsar line, rescales all length-like
    quantities by the current separation :math:`r_{{\\rm sp}}(t)`, and then shifts
    the curve to the pulsar’s instantaneous position in the orbital plane.
    Unitless/angle-like properties are delegated to the underlying normalized
    object. 

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
    x, y : ndarray, shape (N,)
        IBS coordinates in the orbital plane [cm], shifted so the star is at the
        origin and the pulsar lies at ``orbit.vector_sp(t_forbeta)``. 
    s : ndarray
        Signed arclength along the IBS [cm]; increases through the apex.
    s_max, s_max_g : float
        Arclength cut and the location of ``gamma_max`` [cm] (rescaled from
        the normalized values).
    r, r1 : ndarray
        Distances from pulsar→IBS and star→IBS, respectively [cm].
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
    b_opt, b_opt_mid, b_opt_comov, b_opt_mid_comov : ndarray
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
                 winds = None, 
                 abs_gg_filename = None,
                 include_incl_in_los=False):
        self.t_forbeta = t_to_calculate_beta_eff
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.n = n
        self.winds = winds
        self.ug_apex = winds.u_g_density_apex(t=t_to_calculate_beta_eff)
        b_ns_apex, b_opt_apex = winds.magn_fields_apex(t=t_to_calculate_beta_eff)
        self.b_ns_apex, self.b_opt_apex = b_ns_apex, b_opt_apex
        self.b_apex = b_ns_apex + b_opt_apex
        
        self.abs_gg_filename = abs_gg_filename
        self.include_incl_in_los = include_incl_in_los
        self.peek_keys = PEEK_KEYS
        
        self.calculate()
        
    def calculate(self):
        """General calculation of everything"""
        self.calculate_normalized_ibs()
        self.rescale_to_position()
    
    def calculate_normalized_ibs(self):
        """calculate the effective beta from Winds at time t_forbeta and 
            initialize the normalized IBS with this beta.
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

        self.ibs_n = IBS_norm(beta=self.beta, s_max=self.s_max,
                        gamma_max=self.gamma_max, s_max_g=self.s_max_g, n=self.n,
                    unit_los=unit_los_).rotate(pi + _nu_tr)
        
    def rescale_to_position(self):
        """
        Rescale the IBS to the real units at
        the time t_to_calculate_beta_eff [s] and rotate it so that its 
        line of sy_mmetry is S-P line. Rescaled are: x, y, r, s, r1, x_apex.
        The tangent is added pi + nu_tr to (at the stage of 'IBS.rotate').
        To the x and y - coordinates of the IBS there is also added the
        vector of the real s-p distance at the moment (in cm).
        Yes, read this terrible English sentence which I translated from 
        Russian in my head. Suffer.
        ---
        Returns new rescaled ibs_resc:IBS
        """
        _r_sp = self.winds.orbit.r(self.t_forbeta)
        _x_sp, _y_sp = self.winds.orbit.x(self.t_forbeta), self.winds.orbit.y(self.t_forbeta)
        _nu_tr = self.winds.orbit.true_an(self.t_forbeta)
        x_sh, y_sh = self.ibs_n.x, self.ibs_n.y
        x_sh_mid, y_sh_mid = self.ibs_n.x_mid, self.ibs_n.y_mid
        
        x_sh, y_sh = x_sh * _r_sp, y_sh * _r_sp
        x_sh_mid, y_sh_mid = x_sh_mid * _r_sp, y_sh_mid * _r_sp
        
        x_sh += _r_sp * cos(_nu_tr)
        y_sh += _r_sp * sin(_nu_tr)
        x_sh_mid += _r_sp * cos(_nu_tr)
        y_sh_mid += _r_sp * sin(_nu_tr)
        
        self.x = x_sh
        self.y = y_sh
        self.x_mid = x_sh_mid
        self.y_mid = y_sh_mid
        
        for name in ("s", "s_max", "s_max_g", "r", "r1", "x_apex", "ds_dtheta",
                     "s_m", "s_p", "s_mid", "ds", "dx", "dy"):
            setattr(self, name, _r_sp * getattr(self.ibs_n, name))
   
        
        self.r1_mid = np.sqrt(x_sh_mid**2 + y_sh_mid**2)
        self.r_mid = np.sqrt( (x_sh_mid-_x_sp)**2
                              + (y_sh_mid - _y_sp)**2)
        
        self.x_apex_coord = self.r_se/self.r_sp * self.winds.orbit.x(self.t_forbeta)
        self.y_apex_coord = self.r_se/self.r_sp * self.winds.orbit.y(self.t_forbeta)
        self.scatter_angle_apex = vector_angle(self.winds.orbit.vector_sp(self.t_forbeta),
                                               self.unit_los)
        
    
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
    
    # @property
    # def g(self):
    #     """Bulk Lorentz factor along the IBS."""
    #     return IBS.gma(self, s = self.s)
    
    # @property
    # def g_mid(self):
    #     """Bulk Lorentz factor along the IBS-mid"""
    #     return IBS.gma(self, s = self.s_mid)
    
    
    def gg_abs(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in every point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (n, e_phot.size)
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x[i], y = self.y[i],
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for i in range(self.x.size)])
        else:
            gg_res = gg_tab(E=e_phot, x=self.x, y=self.y, 
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    
    def gg_abs_mid(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in every mid point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorbtion/absorb_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorbtion for a `sys_name` provided for `winds` arg.
            
        what_return: str {'abs' or 'tau'}
            What to return: e^-tau or tau. Default 'abs'.
            
        returns : np.ndarray of shape (n-1, e_phot.size)
            
        """
        if self.abs_gg_filename is not None and str(self.abs_gg_filename).strip():
            filename = self.abs_gg_filename
        elif self.winds.sys_name in known_names:
            filename = self.winds.sys_name
        else:
            raise ValueError('Provide abs_gg_filename or winds.sys_name for gg-abs.')
        if analyt:
            gg_res = np.array([gg_analyt(eg = e_phot / 5.11e5,
                         x = self.x_mid[i], y = self.y_mid[i],
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for i in range(self.x_mid.size)])
        else:
            gg_res = gg_tab(E=e_phot, x=self.x_mid, y=self.y_mid,
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    def gg_abs_apex(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in the IBS apex.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorbtion/absorb_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorbtion for a `sys_name` provided for `winds` arg.
            
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
                         x = self.x_apex, y = self.y_apex,
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
        else:
            gg_res = gg_tab(E=e_phot, x=self.x_apex_coord, y=self.y_apex_coord,
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    
    ###########################################################################
    @property
    def ug(self):
        """Photon field energy density on the IBS [erg/cm^3]."""
        return self.winds.u_g_density(r_from_s = self.r1,
                                      r_star = self.winds.Ropt,
                                      T_star = self.winds.Topt,
                                      )
    
    @property
    def ug_mid(self):
        """Photon field energy density on the IBS_mid [erg/cm^3]."""
        return self.winds.u_g_density(r_from_s = self.r1_mid,
                                      r_star = self.winds.Ropt,
                                      T_star = self.winds.Topt,
                                      )
    
    @property
    def ug_comov(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3]."""
        return lor_trans_ug_iso(ug_iso = self.ug, gamma=self.g)
      
    @property
    def ug_mid_comov(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3]."""          
        return lor_trans_ug_iso(ug_iso = self.ug_mid, gamma=self.g_mid)
      
        
    ###########################################################################
    @property
    def b_ns(self):
        """Neutron star-originating magnetic field on the IBS [G]."""
        return self.winds.ns_field_initialized(r_to_p = self.r, t=self.t_forbeta)
    
    @property
    def b_ns_mid(self):
        """Neutron star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.ns_field_initialized(r_to_p = self.r_mid, t=self.t_forbeta)
    
    @property
    def b_ns_comov(self):
        """Neutron star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_ns, gamma=self.g)
    
    @property
    def b_ns_mid_comov(self):
        """Neutron star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_ns_mid, gamma=self.g_mid)
    
    
    ###########################################################################
    @property
    def b_opt(self):
        """Optical star-originating magnetic field on the IBS [G]."""
        return self.winds.opt_field_initialized(r_to_s = self.r1, t=self.t_forbeta)
    
    @property
    def b_opt_mid(self):
        """Optical star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.opt_field_initialized(r_to_s = self.r1_mid, t=self.t_forbeta)
    
    @property
    def b_opt_comov(self):
        """Optical star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_opt, gamma=self.g)
    
    @property
    def b_opt_mid_comov(self):
        """Optical star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_opt_mid, gamma=self.g_mid)
    
    ###########################################################################
    @property
    def b(self):
        """Total magnetic field on the IBS [G]."""
        return self.b_ns + self.b_opt
    
    @property
    def b_mid(self):
        """Total magnetic field on the IBS_mid [G]."""
        return self.b_ns_mid + self.b_opt_mid
    
    @property
    def b_comov(self):
        """Total magnetic field on the IBS in the comoving frame [G]."""
        return self.b_ns_comov + self.b_opt_comov
    
    @property
    def b_mid_comov(self):
        """Total magnetic field on the IBS_mid in the comoving frame [G]."""
        return self.b_ns_mid_comov + self.b_opt_mid_comov
    
    ###########################################################################
    @property
    def T_opt_eff(self):
        """Optical star effective temperature on the IBS [K]. 
        Simply the star temperature everywhere."""
        return self.winds.Topt * np.ones(self.r.size)
    
    @property
    def T_opt_eff_mid(self):
        """Optical star effective temperature on the IBS_mid [K]. 
        Simply the star temperature everywhere."""
        return self.winds.Topt * np.ones(self.r_mid.size)
    
    @property
    def T_opt_eff_comov(self):
        """Optical star effective temperature on the IBS in the comoving frame [K]."""
        return lor_trans_Teff_iso(Teff_iso = self.T_opt_eff, gamma=self.g)
    
    @property
    def T_opt_eff_mid_comov(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K]."""
        return lor_trans_Teff_iso(Teff_iso = self.T_opt_eff_mid, gamma=self.g_mid)
        
    
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
        At which energy [eV] to calculate the gamma-gamma absorbtion, if ibs_color
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
             showtime=None, E_for_gg=1e12):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        
        if ax is None:
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

            line_ = plot_with_gradient(fig=fig, ax=ax, xdata=self.x, ydata=self.y,
                            some_param=color_param, colorbar=to_label, lw=2, ls='-',
                            colorbar_label=ibs_color)

        elif mcolors.is_color_like(ibs_color):
            line_ = ax.plot(self.x, self.y, color=ibs_color, label = label)     
        else:
            raise ValueError(f"""ibs_colos should be either oe of
                             {PEEK_KEYS} or a matpotlib color.""")

        if show_winds:
            if not isinstance(self.winds, Winds):
                raise ValueError("You should provide winds:Winds to show the winds.")
            self.winds.peek(ax=ax, showtime=showtime,
                            plot_rs=False)

        # if rescaled:
        puls_vector = self.winds.orbit.vector_sp(self.t_forbeta)
        _xp, _yp = puls_vector[0], puls_vector[1]
        ax.scatter(_xp, _yp, color='b') # pulsaR
        # ...and the star was alreay plotted in the winds.peek()
        ax.plot( [0, 1.5*_xp], [0, 1.5*_yp], color='k', alpha=0.3, ls='--',)

        ##################################################################################
        if showtime is None:
            showtime = [-self.winds.orbit.T/2, self.winds.orbit.T/2]
        show_cond  = np.logical_and(self.winds.orbit.ttab > showtime[0], 
                                    self.winds.orbit.ttab < showtime[1])
        orb_x, orb_y = self.winds.orbit.xtab[show_cond], self.winds.orbit.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
            ]))
        
        ax.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.winds.orbit.r_periastr) )
        ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        
            
        ax.legend()
        return line_ 
    peek.__doc__ = peek_docs

    
    def __getattr__(self, name):
        ibs_n_ = self.__dict__.get("ibs_n", None)
        if ibs_n_ is None:
            raise AttributeError(name)
        return getattr(ibs_n_, name)
    
    
class IBS3D: #!!!
    """
    An analog of the class IBS for there dimensions. I haven't written the' proper
    documentation yet, so please refer to the 'IBS' docs; here I only give the
    list of parameters.
    
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
    abs_gg_filename : str or None, optional
        Name of the file with tabukated gg-opacity. If None (default),
        IBSEn searches for a file tabulated for winds.sys_name if it is in
        the known names: {known_names}. Can also be one of known names, then 
        searches for a file tabulated for a system with this name.
    
    """
    def __init__(self, t_to_calculate_beta_eff, s_max=1.0, gamma_max=3.0, s_max_g=4.0, n=31, 
                 n_phi=33,
                 winds = None, 
                 abs_gg_filename = None):
        self.t_forbeta = t_to_calculate_beta_eff
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.n = n
        self.n_phi = n_phi
        self.winds = winds
        self.ug_apex = winds.u_g_density_apex(t=t_to_calculate_beta_eff)
        b_ns_apex, b_opt_apex = winds.magn_fields_apex(t=t_to_calculate_beta_eff)
        self.b_ns_apex, self.b_opt_apex = b_ns_apex, b_opt_apex
        self.b_apex = b_ns_apex + b_opt_apex
        
        self.abs_gg_filename = abs_gg_filename
        self.peek_keys = PEEK_KEYS
       
        self._calculate_normalized_ibs()
        self._rescale_to_position()

    
    def _calculate_normalized_ibs(self):
        """calculate the effective beta from Winds at time t_forbeta and 
            initialize the normalized IBS with this beta.
        """
        self.beta = self.winds.beta_eff(self.t_forbeta)
        self.r_sp = self.winds.orbit.r(self.t_forbeta)
        self.r_se = self.winds.dist_se_1d(self.t_forbeta)

        unit_los_ = rotated_vector(self.winds.orbit.nu_los, self.winds.orbit.incl_los)
        self.unit_los = unit_los_
        _nu_tr = self.winds.orbit.true_an(self.t_forbeta)

        # self.ibs_n = IBS_norm3D(beta=self.beta, s_max=self.s_max,
        #     gamma_max=self.gamma_max, s_max_g=self.s_max_g, n=self.n,
        #     n_phi=self.n_phi,
        #     unit_los=unit_los_).rotate(phi=pi + _nu_tr, vec_ax=np.array([0, 0, 1]))
        
        self.ibs_n = IBS_norm3D(beta=self.beta, s_max=self.s_max,
            gamma_max=self.gamma_max, s_max_g=self.s_max_g, n=self.n,
            n_phi=self.n_phi,
            unit_los=unit_los_).rotate_to_ax(new_axis=self.winds.orbit.vector_sp(self.t_forbeta))
        
    def _rescale_to_position(self):
        """
        Rescale the IBS to the real units at
        the time t_to_calculate_beta_eff [s] and rotate it so that its 
        line of sy_mmetry is S-P line. Rescaled are: x, y, r, s, r1, x_apex.
        To the x and y - coordinates of the IBS there is also added the
        vector of the real s-p distance at the moment (in cm).
        Yes, read this terrible English sentence which I translated from 
        Russian in my head. Suffer.
        ---
        Returns new rescaled ibs_resc:IBS
        """
        r_sp_vec = self.winds.orbit.vector_sp(self.t_forbeta)
        _r_sp = absv(r_sp_vec)
        ### rescale ans shift vector stuff:
        for name in ("r_vec", "r1_vec"):
            for suffix in ("", "_mid"):
                norm_vec = getattr(self.ibs_n, name+suffix)
                rescaled_vec = _r_sp * norm_vec
                shifted_rotated_vec = rescaled_vec + r_sp_vec[None, None, :]
                setattr(self, name+suffix, shifted_rotated_vec)
                

        for name, i in zip(("x", "y", "z"), (0, 1, 2)):
            for suffix in ("", "_mid"):
                setattr(self, name+suffix, getattr(self, "r_vec"+suffix)[..., i])

        ### rescale some other stuff        
        for name in ("s", "s_max", "s_max_g", "r", "r1", "r_mid", "r1_mid",  "x_apex", "ds_dtheta",
                     "s_m", "s_p", "s_mid", "ds", "dx", "dy"):
            setattr(self, name, _r_sp * getattr(self.ibs_n, name))
   
        
        self.x_apex_coord = self.r_se/self.r_sp * self.winds.orbit.x(self.t_forbeta)
        self.y_apex_coord = self.r_se/self.r_sp * self.winds.orbit.y(self.t_forbeta)
        self.scatter_angle_apex = vector_angle(self.winds.orbit.vector_sp(self.t_forbeta),
                                               self.unit_los)
        
    
    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_ [cm] for the first phi-angle. This means that only one horn of the
        3D IBS corresponding for phi=0 is taken for 'what'.
 
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
        interpolator = interp1d(self.s[0, :], data[0, ...], kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return interpolator(s_)
    
    
    def gg_abs(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in every point of
        the IBS. The absortion is supposed to be on a photon field of a central
        star which is represented by a BB. Since the gg-absorbtion is tabulated 
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
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for _x, _y in zip(self.x[0, :], self.y[0, :])])
        else:
            gg_res_1horn = gg_tab(E=e_phot, x=self.x[0, :], y=self.y[0, :], 
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        return gg_res
    
    
    def gg_abs_mid(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in every mid point of
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
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
                           for _x, _y in zip(self.x_mid[0, :], self.y_mid[0, :])])
        else:
            gg_res_1horn = gg_tab(E=e_phot, x=self.x_mid[0, :], y=self.y_mid[0, :], 
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        gg_res = np.array([gg_res_1horn for _i in range(self.n_phi)])
        return gg_res
    
    def gg_abs_apex(self, e_phot, analyt=False, what_return='abs'):
        """ gamma-gamma absorbtion coefficient (as e^-tau) in the IBS apex.
        
        e_phot : np.ndarray
            [eV] - energy of the VHE photon.
        
        analyt : bool, optional
            Whether to use the analytical approximation of 
            Sushch and van Soelen, 2023. Default False
            
        filename : str or path or None, optional
            Path to the file with tabulated opacities. File should be inside
            the absorbtion/absorb_tab folder. Can be one of the known names,
            then resolves to the file tabulated for it. If None, tries to 
            read the absorbtion for a `sys_name` provided for `winds` arg.
            
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
                         x = self.x_apex, y = self.y_apex,
                         R_star=self.winds.Ropt, T_star = self.winds.Topt,
                         nu_los=self.winds.orrbit.nu_los,
                         incl_los=self.winds.orbit.incl_los)
        else:
            gg_res = gg_tab(E=e_phot, x=self.x_apex_coord, y=self.y_apex_coord,
                            orb=self.winds.orbit,
                            filename=filename, what_return=what_return)
        return gg_res
    
    ###########################################################################
    @property
    def ug(self):
        """Photon field energy density on the IBS [erg/cm^3]."""
        return self.winds.u_g_density(r_from_s = self.r1,
                                      r_star = self.winds.Ropt,
                                      T_star = self.winds.Topt,
                                      )
    
    @property
    def ug_mid(self):
        """Photon field energy density on the IBS_mid [erg/cm^3]."""
        return self.winds.u_g_density(r_from_s = self.r1_mid,
                                      r_star = self.winds.Ropt,
                                      T_star = self.winds.Topt,
                                      )
    
    @property
    def ug_comov(self):
        """Photon field energy density on the IBS in the comoving frame [erg/cm^3]."""
        return lor_trans_ug_iso(ug_iso = self.ug, gamma=self.g)
      
    @property
    def ug_mid_comov(self):
        """Photon field energy density on the IBS_mid in the comoving frame [erg/cm^3]."""          
        return lor_trans_ug_iso(ug_iso = self.ug_mid, gamma=self.g_mid)
      
        
    ###########################################################################
    @property
    def b_ns(self):
        """Neutron star-originating magnetic field on the IBS [G]."""
        return self.winds.ns_field_initialized(r_to_p = self.r, t=self.t_forbeta)
    
    @property
    def b_ns_mid(self):
        """Neutron star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.ns_field_initialized(r_to_p = self.r_mid, t=self.t_forbeta)
    
    @property
    def b_ns_comov(self):
        """Neutron star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_ns, gamma=self.g)
    
    @property
    def b_ns_mid_comov(self):
        """Neutron star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_ns_mid, gamma=self.g_mid)
    
    
    ###########################################################################
    @property
    def b_opt(self):
        """Optical star-originating magnetic field on the IBS [G]."""
        return self.winds.opt_field_initialized(r_to_s = self.r1, t=self.t_forbeta)
    
    @property
    def b_opt_mid(self):
        """Optical star-originating magnetic field on the IBS_mid [G]."""
        return self.winds.opt_field_initialized(r_to_s = self.r1_mid, t=self.t_forbeta)
    
    @property
    def b_opt_comov(self):
        """Optical star-originating magnetic field on the IBS in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_opt, gamma=self.g)
    
    @property
    def b_opt_mid_comov(self):
        """Optical star-originating magnetic field on the IBS_mid in the comoving frame [G]."""
        return lor_trans_b_iso(B_iso=self.b_opt_mid, gamma=self.g_mid)
    
    ###########################################################################
    @property
    def b(self):
        """Total magnetic field on the IBS [G]."""
        return self.b_ns + self.b_opt
    
    @property
    def b_mid(self):
        """Total magnetic field on the IBS_mid [G]."""
        return self.b_ns_mid + self.b_opt_mid
    
    @property
    def b_comov(self):
        """Total magnetic field on the IBS in the comoving frame [G]."""
        return self.b_ns_comov + self.b_opt_comov
    
    @property
    def b_mid_comov(self):
        """Total magnetic field on the IBS_mid in the comoving frame [G]."""
        return self.b_ns_mid_comov + self.b_opt_mid_comov
    
    ###########################################################################
    @property
    def T_opt_eff(self):
        """Optical star effective temperature on the IBS [K]. 
        Simply the star temperature everywhere."""
        return self.winds.Topt * np.ones(self.r.shape)
    
    @property
    def T_opt_eff_mid(self):
        """Optical star effective temperature on the IBS_mid [K]. 
        Simply the star temperature everywhere."""
        return self.winds.Topt * np.ones(self.r_mid.shape)
    
    @property
    def T_opt_eff_comov(self):
        """Optical star effective temperature on the IBS in the comoving frame [K]."""
        return lor_trans_Teff_iso(Teff_iso = self.T_opt_eff, gamma=self.g)
    
    @property
    def T_opt_eff_mid_comov(self):
        """Optical star effective temperature on the IBS_mid in the comoving frame [K]."""
        return lor_trans_Teff_iso(Teff_iso = self.T_opt_eff_mid, gamma=self.g_mid)
        
    
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
        At which energy [eV] to calculate the gamma-gamma absorbtion, if ibs_color
        is 'gg_tau' or 'gg_abs'. Default 1e12

    Raises
    ------
    ValueError
        If the ibs_color is not one of the recognizable options.

    Returns
    -------
    None.

    """
    def peek(self,  ax=None, show_winds=False,
             ibs_color='k', to_label=True,
             edgecolor='k', linewidth=0.1,
             alpha=0.5, colorbar=True,
             showtime=None, E_for_gg=1e12):
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


        # if show_winds:
        #     if not isinstance(self.winds, Winds):
        #         raise ValueError("You should provide winds:Winds to show the winds.")
        #     self.winds.peek(ax=ax, showtime=showtime,
        #                     plot_rs=False)

        # if rescaled:
        puls_vector = self.winds.orbit.vector_sp(self.t_forbeta)
        _xp, _yp = puls_vector[0], puls_vector[1]
        ax.scatter(_xp, _yp, 0, color='b') # pulsaR
        # ...and the star was alreay plotted in the winds.peek()
        ax.plot( [0, 1.5*_xp], [0, 1.5*_yp], [0, 0], color='k', alpha=0.3, ls='--',)
        vec_disk1, vec_disk2 = self.winds.vectors_of_disk_passage
        xx1, yy1, zz1 = vec_disk1                                                 
        xx2, yy2, zz2 = vec_disk2                                                 
        ax.plot([xx1, xx2], [yy1, yy2], [zz1, zz2], color='orange', ls='--', lw=2)    

        ##################################################################################
        if showtime is None:
            showtime = [-self.winds.orbit.T/2, self.winds.orbit.T/2]
        show_cond  = np.logical_and(self.winds.orbit.ttab > showtime[0], 
                                    self.winds.orbit.ttab < showtime[1])
        orb_x, orb_y = self.winds.orbit.xtab[show_cond], self.winds.orbit.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
            ]))
        ax.plot(orb_x, orb_y, 0*orb_x)
        ax.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.winds.orbit.r_periastr) )
        ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        ax.set_zlim(-1.2*x_scale, 1.2*x_scale) 
        
        plot_surface_quads(ax=ax, coords=self.r_vec, param=color_param, linewidth=linewidth,
                           edgecolor=edgecolor, colorbar=colorbar, phi_close=True,
                           cbar_label=bar_label, alpha=alpha,
                           )
        # ax.scatter(xstar_, ystar_, zstar_, color='b')
        ax.scatter(0, 0, 0, color='r')
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        # ax.set_zlim(-2, 2)
        xlos_, ylos_, zlos_ = self.unit_los * x_scale
        ax.quiver(0, 0, 0, 1.*xlos_, 1.*ylos_, 1.*zlos_, arrow_length_ratio=0.12, linewidth=2, color='g')
        ax.legend()
        # return line_ 
    # peek.__doc__ = peek_docs

    
    def __getattr__(self, name):
        ibs_n_ = self.__dict__.get("ibs_n", None)
        if ibs_n_ is None:
            raise AttributeError(name)
        return getattr(ibs_n_, name)
 
    