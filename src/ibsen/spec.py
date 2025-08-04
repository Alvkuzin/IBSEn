# pulsar/spectrum.py
import numpy as np
from numpy import pi, sin, cos, exp

from ibsen.utils import lor_trans_e_spec_iso, lor_trans_b_iso, lor_trans_ug_iso, loggrid, trapz_loglog
from scipy.integrate import trapezoid
import astropy.units as u
from ibsen.get_obs_data import get_parameters
import ibsen.absorbtion.absorbtion as absb
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d, RegularGridInterpolator

# import time
import naima
from naima.models import Synchrotron, InverseCompton

sed_unit = u.erg / u.s / u.cm**2

def pl(x, p, norm):
    return norm * x**(-p)

def unpack_dist(sys_name=None, dist=None):
    """
    Unpack distance with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        dist: float or None
            Explicit values that override defaults.

    Returns:
        float dist
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
    dist_final = dist if dist is not None else defaults.get('D')

    return dist_final

def integrate_over_ibs_with_weights(arr_xy, x, y_extended, y_eval, weights, y_scale):
    """
    A technical function for integrating the 2d-array over the IBS:
        SED(y) = {\int_x Arr(x, y / y_scale) * weights(x) dx} / {\int_x dx}

    Parameters
    ----------
    arr_xy : 2d np.array
        The array of function Arr(x, y) tabulated at a grid of 
        x \times y_extended. It should be organized so that arr_xy[:, i] is 
        the slice at y_extended[i] and arr_xy[i, :] is the slice at x[i].
        
    x : 1d np.array
        The array of x-coordinates. On this array, arr_xy (see further) is \
            tabulated, and the integration will be performed over this 
            x-array.
    y_extended : 1d np.array
        The array of y-coordinates at which arr_xy is tabulated. The interval \
            over which y_extended varies should contain all values of 
            y_eval / y_scale over all x's.
    y_eval : 1d np.array
        The array of y-coordinates to evaluate the integral at.
    weights : 1d np.array
        An array of weights(x) tabulated at x.
    y_scale : 1d np.array
        An array of y_scale(x) tabulated at x.

    Returns
    -------
    sed_e_up : 1d np.array
        The integral.
    sed_e_down : 2d np.array
        The integrand evaluated at x \times y_eval.

    """
        
    # It's maybe not the best idea to use RGI here, seems like it sometimes
    # interpolates too rough. But I haven't figured out a way to use interp1d 

    
    RGI = RegularGridInterpolator((x, y_extended), arr_xy, bounds_error=False,
    fill_value=0., method = 'linear') 

    E_new_up = (y_eval[None, :] / y_scale[:, None])

    pts_up = np.column_stack([
        np.repeat(x, y_eval.size),
        E_new_up.ravel()      
    ])

    vals_up = RGI(pts_up) 

    I_interp_up = vals_up.reshape(x.size, y_eval.size)  

    div = np.max(x)
    sed_s_e_up = I_interp_up * weights[:, None]

    sed_e_up = trapezoid(sed_s_e_up, x, axis=0) /  div
    return sed_e_up, sed_s_e_up

def calculate_sed_simple(sed_e_apex, rescale):
    """
    Rescales the SED calculated at the apex according to the `rescale` array.
    
    Parameters
    ----------
    sed_e_apex : 1d np.array of size of energy grid.
        SED calculated in the apex.
    rescale : 1d np.array of size of 1 IBS horn
        In each IBS point, SED_apex will be multiplied by rescale[i_ibs].

    Returns
    -------
    sed_s_e : 2d np.array
        An array of SEDs from every point of IBS so that sed_s_e[:, i_e] is
        an i_e'th point of energy and sed_s_e[i_s, :] is the i_s'th point of
        an IBS.

    """
    sed_s_e = np.zeros((rescale.size, sed_e_apex.size))
    for i_ibs in range(0, rescale.size):
        sed_here = sed_e_apex * rescale[i_ibs]
        sed_s_e[i_ibs, :] = sed_here
    return sed_s_e
    

def calculate_sed_nonboosted_1horn(s_1d, e_ext, e_ev, dNe_de_1horn, e_el, simple, apex_only,
                             what_model, distance, B_apex=None, Topt=None, ug_apex=None,
                             b_scale=None, u_scale=None, n_scale=None, lorentz_boost=False,
                             gammas_1horn=None):
    """
    Compute the spectral energy distribution (SED) for synchrotron or inverse Compton emission
    from an electron population distributed along a binary shock.
    
    Parameters
    ----------
    s_1d : 1d np.array
        Shock coordinate array (cm) defining discrete segments.
    e_ext : 1d np.array 
        Energy grid on which initial SED should be calculated (eV).
    e_ev : 1d np.array
        Photon energy grid for the final SED to be calculated on (eV).
    dNe_de_1horn : 2D array
        Electron spectrum per segment (dN/dE).
    e_el : 1d np.array
        Electron energy bin centers (eV).
    simple : bool
        If True, apply simple rescaling of the apex SED.
    apex_only : bool
        If True, compute only the apex SED and skip extended calculation.
    what_model : {'syn', 'ic'}
        Emission mechanism: 'syn' for synchrotron, 'ic' for inverse Compton.
    distance : float
        Distance to the source in cm.
    B_apex : float, optional
        Magnetic field at the apex (G). Required for synchrotron.
    Topt : float, optional
        Seed photon temperature (K). Required for IC.
    ug_apex : float, optional
        Seed photon energy density at apex (erg/cm^3). Required for IC.
    b_scale : float or array-like, optional
        Scaling factor(s) for B-field along the shock.
    u_scale : float or array-like, optional
        Scaling factor(s) for photon energy density.
    n_scale : float, optional
        Normalization scaling for electron density.
    lorentz_boost : bool, optional
        If True, apply Lorentz transformations to fields and electron spectra.
    gammas_1horn : array-like, optional
        Lorentz factors for each segment if boosted.
    
    Returns
    -------
    sed_e_tot : 1d np.ndarray
        Integrated SED over all segments.
    sed_s_e_tot : 2d np.ndarray
        2D array (n_segments x n_e_ext) of SED per segment.
    
    Raises
    ------
    ValueError
        If `what_model` is not 'syn' or 'ic', or if no valid calculation mode.
    
    Notes
    -----
    Supports three modes:
      - apex_only: calculate only at the shock apex.
      - simple: rescale the apex SED along the shock in a simplified manner.
      - full: compute SED at each segment properly, including optional Lorentz boosting.
    

    
    """

    if apex_only or simple:
        dNe_de_1d = trapezoid(dNe_de_1horn, s_1d, axis=0) / np.max(s_1d)
        ok = np.where(dNe_de_1d > 0)
        e_spec_for_naima = naima.models.TableModel(e_el[ok]*u.eV, (dNe_de_1d[ok])/u.eV )
        e_dim = e_ext * u.eV

        if what_model == 'syn':
            sync = Synchrotron(e_spec_for_naima, B = B_apex * u.G, nEed=173)
            sed_synchr = sync.sed(e_dim, distance = distance * u.cm)
            sed_e_apex = sed_synchr / sed_unit
        
        elif what_model == 'ic':
            seed_ph = ['star', Topt * u.K, ug_apex * u.erg / u.cm**3]
            ic = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=71)
            sed_ic = ic.sed(e_dim, distance = distance * u.cm)
            sed_e_apex = sed_ic / sed_unit
        else: 
            raise ValueError('no mechanism')
            
        
        if apex_only:
            sed_e_tot = sed_e_apex
            sed_s_e_tot = sed_e_apex[None, :]

    if simple:
        if what_model == 'syn':
            B_here = B_apex * b_scale
            if lorentz_boost:
                # gamma_here = gammas_1horn[i_ibs]
                B_comov = np.array([
                    lor_trans_b_iso(B_iso=B_here[i_ibs], gamma=gammas_1horn[i_ibs])
                                for i_ibs in range(B_here.size)
                                    ])
               
            else:
                B_comov = B_here
            rescale_simple = (B_comov / B_apex)**2 * n_scale
        if what_model == 'ic':
            ug_here = ug_apex * u_scale
            if lorentz_boost:
                ug_comov = np.array([
                    lor_trans_ug_iso(ug_iso=ug_here[i_ibs], gamma=gammas_1horn[i_ibs])
                                for i_ibs in range(ug_here.size)
                                    ])
               
            else:
                ug_comov = ug_here                
            rescale_simple = ug_comov / ug_apex * n_scale

        sed_s_e_tot = calculate_sed_simple(sed_e_apex=sed_e_apex,
                                           rescale=rescale_simple)
        sed_e_tot = trapezoid(sed_s_e_tot, s_1d, axis=0) / trapezoid(np.ones(s_1d.size), s_1d)

                    
    elif (not simple) and (not apex_only):
        sed_s_e_tot = np.zeros((s_1d.size, e_ext.size))
        for i_ibs in range(0, s_1d.size):
            # rescaling B and u_g to the point on an IBS. I do it even in case
            # simple = True, so that I can retrieve the values later in addition 
            # to the SED 
            
            
            # Calculating B, u_g, and electron spectrum in the frame comoving
            # along the shock with a bulk Lorentz factor of Gammas[i_ibs]
            if what_model == 'syn':
                B_here = B_apex * b_scale[i_ibs]
                if lorentz_boost:
                    gamma_here = gammas_1horn[i_ibs]
                    B_comov = lor_trans_b_iso(B_iso=B_here, gamma=gamma_here)
                    e_vals_comov, dN_de_comov = lor_trans_e_spec_iso(E_lab=e_el,
                        dN_dE_lab=dNe_de_1horn[i_ibs, :], gamma=gamma_here,
                        )
                else:
                    B_comov = B_here
                    e_vals_comov, dN_de_comov = e_el, dNe_de_1horn[i_ibs, :]
                
                # Preparing e_spec so in can be fed to Naima
                if np.max(dN_de_comov) == 0:
                    continue
                ok = np.where(dN_de_comov > 0)
                e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
                E_dim = e_ext * u.eV

                # calculating an actual spectrum
                Sync = Synchrotron(e_spec_for_naima, B = B_comov * u.G)
                sed_syncr = Sync.sed(E_dim, distance = distance * u.cm)
                sed_tot = sed_syncr 
                # and putting a total dimentionLESS spec into SED_s_E
                sed_s_e_tot[i_ibs, :] = sed_tot / sed_unit
                    

                    
            if what_model == 'ic':
                u_g_here = ug_apex * u_scale[i_ibs]
                if lorentz_boost:
                    gamma_here = gammas_1horn[i_ibs]
                    u_g_comov = lor_trans_ug_iso(ug_iso=u_g_here, gamma=gamma_here)
                    e_vals_comov, dN_de_comov = lor_trans_e_spec_iso(E_lab=e_el,
                        dN_dE_lab=dNe_de_1horn[i_ibs, :], gamma=gamma_here)
                else:
                    u_g_comov = u_g_here
                    e_vals_comov, dN_de_comov = e_el, dNe_de_1horn[i_ibs, :]
                
                # Preparing e_spec so in can be fed to Naima
                if np.max(dN_de_comov) == 0:
                    continue
                ok = np.where(dN_de_comov > 0)
                e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
                E_dim = e_ext * u.eV

                seed_ph = ['star', Topt * u.K, u_g_comov * u.erg / u.cm**3]
                ic = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=101)
                sed_ic = ic.sed(E_dim, distance = distance * u.cm)
                # and putting a total dimentionLESS spec into SED_s_E
                sed_s_e_tot[i_ibs, :] = sed_ic / sed_unit
                    
        sed_e_tot = trapezoid(sed_s_e_tot, s_1d, axis=0) / trapezoid(np.ones(s_1d.size), s_1d)
    else:
        raise ValueError('You shouldn\'t be here')
    return sed_e_tot, sed_s_e_tot

def boost_sed_from_1horn(sed_s_e, s_1d, e_ext, e_ev, dopls, delta_power, abs_tot):
    """
    Boost and absorb an SED computed for a single horn of the intrabinary shock (IBS).
    
    Parameters
    ----------
    sed_s_e : array-like, shape (n_segments, n_e_ev)
        Input SED for each segment of one IBS horn.
    s_1d : array-like
        Shock coordinate array (cm) defining discrete segments.
    e_ext : array-like
        Energy grid used for initial SED integration (eV).
    e_ev : array-like
        Photon energy grid for boosted SED evaluation (eV).
    dopls : array-like, shape (2*n_segments,)
        Doppler factors: first n_segments for downstream, next n_segments for upstream.
    delta_power : float
        Power-law exponent applied to Doppler weights.
    abs_tot : array-like, shape (n_e_ev,)
        Total absorption factor to apply after boosting.
    
    Returns
    -------
    sed_tot : np.ndarray
        1D array of the total boosted and absorbed SED (dimensionless).
    sed_s_ : np.ndarray, shape (2*n_segments, n_e_ev)
        2D array of SED per segment for both downstream and upstream horns, after
        boosting and absorption.
    
    Notes
    -----
    1. Splits `dopls` into downstream and upstream arrays.
    2. Integrates the input SED over IBS segments weighted by Doppler factors
       raised to `delta_power` and scaled by the Doppler factor (`y_scale`).
    3. Concatenates downstream (reversed order) and upstream results.
    4. Applies the absorption factor `abs_tot` to both total and per-segment SED.
    

    
    """
    n_ = s_1d.size
    deltas_up = dopls[n_ : 2*n_]
    deltas_down = dopls[:n_]
    
    sed_e_up, sed_s_e_up = integrate_over_ibs_with_weights(arr_xy=sed_s_e,
                x=s_1d, y_extended=e_ext, y_eval=e_ev, 
                weights=deltas_up**delta_power,
                y_scale=deltas_up)
    
    sed_e_down, sed_s_e_down = integrate_over_ibs_with_weights(arr_xy=sed_s_e,
                x=s_1d, y_extended=e_ext, y_eval=e_ev, 
                weights=deltas_down**delta_power,
                y_scale=deltas_down)
    
    sed_tot = sed_e_up + sed_e_down

    sed_s_ = np.zeros((2 * n_, e_ev.size))
    sed_s_[:n_, :] = sed_s_e_down[::-1, :]
    sed_s_[n_ : 2*n_, :] = sed_s_e_up
    
    sed_tot = sed_tot * abs_tot
    sed_s_ = sed_s_ * abs_tot[None, :] 
                
    return sed_tot, sed_s_


class SpectrumIBS: #!!!
    def __init__(self, els, mechanisms=['syn', 'ic'],
                 delta_power=4, lorentz_boost=True, simple=False,
                 abs_photoel=True, abs_gg=False, nh_tbabs=0.8, 
                 distance = None, apex_only=False):
        self.els = els
        self._orb = self.els.ibs.winds.orbit
        self._ibs = self.els.ibs
        
        self.delta_power = delta_power
        self.lorentz_boost = lorentz_boost
        self.simple = simple
        self.abs_photoel = abs_photoel
        self.abs_gg = abs_gg
        self.nh_tbabs = nh_tbabs
        self.mechanisms = mechanisms
        self.apex_only = apex_only
        
        _dist = unpack_dist(self._orb.name, distance)
        self.distance = _dist
                

    def calculate_sed_on_ibs(self, E = np.logspace(2, 14, 1000),
                             to_set_onto_ibs=True, to_return=False,):
        
        _b_1horn, _u_1horn = (self.els._b[self._ibs.n : 2*self._ibs.n], 
                              self.els._u[self._ibs.n : 2*self._ibs.n])

        try:
            dNe_de_IBS, e_vals = self.els.dNe_de_IBS, self.els.e_vals
        except:
            print('no dNe_de_IBS in els, calculating...')
            dNe_de_IBS, e_vals = self.els.calculate(to_return=True)
            
        ug_a = self.els.u_g_apex
        rsp = self.els.r_sp
        _Topt = self._ibs.winds.Topt
        
        _abs_ph = np.ones(E.size)
        _abs_gg = np.ones(E.size)
        
        if self.abs_photoel:
            _abs_ph = absb.abs_photoel(E=E, Nh = self.nh_tbabs)
        if self.abs_gg:
            if self._orb.name != 'psrb': 
                raise Warning('abs_gg is only implemented for psrb orbit. Using abs_gg for psrb orbit.')
            _abs_gg = absb.abs_gg_tab(E=E,
                nu_los = self._orb.nu_los, t = self._ibs.t_forbeta, Teff=_Topt)
            
            
        # -------------------------------------------------------------------------
        # (1) for each segment of IBS, we calulate a spectrum and put it into
        # the 2-dimentional array SED_s_E. The E_ph is always the same and it is 
        # E but extended: min(E) / max(delta) < E_ph < max(E) * max(delta)
        # if simple=True, then the simple apex-spectrum rescaling and integration
        # is performed, without calculating a spec for each s-segment
        # -------------------------------------------------------------------------
        dopls = self._ibs.real_dopl # for both horns
        
        d_max = np.max(dopls)
        Nphot = int(2 * E.size * 
                    np.log10(d_max**2 * 1.21 * np.max(E) / np.min(E)) /
                    np.log10(np.max(E) / np.min(E)) )
        E_ext = np.logspace(np.log10(np.min(E)/d_max/1.1),
                            np.log10(np.max(E)*d_max*1.1), Nphot)
        # this SED_s_E we calculate only for 1 horn
        if self._ibs.one_horn:
            # s_1d_dim = self._ibs.s * rsp
            raise ValueError('IBS should be two-horned.')
        nibs = self._ibs.n
        s_1d_dim = self._ibs.s[nibs : 2*nibs] * rsp
        dNe_de_IBS_1horn = dNe_de_IBS[nibs : 2*nibs, :]
            
        # _n = 0.5 * (1 - cos( self._ibs.s_interp(s_=s_1d_dim/rsp, what = 'theta') ))
        Ntot = trapezoid(dNe_de_IBS_1horn, e_vals, axis=1)
        Navg = trapezoid(Ntot, s_1d_dim) / trapezoid(np.ones(s_1d_dim.size), s_1d_dim)
        _n = Ntot / Navg 

        gammas_1horn = self.els.ibs.gma(s = s_1d_dim/rsp)
        sed_tot = np.zeros(E.size)
        sed_s_ = np.zeros((2 * s_1d_dim.size, E.size))
        for mechanism in self.mechanisms:
            if mechanism.lower() in ('s', 'sy', 'syn', 'synchr', 'synchrotron'):
                sed_sy, sed_s_sy = calculate_sed_nonboosted_1horn(s_1d=s_1d_dim,
                        e_ext=E_ext, e_ev=E, dNe_de_1horn=dNe_de_IBS_1horn,
                        e_el=e_vals, simple=self.simple, 
                        apex_only=self.apex_only, what_model='syn',
                        distance=self.distance, B_apex=self.els.B_apex, 
                        b_scale=_b_1horn, n_scale=_n, lorentz_boost=self.lorentz_boost,
                        gammas_1horn=gammas_1horn) # one horn values
                if not self.apex_only:
                    sed_sy, sed_s_sy = boost_sed_from_1horn(sed_s_e=sed_s_sy,
                        s_1d=s_1d_dim, e_ext=E_ext, e_ev=E, dopls=dopls,
                        delta_power=self.delta_power, abs_tot=_abs_gg*_abs_ph) # two horn values now
                sed_tot += sed_sy
                sed_s_ += sed_s_sy
                self.sed_sy = sed_sy
                self.sed_s_sy = sed_s_sy
                
            elif mechanism.lower() in ('i', 'ic', 'inv', 'inverse_compton',
                                'inverse compton'):
                sed_ic, sed_s_ic = calculate_sed_nonboosted_1horn(s_1d=s_1d_dim,
                        e_ext=E_ext, e_ev=E, dNe_de_1horn=dNe_de_IBS_1horn,
                        e_el=e_vals, simple=self.simple, 
                        apex_only=self.apex_only, what_model='ic',
                        distance=self.distance,  Topt=_Topt, ug_apex=ug_a,
                        u_scale=_u_1horn, n_scale=_n, lorentz_boost=self.lorentz_boost,
                        gammas_1horn=gammas_1horn) # one horn values
                if not self.apex_only:
                    sed_ic, sed_s_ic = boost_sed_from_1horn(sed_s_e=sed_s_ic,
                        s_1d=s_1d_dim, e_ext=E_ext, e_ev=E, dopls=dopls,
                        delta_power=self.delta_power, abs_tot=_abs_gg*_abs_ph) # two horn values now
                sed_tot += sed_ic
                sed_s_ += sed_s_ic
                self.sed_ic = sed_ic
                self.sed_s_ic = sed_s_ic
                
            else:
                raise ValueError('I don\'t know this model. Try `syn` or `IC`.' )
        self.sed = sed_tot
        self.sed_s = sed_s_
        self.e_ph = E
        self.sed_spl = interp1d(E, sed_tot)
                
        if to_return:    
            return E, sed_tot, sed_s_
    
    
    def flux(self, e1, e2):
        try:
            sed_spl_ = self.sed_spl
        except:
            raise ValueError('The specrum has not been set yet.')
        _E = loggrid(e1, e2, n_dec = 250)
        sed_here = sed_spl_(_E)
        return trapz_loglog(sed_here / _E, _E)
            
    
    def fluxes(self,  bands):
        fluxes_ = []
        for band in bands:
            e1, e2 = band
            fluxes_.append(SpectrumIBS.flux(self, e1, e2))
        return np.array(fluxes_)
    
    def index(self, e1, e2):
        try:
            sed_spl_ = self.sed_spl
        except:
            raise ValueError('The specrum has not been set yet.')
        _E = loggrid(e1, e2, n_dec = 50)
        sed_here = sed_spl_(_E)
        popt, pcov = curve_fit(f = pl, xdata = _E,
                               ydata = sed_here,
                               p0=(0.5, 
                                   sed_here[0] * _E[0]**0.5
                                   ))
        return popt[0] + 2 
    
    def indexes(self, bands):
        indexes_ = []
        for band in bands:
            e1, e2 = band
            indexes_.append(SpectrumIBS.index(self, e1, e2))
        return np.array(indexes_)
        

    def peek(self, ax=None, 
            to_label=True,
        show_many = True,
        **kwargs):
    
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))    

        if self.sed is None:
            raise ValueError("You should call `calculate()` first to set SED")
        
        
        emiss_s = trapezoid(self.sed_s, self.e_ph, axis=1)

        ax[0].plot(self.e_ph, self.sed, label=None, **kwargs)
        ax[1].plot(self._ibs.s, emiss_s/np.max(emiss_s), **kwargs)



        if show_many:
            _n = self._ibs.n
            for i_s in (int(_n * 0.15),
                        int(_n * 0.7),
                        int(_n*1.3),
                        int(_n*1.85),
                    ):
                ilo, ihi = int(i_s-_n/10), int(i_s+_n/10)
                label_interval = f"{(self._ibs.s[ilo] / self._ibs.s_max) :.2f}-{(self._ibs.s[ihi] / self._ibs.s_max) :.2f}"
                label_s = fr"$s = ({label_interval})~ s_\mathrm{{max}}$"
                int_sed_here = (trapezoid(self.sed_s[ilo : ihi, :], self._ibs.s[ilo:ihi], axis=0) / 
                                (self._ibs.s[ihi] - self._ibs.s[ilo]) 
                                )
   
                ax[0].plot(self.e_ph, int_sed_here, alpha=0.3,
                           label=label_s, **kwargs)

            
        if to_label:
            ax[0].legend()
            # ax[1].legend()
        
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel(r'$E_\gamma$ [eV]')
        ax[0].set_ylabel(r'$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')
        ax[0].set_title(r'SED')

        ax[0].set_ylim(np.nanmax(self.sed) * 1e-3, np.nanmax(self.sed) * 2)
        ax[1].set_ylim(1e-3, 1.4)
        

        ax[1].set_xlabel(r'$s$')
        # ax[1].set_ylabel(r'Emissivity')
        ax[1].set_yscale('log')
        ax[1].set_title(r'Emissivity along IBS')
        
    
    
if __name__ == "__main__":
    from ibsen.orbit import Orbit

    DAY = 86400.
    AU = 1.5e13
    
    sys_name = 'psrb' 
    orb = Orbit(sys_name = sys_name, n=1003)
    from ibsen.winds import Winds
    winds = Winds(orbit=orb, sys_name = sys_name, alpha=-10/180*pi, incl=23*pi/180,
              f_d=165, f_p=0.1, delta=0.02, np_disk=3, rad_prof='pl', r_trunk=None,
             height_exp=0.25)     
    
    from ibsen.ibs import IBS
    from ibsen.el_ev import ElectronsOnIBS
    # from scipy.integrate import trapezoid
    
    t = 15 * DAY
    Nibs = 41
    ibs = IBS(beta=None,
              winds=winds,
              gamma_max=3.,
              s_max=1.,
              s_max_g=4.,
              n=Nibs,
              one_horn=False,
              t_to_calculate_beta_eff=t) # the same IBS as before
    
    els = ElectronsOnIBS(Bp_apex=1, ibs=ibs, cooling='adv', eta_a = 1e20,
                     to_inject_e = 'ecpl',) 
    els.calculate(to_set_onto_ibs=True, to_return=False)
    print('el_ev calculated')
    
    spec = SpectrumIBS(els=els, mechanisms=['i', 's'], simple=False,
                       apex_only=False, lorentz_boost=True)
    spec.calculate_sed_on_ibs()
    print('spec calculated')
    spec.peek()
