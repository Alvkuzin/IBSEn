# pulsar/spectrum.py
import numpy as np
from numpy import pi, sin, cos, exp

from ibsen.utils import loggrid, trapz_loglog, \
        interplg, fill_nans, fill_nans_1d, avg
        
from scipy.integrate import trapezoid
import astropy.units as u
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params, index
import ibsen.absorbtion.absorbtion as absb
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import naima
from naima.models import Synchrotron, InverseCompton

sed_unit = u.erg / u.s / u.cm**2
from astropy import constants as const

EV_TO_ERG = float(const.e.value) * 1e7
M_E = float(const.m_e.cgs.value)
C_LIGHT = float(const.c.cgs.value)
MC2E = M_E * C_LIGHT**2
def pl(x, p, norm):
    return norm * x**(-p)


def value_on_ibs_with_weights(arr_xy, x, y_extended, y_eval, weights, y_scale,
                                    toboost=True):
    """
    A technical function for constructing the 2d-array on the IBS:
        SED(y) = Arr(x, y / y_scale(x) ) * weights(x), given 
        the `Arr` pre-calculated at a grid [x, y_extended].

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
    sed_s_e_up : 2d np.array
        The quantity evaluated at x \times y_eval.

    """
        
    # It's maybe not the best idea to use RGI here, seems like it sometimes
    # interpolates too rough. But I haven't figured out a way to use interp1d 
    RGI = RegularGridInterpolator((x, y_extended), arr_xy, bounds_error=False,
                                  method = 'linear') 
    E_new_up = (y_eval[None, :] / y_scale[:, None])

    pts_up = np.column_stack([
        np.repeat(x, y_eval.size),
        E_new_up.ravel()      
    ])

    vals_up = RGI(pts_up) 
    I_interp_up = vals_up.reshape(x.size, y_eval.size)  
    sed_s_e_up = I_interp_up * weights[:, None]
    return sed_s_e_up

def calculate_sed_1zone_naima(e_photon, sed_function, dne_de, e_el, 
                          **sed_function_kwargs):
    """
    Helper to calculate an SED using Naima emission models.
        1. Prepares a naima-readable electron population function from electron
        spectrum (dne_de, e_el);
        2. Initializes an emisison model as sed_object = sed_function(e_spec_for_naima,
                                                         **sed_function_kwargs);
        3. Calculates the SED at the photon energies e_photon as a method 
        of sed_object

    Parameters
    ----------
    e_photon : 1d np.narray (Nph, )
        Photon energies [eV].
    sed_function : callable
        Naima lepton emission model function: Synchrotron, InverseCompton.
    dne_de : 1d np.narray (Ne, )
        Relativistic electrons spectrum dne/de [1/eV].
    e_el : 1d np.narray (Ne, )
        Electron energies at whch dne/de is passed.
    **sed_function_kwargs : 
        Kwargs for the Naima model.

    Returns
    -------
    1d np.narray (Nph, )
        SED calculated at photon energies e_photon.

    """
    ok = (dne_de > 0) & np.isfinite(dne_de)
    if e_el[ok].size < 2:
        raise ValueError("Less than 2 OK e-energies for Naima")
    e_spec_for_naima = naima.models.TableModel(e_el[ok]*u.eV,
                                               (dne_de[ok])/u.eV )
    sed_object = sed_function(e_spec_for_naima, **sed_function_kwargs)
    return sed_object.sed(e_photon * u.eV) / sed_unit

docstr_specibs =  f"""
    Spectral energy distribution (SED) from an intrabinary shock (IBS).

    Builds broadband SEDs produced along the IBS at a chosen orbital epoch,
    using the precomputed electron distribution on the shock and optionally
    applying Lorentz boosting (segment-wise Doppler factors) and line-of-sight
    absorption. Supports synchrotron and inverse-Compton (IC; isotropic or
    anisotropic) emission, integrates segment contributions over both horns,
    and exposes helpers for band fluxes, photon indexes, and quick-look plots.
    Energies are in eV; SEDs are in erg /s/ cm2. 
    Parameters
    ----------
    els : ElectronsOnIBS
        Electron distribution and IBS geometry at the chosen epoch (provides
        per-segment spectra, B/u scaling, Doppler factors, and arc-length grid).
    mechanisms : list of {'syn','ic'}, optional
        Emission mechanisms to include. Case-insensitive aliases accepted
        (e.g., 's', 'synchrotron', 'i', 'inverse_compton'). Default ['syn','ic'].
    method: str {'full', 'simple', 'apex'}, optional
        How to calculate the spectrum of the IBS. Default 'full'.
            - if 'full', Naima calculates SEDs in every mid-point of the
        IBS with its values of fields, on the populations of electrons 
        corresponding to this IBS point. The total SED is calculated by  
        summing up all SEDs (with doppler-boosting, of course).
            - if 'simple', Naima calculates one SED and then rescales it to 
        every point of the IBS. Tyypically with some effective 
        values of fields, temperatures, scattering angles, and some 
        averaged electron population, etc, are used. Then this
        SED is attached to all mid-points of the IBS, and then properly
        doppler-boosted. 
            - if 'apex', the values of fields in the apex are used. The 
        electron population is summed over the whole IBS.
        
    ic_ani : bool, optional
        If True, compute anisotropic IC using the IBS scattering angle; else
        isotropic IC. Default False.
    sys_name : {known_names} or None, optional
        If str, load default distance to the system via ``get_parameters(sys_name)``.
         Explicit keyword arguments
        below override any defaults. If None, distance must be
        given explicitly or be in a dict `sys_params`.
    sys_params : dict or None, optional
        If dict, load distance from this dictionary. Explicit keywords
        below override any defaults. If None, distance must be
        given explicitly or be in `sys_name`.
    delta_power : float, optional
        Exponent of the Doppler weight for SEDs used when summing up segments,
        i.e. weight ∝ δ^{{delta_power}}. Default 4. 
    lorentz_boost : bool, optional
        If True, transform fields and electron spectra to the comoving frame
        before radiation calculation and use comoving scattering angles for
        anisotropic IC. Default True. 

    abs_photoel : bool, optional
        Apply photoelectric absorption (TBabs-like) to the final SED. Default False.
    abs_gg : bool, optional
        Apply γγ absorption along the line of sight (currently implemented for
        'psrb' system). Default False. 
    nh_tbabs : float, optional
        Hydrogen column density for photoelectric absorption (10²² cm⁻² units as
        used by the helper). Default 0.8. 
    distance : float or None, optional
        Source distance [cm]. If None, taken from system defaults via
        ``unpack_dist(orbit.name, distance)``.

    Attributes
    ----------
    els : ElectronsOnIBS
        Electron/IBS container used for radiation.
    _orb : Orbit
        Orbit of the system (from ``els.ibs.winds.orbit``).
    _ibs : IBS
        IBS geometry at the evaluation time.
    distance : float
        Adopted distance [cm].
    ic_ani, delta_power, lorentz_boost, simple, abs_photoel, abs_gg, nh_tbabs, apex_only
        Stored configuration flags/parameters.
    e_ph : ndarray
        Photon-energy grid used for the last computed SED [eV].
    sed : ndarray
        Total SED integrated over the IBS [erg s⁻¹ cm⁻²].
    sed_s : ndarray
        Per-segment SED array with shape (n_segments, e_ph.size) [erg s⁻¹ cm⁻²].
    sed_sy, sed_ic : ndarray, optional
        Mechanism-separated totals (set if those mechanisms were computed). 
    calculated : bool
        Whether the SED was calculated.

    Methods
    -------
    sed_nonboosted_apex(self, e_ext, emiss_mechanism)
        Compute an SED using the apex fields values and a total el spectrum.
    sed_nonboosted_simple(self, e_ext, emiss_mechanism)
        Compute an SED over the IBS using a simple and fast approach.
    sed_nonboosted_apex(self, e_ext, emiss_mechanism)
        Compute an SED over the IBS using the currect approach.
    calculate(e_ph=np.logspace(2,14,1000), to_return=False)
        Compute and store SED(s) on energy grid e_ph;
        optionally return (e_ph, sed_tot, sed_s_).
    flux(e1, e2, epow=1)
        Band integral over [e1, e2] of order ``epow`` of dN/dE (derived from SED),
        returned in erg^{{epow}} s⁻¹ cm⁻². Raises ValueError if interpolation fails.
    fluxes(bands, epows=None)
        Vectorized band fluxes for many (e1,e2) intervals; ``epows`` may be None,
        a scalar, or an iterable matching ``bands``. 
    index(e1, e2)
        Photon index γ fitted over [e1, e2] assuming dN/dE ∝ E^{{-γ}}; returns NaN
        if the fit fails. 
    indexes(bands)
        Vectorized photon indexes for many bands. 
    peek(ax=None, to_label=True, show_many=True, **kwargs)
        Quick-look plot: total SED and emissivity (energy-integrated per segment).

    Notes
    -----
    * **Segment treatment.** If ``method!=`apex`', each segment’s SED is computed
      (synchrotron or IC) using per-segment electron spectra and local B/u
      values.
      For symmetric mechanisms (syn and ic), the opposite horn is filled by symmetry; for
      anisotropic IC, both horns are computed explicitly. 
    * **Boosting & integration.** Segment SEDs are Doppler-boosted, absorbed,
      and summed along arclength; The internal energy grid are extended
      by the maximum Doppler factor to avoid edge losses. 
    * **Units.** Input energies e_ph, e1, e2 are in eV. Stored/returned SEDs are in
      erg s⁻¹ cm⁻². The band integral uses a log-grid and applies the necessary
      eV→erg factor for general moment ``epow``. 

    """
class SpectrumIBS: #!!!
    __doc__ = docstr_specibs
    def __init__(self, els, mechanisms=['syn', 'ic'], 
                 method='full',
                 ic_ani=False, 
                 sys_name=None, sys_params=None,
                 delta_power=3, lorentz_boost=True,
                 abs_photoel=False, abs_gg=False, 
                 nh_tbabs=0.8, 
                 distance = None):
        self.calculated = False
        self.els = els
        self._orb = self.els.ibs.winds.orbit
        self._ibs = self.els.ibs
        self.method = method
        self.ic_ani = ic_ani
        self.delta_power = delta_power
        self.lorentz_boost = lorentz_boost
        self.abs_photoel = abs_photoel
        self.abs_gg = abs_gg
        
        self.nh_tbabs = nh_tbabs
        self.mechanisms = mechanisms
        
        _dist = unpack_params(('D', ),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names, get_defaults_func=get_parameters,
                               D=distance)
        self.distance = _dist
        self._set_sed_parameters()
        
    def _set_sed_parameters(self):
        """
        fetches parameters from `ibs` and `els` needed for SED calculation.
        Only the parameters set here should be used for SED. 
        
        E.g.,instead of using self.els.ibs.b_mid somewhere in synchrotron SED
        computation, set here self.b = self.els.ibs.b_mid and use only that.
        """
        if not self.els.calculated:
            print('no dNe_deds_IBS in els, calculating...')
            self.els.calculate()
            
        self.dopls = self._ibs.dopl_mid # for both horns
        self.i_apex = int(  (self._ibs.s_mid.size-1)  / 2) # apex point at IBS    
        self.ibs_size = self._ibs.s_mid.size
        
        if self.lorentz_boost:
            b_2horns = self._ibs.b_mid_comov
            u_2horns = self._ibs.ug_mid_comov
            temp_2horns = self._ibs.T_opt_eff_mid_comov
            scat_ang_2horns = self._ibs.scattering_angle_mid_comov
            e_vals = self.els.e_vals
            dne_de_mid = self.els.dNe_de_mid
            ne_i_mid = self.els.n_i_mid
        else:
            b_2horns = self._ibs.b_mid
            u_2horns = self._ibs.ug_mid
            temp_2horns = self._ibs.T_opt_eff_mid
            scat_ang_2horns = self._ibs.scattering_angle_mid
            e_vals = self.els.e_vals_comov
            dne_de_mid = self.els.dNe_de_mid_comov
            ne_i_mid = self.els.n_i_mid_comov
            
        self.b = b_2horns
        self.u = u_2horns
        self.temp_eff = temp_2horns
        self.scat_ang = scat_ang_2horns
    
        self.e_el = e_vals
        self.dne_de = dne_de_mid
        self.ne_i = ne_i_mid
        
    def _key_from_mechanism(mechanism, ic_ani):
        """
        Turns user-passed key for mechanism and boolean ic_ani into internal
        key for the emisison mechanism calculation: 'syn', 'ic', 'ic_ani'.

        """
        if mechanism.lower() in ('s', 'sy', 'syn', 'synchr', 'synchrotron'):
            emiss_key = 'syn'
            
        elif mechanism.lower() in ('i', 'ic', 'inv', 'inverse_compton',
                            'inverse compton'):
            if ic_ani:
                emiss_key = 'ic_ani'
            else:
                emiss_key = 'ic'
        
        else:
            raise ValueError('I don\'t know this model. Try `Syn` or `IC`.' )
        return emiss_key
        
    def _set_absorbtion(self, e_ph):
        """
        Sets absorbtion coefficient e^-tau for energies `e_ph` in every point of
        IBS as self.abs_tot : array of shape (s_mid.size, e_ph.size).   

        Parameters
        ----------
        e_ph : np.ndarray
            Photon energies [eV].

        Returns
        -------
        None.

        """
        _abs_ph = np.ones(e_ph.size)
        _abs_gg = np.ones((self.ibs_size, e_ph.size))
        
        if self.abs_photoel:
            _abs_ph = absb.abs_photoel(E=e_ph, Nh = self.nh_tbabs)
        if self.abs_gg:          
            _abs_gg = self._ibs.gg_abs_mid(e_ph)  # array of shape (s_mid.size, e_ph.size)            
        abs_tot=_abs_gg*_abs_ph[None, :]
        self.abs_tot = abs_tot
    
    def _extended_photon_energies(self, e_ph):
        """
        Returns an auxillary grid of photon energies e_ext. 
        Parameters
        ----------
        e_ph : np.ndarray
            Photon energies [eV].

        Returns
        -------
        None.

        """
        
        d_max = max(np.max(self.dopls), 1/np.min(self.dopls))
        if self.method != 'apex':
            ### Introduce an auxillary extended grid over photon energries.
            ### Coef `2` is empirically found to be OK for later interpolation.
            ndec_ = int(2 * e_ph.size / np.log10(np.max(e_ph) / np.min(e_ph)))
            e_ext = loggrid(np.min(e_ph)/d_max/1.2, np.max(e_ph)*d_max*1.2,
                            ndec_)
        else:
            e_ext = e_ph
        return e_ext
    
    def sed_nonboosted_apex(self, e_ext, emiss_mechanism):
        """
        Calculates a non-boosted SED using the values of fields in the apex.
        The electron population is summed over the whole IBS.

        Parameters
        ----------
        e_ext : np.ndarray
            The photon energies [eV].
        emiss_mechanism : str {'syn', 'ic', 'ic_ani'}
            How to calculate the emission.

        Returns
        -------
        np.ndarray (e_ext.size, )
            SED calculated at e_ext.

        """
        dne_de_tot = np.sum(self.dne_de, axis=0)
        n_ = self.i_apex
        if emiss_mechanism == 'syn':
            emiss_function = Synchrotron
            kwargs = dict(B = self.b[n_] * u.G, nEed=71)
        if emiss_mechanism == 'ic':
            emiss_function = InverseCompton
            kwargs = dict(seed_photon_fields=[['star',
                                self.temp_eff[n_] * u.K,
                                self.u[n_] * u.erg / u.cm**3]],
                          nEed=177)
        if emiss_mechanism == 'ic_ani':
            emiss_function = InverseCompton
            kwargs = dict(seed_photon_fields=[['star',
                                self.temp_eff[n_] * u.K,
                                self.u[n_] * u.erg / u.cm**3,
                                self.scat_ang[n_] * u.rad
                                ]],
                          nEed=177)
        sed_apex = calculate_sed_1zone_naima(e_photon=e_ext,
                                    sed_function=emiss_function,
                                    dne_de=dne_de_tot,
                                    e_el=self.e_el,
                                    **kwargs)
        return sed_apex # shape (e_ext.size, )
    
    def sed_nonboosted_simple(self, e_ext, emiss_mechanism):
        """
        Calculates a non-boosted SED using effective values of fields
        and the total electron population summed over the whole IBS. The
        SED is then rescaled to every point of the IBS. The effective
        values are chosen as to give the best results for 
        energies around 1 keV (for synchrotron) and around 1 TeV (for IC).

        For synchrotron, the effective B is calculated as ssuming
        that in the energy range around 1 keV (for photons), 
        electron population is powerlaw like (the effective index p is
        found), and that emissivity \propto B^(p+1)/2. So

        B_eff^(p+1)/2  = sum( B^(p+1)/2 * dopl^delta_power ) / sum( dopl^delta_power ). 
          
         For IC,
        in analogy, we assume that the emissivity \propto u^(p+1)/2
        (without any reason, really, just by analogy). 
        
        Synchrotron and IC emission are then rescaled as
        sed_sy(s) = sed_eff_sy * (B(s) / B_eff)^((p_eff+1)/2) * n_i(s) / n_i_tot.
        sed_ic(s) = sed_eff_ic * (u(s) / B_eff)^((p_eff+1)/2) * n_i(s) / n_i_tot.
        
        These approximations seem to work with an accuracy of
         a few percent for
        'syn' around 1 keV and for 'ic' around 1 TeV, while for 
        'ic_ani' the accuracy is worse, up to 10-30%.

        Parameters
        ----------
        e_ext : np.ndarray
            The photon energies [eV].
        emiss_mechanism : str {'syn', 'ic', 'ic_ani'}
            How to calculate the emission.

        Returns
        -------
        np.ndarray (e_ext.size, )
            SED calculated at e_ext.

        """
        dne_de_tot = np.sum(self.dne_de, axis=0)
        _pow = 0.5*(self.els.p_e+1)
        e_el_sy = (3*3/1.6e-11)**0.5 * MC2E / EV_TO_ERG
        e_el_ic = (3e12 / 3 / 4)**0.5 * MC2E / EV_TO_ERG
        pe_b = index(dne_de_tot, self.e_el, e_el_sy/5, e_el_sy*5)
        pe_u = index(dne_de_tot, self.e_el, e_el_ic/5, e_el_ic*5)
        
        _pow_b = 0.5*(pe_b+1) if not np.isnan(pe_b) else _pow
        _pow_u = 0.5*(pe_u+1) if not np.isnan(pe_u) else _pow
        sy_r = (self.e_el > e_el_sy/5)  & (self.e_el < e_el_sy*5)
        n_i_syn = trapz_loglog(self.dne_de[:, sy_r], self.e_el[sy_r], axis=1)
        n_norm = n_i_syn / np.sum(n_i_syn) 
        
        if emiss_mechanism == 'syn':
            emiss_function = Synchrotron
            b_eff = (np.sum(self.b**_pow * self.dopls**self.delta_power) 
                     / np.sum(self.dopls**self.delta_power ))**(1/_pow)
            kwargs = dict(B = b_eff * u.G, nEed=71)
            rescale_coef = (self.b / b_eff) ** _pow_b * n_norm
            
        if emiss_mechanism in ('ic', 'ic_ani'):
            emiss_function = InverseCompton
            u_eff = (np.sum(self.u**_pow_u * self.dopls**self.delta_power) 
                     / np.sum(self.dopls**self.delta_power ))**(1/_pow_u)
            temp_eff_eff = avg(self.temp_eff, power=_pow_u, 
                               weights=self.dopls**self.delta_power )
            rescale_coef = (self.u / u_eff) ** _pow_u * n_norm
            
        if emiss_mechanism == 'ic':
            kwargs = dict(seed_photon_fields=[['star',
                                temp_eff_eff * u.K,
                                u_eff * u.erg / u.cm**3]],
                          nEed=177)
            
        if emiss_mechanism == 'ic_ani':
            scat_ang_eff = avg(self.scat_ang, power=_pow_u, 
                               weights=self.dopls**self.delta_power * self.ne_i)
            kwargs = dict(seed_photon_fields=[['star',
                                temp_eff_eff * u.K,
                                u_eff * u.erg / u.cm**3,
                                scat_ang_eff * u.rad
                                ]],
                          nEed=177)
            
        sed_eff = calculate_sed_1zone_naima(e_photon=e_ext,
                                    sed_function=emiss_function,
                                    dne_de=dne_de_tot,
                                    e_el=self.e_el,
                                    **kwargs)

        return sed_eff[None, :] * rescale_coef[:, None]
    
    def sed_nonboosted_full(self, e_ext, emiss_mechanism):
        """
        Calculates a non-boosted SED properly in every point of the IBS,
        using the local values of fields and the local electron population.

        Parameters
        ----------
        e_ext : np.ndarray
            The photon energies [eV].
        emiss_mechanism : str {'syn', 'ic', 'ic_ani'}
            How to calculate the emission.

        Returns
        -------
        np.ndarray (e_ext.size, )
            SED calculated at e_ext.

        """
        sed_s = np.zeros((self.ibs_size, e_ext.size))
        #### we iterate over the whole IBS. If mechanism is isotropic Sy or IC,
        #### we calculate everything properly for one horn (lower) and then
        #### for the upper horn we appoint the corresponding values. 
        ####
        #### If the mechanism in non-isotropic ic_ani (or generally any mechanism
        #### which is not axisymmetric), we calculate SED properly and 
        #### independently for both horns.
        for i_ibs in range(0, self.ibs_size):
            if i_ibs == self.i_apex:
                # in apex there are no electrons in our approach...
                sed_s[i_ibs, :] = np.zeros(e_ext.size)
                continue
            if emiss_mechanism in ('syn', 'ic'):
                ### if the emission mechanism is symmetric between lower and
                ### upper horn (axial symmetry around the S-P line), then,
                ### if we're at the upper horn, appoint the symmetric values
                ### from the lower horn and continue
                
                if i_ibs > self.i_apex:  
                    sed_s[i_ibs, :] = sed_s[2*self.i_apex - i_ibs, :]
                    continue
                
            if emiss_mechanism == 'syn':
                emiss_function = Synchrotron
                kwargs = dict(B = self.b[i_ibs] * u.G, nEed=71)
            if emiss_mechanism == 'ic':
                emiss_function = InverseCompton
                kwargs = dict(seed_photon_fields=[['star',
                                    self.temp_eff[i_ibs] * u.K,
                                    self.u[i_ibs] * u.erg / u.cm**3]],
                              nEed=177)
            if emiss_mechanism == 'ic_ani':
                emiss_function = InverseCompton
                kwargs = dict(seed_photon_fields=[['star',
                                    self.temp_eff[i_ibs] * u.K,
                                    self.u[i_ibs] * u.erg / u.cm**3,
                                    self.scat_ang[i_ibs] * u.rad
                                    ]],
                              nEed=177)
            sed_s[i_ibs, :] = calculate_sed_1zone_naima(e_photon=e_ext,
                                        sed_function=emiss_function,
                                        dne_de=self.dne_de[i_ibs, :],
                                        e_el=self.e_el,
                                        **kwargs)
        return sed_s

    def calculate(self, e_ph = np.logspace(2, 14, 1000),
                             to_return=False,):
        """
        Calculates emission from the intrabinary shock (IBS) for given
        mechanisms. The SED is calculated for both horns of the IBS and
        then summed up, doppler-boosted and absorbed.
        The SED is calculated in erg/s/cm2, and e_ph is in eV.

        Parameters
        ----------
        e_ph : 1d np.array, optional
            Photon energies to calculate the SEDs on. 
            The default is np.logspace(2, 14, 1000).
        to_return : bool, optional
            Whether to return e_ph, sed (total, 1d),
              sed-s (in every IBS segmens, 2d). The default is False.

        Raises
        ------
        ValueError
            If the radiation mechanism is not {'syn', 'ic'}.
            
        ValueError
            If the calculation method is not {'apex', 'simple', 'full'}.

        Returns
        -------
        e_ph : np.array
            Photon energies (eV).
        sed_tot : np.array of size of e_ph
            SED summed over the whole IBS (erg/s/cm2).
        sed_s_ : 2d np.array of ibs.s.size x e_ph.size
            SED in every segment of IBS (erg/s/cm2).

        """
        
        self._set_absorbtion(e_ph=e_ph)
        # -------------------------------------------------------------------------
        # for each segment of IBS, we calulate a spectrum and put it into
        # the 2-dimentional array sed_s_. The E_ph is always the same and it is 
        # e_ph but extended: ~ min(e_ph) / max(delta) < E_ph < max(e_ph) * max(delta)
        # -------------------------------------------------------------------------
        e_ext = self._extended_photon_energies(e_ph=e_ph)
        
        sed_tot = np.zeros(e_ph.size)
        sed_s_ = np.zeros((self._ibs.s_mid.size, e_ph.size))

        for mechanism in self.mechanisms:
            emiss_key = SpectrumIBS._key_from_mechanism(mechanism, self.ic_ani)
            if self.method == 'apex':
                sed_apex = self.sed_nonboosted_apex(e_ext=e_ext,
                                                    emiss_mechanism=emiss_key)
                sed_here = sed_apex * self.abs_tot[self.i_apex, :]
                sed_s_here = sed_here[None, :]
                
            elif self.method in ('full', 'simple'):
                if self.method == 'simple':
                    sed_s_nonboosted = self.sed_nonboosted_simple(e_ext=e_ext,
                                                    emiss_mechanism=emiss_key)
                if self.method == 'full':
                    sed_s_nonboosted = self.sed_nonboosted_full(e_ext=e_ext,
                                                    emiss_mechanism=emiss_key)
                sed_s_simple_boosted = value_on_ibs_with_weights(
                                        arr_xy=sed_s_nonboosted,
                                        x=self._ibs.s_mid,
                                        y_extended=e_ext,
                                        y_eval=e_ph,
                                        weights=self.dopls ** self.delta_power,
                                        y_scale = self.dopls)
                sed_s_here = sed_s_simple_boosted * self.abs_tot
                sed_here = np.sum(sed_s_here, axis=0)
            else:
                raise ValueError("""I don\'t know this method.
                                 Try `apex`, `simple`, or 'full'.""")
            
            sed_s_here = fill_nans(sed_s_here)
            sed_here = fill_nans_1d(sed_here)
            sed_tot += sed_here
            sed_s_ += sed_s_here 
           
            if emiss_key == 'syn':
                self.sed_sy = sed_here
                self.sed_s_sy = sed_s_here
            if emiss_key in ('ic', 'ic_ani'):
                self.sed_ic = sed_here
                self.sed_s_ic = sed_s_here
                
        ### now SEDs are in erg / s / cm2 
        ### but e_ph is in eV
        # sed_tot_fin = fill_nans_1d(sed_tot)
        # sed_s_fin = fill_nans(sed_s_)
        # sed_tot_fin_good = np.isfinite(sed_tot_fin) & 
        self.sed = fill_nans_1d(sed_tot)
        self.sed_s = fill_nans(sed_s_)
        self.e_ph = e_ph
        self.calculated = True
        if to_return:    
            return e_ph, sed_tot, sed_s_
    
    
    def flux(self, e1, e2, epow=1):
        """
        Flux in the band [e1, e2]: flux = \int_e1^e2 e^epow dN/de de

        Parameters
        ----------
        e1 : float
            Lower energy [eV].
        e2 : float
            Upper energy [eV].
        epow : float, optional
            Which moment of dN/de to integrate. epow=0 gives the number of
            photons in [e1, e2] [s^-1]. epow=1 gives flux [erg/s].
            The default is 1.

        Returns
        -------
        float
            Flux in the band [e1, e2]. More precisely, a moment of order epow
            of the photon distribution dN/de.

        """
        
        _mask = np.logical_and(self.e_ph >= e1/1.2, self.e_ph <= e2*1.2)
        _good = _mask & np.isfinite(self.sed)
        # Below, there's the interpolation. Without `fill_value`, the interpolation sed_here 
        # later in this funcion sometimes raises an error saying that e1, e2 are out of
        # interpolation bands. That should not happen, but apparently IS happening 
        # because of `_good` mask and random NaNs that are often in IC SED. Idk. 
        # Now we are filling everything to the left with the leftmost SED value
        # and the same with the right from the good interval: e_ph[_good].
        # It's gonna be bad only if there are a LOT of NaNs in a SED, which
        # should not be the case...

        ### we are shooting ourselves in a knee, potentially
        _E = loggrid(e1, e2, n_dec = 23) # eV
        sed_here = interplg(_E, 
                            self.e_ph[_good], 
                            self.sed[_good],
                            fill_value=(np.log10(self.sed[_good][0]),
                                        np.log10(self.sed[_good][-1])),
                            bounds_error=False,) 
                                
        return trapz_loglog(sed_here / _E**2 * _E**epow,
                            _E) * EV_TO_ERG**(epow-1) # erg^epow /s/cm^2
 
    
    def fluxes(self, bands, epows=None):
        """
        Compute band fluxes for multiple (e1, e2) bands.
    
        Parameters
        ----------
        bands : iterable of (e1, e2)
            Each element is a 2-tuple or 2-list with lower/upper energy.
        epows : None | scalar | iterable of scalars
            If None, use epow=1 for all bands.
            If scalar, use that same epow for all bands.
            If iterable, must have the same length as bands (per-band epow).
    
        Returns
        -------
        np.ndarray
            One flux value per band, in the same order as input.
        """
        bands = list(bands)
        n = len(bands)
    
        if epows is None:
            epows_list = [1] * n
        else:
            # accept scalar epows or a sequence matching length of bands
            try:
                m = len(epows)  # sequence?
            except TypeError:
                epows_list = [epows] * n  # scalar -> repeat
            else:
                if m != n:
                    raise ValueError("epows must be None, a scalar, or the same length as bands")
                epows_list = list(epows)
    
        fluxes_ = [
            SpectrumIBS.flux(self, e1, e2, epow=epow)
            for (e1, e2), epow in zip(bands, epows_list)
        ]
        return np.asarray(fluxes_)
    
    def index(self, e1, e2):
        """
        Photon index of a spectrum dN/de. Fits a dN/de in a given range 
        [e1, e2] with a powerlaw.

        Parameters
        ----------
        e1 : float
            Lower energy [eV].
        e2 : float
            Upper energy [eV].

        Returns
        -------
        float | np.nan
            If the fit is successful, the photon index is returned. If the 
            curve_fit raised an error, np.nan is returned.

        """
            
        _mask = np.logical_and(self.e_ph >= e1/1.2, self.e_ph <= e2*1.2)
        _good = _mask & np.isfinite(self.sed)

        _E = loggrid(e1, e2, n_dec = 61) # eV
        sed_here = interplg(_E, 
                            self.e_ph[_good], 
                            self.sed[_good],
                            fill_value=(np.log10(self.sed[_good][0]),
                                        np.log10(self.sed[_good][-1])),
                            bounds_error=False,) 
        try:
            popt, pcov = curve_fit(f = pl, xdata = _E,
                                   ydata = sed_here,
                                   p0=(0.5, 
                                       sed_here[15] * _E[15]**0.5
                                       ))
            return popt[0] + 2. 
        except:
            return np.nan
        
    def indexes(self, bands):
        """
        Compute photon indexes for multiple [e1, e2] bands.

        Parameters
        ----------
        bands : iterable of (e1, e2)
            Each element is a 2-tuple or 2-list with lower/upper energy.

        Returns
        -------
        np.ndarray
            One index value per band, in the same order as input.

        """
        indexes_ = []
        for band in bands:
            e1, e2 = band
            indexes_.append(SpectrumIBS.index(self, e1, e2))
        return np.array(indexes_)
        

    def peek(self, ax=None, 
            to_label=False,
        show_many = True,
        **kwargs):
        """
        Quick look at the SED and emissivity along the IBS.
        Draws on ax[0] the SED, and on ax[1] the emissivity along the IBS.

        Parameters
        ----------
        ax : ax object of pyplot, optional
            The axis to draw on. None or the axis with at least
             1 row and 2 colummns. If None, the ax obect is created.
               The default is None.
        to_label : bool, optional
            Whether to put legend on the ax[0]. The default is True.
        show_many : bool, optional
            Whether to show several SEDs from different parts of the IBS
            in addition to the total SED. The default is True.
        **kwargs : 
            kwargs for ax[0] and ax[1] (same for both).

        Raises
        ------
        ValueError
            If SED was not calculated before with calculate().

        """
    
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))    

        if self.sed is None:
            raise ValueError("You should call `calculate()` first to set SED")
        
        ax[0].plot(self.e_ph, self.sed, label=None, **kwargs)
        
        emiss_to_integr = np.where(np.isfinite(self.sed_s), self.sed_s, 0)
        # emiss_to_integr[]
        emiss_s = trapezoid(emiss_to_integr, self.e_ph, axis=1)
        # emiss_s_ = emiss_s
        # emiss_s_[np.isinf(emiss_s)]=np.nan
        # emiss_s_[np.isinf(emiss_s)]=0
        
        # print(np.nanmax(emiss_s_))
        ax[1].plot(self._ibs.s_mid, emiss_s/np.nanmax(emiss_s), **kwargs)
        # ax[1].plot(self._ibs.s_mid, emiss_s, **kwargs)
        
        if show_many:
            _n = self._ibs.n-1
            for i_s in (int(_n * 0.15),
                        int(_n * 0.7),
                        int(_n*1.3),
                        int(_n*1.85),
                    ):
                ilo, ihi = int(i_s-_n/10), int(i_s+_n/10)
                label_interval = f"{(self._ibs.s_mid[ilo] / self._ibs.s_max) :.2f}-{(self._ibs.s_mid[ihi] / self._ibs.s_max) :.2f}"
                label_s = fr"$s = ({label_interval})~ s_\mathrm{{max}}$"
                # int_sed_here = (trapezoid(self.sed_s[ilo : ihi, :], self._ibs.s[ilo:ihi], axis=0) / 
                #                 (self._ibs.s[ihi] - self._ibs.s[ilo]) 
                #                 )
                int_sed_here = np.sum(self.sed_s[ilo : ihi, :], axis=0)
                
   
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
        isfinite_sed = np.isfinite(self.sed)
        ax[0].set_ylim(np.nanmax(self.sed[isfinite_sed]) * 1e-3,
                       np.nanmax(self.sed[isfinite_sed]) * 2)
        ax[1].set_ylim(1e-3, 1.4)
        

        ax[1].set_xlabel(r'$s$')
        # ax[1].set_ylabel(r'Emissivity')
        ax[1].set_yscale('linear')
        ax[1].set_title(r'Emissivity along IBS')
        
        
