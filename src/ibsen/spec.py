# ibsen/spec.py
import numpy as np
from ibsen.utils import loggrid, trapz_loglog, \
        interplg,  avg, index_simple, unpack_params, index
        
import astropy.units as u
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.el_ev import e_syn_el, e_ic_el
import ibsen.absorption.absorption as absb
import naima
from naima.models import Synchrotron, InverseCompton

sed_unit = u.erg / u.s / u.cm**2
from astropy import constants as const

EV_TO_ERG = float(const.e.value) * 1e7
M_E = float(const.m_e.cgs.value)
K_BOLTZ = float(const.k_B.cgs.value)
C_LIGHT = float(const.c.cgs.value)
MC2E = M_E * C_LIGHT**2

def _to_iter(required_n, arr=None, default_for_None=1.0, descr_for_errors='this arr'):
    if arr is None:
        res = [default_for_None for _i in range(required_n)]
    elif isinstance(arr, float) or isinstance(arr, int):
        res = [float(arr) for _i in range(required_n)]
    elif isinstance(arr, np.ndarray):
        if arr.ndim==1 and arr.shape[0]==required_n:
            res = arr
        else:
            raise ValueError(f"If '{descr_for_errors}' is an np.array, it should be a 1d array of length {required_n}.")
    elif  isinstance(arr, list) or isinstance(arr, tuple):
        if len(arr)==required_n:
            res = arr
        else:
            raise ValueError(f"If '{descr_for_errors}' is an iterable, it should be of length {required_n}.")
    else:
        raise ValueError(f"'{descr_for_errors}' should be either iterable of length {required_n}, or int/float, or None.")
    return res

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

def _cell_signature_matrix(spatial_shape, *fields, decimals=None):
    """
    Build a 2D signature matrix: one row per spatial cell.

    Each field may be either:
      - scalar over space: shape == spatial_shape
      - vector over space:  shape == spatial_shape + (k,)

    The rows are concatenated into a single numeric signature.
    """
    mats = []
    for f in fields:
        a = np.asarray(f)

        if a.shape == spatial_shape:
            mats.append(a.reshape(-1, 1))
        elif a.shape[:-1] == spatial_shape:
            mats.append(a.reshape(-1, a.shape[-1]))
        else:
            raise ValueError(
                f"Incompatible shape {a.shape}, expected {spatial_shape} "
                f"or {spatial_shape + (a.shape[-1],)}"
            )

    sig = np.concatenate(mats, axis=1)

    # Optional approximate symmetry detection:
    # if decimals is None -> exact matching
    # if decimals is set -> round before unique
    if decimals is not None:
        sig = np.round(sig, decimals=decimals)

    _, first_idx, inv = np.unique(sig, axis=0, return_index=True, return_inverse=True)
    return first_idx, inv


def doppler_transform_sed(sed_prime, delta, weights, e_big, e_out):
    """
    Given the SED calculated in an extended zone (..., Ne), in co-moving
    frames on a photon energies `e_big`, calculates the Doppler-boosted SED in 
    a lab frame on photon energies `e_out`:
        
        SED(E) = SED_prime(E/delta) * weight.
    
    Parameters
    ----------
    sed_prime : array, shape (..., Ne_big)
        Comoving-frame SED in each cell.
    delta : array, shape (...)
        Doppler factor in each cell.
    weights : array, shape (...)
        The weights at which to multiply the lab-frame SED.
    e_big : array, shape (Ne_big,)
        Energy grid on which sed_prime is tabulated.
    e_out : array, shape (Ne,)
        Desired lab-frame energy grid.

    Returns
    ----------
    sed_lab : array, shape (..., Ne)
        Lab-frame SED.
        
    Notes
    ----------
    
    Puts all spatial dimensions in one row, obtaining the array 
    (N_spat_tot, Ne_big), then interpolates, in a self-written manner, over
    energies to recalculate SED(e) --> SED(e/delta_{ij...}). Then multiplies
    by `weights`, which are typically expected to be delta^(3...4). 
    
    The interpolation is in a log-log scale. The SED_prime values are clipped
    from below at the level of 2x numpy minimum non-zero value.
    
    If the inputs: `sed_prime` or `delta` or `weights`, contain NaNs/infs,
    the behaviour is undefined.
        
    """
    
    spatial_shape = delta.shape
    M = delta.size
    Ne_big = e_big.size
    Ne = e_out.size

    sed2 = sed_prime.reshape(M, Ne_big)
    d = delta.reshape(M)
    w = weights.reshape(M)
    _tiny = np.finfo(float).tiny

    # Work in log-energy space
    logE_big = np.log(e_big)
    logE_out = np.log(e_out)
    
    # Target comoving-frame energies for each cell and each output energy
    logEt = logE_out[None, :] - np.log(d)[:, None]   # shape (M, Ne)
    
    # Find bracketing indices in log-energy grid
    idx = np.searchsorted(logE_big, logEt, side="left")
    idx = np.clip(idx, 1, Ne_big - 1)
    
    i0 = idx - 1
    i1 = idx
    
    # Bracketing coordinates in log-energy
    x0 = logE_big[i0]
    x1 = logE_big[i1]
    
    # Bracketing SED values
    y0 = np.take_along_axis(sed2, i0, axis=-1)
    y1 = np.take_along_axis(sed2, i1, axis=-1)
    
    # Log-log interpolation:
    # interpolate log(SED) linearly as a function of log(E)
    logy0 = np.log(np.maximum(y0, 2.*_tiny))
    logy1 = np.log(np.maximum(y1, 2.*_tiny))
    
    t = (logEt - x0) / (x1 - x0)
    log_sed_interp = (1.0 - t) * logy0 + t * logy1
    
    # Back to linear space and apply weights
    sed_lab = w[:, None] * np.exp(log_sed_interp)

    return sed_lab.reshape(spatial_shape + (Ne,))

def calculate_sed_1zone_naima(e_photon, sed_function, dne_de, e_el, distance,
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
    distance : float
        Distance to the source in cm. Can be 0, then the luminosity is returned.
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
    return sed_object.sed(e_photon * u.eV, distance = distance * u.cm) / sed_unit

docstr_specibs =  f"""
    Spectral energy distribution (SED) from an intrabinary shock (IBS).

    Builds broadband SEDs produced along the IBS at a chosen orbital epoch,
    using the precomputed electron distribution on the shock and optionally
    applying Lorentz boosting of values and line-of-sight
    absorption. Supports synchrotron and inverse-Compton (IC; isotropic or
    anisotropic) emission, sums up segment contributions over the whole IBS,
    and defines helpers for band fluxes, photon indexes, and quick-look plots.
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
        every point of the IBS. Some effective 
        values of fields, temperatures, scattering angles, and some 
        averaged electron population, etc, are used. Then this
        SED is rescaled to all mid-points of the IBS, and then properly
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
        i.e. weight \propto \delta^{{delta_power}}. Default 3. 
    lorentz_boost : bool, optional
        If True, transform fields and electron spectra to the comoving frame
        before radiation calculation and use comoving scattering angles for
        anisotropic IC. Default True. 

    abs_photoel : bool, optional
        Apply photoelectric absorption (TBabs-like) to the final SED. Default False.
    abs_gg : bool, optional
        Apply γγ absorption along the line of sight. Default False. 
    nh_tbabs : float, optional
        Hydrogen column density for photoelectric absorption (10^22 cm-2 units as
        used by XSpec). Default 0.8. 
    distance : float or None, optional
        Source distance [cm]. If None, taken from system defaults via
        ``unpack_dist(orbit.name, distance)``.
    mode : str {'rgi', 'int'}, optional
        Whether to use RegularGridInterpolator (vectorized but lin-lin
            interplation) or non-vectorized interp1d in log-log scale for
            each point at the IBS independently.
    ne_mult : float, optional
        The density of nods at the auxilary energy grid. ne_mult=1 means
        the same density as the input e_ph of calculate(e_ph).
        ne_mult < 1 (slightly rarer than the input) is +- OK, as far as 
        ne_mult >= 0.3-0.5, while ne_mult > 1 is redundant.
        Default 0.7
    nEed_syn : int, optional
        `nEed` Naima input for Synchrotron. If None, set to 51. Default None 
    nEed_ic : int, optional
        `nEed` Naima input for InverseCompton. If None, set to 171. 
        Default None 
    
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
        Total SED integrated over the IBS [erg s-1 cm-2].
    sed_s : ndarray
        Per-segment SED array with shape (n_segments, e_ph.size) [erg s-1 cm-2].
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
    sed_nonboosted_full(self, e_ext, emiss_mechanism)
        Compute an SED over the IBS using the fully correct method.
    calculate(e_ph=np.logspace(2,14,1000), to_return=False)
        Compute and store SED(s) on energy grid e_ph;
        optionally return (e_ph, sed_tot, sed_s_).
    flux(e1, e2, epow=1)
        Band integral over [e1, e2] of order ``epow`` of dN/dE (derived from SED),
        returned in erg^{{epow}} s-1 cm-2. Raises ValueError if interpolation fails.
    fluxes(bands, epows=None)
        Vectorized band fluxes for many (e1,e2) intervals; ``epows`` may be None,
        a scalar, or an iterable matching ``bands``. 
    index(e1, e2)
        Photon index γ fitted over [e1, e2] assuming dN/dE \propto E^{{-index}}; returns NaN
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
      erg s-1 cm-2. The band integral uses a log-grid and applies the necessary
      eV --> erg factor for general moment ``epow``. 

    """
class SpectrumIBS: #!!!
    __doc__ = docstr_specibs
    def __init__(self, els, 
                 mechanisms=['syn', 'ic'], 
                 method='full',
                 ic_ani=False, 
                 sys_name=None, sys_params=None,
                 delta_power=3, lorentz_boost=True,
                 abs_photoel=False, abs_gg=False, 
                 nh_tbabs=0.8, 
                 distance = None,
                 mode='int',
                 ne_mult=0.7,
                 nEed_syn = None,
                 nEed_ic = None):
        self.calculated = False
        self.els = els
        try:
            self._orb = self.els.ibs.winds.orbit
            self._ibs = self.els.ibs
        except:
            print("No attributes 'ibs' and 'winds.orbit' here!")
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
        self.mode = mode
        self.ne_mult = ne_mult
        self.nEed_syn = nEed_syn if nEed_syn is not None else 51
        self.nEed_ic = nEed_ic if nEed_ic is not None else 171
        
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

            
        self.dopls = self._ibs.dopl_mid 
        self.spatial_shape = self.dopls.shape
        if self.lorentz_boost:
            b_2horns = self._ibs.b_mid_comov
            u_2horns = self._ibs.ug_mid_comov
            temp_2horns = self._ibs.T_opt_eff_mid_comov
            scat_ang_2horns = self._ibs.scattering_angle_mid_comov
            e_vals = self.els.e_vals_comov
            dne_de_mid = self.els.dNe_de_mid_comov
            ne_i_mid = self.els.n_i_mid_comov

        else:
            b_2horns = self._ibs.b_mid
            u_2horns = self._ibs.ug_mid
            temp_2horns = self._ibs.T_opt_eff_mid
            scat_ang_2horns = self._ibs.scattering_angle_mid
            e_vals = self.els.e_vals
            dne_de_mid = self.els.dNe_de_mid
            ne_i_mid = self.els.n_i_mid
            
        self.b = b_2horns
        self.u = u_2horns
        self.pe_default = self.els.p_e
        
        self.temp_eff = temp_2horns
        self.scat_ang = scat_ang_2horns
        
        self.Topt = self._ibs.winds.star.Topt
        self.b_apex = self._ibs.b_apex
        self.u_apex = self._ibs.ug_apex
        self.scat_ang_apex = self._ibs.scatter_angle_apex
        
        
        self.e_el = e_vals
        self.dne_de = dne_de_mid
        self.ne_i = ne_i_mid
        
        
    def _set_absorption(self, e_ph):
        """
        Sets absorption coefficient e^-tau for energies `e_ph` in every point of
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
        _abs_gg = np.ones(self.spatial_shape + (e_ph.size, ))
        _abs_gg_apex = np.ones(e_ph.size)
        
        if self.abs_photoel:
            _abs_ph = absb.abs_photoel(E=e_ph, Nh = self.nh_tbabs)
        if self.abs_gg:          
            _abs_gg = self._ibs.gg_abs_mid(e_ph)  # array of shape (s_mid.size, e_ph.size) or (n_phi, n-1, e_ph.size)           
            _abs_gg_apex = self._ibs.gg_abs_apex(e_ph)
        # abs_tot=
        self.abs_tot = _abs_gg * _abs_ph[..., :]
        self.abs_tot_apex = _abs_gg_apex * _abs_ph
    
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
            ### Introduce an auxillary extended grid over photon energries
            ndec_ = int(self.ne_mult * e_ph.size /
                        np.log10(np.max(e_ph) / np.min(e_ph))
                        )
            e_ext = loggrid(np.min(e_ph)/d_max/1.05, np.max(e_ph)*d_max*1.05,
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
        dne_de_tot = np.sum(self.dne_de, axis=tuple(range(self.dne_de.ndim-1)))

        if emiss_mechanism == 'syn':
            emiss_function = Synchrotron
            kwargs = dict(B = self.b_apex * u.G, nEed=self.nEed_syn)
        if emiss_mechanism == 'ic':
            emiss_function = InverseCompton
            kwargs = dict(seed_photon_fields=[['star',
                                self.Topt * u.K,
                                self.u_apex * u.erg / u.cm**3]],
                          nEed=self.nEed_ic)
        if emiss_mechanism == 'ic_ani':
            emiss_function = InverseCompton
            kwargs = dict(seed_photon_fields=[['star',
                                self.Topt * u.K,
                                self.ug_apex * u.erg / u.cm**3,
                                self.scat_ang_apex * u.rad
                                ]],
                          nEed=self.nEed_ic)
        sed_apex = calculate_sed_1zone_naima(e_photon=e_ext,
                                    sed_function=emiss_function,
                                    dne_de=dne_de_tot,
                                    e_el=self.e_el,
                                    distance = self.distance,
                                    **kwargs)
        return sed_apex # shape (e_ext.size, )
    
    def sed_nonboosted_simple(self, e_ext, emiss_mechanism):
        """
        Calculates a non-boosted SED using effective values of fields
        and the total electron population summed over the whole IBS. The
        SED is then rescaled to every point of the IBS. The effective
        values are chosen as to give the best results for 
        energies around 1 keV (for synchrotron) and around 1 TeV (for IC).

        For synchrotron, the effective B is calculated assuming
        that in the energy range around 1 keV (for photons), 
        electron population is powerlaw like (the effective index p is
        found), and that emissivity \propto B^(p+1)/2. So

        B_eff^(p+1)/2  = average( B^(p+1)/2) with weights = dopl^delta_power * n_e_relevant 
          
         For IC,
        in analogy, we assume that the emissivity \propto u^(p+1)/2
        (without any reason, really, just by analogy). 
        
        Synchrotron and IC emission are then rescaled as
        sed_sy(s) = sed_eff_sy * (B(s) / B_eff)^((p_eff+1)/2) * n_i(s) / n_i_tot.
        sed_ic(s) = sed_eff_ic * (u(s) / B_eff)^((p_eff+1)/2) * n_i(s) / n_i_tot.
        
        These approximations seem to work with an accuracy of
         ~10 percent for
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
        np.ndarray (spatial_shape, e_ext.size)
            SED calculated at e_ext.

        """
        dne_de_tot = np.sum(self.dne_de, axis=tuple(range(self.dne_de.ndim-1)))
        
        _pow = 0.5*(self.pe_default+1.)
        _b_avg = np.sqrt(np.average(self.b**1.5))
        _e_soft_avg = np.average( (self.temp_eff * K_BOLTZ * u.erg).to('eV').value)
        e_el_sy = e_syn_el(e_ph=3e4, b=_b_avg, return_dim='eV')
        e_el_ic = e_ic_el(e_ph=3e12, e_soft=_e_soft_avg, return_dim='eV', numerical_coef=4.0)
        pe_b = index(dne_de_tot, self.e_el, e_el_sy/5, e_el_sy*5)
        pe_u = index(dne_de_tot, self.e_el, e_el_ic/5, e_el_ic*5)
        
        _pow_b = 0.5*(pe_b+1) if not np.isnan(pe_b) else _pow
        _pow_u = 0.5*(pe_u+1) if not np.isnan(pe_u) else _pow
        sy_r = (self.e_el > e_el_sy/5)  & (self.e_el < e_el_sy*5)
        n_i_syn = trapz_loglog(self.dne_de[..., sy_r], self.e_el[sy_r], axis=-1)
        ic_r = (self.e_el > e_el_ic/5)  & (self.e_el < e_el_ic*5)
        n_i_ic = trapz_loglog(self.dne_de[..., ic_r], self.e_el[ic_r], axis=-1)
        
        n_norm_sy = n_i_syn / np.sum(n_i_syn) 
        n_norm_ic = n_i_ic / np.sum(n_i_ic) 
        
        
        if emiss_mechanism == 'syn':
            emiss_function = Synchrotron
            b_eff = avg(self.b, power=_pow_b, weights=self.dopls**self.delta_power  * n_i_syn)
            kwargs = dict(B = b_eff * u.G, nEed=self.nEed_syn)
            rescale_coef = (self.b / b_eff) ** _pow_b * n_norm_sy
            
        if emiss_mechanism in ('ic', 'ic_ani'):
            emiss_function = InverseCompton
            u_eff = avg(self.u, power=_pow_u, weights=self.dopls**self.delta_power  * n_i_ic)
            temp_eff_eff = avg(self.temp_eff, power=_pow_u, weights=self.dopls**self.delta_power )
            rescale_coef = (self.u / u_eff) ** _pow_u * n_norm_ic
            
        if emiss_mechanism == 'ic':
            kwargs = dict(seed_photon_fields=[['star',
                                temp_eff_eff * u.K,
                                u_eff * u.erg / u.cm**3]],
                          nEed=self.nEed_ic)
            
        if emiss_mechanism == 'ic_ani':
            scat_ang_eff = avg(self.scat_ang, power=_pow_u, 
                               weights=self.dopls**self.delta_power * n_i_ic)
            
            kwargs = dict(seed_photon_fields=[['star',
                                temp_eff_eff * u.K,
                                u_eff * u.erg / u.cm**3,
                                scat_ang_eff * u.rad
                                ]],
                          nEed=self.nEed_ic)
            
        sed_eff = calculate_sed_1zone_naima(e_photon=e_ext,
                                    sed_function=emiss_function,
                                    dne_de=dne_de_tot,
                                    e_el=self.e_el,
                                    distance = self.distance,
                                    **kwargs)
        return rescale_coef[..., None] * sed_eff
 
    def sed_nonboosted_full(self, e_ext, emiss_mechanism, decimals=None):
        """
        Calculates the non-boosted SED in every point of the emission zone,
        but only once for each unique set of local parameters.
    
        Parameters
        ----------
        e_ext : np.ndarray
            Photon energies [eV].
        emiss_mechanism : str {'syn', 'ic', 'ic_ani'}
            Emission mechanism.
        decimals : int or None
            If None, cells are considered identical only if all parameters match
            exactly. If an integer is given, parameters are rounded to that many
            decimals before deduplication.
    
        Returns
        -------
        np.ndarray
            SED on the full spatial grid, shape (..., e_ext.size).
        """
        e_ext = np.asarray(e_ext)
    
        # Spatial shape is the shape of the scalar fields
        spatial_shape = np.shape(self.b)
        M = int(np.prod(spatial_shape))
    
        flat_dne = self.dne_de.reshape(M, self.dne_de.shape[-1])
    
        if emiss_mechanism == "syn":
            emiss_function = Synchrotron
    
            # Parameters that define the SED for a cell
            sig_fields = (self.b, self.dne_de)
    
            def build_kwargs(i):
                return dict(
                    B=self.b.reshape(M)[i] * u.G,
                    nEed=self.nEed_syn,
                )
    
        elif emiss_mechanism == "ic":
            emiss_function = InverseCompton
            sig_fields = (self.u, self.temp_eff, self.dne_de)
    
            def build_kwargs(i):
                return dict(
                    seed_photon_fields=[[
                        "star",
                        self.temp_eff.reshape(M)[i] * u.K,
                        self.u.reshape(M)[i] * u.erg / u.cm**3,
                    ]],
                    nEed=self.nEed_ic,
                )
    
        elif emiss_mechanism == "ic_ani":
            emiss_function = InverseCompton
            sig_fields = (self.u, self.temp_eff, self.scat_ang, self.dne_de)
    
            def build_kwargs(i):
                return dict(
                    seed_photon_fields=[[
                        "star",
                        self.temp_eff.reshape(M)[i] * u.K,
                        self.u.reshape(M)[i] * u.erg / u.cm**3,
                        self.scat_ang.reshape(M)[i] * u.rad,
                    ]],
                    nEed=self.nEed_ic,
                )
    
        else:
            raise ValueError(f"Unknown emiss_mechanism={emiss_mechanism!r}")
    
        # Find unique parameter combinations
        first_idx, inv = _cell_signature_matrix(spatial_shape, *sig_fields, decimals=decimals)
    
        # Compute each unique SED once
        sed_unique = np.empty((len(first_idx), e_ext.size), dtype=float)
    
        for k, i in enumerate(first_idx):
            # If there are no electrons in this cell, set SED to zero
            dne = flat_dne[i]
            _ok = np.isfinite(dne) & (dne > 0)
            if dne[_ok].size < 2:
                sed_unique[k] = 0.0
                continue
            
            sed_unique[k, :] = calculate_sed_1zone_naima(
                e_photon=e_ext,
                sed_function=emiss_function,
                dne_de=flat_dne[i],
                e_el=self.e_el,
                distance=self.distance,
                **build_kwargs(i),
            )
    
        # Scatter back to all cells
        sed_flat = sed_unique[inv]
        return sed_flat.reshape(spatial_shape + (e_ext.size,))

    def calculate(self, e_ph = np.logspace(2, 14, 1000),
                             to_return=False,
                             ):
        """
        Calculates emission from the intrabinary shock (IBS) for given
        mechanisms. The SED is calculated for the whole emission zone and
        then summed up, doppler-boosted, and absorbed.
        The SED is calculated in erg/s/cm2, and e_ph is in eV.

        Parameters
        ----------
        e_ph : 1d np.array, optional
            Photon energies to calculate the SEDs on. 
            The default is np.logspace(2, 14, 1000).
        to_return : bool, optional
            Whether to return e_ph, sed (total, shape (e_ph.size, )),
              sed-s (in every IBS segmens, (spatial_shape, e_ph.size)). 
              The default is False.

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
            SED summed over the emisison zone (erg/s/cm2).
        sed_s_ : N+1-dim np.array of (..., e_ph.size) where N is the dimension
            of the emission zone.
            SED in every segment of the emission zone (erg/s/cm2).

        """
        
        self._set_absorption(e_ph=e_ph)
        # -------------------------------------------------------------------------
        # for each segment of IBS, we calulate a spectrum and put it into
        # the array sed_s_ (..., Ne_ext). The E_ph is the same for all cells and it is 
        # extended e_ph: ~ min(e_ph) / max(delta) < E_ph < max(e_ph) * max(delta)
        # -------------------------------------------------------------------------
        e_ext = self._extended_photon_energies(e_ph=e_ph)
        sed_tot = np.zeros(e_ph.size)
        sed_s_ = np.zeros(self.spatial_shape + (e_ph.size,))

        for mechanism in self.mechanisms:
            emiss_key = _key_from_mechanism(mechanism, self.ic_ani)
            if self.method == 'apex':
                sed_apex = self.sed_nonboosted_apex(e_ext=e_ext,
                                                    emiss_mechanism=emiss_key)
                sed_here = sed_apex * self.abs_tot_apex
                sed_s_here = sed_here[None, :]
                
            elif self.method in ('full', 'simple'):
                if self.method == 'simple':
                    sed_s_nonboosted = self.sed_nonboosted_simple(e_ext=e_ext,
                                                    emiss_mechanism=emiss_key)
                if self.method == 'full':
                    sed_s_nonboosted = self.sed_nonboosted_full(e_ext=e_ext,
                                                emiss_mechanism=emiss_key,
                                                decimals=6)
                    
                sed_s_nonabs_boosted = doppler_transform_sed(
                    sed_prime=sed_s_nonboosted,
                    delta=self.dopls, 
                    weights=self.dopls ** self.delta_power,
                    e_big = e_ext, 
                    e_out = e_ph)
                
                sed_s_here = sed_s_nonabs_boosted * self.abs_tot
                sed_here = np.sum(sed_s_here, axis=tuple(np.arange(sed_s_here.ndim - 1)))
                
            else:
                raise ValueError("""I don\'t know this method.
                                 Try `apex`, `simple`, or 'full'.""")
            

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
        self.sed = sed_tot
        self.sed_s = sed_s_
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
            photons in [e1, e2] [s^-1]; epow=1 gives flux [erg/s].
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
        _E_erg = (_E * u.eV).to("erg").value
        sed_here = interplg(_E, 
                            self.e_ph[_good], 
                            self.sed[_good],
                            fill_value=(np.log10(self.sed[_good][0]),
                                        np.log10(self.sed[_good][-1])),
                            bounds_error=False,) 
                                
        return trapz_loglog(sed_here / _E_erg**2 * _E_erg**epow, _E_erg)  # erg^epow /s/cm^2
 
    
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
    
        epows_list = _to_iter(required_n=n, arr=epows, default_for_None=1.0, 
                              descr_for_errors='epows')
    
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
        ind_ = index_simple(self.sed[_good] / self.e_ph[_good]**2,
                            self.e_ph[_good])
        return ind_
        
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
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))    

        if self.sed is None:
            raise ValueError("You should call `calculate()` first to set SED")
        
        ax[0].plot(self.e_ph, self.sed, label=None, **kwargs)
        
        
        emiss_to_integr = np.where(np.isfinite(self.sed_s), self.sed_s, 0)
        s_plot = self._ibs.s_mid
        if s_plot.ndim==2:
            _where = (self._ibs.phis<np.pi)
            emiss_to_integr_m, emiss_to_integr_p = emiss_to_integr.copy(), emiss_to_integr.copy()
            emiss_to_integr_p[_where, :, :] = 0
            emiss_to_integr_m[~_where, :, :] = 0
            emiss_to_integr_p = np.sum(emiss_to_integr_p, axis=0)
            emiss_to_integr_m = np.sum(emiss_to_integr_m, axis=0)
            emiss_to_integr = np.concatenate((emiss_to_integr_p[::-1], 
                                              emiss_to_integr_m))
            s_plot = np.concatenate((-self._ibs.s_mid[0][::-1], self._ibs.s_mid[0]))
            
        emiss_s = trapz_loglog(emiss_to_integr, self.e_ph, axis=-1)
        ax[1].plot(s_plot, emiss_s/np.nanmax(emiss_s), **kwargs)
        
        if show_many:
            _n = self._ibs.n-1
            for i_s in (int(_n * 0.15),
                        int(_n * 0.7),
                        int(_n*1.3),
                        int(_n*1.85),
                    ):
                ilo, ihi = int(i_s-_n/10), int(i_s+_n/10)
                label_interval = f"{(s_plot[ilo] / self._ibs.s_max_cm) :.2f}-{(s_plot[ihi] / self._ibs.s_max_cm) :.2f}"
                label_s = fr"$s = ({label_interval})~ s_\mathrm{{max}}$"
                int_sed_here = np.sum(emiss_to_integr[ilo : ihi, :], axis=0)
                
   
                ax[0].plot(self.e_ph, int_sed_here, alpha=0.3,
                           label=label_s, **kwargs)

            
        if to_label:
            ax[0].legend()
        
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
        
        
