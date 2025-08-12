# pulsar/spectrum.py
import numpy as np
from numpy import pi, sin, cos, exp

from ibsen.utils import lor_trans_e_spec_iso, lor_trans_b_iso, \
    lor_trans_ug_iso, loggrid, trapz_loglog, lor_trans_Teff_iso, \
        interplg
from scipy.integrate import trapezoid
import astropy.units as u
from ibsen.get_obs_data import get_parameters
import ibsen.absorbtion.absorbtion as absb
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt

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

def fill_nans(sed):
    """
    Replace NaN values in a 2D SED array by the half-sum of their immediate
    neighbors along the energy axis (axis=1).

    Parameters
    ----------
    sed : array-like, shape (n_s, n_e)
        Input 2D array with NaNs to be filled.

    Returns
    -------
    filled : np.ndarray, shape (n_s, n_e)
        A copy of `sed` where each NaN at position [:, j] (1 <= j < n_e-1)
        has been replaced by 0.5*(sed[:, j-1] + sed[:, j+1]) whenever both neighbors
        are non-NaN. Edge columns are left unchanged.
    """
    sed_ = np.array(sed, dtype=float, copy=True)
    nan_mask = np.isfinite(sed)
    
    # Shifted arrays for left and right neighbors
    left  = np.roll(sed,  1, axis=1)
    right = np.roll(sed, -1, axis=1)
    
    # Candidate fill values
    fill_values = 0.5 * (left + right)
    
    # Valid positions: not edges, original is NaN, and neighbors are non-NaN
    valid = nan_mask.copy()
    valid[:,  0] = False
    valid[:, -1] = False
    valid &= ~np.isfinite(left) & ~np.isfinite(right)
    
    # Fill NaNs
    sed_[valid] = fill_values[valid]
    return sed_    

def fill_nans_1d(arr):
    """
    Replace NaNs in a 1D array by the half-sum of their immediate neighbors.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input 1D array with possible NaNs.

    Returns
    -------
    filled : np.ndarray, shape (n,)
        A copy of `arr` where each NaN at position i (1 <= i < n-1)
        is replaced by 0.5*(arr[i-1] + arr[i+1]) whenever both neighbors
        are non-NaN. Edge elements (i=0 and i=n-1) remain unchanged.
    """
    a = np.array(arr, dtype=float, copy=True)
    nan_mask = np.isfinite(a)
    
    # neighbors
    left  = np.roll(a,  1)
    right = np.roll(a, -1)
    
    # fill values
    fill_vals = 0.5 * (left + right)
    
    # valid positions: not edges, original is NaN, neighbors are valid
    valid = nan_mask.copy()
    valid[0] = False
    valid[-1] = False
    valid &= ~np.isfinite(left) & ~np.isfinite(right)
    
    a[valid] = fill_vals[valid]
    return a

def integrate_over_ibs_with_weights(arr_xy, x, y_extended, y_eval, weights, y_scale,
                                    ):
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
        The integral.sed_e_up
    sed_e_down : 2d np.array
        The integrand evaluated at x \times y_eval.

    """

    # if not toboost:
    #     # a = interp1d(x, y)
    #     # _spl = interp1d(y_extended, arr_xy[0, :])
    #     # sed_e_up = _spl(y_eval)
    #     sed_e_up = interplg(x=y_eval, xdata=y_extended, ydata=arr_xy[0, :])
    #     sed_s_e_up = sed_e_up[None, :]
    # if toboost:
        
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

    div = trapezoid(np.ones(x.size), x)
    sed_s_e_up = I_interp_up * weights[:, None]

    # sed_e_up = trapezoid(sed_s_e_up, x, axis=0) /  div
    sed_e_up = np.sum(sed_s_e_up, axis=0)
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
        sed_s_e[i_ibs, :] = fill_nans_1d(sed_here)
    return sed_s_e
    

def calculate_sed_nonboosted_2horns(s_1d_2horns, e_ext, e_ev, dNe_de_2horns, e_el, simple, apex_only,
                             what_model, distance, B_apex=None, Topt=None, ug_apex=None,
                             b_scale_2horns=None, u_scale_2horns=None, n_scale_2horns=None, lorentz_boost=False,
                             gammas_2horns=None,  scatter_angs_2horns=None):
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
        2D array (2 * n_segments x n_e_ext) of SED per segment.
    
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
    n_ = int(s_1d_2horns.size/2)
    # print(n_)
    s_1d_1horn = s_1d_2horns[n_:2*n_]
    dNe_de_1horn = dNe_de_2horns[n_:2*n_, :] 
    b_scale_1horn = b_scale_2horns[n_:2*n_] if b_scale_2horns is not None else None
    u_scale_1horn = u_scale_2horns[n_:2*n_] if u_scale_2horns is not None else None
    n_scale_1horn = n_scale_2horns[n_:2*n_] if n_scale_2horns is not None else None
    gammas_1horn = gammas_2horns[n_:2*n_] if gammas_2horns is not None else None
    scatter_angs_1horn = scatter_angs_2horns[n_:2*n_] if scatter_angs_2horns is not None else None  

    

    if apex_only or simple:
        
        # dNe_de_1d = trapezoid(dNe_de_1horn, s_1d_1horn, axis=0) / trapezoid(np.ones(n_), s_1d_1horn)
        ### for apex_only, we will calculate SED one time, so we need the 
        ### TOTAL number of emitting particles, but in simple we will 
        ### `calculate` SED in all marts of IBS, and then sum it up, so we
        ### need an average e-spectrum.
        if apex_only:
            dNe_de_1d = np.sum(dNe_de_1horn, axis=0)
        if simple:
            dNe_de_1d = np.sum(dNe_de_1horn, axis=0) / n_
        ok = np.where(dNe_de_1d > 0)
        e_spec_for_naima = naima.models.TableModel(e_el[ok]*u.eV, (dNe_de_1d[ok])/u.eV )
        e_dim = e_ext * u.eV

        if what_model == 'syn':
            sync = Synchrotron(e_spec_for_naima, B = B_apex * u.G, nEed=73)
            sed_synchr = sync.sed(e_dim, distance = distance * u.cm)
            sed_e_apex = fill_nans_1d(sed_synchr / sed_unit)
            # print(sed_e_apex)
        
        elif what_model == 'ic' or what_model == 'ic_ani':
            
            if what_model == 'ic' :
                seed_ph = ['star', Topt * u.K, ug_apex * u.erg / u.cm**3]
            else:
                scatter_apex = scatter_angs_1horn[0]
                seed_ph = ['star', Topt * u.K, ug_apex * u.erg / u.cm**3,
                           scatter_apex * u.rad]
            ic = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=71)
            sed_ic = ic.sed(e_dim, distance = distance * u.cm)
            sed_e_apex = fill_nans_1d(sed_ic / sed_unit)

            
        else: 
            raise ValueError('no mechanism')
            
        
    if apex_only:
        # sed_e_apex = sed_e_apex
        sed_e_tot = sed_e_apex
        sed_s_e_tot = sed_e_apex[None, :]

    elif simple:
        if what_model == 'syn':
            B_1horn = B_apex * b_scale_1horn
            if lorentz_boost:
                # gamma_here = gammas_1horn[i_ibs]
                B_comov = np.array([
                    lor_trans_b_iso(B_iso=B_1horn[i_ibs], gamma=gammas_1horn[i_ibs])
                                for i_ibs in range(B_1horn.size)
                                    ])
               
            else:
                B_comov = B_1horn
            rescale_simple = (B_comov / B_apex)**2 * n_scale_1horn
            # print(rescale_simple)
        if what_model == 'ic' or what_model == 'ic_ani':
            ug_1horn = ug_apex * u_scale_1horn
            if lorentz_boost:
                ug_comov = np.array([
                    lor_trans_ug_iso(ug_iso=ug_1horn[i_ibs], gamma=gammas_1horn[i_ibs])
                                for i_ibs in range(ug_1horn.size)
                                    ])
               
            else:
                ug_comov = ug_1horn                
            rescale_simple = ug_comov / ug_apex * n_scale_1horn
            
        #### for `simple`, we calculate the SED for 1 horn and attach it to both
        #### horns due to symmetry considerations 
        sed_s_e_tot_1horn = calculate_sed_simple(sed_e_apex=sed_e_apex,
                                           rescale=rescale_simple)

        sed_s_e_tot = np.concatenate((sed_s_e_tot_1horn[::-1, :], sed_s_e_tot_1horn))
        # sed_s_e_tot = fill_nans(sed_s_e_tot)

        # sed_e_tot = trapezoid(sed_s_e_tot, s_1d_2horns, axis=0) /(np.max(s_1d_2horns) - np.min(s_1d_2horns))
        sed_e_tot = np.sum(sed_s_e_tot, axis=0)
        # print(sed_e_tot)
                    
    elif (not simple) and (not apex_only):
        sed_s_e_tot = np.zeros((2*n_, e_ext.size))
        #### we iterate over the whole IBS. If mechanism is isotropic Sy or IC,
        #### we calculate everything properly for one horn (lower) and then
        #### for the upper horn we appoint the corresponding values. 
        ####
        #### If the mechanism in non-isotropic ic_ani (or generally any mechanism
        #### which depends on the horn), we calculate SED properly and 
        #### independently for both horns.
        for i_ibs in range(0, 2*n_):
            # if i_ibs == 0 or i_ibs == 2*n_-1:
            #     continue
            # rescaling B and u_g to the point on an IBS. I do it even in case
            # simple = True, so that I can retrieve the values later in addition 
            # to the SED 
            
            # Then calculating B, u_g, electron spectrum, and scattering angle
            # in the frame comoving
            # along the shock with a bulk Lorentz factor of Gammas[i_ibs]
            if what_model == 'syn':
                if i_ibs >= n_:  
                    # if we're at the upper horn, appoint the symmetric values
                    # from the lower horn and continue
                    sed_s_e_tot[i_ibs, :] = sed_s_e_tot[2*n_ - 1 - i_ibs, :]
                    continue 
                #### else --- calculate properly
                B_2horns = B_apex * b_scale_2horns[i_ibs]
                if lorentz_boost:
                    gamma_here = gammas_2horns[i_ibs]
                    B_comov = lor_trans_b_iso(B_iso=B_2horns, gamma=gamma_here)
                    e_vals_comov, dN_de_comov = lor_trans_e_spec_iso(E_lab=e_el,
                        dN_dE_lab=dNe_de_2horns[i_ibs, :], gamma=gamma_here,
                        )
                else:
                    B_comov = B_2horns
                    e_vals_comov, dN_de_comov = e_el, dNe_de_2horns[i_ibs, :]
                
                # Preparing e_spec so in can be fed to Naima
                if np.max(dN_de_comov) == 0:
                    continue
                ok = np.where(dN_de_comov > 0)
                e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
                E_dim = e_ext * u.eV

                # calculating an actual spectrum
                Sync = Synchrotron(e_spec_for_naima, B = B_comov * u.G)
                sed_synchr = Sync.sed(E_dim, distance = distance * u.cm)
                # sed_tot = sed_syncr 
                # and putting a total dimentionLESS spec into SED_s_E
                sed_s_e_tot[i_ibs, :] = fill_nans_1d(sed_synchr / sed_unit)
                    
 
            elif what_model == 'ic' or what_model == 'ic_ani':
                if i_ibs >= n_:  
                    # if we're at the upper horn, appoint the symmetric values
                    # from the lower horn and continue
                    sed_s_e_tot[i_ibs, :] = sed_s_e_tot[2*n_ - 1 - i_ibs, :]
                    continue 
                #### else --- calculate properly
                u_g_2horns = ug_apex * u_scale_2horns[i_ibs]
                if lorentz_boost:
                    gamma_here = gammas_2horns[i_ibs]
                    u_g_comov = lor_trans_ug_iso(ug_iso=u_g_2horns, gamma=gamma_here)
                    Teff_comov = lor_trans_Teff_iso(Topt, gamma=gamma_here)
                    e_vals_comov, dN_de_comov = lor_trans_e_spec_iso(E_lab=e_el,
                        dN_dE_lab=dNe_de_2horns[i_ibs, :], gamma=gamma_here)
                else:
                    u_g_comov = u_g_2horns
                    e_vals_comov, dN_de_comov = e_el, dNe_de_2horns[i_ibs, :]
                    Teff_comov = Topt
                # print(Teff_comov)
                # Preparing e_spec so in can be fed to Naima
                if np.max(dN_de_comov) == 0:
                    continue
                ok = np.where(dN_de_comov > 0)
                e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
                E_dim = e_ext * u.eV
                if what_model == 'ic':
                    seed_ph = ['star', Teff_comov * u.K, u_g_comov * u.erg / u.cm**3]
                if what_model == 'ic_ani':
                    seed_ph = ['star', Teff_comov * u.K, u_g_comov * u.erg / u.cm**3,
                               scatter_angs_2horns[i_ibs] * u.rad]
                    
                ic = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=101)
                sed_ic = ic.sed(E_dim, distance = distance * u.cm)
                # and putting a total dimentionLESS spec into SED_s_E
                sed_s_e_tot[i_ibs, :] = fill_nans_1d(sed_ic / sed_unit)
            else:
                raise ValueError('I don\'t know this model. Try `syn` or `ic`.' )
        sed_s_e_tot = fill_nans(sed_s_e_tot)
        # sed_e_tot = trapezoid(sed_s_e_tot, s_1d_2horns, axis=0) / (np.max(s_1d_2horns) - np.min(s_1d_2horns))
        sed_e_tot = np.sum(sed_s_e_tot, axis=0)
    else:
        raise ValueError('You shouldn\'t be here')
        
    return sed_e_tot, sed_s_e_tot

def boost_sed_from_2horns(sed_s_e, s_1d_2horns, e_ext, e_ev, dopls_2horns,
                          delta_power, abs_tot=None, toboost=True):
    """
    Boost and absorb an SED computed for a single horn of the intrabinary shock (IBS).
    
    Parameters
    ----------
    sed_s_e : array-like, shape (2 * n_segments, n_e_ev)
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
    1. Integrates the input SED over IBS segments weighted by Doppler factors
       raised to `delta_power` and scaled by the Doppler factor (`y_scale`).
    2. Applies the absorption factor `abs_tot` to both total and per-segment SED.
    

    
    """
    ##### we should move the condition with `if toboost...` here from
    ##### integrate_over_...
    if toboost:
        sed_tot, sed_s_ = integrate_over_ibs_with_weights(arr_xy=sed_s_e,
                    x=s_1d_2horns, y_extended=e_ext, y_eval=e_ev, 
                    weights=dopls_2horns**delta_power,
                    y_scale=dopls_2horns)
    else:
        sed_tot = sed_s_e[0, :]
        sed_s_ = sed_tot[None, :]

    if abs_tot is not None:
        sed_tot = sed_tot * abs_tot
        sed_s_ = sed_s_ * abs_tot[None, :] 
                
    return sed_tot, sed_s_


class SpectrumIBS: #!!!
    def __init__(self, els, mechanisms=['syn', 'ic'], ic_ani=False,
                 delta_power=4, lorentz_boost=True, simple=False,
                 abs_photoel=True, abs_gg=False, nh_tbabs=0.8, 
                 distance = None, apex_only=False):
        self.els = els
        self._orb = self.els.ibs.winds.orbit
        self.set_ibs()
        self.ic_ani = ic_ani
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
        
    def set_ibs(self):
        # check if the IBS was rescaled, and if not, rescale and use rescaled
        if abs(self.els.ibs.x_apex) > 1e5:
            # assume then that the IBS was rescaled
            self._ibs = self.els.ibs
        else:
            ibs1 = self.els.ibs.rescale_to_position()
            self._ibs = ibs1
                

    def calculate_sed_on_ibs(self, E = np.logspace(2, 14, 1000),
                             to_set_onto_ibs=True, to_return=False,):
        
        # _b_1horn, _u_1horn = (self.els._b[self._ibs.n : 2*self._ibs.n], 
        #                       self.els._u[self._ibs.n : 2*self._ibs.n])
        _b_2horns, _u_2horns = self.els._b, self.els._u
        

        try:
            dNe_de_IBS, e_vals = self.els.dNe_de_IBS, self.els.e_vals
        except:
            print('no dNe_de_IBS in els, calculating...')
            dNe_de_IBS, e_vals = self.els.calculate(to_return=True)
            
        # ug_a = self.els.u_g_apex
        # rsp = self.els.r_sp
        # _Topt = self._ibs.winds.Topt
        
        _abs_ph = np.ones(E.size)
        _abs_gg = np.ones(E.size)
        
        if self.abs_photoel:
            _abs_ph = absb.abs_photoel(E=E, Nh = self.nh_tbabs)
        if self.abs_gg:
            if self._orb.name != 'psrb': 
                print('abs_gg is only implemented for psrb orbit. Using abs_gg for psrb orbit.')
            _abs_gg = absb.abs_gg_tab(E=E,
                nu_los = self._orb.nu_los, t = self._ibs.t_forbeta, 
                Teff=self._ibs.winds.Topt)
            
            
        # -------------------------------------------------------------------------
        # (1) for each segment of IBS, we calulate a spectrum and put it into
        # the 2-dimentional array SED_s_E. The E_ph is always the same and it is 
        # E but extended: min(E) / max(delta) < E_ph < max(E) * max(delta)
        # if simple=True, then the simple apex-spectrum rescaling and integration
        # is performed, without calculating a spec for each s-segment
        # -------------------------------------------------------------------------
        dopls = self._ibs.real_dopl # for both horns
        
        d_max = max(np.max(dopls), 1/np.min(dopls))
        if not self.apex_only:
            Nphot = int(2 * E.size * 
                        np.log10(d_max**2 * 1.21 * np.max(E) / np.min(E)) /
                        np.log10(np.max(E) / np.min(E)) )
            E_ext = np.logspace(np.log10(np.min(E)/d_max/1.1),
                                np.log10(np.max(E)*d_max*1.1), Nphot)
        else:
            E_ext = E
        # this SED_s_E we calculate only for 1 horn
        if self._ibs.one_horn:
            # s_1d_dim = self._ibs.s * rsp
            raise ValueError('IBS should be two-horned.')
        # nibs = self._ibs.n
        # s_1d_dim = self._ibs.s[nibs : 2*nibs] #* rsp
        s_2horns = self._ibs.s
        # dNe_de_IBS_1horn = dNe_de_IBS[nibs : 2*nibs, :]
            
        # _n = 0.5 * (1 - cos( self._ibs.s_interp(s_=s_1d_dim/rsp, what = 'theta') ))
        Ntot = trapezoid(dNe_de_IBS, e_vals, axis=1)
        Navg = trapezoid(Ntot, s_2horns) / (np.max(s_2horns)-np.min(s_2horns))
        _n_norm_2horns = Ntot / Navg 

        # gammas_1horn = self.els.ibs.gma(s = s_1d_dim/rsp)
        gammas_2horns = self._ibs.g
        
        sed_tot = np.zeros(E.size)
        sed_s_ = np.zeros((s_2horns.size, E.size))
            
        """
        sed_s_e, s_1d_2horns, e_ext, e_ev, dopls_2horns, delta_power, abs_tot=None
        """
        # print('b 2 horns = ', _b_2horns)
        for mechanism in self.mechanisms:
            if mechanism.lower() in ('s', 'sy', 'syn', 'synchr', 'synchrotron'):
                sed_sy, sed_s_sy = calculate_sed_nonboosted_2horns(s_1d_2horns=s_2horns,
                        e_ext=E_ext, e_ev=E, dNe_de_2horns=dNe_de_IBS,
                        e_el=e_vals, simple=self.simple, 
                        apex_only=self.apex_only, what_model='syn',
                        distance=self.distance, B_apex=self.els.B_apex, 
                        b_scale_2horns=_b_2horns, n_scale_2horns=_n_norm_2horns,
                        lorentz_boost=self.lorentz_boost,
                        gammas_2horns=gammas_2horns) 
                # if not self.apex_only:
                sed_sy, sed_s_sy = boost_sed_from_2horns(sed_s_e=sed_s_sy,
                    s_1d_2horns=s_2horns, e_ext=E_ext, e_ev=E, dopls_2horns=dopls,
                    delta_power=self.delta_power, abs_tot=_abs_gg*_abs_ph,
                    toboost=(not self.apex_only)) 
                sed_tot += sed_sy
                sed_s_ += sed_s_sy
                self.sed_sy = sed_sy
                self.sed_s_sy = sed_s_sy
                
            elif mechanism.lower() in ('i', 'ic', 'inv', 'inverse_compton',
                                'inverse compton'):
                if self.ic_ani:
                    _model_name = 'ic_ani'
                    if self.lorentz_boost:
                        scatter_2horns = self._ibs.scattering_angle_comoving
                    else:
                        scatter_2horns = self._ibs.scattering_angle
                else:
                    _model_name = 'ic'
                    scatter_2horns = None
                sed_ic, sed_s_ic = calculate_sed_nonboosted_2horns(s_1d_2horns=s_2horns,
                        e_ext=E_ext, e_ev=E, dNe_de_2horns=dNe_de_IBS,
                        e_el=e_vals, simple=self.simple, 
                        apex_only=self.apex_only, what_model=_model_name,
                        distance=self.distance, n_scale_2horns=_n_norm_2horns,
                        Topt=self._ibs.winds.Topt, ug_apex=self.els.u_g_apex,
                        u_scale_2horns=_u_2horns,
                        lorentz_boost=self.lorentz_boost,
                        gammas_2horns=gammas_2horns, scatter_angs_2horns=scatter_2horns) 
                # print(sed_ic)
                # if not self.apex_only:
                sed_ic, sed_s_ic = boost_sed_from_2horns(sed_s_e=sed_s_ic,
                    s_1d_2horns=s_2horns, e_ext=E_ext, e_ev=E, dopls_2horns=dopls,
                    delta_power=self.delta_power, abs_tot=_abs_gg*_abs_ph,
                    toboost=(not self.apex_only)) 
                # print(sed_ic)
                sed_tot += sed_ic
                sed_s_ += sed_s_ic
                self.sed_ic = sed_ic
                self.sed_s_ic = sed_s_ic
                
            else:
                raise ValueError('I don\'t know this model. Try `Syn` or `IC`.' )
        self.sed = sed_tot
        self.sed_s = sed_s_
        self.e_ph = E
        # self.sed_spl = interp1d(E, sed_tot)
        # print(self.sed)
        if to_return:    
            return E, sed_tot, sed_s_
    
    
    def flux(self, e1, e2):
        # try:
        #     sed_spl_ = self.sed_spl
        # except:
        #     raise ValueError('The specrum has not been set yet.')
        try:
            _mask = np.logical_and(self.e_ph >= e1/1.15, self.e_ph < e2*1.15)
            _spl_sed_in_this_band = interp1d(self.e_ph[_mask], self.sed[_mask])
        except:
            raise ValueError('Cannot create a spline for flux calculation.')
        _E = loggrid(e1, e2, n_dec = 59)
        sed_here = _spl_sed_in_this_band(_E)
        # return trapz_loglog(sed_here / _E, _E)
        return trapezoid(sed_here / _E, _E)

            
    
    def fluxes(self,  bands):
        fluxes_ = []
        for band in bands:
            e1, e2 = band
            fluxes_.append(SpectrumIBS.flux(self, e1, e2))
        return np.array(fluxes_)
    
    def index(self, e1, e2):
        # try:
        #     sed_spl_ = self.sed_spl
        # except:
        #     raise ValueError('The specrum has not been set yet.')
            
        try:
            _mask = np.logical_and(self.e_ph >= e1/1.2, self.e_ph <= e2*1.2)
            _spl_sed_in_this_band = interp1d(self.e_ph[_mask], self.sed[_mask])
        except:
            raise ValueError('Cannot create a spline for index calculation.')
        _E = loggrid(e1, e2, n_dec = 51)
        sed_here = _spl_sed_in_this_band(_E)
        try:
            popt, pcov = curve_fit(f = pl, xdata = _E,
                                   ydata = sed_here,
                                   p0=(0.5, 
                                       sed_here[0] * _E[0]**0.5
                                       ))
            return popt[0] + 2. 
        except:
            return np.nan
        
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
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))    

        if self.sed is None:
            raise ValueError("You should call `calculate()` first to set SED")
        
        
        emiss_s = trapezoid(self.sed_s, self.e_ph, axis=1)

        ax[0].plot(self.e_ph, self.sed, label=None, **kwargs)
        emiss_s_ = emiss_s
        emiss_s_[np.isinf(emiss_s)]=np.nan
        # print(np.nanmax(emiss_s_))
        ax[1].plot(self._ibs.s, emiss_s/np.nanmax(emiss_s_), **kwargs)



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
    winds = Winds(orbit=orb, sys_name = sys_name, alpha=-8/180*pi, incl=23*pi/180,
              f_d=150, f_p=0.1, delta=0.01, np_disk=3, rad_prof='pl')     
    
    from ibsen.ibs import IBS
    from ibsen.el_ev import ElectronsOnIBS
    # from scipy.integrate import trapezoid
    # fig, ax = plt.subplots(2, 1)
    t = -30 * DAY
    Nibs = 80
    ibs = IBS(beta=None,
              winds=winds,
              gamma_max=1.4,
              s_max=1.,
              s_max_g=4.,
              n=Nibs,
              one_horn=False,
              t_to_calculate_beta_eff=t) 
    ibs1 = ibs.rescale_to_position()
    # ibs1.peek(show_winds=True, to_label=False, showtime=(-100*DAY, 100*DAY),
    #          ibs_color='doppler')
    els = ElectronsOnIBS(Bp_apex=7.768, ibs=ibs, cooling='stat_ibs', eta_a = 1e20,
                     to_inject_e = 'ecpl', p_e=1.7) 
    els.calculate(to_return=False)
    print('el_ev calculated')
    
    spec = SpectrumIBS(els=els, mechanisms=['i', 's'], simple=False,
                       apex_only=False, lorentz_boost=True)
    E = np.concatenate((
        loggrid(2.9e2, 1.1e4, 61),
        ))
    spec.calculate_sed_on_ibs(E=E
                              )
    print('spec calculated')
    print(spec.flux(3e2, 1e4))
    print(spec.index(3e2, 1e4))
    
    # print(spec.flux(4e11, 1e13))
    # print(spec.index(4e11, 1e13))
    
    
    spec.peek()
