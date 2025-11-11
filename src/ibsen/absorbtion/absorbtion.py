import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from scipy.interpolate import  interp1d
from scipy.integrate import  tplquad, dblquad
import xarray as xr
from pathlib import Path
from ibsen.utils import n_from_v, rotated_vector, absv, vector_angle, mydot
from joblib import Parallel, delayed
import multiprocessing

c_light = 2.998e10
sigma_b = 5.67e-5
h_planck_red = 1.055e-27
k_b = 1.381e-16
Rsun = 7e10
Msun = 2e33
Ropt = 10 * Rsun
DAY = 86400.

m_e = 9.109e-28
h_planck = h_planck_red * 2 * pi
e_char = 4.803e-10
MC2E = m_e * c_light**2
sigma_t = 8/3 * pi * (e_char**2 / MC2E)**2



_here = Path(__file__).parent          
_tabdata = _here / "absorb_tab" 

### ----------------- For photoelectric absorbtion ------------------------ ###
file_phel = _tabdata / 'sgm_tbabs_new.txt'
da_phel = np.genfromtxt(file_phel, skip_header=1, delimiter = ', ')
_e, _sgm = da_phel[:, 0], 10**da_phel[:, 1]
per_ = np.argsort(_e)
_e, _sgm = _e[per_], _sgm[per_]
spl_phel = interp1d(np.log10(_e), np.log10(_sgm), 'linear')

### ----------------- For gg-absorbtion PSR B1259-63 ----------------------- ###
name_gg_psrb = _tabdata / "gg_abs_psrb.nc"
ds_gg_psrb = xr.open_dataset(name_gg_psrb)
ds_gg_psrb = ds_gg_psrb.rename({'__xarray_dataarray_variable__': 'data'})

def sigma_gg(e_star, e_g, mu):
    """
    Cross-section of anisotropic gamma gamma --> e+ e- conversion.
    Simple analytic expression from 1703.00680 (they cite 
    **(Jauch & Rohrlich 1976)**) that should work for any multidimentional
    e_star, e_g, mu

    Parameters
    ----------
    e_star : np.ndarray
        Star seed photon energy in units of m_e c^2.
    e_g : np.ndarray
        Target photon energy in units of m_e c^2.
    mu : np.ndarray
        The scattering angle in [rad] but I'm slightly confused in which frame.
        I hope, in the lab frame...

    Returns
    -------
    np.ndarray
        gamma-gamma cross-section [cm^2].

    """
    b =  (1 - 2 / e_g / e_star / (1 - mu))**0.5
    return 3/16 * (1 - b**2) * ( (3 - b**4) * np.log( (1 + b) / (1 - b) ) - 2 * b * (2 - b**2) ) * sigma_t



def abs_photoel(E, Nh): 
    """    
    Photoelectric absorbtion TBabs
    Parameters
    ----------
    E : np.ndarray or float
        Photon energy [eV].
    Nh : float
        Hydrogen column densiry as in XSPEC [10^22 g cm^-2].

    Returns
    -------
    Dimentionless multiplicative absorbtion coef = exp(-tau):  0 < coef < 1.

    """

    e_min, e_max = np.min(_e), np.max(_e)
    E_kev = E / 1e3 #/ 1.6e-9
    if isinstance(E_kev, np.ndarray):
        E_low = E_kev[E_kev <= e_min]
        E_good = E_kev[np.logical_and(E_kev < e_max, E_kev > e_min)]
        E_high = E_kev[E_kev > e_max]
        # sgm_sm = 10**spl_sgm(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
        sgm_sm = 10**spl_phel(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
        
    
        a_low = np.zeros(E_low.size)
        a_good = np.exp(-sgm_sm * Nh * 1e22)
        a_high = np.zeros(E_high.size) + 1
        absorb = np.concatenate((a_low, a_good, a_high))
    else:
        if E_kev < e_min:
            absorb = 0
        elif E_kev > e_max:
            absorb = 1
        else:
            sgm_sm = 10**spl_phel(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
            absorb = np.exp(-sgm_sm * Nh * 1e22)
    return absorb

def f_helper(mu0, d0):
    """
    Phase-dependent helper for a Suchch & van Soelen analytical gg-absorbtion
    """
    sqrt_ = np.sqrt(1-mu0**2)
    return 1/d0 * (
        1/sqrt_ * (pi/2 - np.arctan(mu0/sqrt_) ) - 1
        )

def phi_helper(eg, R_star, T_star):
    """
    Phase-independent helper for a Suchch & van Soelen analytical gg-absorbtion
    """
    overall_coef = 64*pi / 3 / (h_planck * c_light)**3 * sigma_t * R_star**2
    exp_ = np.exp(-2 * MC2E / eg / k_b / T_star)
    return (  overall_coef * 
              (MC2E / eg)**3 * 
              exp_ / (1 - exp_)
              )

def gg_analyt(eg, x, y, R_star, T_star, nu_los, incl_los):
    """
    Gamma-gamma absorbtion coefficient (e^-tau) on the seed photon field  of 
    the star.

    Parameters
    ----------
    eg : np.ndarray (Ne, )
        The photon energy E_gamma / (m_e c^2).
    x : float
        X-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    y : float
        Y-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    R_star : float
        Optical star radius [cm].
    T_star : float
        Optical star effective temperature.
    nu_los : float
        The angle in the pulsar plane between the direction from
        optical star  to periastron and
        a projection of the LoS onto the orbit [rad]. Connected to the longtitude of 
        periastron w as: w = 3pi/2 - nu_los. So for PSRB w=138 deg and
        nu_los = 132 deg = 2.30 rad.
    incl_los : float
        The orbit inclination: the angle between the line of sight and the 
        angular velosity of the pulsar.

    Returns
    -------
    np.ndarray (Ne, )
        Coeffifient e^tau <1 for the gamma-gamma absorbtion.
        
    Notes:
        There is a pseuso-vectorization for the x- and y-coordinates. It works 
        only for a scalar energy `eg`(e.g., eg=1e6), then you can pass x and y
        coordinates as np.arrays or lists ot tuples of the same length (Nx,).
        In this case, the return is np.ndarray (Nx,).
        Currently you cannot pass both `eg` and `x`/`y` as vectors. 

    """
    if isinstance(x, np.ndarray) or isinstance(x, list) or (isinstance(x, tuple)):
        x, y = np.asarray(x), np.asarray(y)
        r_init=[]
        mu_init=[]
        for x_, y_ in zip(x, y):
            vec_init = np.array([x_, y_, 0])
            r_init.append(absv(vec_init))
            n_init = n_from_v(vec_init)
            n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
            mu_init.append( mydot(n_init, n_los) )
        r_init, mu_init = [np.array(ar) for ar in (r_init, mu_init)]
    else:
        vec_init = np.array([x, y, 0])
        r_init = absv(vec_init)
        n_init = n_from_v(vec_init)
        n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
        mu_init = mydot(n_init, n_los) 
    return np.exp(-2 * pi * 
                  phi_helper(eg, R_star, T_star) *
                  f_helper(mu_init, r_init)
                  )
    

def tau_gg_iso_2d(eg, x, y, R_star, T_star, incl_los, nu_los, fast=False):
    """
    Optical depth due to gamma-gamma pair production for a photon of energy
    eg (in units of electron rest-energy) emitted from the position (x, y)
    in the pulsar orbit plane. X-axis is directed from the optical star to
    the periastron of the pulsar. Z-axis is aligned with the direction of pulsar 
    orbital velosity; Y-axis is chosen so that [e_X x e_Y] = e_Z.
    

    Parameters
    ----------
    eg : float
        Photon energy in units of electron rest energy: E_g / (m_e c^2).
    x : float
        x coordinate of the emission.
    y : float
        y coordinte of emission
    R_star : float
        The radius of a star.
    T_star : float
        The effective temperature of a star.
    incl_los : float
        Inclination of the LoS to the normal to the pulsar plane.
    nu_los : float
        The angle between the direction of X-axis and the projection of the
        LoS onto the orbital plane.
        

    Returns
    -------
    tau_gg : float
        Optical depth at energy eg.

    """
    vec_init = np.array([x, y, 0])
    r_init = absv(vec_init)
    n_init = n_from_v(vec_init)
    n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
    mu_init = mydot(n_init, n_los)
    dist_to_s = lambda l_: np.sqrt(r_init**2 + l_**2 + 2 * r_init * l_ * mu_init) 
    exp_ = lambda e_: np.exp(-MC2E * e_ / k_b / T_star)
    n_ph_here_reduced = lambda e_: (
         e_**2 *
        exp_(e_) / (1 - exp_(e_)) 
        )
    mu_imp = lambda l_: np.sqrt(1.0 - r_init**2 / dist_to_s(l_)**2 *
                                (1 - mu_init**2)
                                )
                                    
    sigma_here = lambda e_, l_: (
        sigma_gg(e_star = e_ , e_g = eg, mu = mu_imp(l_))
                                      )
    
    overall_coef = 2*pi * (MC2E / h_planck / c_light)**3
    under_int = lambda e_, l_: ( n_ph_here_reduced(e_) *
                                     sigma_here(e_, l_) *
                                     (1 - mu_imp(l_)) * 
                                     R_star**2 / dist_to_s(l_)**2
                                     )

    low_inner = lambda l_: 2 / eg / (1 - mu_imp(l_)) #* (1 + 1e-6)
    # hi_inner = lambda l_: low_inner(l_) * 1e3
    res = 0
    ### improving integration by refining the grid over e and l
    if not fast:
        for dist_edges in ((0., 1.), (1., 10.), (10., 100.)):
            for e_edges in ((1., 3.), (3., 10.), (10., 1000.)):
                d_l, d_u = dist_edges
                e_l, e_u = e_edges
                low_l = lambda l_: low_inner(l_) * e_l
                up_l = lambda l_: low_inner(l_) * e_u
                
                res += dblquad( under_int, d_l * r_init, d_u * r_init,
                 low_l, up_l, epsrel = 1e-3)[0] * overall_coef
    if fast: ### accuracy of about 20%
        res = dblquad( under_int, 0, 100 * r_init,
         low_inner, lambda l_: low_inner(l_)*1e3, epsrel = 1e-3)[0] * overall_coef
    return res

def tabulate_absgg(orb, nrho, nphi, ne, Topt, Ropt, rhomin=0.1, rhomax=5.5, emin=1e9,
                   emax=1e14, to_save=True, namefile='gg_abs', to_return=False,
                   n_cores=None, fast=False):
    """
    Precompute and tabulate gamma–gamma absorption optical depths on a (log10 rho, phi, log10 E) grid,
    and optionally save the 3D table to a NetCDF file via xarray.
    
    The grid spans:
      - radial scaling : rho in [rhomin, rhomax] (log-spaced),
      - orbital angle  : phi in [0, 2π) (lin-spaced, last bin kept < 2π),
      - photon energy  : E in [emin, emax] eV (log-spaced).
    
    For each (rho, phi) the function computes the true orbital separation and
    evaluates tau(E) by parallelizing over the energy axis.
    
    Parameters
    ----------
    orb : Orbit
    nrho : int
        Number of grid points in rho (logarithmic).
    nphi : int
        Number of grid points in phi (linear).
    ne : int
        Number of grid points in energy E (logarithmic).
    Topt : float
        Stellar effective temperature, [K].
    Ropt : float
        Stellar radius, in cm.
    rhomin : float, optional
        Minimum rho (dimensionless) for the grid. Default is 0.1.
    rhomax : float, optional
        Maximum rho (dimensionless) for the grid. Default is 5.5.
    emin : float, optional
        Minimum photon energy in eV for the grid. Default is 1e9.
    emax : float, optional
        Maximum photon energy [eV] for the grid. Default is 1e14.
    to_save : bool, optional
        If True, write the result to `<namefile>.nc` (NetCDF) under the `_tabdata` directory.
        Default is True.
    namefile : str, optional
        Basename of the output file (without extension). Default is 'gg_abs'.
    to_return : bool, optional
        If True, also return the raw arrays `(res, xx, yy, es)`. Default is False.
    n_cores : 'all', or int, or None, optional
        Number of CPU cores to use for energy-axis parallelization. If None, the
        computation is not parallized. If `all`, uses all available cores. 
        Internally capped at 20. Default is None.
    fast : bool, optional
        Forwarded to `tau_gg_iso_2d` to select a faster/approximate path, if supported.
        Default is False.
    
    Returns
    -------
    res : np.ndarray, shape (nrho, nphi, ne)
        Optical depth `tau` on the grid (dimensionless).
        Returned only if `to_return=True`.
    xx : np.ndarray, shape (nrho, nphi)
        Cartesian x positions corresponding to each (rho, phi), in cm.
        Returned only if `to_return=True`.
    yy : np.ndarray, shape (nrho, nphi)
        Cartesian y positions corresponding to each (rho, phi), in cm.
        Returned only if `to_return=True`.
    es : np.ndarray, shape (ne,)
        Energy grid in eV. Returned only if `to_return=True`.
    
    Other Parameters
    ----------------
    _tabdata : pathlib.Path
        Module-level path where the table is saved (used when `to_save=True`).
    
    Notes
    -----
    - The phi grid is `np.linspace(0, 2π * (1 - 1e-4), nphi, endpoint=True)`, so the last bin
      stays just below `2π` to avoid duplicating the `phi=0` line.
    - Energy is passed to `tau_gg_iso_2d` as `eg = E / (m_e c^2)`, using `5.11e5 eV` for `m_e c^2`.
    - The output file contains an `xarray.DataArray` with coordinates:
      `('logrho', log10(rhos))`, `('phi', phis)`, `('loge', log10(es))`.
    - Parallelization is along the energy axis for each (rho, phi) pair.

    """
    _eps = 1e-4
    rhos = np.geomspace(rhomin, rhomax, nrho)
    phis = np.linspace(0, 2*pi * (1.-_eps), nphi, endpoint=True) # !!!
    r_reals = orb.r(t=orb.t_from_true_an(nu=phis)) # true orb separations at nu_true=phis
    es = np.geomspace(emin, emax, ne)
    res = np.zeros((rhos.size, phis.size, es.size))
    xx = np.zeros((rhos.size, phis.size))
    yy = np.zeros((rhos.size, phis.size))
    
    for ir, rho in enumerate(rhos):
        for iphi, phi in enumerate(phis):
            x, y = r_reals[iphi] * cos(phi) * rho, r_reals[iphi] * sin(phi) * rho
            xx[ir, iphi] = x
            yy[ir, iphi] = y
            
            def to_parall(ie):
                return tau_gg_iso_2d(eg = es[ie]/5.11e5, x=x, y=y, R_star=Ropt,
                            T_star=Topt, incl_los=orb.incl_los,
                            nu_los=orb.nu_los, fast=fast)
            if n_cores is None:
                res[ir, iphi, :] = np.array([
                    to_parall(i_e) for i_e in range(es.size)
                    ])
            elif isinstance(n_cores, int) or n_cores == 'all':
                if isinstance(n_cores, int):
                    n_cores_use = n_cores
                if n_cores == 'all':
                    n_cores_use = multiprocessing.cpu_count()
                n_cores_use = np.min(n_cores, 20)    
                tau = Parallel(n_jobs = n_cores_use)(delayed(to_parall)(i_e) for i_e in range(es.size))
                tau = np.array(tau)
                res[ir, iphi, :] = tau
            else:
                raise ValueError("n_cores should be None, or int, or `all`.")
    if to_save:
        da = xr.DataArray(res, coords=[('logrho', np.log10(rhos)),
                                       ('phi', phis),
                                       ('loge', np.log10(es))
                                       ],)
        # Save to NetCDF (or HDF5 if preferred)
        path_here = _tabdata / str(namefile + '.nc')
        da.to_netcdf(path_here)
    if to_return:
        # rr, ff = np.meshgrid()
        return res, xx, yy, es

def gg_tab(E, x, y, orb, filename='psrb', what_return='abs'):
    """
    Interpolate gamma–gamma absorption from a precomputed (log10 rho, phi, log10 E) table
    at user-provided energy(ies) and sky position(s).
    
    This function loads an xarray table (either a built-in preset or a file),
    converts (x, y) to (rho, phi) using the orbital separation at the line-of-sight
    true anomaly, and interpolates tau(E, rho, phi). For large energy arrays,
    it optionally accelerates evaluation by interpolating on a coarse energy grid
    over the overlap with the table's energy range and then re-interpolating.
    
    Parameters
    ----------
    E : float or array_like
        Photon energy or energies in eV (scalar or 1-D array).
    x : float or array_like
        Cartesian x-coordinate(s) in cm (cgs). Scalar or 1-D array.
    y : float or array_like
        Cartesian y-coordinate(s) in cm (cgs). Scalar or 1-D array.
    orb : Orbit
    filename : str, optional
        Name of the table to load. If `'psrb'` (default), use the preset dataset `ds_gg_psrb`.
        Otherwise `filename` is resolved under `_tabdata` and opened as NetCDF,
        with the data variable renamed to `'data'`.
    what_return : {'abs', 'tau'}, optional
        If `'tau'`, return optical depth tau (dimensionless).
        If `'abs'` (default), return absorption factor `exp(-tau)`.
    
    Returns
    -------
    out : float or np.ndarray
        Interpolated result. Shape depends on inputs:
        - scalar `(x, y)` and scalar `E` → scalar,
        - scalar `(x, y)` and array `E` → shape `(len(E),)`,
        - array `(x, y)` and scalar `E` → shape `(len(x),)`,
        - array `(x, y)` and array `E` → shape `(len(x), len(E))`.
        If `what_return='tau'`, values are optical depths (dimensionless);
        if `'abs'`, values are `exp(-tau)`.
    
    Notes
    -----
    - Energies outside the table range `[1e9, 1e14]` eV are filled with zeros in tau
      (i.e., absorption factor 1), and interpolation is performed only on the overlap.
    - For large problems, a coarse energy grid (default 47 points) is used internally
      when `(len(E) >= 3) and (len(rho) >= 3) and (len(x)^2 * len(E) > 5000)`, followed by
      a 1-D interpolation back to the requested energies.
    - Any small negative values due to cubic interpolation are clipped to zero.
"""
    if filename == 'psrb':
        ds_gg = ds_gg_psrb
    if filename != 'psrb':
        name_gg = _tabdata / filename
        ds_gg = xr.open_dataset(name_gg)
        ds_gg = ds_gg.rename({'__xarray_dataarray_variable__': 'data'})

    # Make E guaranteed 1-D; remember if caller passed a scalar
    _E_in = E
    E = np.atleast_1d(np.asarray(E, dtype=float))
    _E_is_scalar = np.asarray(_E_in).ndim == 0

    if isinstance(x, float) and isinstance(y, float):
        # scalar (x, y); E may be scalar or array
        _r = (x**2 + y**2)**0.5
        _phi_pm = np.arctan2(y, x)
        _phi = _phi_pm if _phi_pm >= 0 else _phi_pm + 2*np.pi
        rs = orb.r(orb.t_from_true_an(_phi))
        _rho = _r / rs
        taus = ds_gg['data'].interp(
            loge=np.log10(E),
            logrho=np.log10(_rho),
            phi=_phi,
            method='linear'
        ).values
    else:
        # vector (x, y); E may be scalar or array
        _x, _y = np.asarray(x), np.asarray(y)
        if _x.size == 1 and _y.size != 1: _x = np.zeros(_y.size) + _x
        if _x.size != 1 and _y.size == 1: _y = np.zeros(_x.size) + _y
        _r = (_x**2 + _y**2)**0.5
        _phi_pm = np.arctan2(_y, _x)
        _phi = np.where(_phi_pm < 0, _phi_pm + 2*np.pi, _phi_pm)
        rs = orb.r(orb.t_from_true_an(_phi))
        _rho = _r / rs

        # Overlap-only energy interpolation; zeros outside [1e9, 1e14]
        E_MIN_TAB, E_MAX_TAB = 1e9, 1e14
        mask_overlap = (E >= E_MIN_TAB) & (E <= E_MAX_TAB)
        E_overlap = E[mask_overlap]

        taus = np.zeros((_x.size, E.size), dtype=float)

        condition_for_simple = (E.size >= 3) & (_rho.size >= 3) & (_x.size**2 * E.size > 5000)

        if np.any(E_overlap):
            if condition_for_simple:
                e_lo = float(np.min(E_overlap))
                e_hi = float(np.max(E_overlap))
                if e_lo == e_hi:
                    e_to_interp = np.array([e_lo], dtype=float)
                else:
                    e_to_interp = np.geomspace(e_lo, e_hi, 47)
                loge_query = np.log10(e_to_interp)
            else:
                e_to_interp = None
                loge_query = np.log10(E_overlap)

            res = ds_gg['data'].interp(
                logrho=np.log10(_rho),
                phi=_phi,
                loge=loge_query,
                method='cubic'
            ).values

            taus_q = np.diagonal(res).copy().T  # shape: (_x.size, len(loge_query))

            if condition_for_simple and (e_to_interp is not None) and (e_to_interp.size >= 1):
                kind = 'cubic' if e_to_interp.size >= 4 else 'linear'
                interp = interp1d(
                    np.log10(e_to_interp), taus_q, kind=kind, axis=1,
                    bounds_error=False, fill_value='extrapolate'
                )
                taus[:, mask_overlap] = interp(np.log10(E_overlap))
            else:
                taus[:, mask_overlap] = taus_q
        # else: no overlap -> taus stays zeros

    # Clean up NaNs / negatives
    taus = np.asarray(taus, dtype=float)
    taus[np.isnan(taus)] = 0.0
    taus[taus < 0] = 0.0  # cubic can give tiny negatives

    # Form output according to what_return
    if what_return == 'tau':
        out = taus
    elif what_return == 'abs':
        out = np.exp(-taus)
    else:
        raise ValueError('`what_return` should be {`abs`, `tau`}.')

    # --- Squeeze energy axis if E was scalar ---
    if _E_is_scalar:
        if isinstance(x, float) and isinstance(y, float):
            # scalar x,y + scalar E -> scalar
            return float(np.asarray(out).reshape(-1)[0])
        else:
            # array x/y + scalar E -> shape (x.size,)
            return np.asarray(out)[..., 0]

    return out
    
