import numpy as np
from numpy import pi, sin, cos
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import  dblquad
from ibsen.get_obs_data import known_names
import xarray as xr
from pathlib import Path
from ibsen.utils import n_from_v, rotated_vector, absv, mydot
from joblib import Parallel, delayed
import multiprocessing
from astropy import constants as const

C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)
HBAR = float(const.hbar.cgs.value)
H_PLANCK = 2 * np.pi * HBAR
K_BOLTZ = float(const.k_B.cgs.value)
M_E = float(const.m_e.cgs.value)
E_ELECTRON = 4.803204e-10
MC2E = M_E * C_LIGHT**2
MC2_EV = float((const.m_e * const.c**2).to("eV").value)
sigma_t = 8/3 * pi * (E_ELECTRON**2 / MC2E)**2

DAY = 86400.

_here = Path(__file__).parent          
_tabdata = _here / "absorp_tab" 

### ----------------- For photoelectric absorption ------------------------ ###
file_phel = _tabdata / 'sgm_tbabs_new.txt'
da_phel = np.genfromtxt(file_phel, skip_header=1, delimiter = ', ')
_e, _sgm = da_phel[:, 0], 10**da_phel[:, 1] # e in keV; sigma in Mbarn
per_ = np.argsort(_e)
_e, _sgm = _e[per_], _sgm[per_]
spl_phel = interp1d(np.log10(_e), np.log10(_sgm), 'linear')

file_phe_sol = _tabdata / 'tbabs_se3_solar.txt'
da_phe_sol = np.genfromtxt(file_phe_sol, skip_header=1, delimiter = ', ')
_e_sol, _sgm_sol = da_phe_sol[:, 0], da_phe_sol[:, 1] / da_phe_sol[:, 0]**3 / 1e6 # e in keV; sigma in Mbarn
per_sol = np.argsort(_e_sol)
_e_sol, _sgm_sol = _e_sol[per_sol], _sgm_sol[per_sol]
spl_phe_sol = interp1d(np.log10(_e_sol), np.log10(_sgm_sol), 'linear')

file_phe_wilm = _tabdata / 'tbabs_se3_wilm.txt'
da_phe_wilm = np.genfromtxt(file_phe_wilm, skip_header=1, delimiter = ', ')
_e_wilm, _sgm_wilm = da_phe_wilm[:, 0], da_phe_wilm[:, 1] / da_phe_wilm[:, 0]**3 / 1e6  # e in keV; sigma in Mbarn
per_wilm = np.argsort(_e_wilm)
_e_wilm, _sgm_wilm = _e_wilm[per_wilm], _sgm_wilm[per_wilm]
spl_phe_wilm = interp1d(np.log10(_e_wilm), np.log10(_sgm_wilm), 'linear')



### ----------------- For gg-absorption PSR B1259-63 ----------------------- ###
# name_gg_psrb = _tabdata / "gg_abs_psrb.nc"
# ds_gg_psrb = xr.open_dataset(name_gg_psrb)
# ds_gg_psrb = ds_gg_psrb.rename({'__xarray_dataarray_variable__': 'data'})

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



def abs_photoel(E, Nh, abund='wilm'): 
    """    
    Photoelectric absorption TBabs: https://arxiv.org/pdf/astro-ph/0008425
    Parameters
    ----------
    E : np.ndarray or float
        Photon energy [eV].
    Nh : float
        Hydrogen column densiry as in XSPEC [10^22 g cm^-2].
    abuns : str
        Which element abundance to use. Use 'wilm' for the `ISM` abundance from
        the cited paper from tab. B2; 'solar' for the solar abundance from 
        there.

    Returns
    -------
    Dimentionless multiplicative absorption coef = exp(-tau):  0 < coef < 1.

    """
    abund = abund.lower()
    if abund == 'old':
        e_use, logsgm_spl_use = _e, spl_phel
    elif abund in ('wilm', 'wilms', 'ism'):
        e_use, logsgm_spl_use = _e_wilm, spl_phe_wilm
    elif abund in ('sol', 'solar'):
        e_use, logsgm_spl_use = _e_sol, spl_phe_sol
        
    e_min, e_max = np.min(e_use), np.max(e_use)
    E_kev = np.asarray(E) / 1e3 
    absorp = np.where(E_kev < e_min, 0.0, 1.0)
    where_defined = (E_kev >= e_min) & (E_kev <= e_max)
    
    if np.any(where_defined):
        sgm_sm = 10**logsgm_spl_use(np.log10(E_kev[where_defined])) * 1e6 * 1e-24
        absorp[where_defined] = np.exp(-sgm_sm * Nh * 1e22)
    
    return absorp.item() if np.ndim(E_kev) == 0 else absorp

def f_helper(mu0, d0):
    """
    Phase-dependent helper for a Suchch & van Soelen analytical gg-absorption
    """
    sqrt_ = np.sqrt(1-mu0**2)
    return 1/d0 * (
        1/sqrt_ * (pi/2 - np.arctan(mu0/sqrt_) ) - 1
        )

def phi_helper(eg, R_s, T_s):
    """
    Phase-independent helper for a Suchch & van Soelen analytical gg-absorption
    """
    overall_coef = 64*pi / 3 / (H_PLANCK * C_LIGHT)**3 * sigma_t * R_s**2
    exp_ = np.exp(-2 * MC2E / eg / K_BOLTZ / T_s)
    return (  overall_coef * 
              (MC2E / eg)**3 * 
              exp_ / (1 - exp_)
              )

def gg_analyt(eg, x, y, R_s, T_s, nu_los, incl_los):
    """
    Gamma-gamma absorption by Suchch & van Soelen analytical approximation
    on the seed photon field  of the star.

    Parameters
    ----------
    eg : np.ndarray (Ne, )
        The photon energy E_gamma / (m_e c^2).
    x : float
        X-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    y : float
        Y-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    R_s : float
        Optical star radius [cm].
    T_s : float
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
        Coeffifient e^-tau <1 for the gamma-gamma absorption.
        
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
                  phi_helper(eg, R_s, T_s) *
                  f_helper(mu_init, r_init)
                  )
    

def tau_gg_iso_2d(eg, x, y, R_s, T_s, incl_los, nu_los, fast=False, z=0.):
    """
    Optical depth due to gamma-gamma pair production for a photon of energy
    eg (in units of electron rest-energy) emitted from the position (x, y, z)
    relative to the star in the pulsar orbit plane 
    (z=0; however you can specify arbitrary z). 
    X-axis is directed from the optical star to
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
    R_s : float
        The radius of a star.
    T_s : float
        The effective temperature of a star.
    incl_los : float
        Inclination of the LoS to the normal to the pulsar plane.
    nu_los : float
        The angle between the direction of X-axis and the projection of the
        LoS onto the orbital plane.
    fast : bool, optional
        Whether to calculate fast or properly. Default False
    z : float, optional
        The z-coordinate of the emission point. Default 0 
        

    Returns
    -------
    tau_gg : float
        Optical depth at energy eg.

    """
    vec_init = np.array([x, y, z])
    r_init = absv(vec_init)
    n_init = n_from_v(vec_init)
    n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
    mu_init = mydot(n_init, n_los)
    return tau_gg_mu_d(
        eg=eg, r_init=r_init, mu_init=mu_init,
        R_s=R_s, T_s=T_s, fast=fast,
    )

def tau_gg_mu_d(eg, r_init, mu_init, R_s, T_s, fast=False):
    """
    Optical depth due to gamma-gamma pair production for a photon of energy
    eg (in units of electron rest-energy) emitted from the position at the
    distance r_init from the star, with an initial cos psi = mu_init.

    Parameters
    ----------
    eg : float
        Photon energy in units of electron rest energy: E_g / (m_e c^2).
    r_init : float
        Distance from the emission point to the star
    mu_init : float
        Cosine of the initial scattering angle (between the line
            from the star to the emission Zone and the direction of LoS)
    R_s : float
        The radius of a star.
    T_s : float
        The effective temperature of a star.
    fast : bool, optional
        Whether to calculate fast or properly. Default False

    Returns
    -------
    tau_gg : float
        Optical depth at energy eg.

    """

    r_init = float(r_init)
    mu_init = float(mu_init)
    if not np.isfinite(r_init) or r_init <= 0.:
        raise ValueError("`r_init` must be a positive finite distance.")
    if not np.isfinite(mu_init) or mu_init < -1. or mu_init > 1.:
        raise ValueError("`mu_init` must be finite and lie in [-1, 1].")
    if mu_init == 1.:
        return 0.0
    if mu_init == -1.:
        # The ray passes through the stellar centre. In the point-source
        # radiation-field approximation the optical depth is singular.
        return np.inf

    dist_to_s = lambda l_: np.sqrt(r_init**2 + l_**2 + 2. * r_init * l_ * mu_init)
    exp_ = lambda e_: np.exp(-MC2E * e_ / K_BOLTZ / T_s)
    n_ph_here_reduced = lambda e_: (
         e_**2 *
        exp_(e_) / (1 - exp_(e_)) 
        )
    # Signed cosine between the gamma ray and a stellar photon at distance l.
    # The former sqrt expression was its absolute value and therefore gave the
    # wrong collision angle before closest approach when mu_init < 0.
    mu_imp = lambda l_: np.clip(
        (l_ + r_init * mu_init) / dist_to_s(l_), -1.0, 1.0
    )
    # mu_imp = lambda l_: np.sqrt(1. - r_init**2 / dist_to_s(l_)**2 * (1. - mu_init**2) )
                                    
    sigma_here = lambda e_, l_: (
        sigma_gg(e_star = e_ , e_g = eg, mu = mu_imp(l_))
                                      )
    
    overall_coef = 2. * pi * (MC2E / H_PLANCK / C_LIGHT)**3
    under_int = lambda e_, l_: ( n_ph_here_reduced(e_) *
                                     sigma_here(e_, l_) *
                                     (1 - mu_imp(l_)) * 
                                     R_s**2 / dist_to_s(l_)**2
                                     )

    low_inner = lambda l_: 2 / eg / (1 - mu_imp(l_)) #* (1 + 1e-6)
    res = 0.0
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
        res = dblquad( under_int, 0., 100. * r_init,
         low_inner, lambda l_: low_inner(l_)*1e3, epsrel = 1e-3)[0] * overall_coef
    return res


def _gg_table_path(filename):
    """Resolve a gamma-gamma table basename or path."""
    path = Path(filename)
    if path.parent == Path('.') and path.name in known_names:
        path = Path(f'gg_abs_{path.name}_d_mu.nc')
    if path.suffix != '.nc':
        path = path.with_suffix('.nc')
    return path if path.is_absolute() else _tabdata / path


def tabulate_absgg_d_mu(nr, nmu, ne, T_s, R_s, rmin=1., rmax=1000.,
                        emin=1e9, emax=1e14, to_save=True,
                        filename='gg_abs_d_mu', to_return=False,
                        n_cores=None, fast=False):
    """Tabulate gamma-gamma optical depth on ``(log10(r), mu, log10(E))``.

    ``rmin`` and ``rmax`` are in stellar radii; the saved ``logr`` coordinate
    contains the physical distance in cm. ``mu`` is
    ``dot(r_init / |r_init|, n_los)``. The table therefore depends on the
    stellar properties but not on an orbit or line-of-sight orientation.

    When ``to_return`` is true, return ``(tau, r, mu, E)`` with shapes
    ``(nr, nmu, ne)``, ``(nr,)``, ``(nmu,)``, and ``(ne,)`` respectively.
    """
    if min(nr, nmu, ne) < 2:
        raise ValueError("`nr`, `nmu`, and `ne` must each be at least 2.")

    r_inits = np.geomspace(rmin * R_s, rmax * R_s, nr)
    mu_inits = np.linspace(-1., 1., nmu)
    es = np.geomspace(emin, emax, ne)
    res = np.zeros((r_inits.size, mu_inits.size, es.size))

    if n_cores == 'all':
        n_cores_use = min(multiprocessing.cpu_count(), 20)
    elif isinstance(n_cores, int) and not isinstance(n_cores, bool) and n_cores > 0:
        n_cores_use = min(n_cores, 20)
    elif n_cores is None:
        n_cores_use = None
    else:
        raise ValueError("`n_cores` must be None, a positive integer, or 'all'.")

    for ir, r_init in enumerate(r_inits):
        for imu, mu_init in enumerate(mu_inits):
            def to_parall(ie):
                return tau_gg_mu_d(eg=es[ie] / MC2_EV,
                                   r_init=r_init,
                                   mu_init=mu_init,
                                   R_s=R_s,
                                   T_s=T_s,
                                   fast=fast)
            if n_cores_use is None:
                res[ir, imu, :] = [to_parall(ie) for ie in range(es.size)]
            else:
                res[ir, imu, :] = Parallel(n_jobs=n_cores_use)(
                    delayed(to_parall)(ie) for ie in range(es.size)
                )

    if to_save:
        da = xr.DataArray(
            res.astype("float32"),
            coords=[
                ('logr', np.log10(r_inits)),
                ('mu', mu_inits),
                ('loge', np.log10(es)),
            ],
            name="data",
            attrs={
                'coordinate_system': 'r_init_mu_init',
                'r_unit': 'cm',
                'energy_unit': 'eV',
                'R_s_cm': float(R_s),
                'T_s_K': float(T_s),
                'mu_definition': 'dot(r_hat, n_los)',
                'singular_mu_minus_one': 1,
            },
        )
        path_here = _tabdata / str(filename + '.nc')
        da.to_netcdf(path_here, encoding={'data': {'dtype': 'float32'}})

    if to_return:
        return res, r_inits, mu_inits, es


_gg_d_mu_interpolator_cache = {}


def _load_gg_d_mu_interpolator(filename, method):
    """
    Util for loading tabulated gg-opacities and creating an interpolator. 
    """
    path = _gg_table_path(filename)
    cache_key = (str(path.resolve()), path.stat().st_mtime_ns, method)
    cached = _gg_d_mu_interpolator_cache.get(cache_key)
    if cached is not None:
        return cached

    with xr.open_dataset(path) as ds:
        if 'data' in ds.data_vars:
            da = ds['data']
        elif len(ds.data_vars) == 1:
            da = next(iter(ds.data_vars.values()))
        else:
            raise ValueError(f"Cannot identify the opacity data variable in {path}.")
        required = ('logr', 'mu', 'loge')
        if not all(dim in da.dims for dim in required):
            raise ValueError(
                f"{path} is not an (r_init, mu_init, E) table; expected dimensions {required}."
            )
        da = da.transpose(*required).load()
        axes = tuple(np.asarray(da.coords[name].values, dtype=float) for name in required)
        values = np.asarray(da.values, dtype=float)
        attrs = dict(da.attrs)

    # tau(mu=-1) is singular in the point-source approximation. It is handled
    # explicitly in gg_tab_d_mu; use the neighbouring finite slice solely to
    # keep the regular-grid interpolator well defined near that boundary.
    singular_mu_minus_one = bool(attrs.get('singular_mu_minus_one', False))
    if singular_mu_minus_one and axes[1][0] == -1.:
        values = values.copy()
        values[:, 0, :] = values[:, 1, :]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"The interpolation data in {path} contain non-finite values.")

    interpolator = RegularGridInterpolator(
        axes, values, method=method, bounds_error=False, fill_value=np.nan,
    )
    result = interpolator, axes, singular_mu_minus_one
    _gg_d_mu_interpolator_cache.clear()
    _gg_d_mu_interpolator_cache[cache_key] = result
    return result


def gg_tab_d_mu(E, vec_init, n_los, filename='gg_abs_d_mu',
                what_return='abs', method='linear'):
    """Interpolate a gamma-gamma table using arbitrary 3-D emission vectors.

    Parameters
    ----------
    E : float or array_like, shape (Ne,)
        Photon energies in eV. A scalar is treated as a one-element energy axis.
    vec_init : array_like, shape (..., 3)
        Emission positions relative to the star, in cm.
    n_los : array_like, shape (3,)
        Direction from the source toward the observer; it need not be normalized.
    filename : str or path-like, optional
        Table written by :func:`tabulate_absgg_d_mu`.
    what_return : {'abs', 'tau'}, optional
        Return transmission ``exp(-tau)`` or optical depth tau.
    method : str, optional
        Method accepted by :class:`scipy.interpolate.RegularGridInterpolator`.

    Returns
    -------
    ndarray, shape (..., Ne)
        The spatial shape of ``vec_init`` followed by an energy axis. Optical
        depth is zero outside the tabulated energy interval and for radii above
        the largest tabulated radius.
    """
    if what_return not in ('abs', 'tau'):
        raise ValueError("`what_return` must be either 'abs' or 'tau'.")

    energies = np.asarray(E, dtype=float)
    if energies.ndim > 1:
        raise ValueError("`E` must be a scalar or a one-dimensional array.")
    energies = np.atleast_1d(energies)

    vectors = np.asarray(vec_init, dtype=float)
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError("`vec_init` must have shape (3,) or (..., 3).")

    spatial_shape = vectors.shape[:-1]
    flat_vectors = vectors.reshape(-1, 3)
    radii = absv(flat_vectors)
    if np.any(radii == 0.):
        raise ValueError("Emission positions at the stellar centre are undefined.")

    los = np.asarray(n_los, dtype=float)
    los = n_from_v(los)
    mu = np.clip(mydot(n_from_v(flat_vectors), los), 
                 -1.0, 1.0)

    interpolator, axes, singular_mu_minus_one = _load_gg_d_mu_interpolator(
        filename, method
    )
    logr_axis, _, loge_axis = axes
    r_min, r_max = 10.**logr_axis[[0, -1]]
    e_min, e_max = 10.**loge_axis[[0, -1]]
    radius_at_min = np.isclose(radii, r_min, rtol=1e-6, atol=0.)
    if np.any((radii < r_min) & ~radius_at_min):
        raise ValueError(
            f"An emission radius is below the table minimum ({r_min:.6g} cm)."
        )

    tau = np.zeros((radii.size, energies.size), dtype=float)
    position_indices = np.flatnonzero(
        (radii <= r_max) | np.isclose(radii, r_max, rtol=1e-6, atol=0.)
    )
    energy_indices = np.flatnonzero(
        ((energies >= e_min) | np.isclose(energies, e_min, rtol=1e-6, atol=0.))
        & ((energies <= e_max) | np.isclose(energies, e_max, rtol=1e-6, atol=0.))
    )
    if position_indices.size and energy_indices.size:
        log_r = np.clip(np.log10(radii[position_indices]), logr_axis[0], logr_axis[-1])
        mu_query = mu[position_indices]
        log_e = np.clip(np.log10(energies[energy_indices]), loge_axis[0], loge_axis[-1])
        points = np.column_stack((
            np.repeat(log_r, log_e.size),
            np.repeat(mu_query, log_e.size),
            np.tile(log_e, log_r.size),
        ))
        values = interpolator(points).reshape(log_r.size, log_e.size)
        if np.any(np.isnan(values)):
            raise ValueError("Interpolation unexpectedly produced NaN inside the table bounds.")
        tau[np.ix_(position_indices, energy_indices)] = np.maximum(values, 0.)

        if singular_mu_minus_one:
            behind = np.isclose(mu_query, -1., rtol=0., atol=1e-14)
            if np.any(behind):
                tau[np.ix_(position_indices[behind], energy_indices)] = np.inf

    result = tau if what_return == 'tau' else np.exp(-tau)
    return result.reshape(spatial_shape + (energies.size,))

"""
Some legacy functions for tabulating and accessing data on the orbital 
plane.
"""

def tabulate_absgg(orb, nrho, nphi, ne, T_s, R_s, rhomin=0.1, rhomax=5.5, emin=1e9,
                   emax=1e14, to_save=True, filename='gg_abs', to_return=False,
                   n_cores=None, fast=False):
    """
    LEGACY FUNCTION. DO NOT USE IF YOU DON'T KNOW WHAT YOU'RE DOING.

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
    T_s : float
        Stellar effective temperature, [K].
    R_s : float
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
        If True, write the result to `<filename>.nc` (NetCDF) under the `_tabdata` directory.
        Default is True.
    filename : str, optional
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
    phis = np.linspace(0, 2*pi * (1.-_eps), nphi, endpoint=True) 
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
                return tau_gg_iso_2d(eg = es[ie]/5.11e5, x=x, y=y, R_s=R_s,
                            T_s=T_s, incl_los=orb.incl_los,
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
                n_cores_use = min(n_cores_use, 20)    
                tau = Parallel(n_jobs = n_cores_use)(delayed(to_parall)(i_e) for i_e in range(es.size))
                tau = np.array(tau)
                res[ir, iphi, :] = tau
            else:
                raise ValueError("n_cores should be None, or int, or `all`.")
    if to_save:
        da = xr.DataArray(
            res.astype("float32"),
            coords=[('logrho', np.log10(rhos).astype("float32")),
                    ('phi', phis.astype("float32")),
                    ('loge', np.log10(es).astype("float32"))],
            name="data",
                        )
        encoding = {
            da.name: {
                "dtype": "float32",
            }
        }
        # Save to NetCDF (or HDF5 if preferred)
        path_here = _tabdata / str(filename + '.nc')
        da.to_netcdf(path_here, encoding=encoding)
    if to_return:
        # rr, ff = np.meshgrid()
        return res, xx, yy, es

def gg_tab(E, x, y, orb, filename='psrb', what_return='abs'):
    """
    LEGACY FUNCTION. DO NOT USE IF YOU DON'T KNOW WHAT YOU'RE DOING.
    
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
        Name of the table to load. If recognized as the system name,
        use the preset dataset `ds_gg_<filename>`.
        Otherwise `filename` is resolved under `_tabdata` and opened as NetCDF.
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
    if filename in known_names:
        name_gg = _tabdata / (f'gg_abs_{filename}.nc')
        ds_gg = xr.open_dataset(name_gg)
    else:
        name_gg = _tabdata / (filename + '.nc')
        ds_gg = xr.open_dataset(name_gg)
    try:
        ds_gg = ds_gg.rename({'__xarray_dataarray_variable__': 'data'})
    except:
        _a=1
        # ds_gg = ds_gg.rename({'__xarray_dataarray_variable__': 'data'})

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
    
