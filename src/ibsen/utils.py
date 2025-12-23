# ibsen/utils.py
import numpy as np
# from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from numpy import pi, sin, cos, exp
import warnings
# import astropy.units as u
from astropy import constants as const


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

G = float(const.G.cgs.value)
DAY = 86400.

def lin(x, k, b):
    return k * x + b

def linear_slope(x, y):
    """
    Analytical linear fit of the data x, y by y=kx+b; returns slope k
    """
    _n = x.size
    return ((_n * np.sum(x*y) - np.sum(x) * np.sum(y)) 
            / (_n * np.sum(x**2) - np.sum(x)**2))

def ecpl(E, ind, ecut, norm):
    """
    Exponential cut-off power-law.
    norm * E^(-ind) * exp(-E/ecut).
    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    ecut : np.ndarray
        Exponential cut-off energy.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Exponential cut-off power-law.

    """
    return norm * E**(-ind) * exp(-E / ecut)

def pl(E, ind, norm):
    """
    Power law norm * E^(-ind).

    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Power-law.

    """
    return norm * E**(-ind)

def secpl(E, ind, ecut, beta_e, norm):
    """
    Super-exponential cut-off power-law.
    norm * E^(-ind) * exp(-(E/ecut)^beta_e).
    Parameters
    ----------
    E : np.ndarray
        Energy.
    ind : np.ndarray
        Power-law index.
    ecut : np.ndarray
        Exponential cut-off energy.
    beta_e : np.ndarray
        Super-exponential index.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Super-exponential cut-off power-law.

    """
    return norm * E**(-ind) * exp(- (E / ecut)**beta_e )

def bkpl(E, g1, g2, ebr, norm):
    """
    Broken power law: 
            norm * (e/ebr)**(-g1),      for e < ebr,
        = 
            norm * (e/ebr)**(-g2),      for e >= ebr/
            
    Parameters
    ----------
    E : np.ndarray
        Energy.
    g1 : np.ndarray
        Index for E < e_break.
    g2 : np.ndarray
        Index for E > e_break.
    ebr : np.ndarray
        Break energy.
    norm : np.ndarray
        Overall normalization.

    Returns
    -------
    np.ndarray
        Broken power-law.

    """
    ifsinglevalue = (isinstance(E, float) or isinstance(E, int))
    e = np.asarray(E)
    e1 = e[e<ebr]
    e2 = e[e>=ebr]
    res1 = norm * (e1/ebr)**(-g1)
    res2 = norm * (e2/ebr)**(-g2)
    res = np.concatenate((res1, res2))
    if ifsinglevalue:
        return res[0]
    else:
        return res

def unpack_params(
    param_names,
    *,
    orb_type=None,
    sys_params=None,
    known_types=None,
    get_defaults_func=None,
    overrides=None,
    allow_missing=False,
    missing=None,
    return_dict=False,
    **kwargs
):
    """
    Generic parameter unpacker with priority to explicit overrides, then defaults.

    Parameters
    ----------
    param_names : sequence of str
        Names to extract (order defines the returned tuple order).
    orb_type : str or None, optional
        If a known type, defaults are loaded via `get_defaults_func(orb_type)`
        (or global `get_parameters(orb_type)` if available).
        If None or unknown, fall back to `sys_params`.
    sys_params : dict or None, optional
        Fallback defaults dictionary when `orb_type` is None/unknown or when
        no type-specific defaults are available.
    known_types : iterable of str, optional
        Set/list of valid types for which `get_defaults_func` will be called.
    get_defaults_func : callable or None, optional
        Function taking `orb_type -> dict`. Defaults to `get_parameters`
        if available; else may be omitted if explicit values suffice.
    overrides : dict or None, optional
        Explicit overrides mapping; merged with `**kwargs` (kwargs win on conflicts).
    allow_missing : bool, optional
        If True, missing values are filled with `missing` instead of raising.
    missing : any, optional
        Value used when `allow_missing=True` and a parameter is not found.
    return_dict : bool, optional
        If True, return a dict; otherwise return a tuple in `param_names` order.
    **kwargs
        Additional explicit overrides (e.g., T=..., e=...).

    Returns
    -------
    tuple or dict
        Tuple in the same order as `param_names`, or a dict if `return_dict=True`.

    Raises
    ------
    ValueError
        If a required parameter is missing and `allow_missing` is False.
    TypeError
        If `orb_type` is not a string or None.
    """
    if orb_type is not None and not isinstance(orb_type, str):
        raise TypeError("`orb_type` must be str or None.")

    # 1) Collect explicit overrides: overrides dict + kwargs (kwargs win)
    explicit = {}
    if overrides:
        explicit.update(overrides)
    for k, v in kwargs.items():
        if k in param_names:
            explicit[k] = v

    # 2) Determine which parameters still need defaults
    needed = [name for name in param_names
              if (name not in explicit) or (explicit[name] is None)]

    # If everything is provided explicitly (and not None), we can skip defaults.
    if not needed:
        out = {name: explicit[name] for name in param_names}
        return out if return_dict else tuple(out[n] for n in param_names)

    # 3) We do need some defaults; resolve a defaults dict
    defaults = {}

    if orb_type is not None and orb_type in set(known_types):
        # Try type-specific defaults via provided function or global get_parameters
        fn = get_defaults_func if get_defaults_func is not None else globals().get('get_parameters', None)
        if fn is not None:
            defaults = dict(fn(orb_type) or {})
        elif sys_params is not None:
            # Fall back to sys_params if available
            defaults = dict(sys_params or {})
        else:
            # No defaults source at all. We may still allow missing.
            if not allow_missing:
                raise ValueError(
                    "No defaults source available for orb_type "
                    f"`{orb_type}` and `allow_missing` is False."
                )
            defaults = {}
    else:
        # Type unknown or None: just use sys_params (or empty dict)
        defaults = dict(sys_params or {})

    # 4) Build result with precedence: explicit (not None) -> defaults.get -> missing/error
    out = {}
    for name in param_names:
        if name in explicit and explicit[name] is not None:
            out[name] = explicit[name]
        elif name in defaults and defaults[name] is not None:
            out[name] = defaults[name]
        elif allow_missing:
            out[name] = missing
        else:
            raise ValueError(f"Missing required parameter `{name}`")

    return out if return_dict else tuple(out[n] for n in param_names)


def vectorize_func(func_simple):
    """
    Vectorises a function func_simple of *args, so that each can be either
    a single vector or a tuple of vectors. In the second case, the lengths
    of tuples should be the same.

    Parameters
    ----------
    func_simple : callable
        Function of some vector arguments.

    Raises
    ------
    ValueError
        If any argument is not a vector or a list of vectors.

    Returns
    -------
    callable
        Vectorized function.

    """
    def wrapper(*args):
        if all(isinstance(x, np.ndarray) for x in args):
            return func_simple(*args)
        for x in args:
            if isinstance(x, list):
                n = len(x)
                break
        else:
            raise ValueError("No list argument found!")
        result = []
        for i in range(n):
            new_args = [x[i] if isinstance(x, list) else x for x in args]
            result.append(func_simple(*new_args))
        return result
    return wrapper    

def unpack(query, dictat):
    """
    Unpacking values from a string of names.

    Parameters
    ----------
    query : str
        A string of names from the dictionary to unpack, e.g. 
        "a b c".
    dictat : dict
        A dictionary from which to get values.

    Returns
    -------
    list_ : list
        A list of values [dictat['a'], ...].

    """
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

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

def thresh_crit(x, xarr, yarr):
    """
    Given arrays (xarr, yarr), finds the value of y corresponding to a (scalar)
    x, treating (xarr, yarr) either as a multiple steps function OR, if 
    xarr[i]/yarr[i] are lists [x1, x2]/[y1, y2], the linear function between
    (x1, y1) and (x2, y2).

    Iterates through paired thresholds/intervals in ``xarr`` and values in ``yarr``
    to compute an output ``p`` for a given scalar ``x``. Each element of ``xarr``
    can be either:

    - a **scalar threshold**: if ``xarr[i] <= x``, set ``p = yarr[i]`` and continue;
      otherwise stop and return the current ``p``.
    - a **list/iterable interval** (e.g., ``[xmin, xmax]``): if ``x < min(xarr[i])``,
      stop and return the current ``p``; if ``x >= max(xarr[i])``, set
      ``p = max(yarr[i])`` and continue; otherwise (``min <= x < max``) linearly
      interpolate between ``min(yarr[i])`` and ``max(yarr[i])`` over the interval.

    The search proceeds from ``i = 0`` upward and may exit early when ``x`` falls
    before the current scalar/interval. The default value is ``p = 1`` if no
    entries are applicable.

    Parameters
    ----------
    x : float
        Query point at which to evaluate the piecewise criterion.
    xarr : list of (float or list-like)
        Sequence describing breakpoints. Each element is either a scalar threshold
        (treated as ``(-inf, threshold]`` w.r.t. updating ``p``) or a list-like
        interval (e.g., ``[xmin, xmax]``) over which interpolation is applied.
        **Assumed sorted in ascending order** of position along the x-axis and
        non-overlapping.
    yarr : list of (float or list-like)
        Values associated with each entry of ``xarr``. For scalar thresholds,
        ``yarr[i]`` should be a numeric scalar assigned to ``p`` when applicable.
        For intervals, ``yarr[i]`` should be a list-like of two numeric endpoints
        ``[pmin, pmax]`` (order does not matter; ``min``/``max`` are used) which
        define the interpolation range.

    Returns
    -------
    p : float
        The mapped value at ``x`` after processing thresholds/intervals in order.
        Defaults to ``1`` if no rule applies.

    Notes
    -----
    - The function updates ``p`` **in order** and may update it multiple times
      before returning; the last applicable rule wins.
    - For interval entries, linear interpolation is computed as::

          p = pmin + (pmax - pmin) * (x - xmin) / (xmax - xmin)

      where ``xmin = min(xarr[i])`` and ``xmax = max(xarr[i])``.
    - This implementation requires ``numpy`` as ``np`` in scope.

    Assumptions
    -----------
    - ``xarr`` entries are ordered from low to high along the axis and do not
      overlap in a way that would contradict early stopping.
    - Interval lengths are strictly positive (``xmax > xmin``).

    Raises
    ------
    ZeroDivisionError
        If an interval has zero length (``xmax == xmin``).
    TypeError, ValueError
        If elements of ``xarr``/``yarr`` are not numeric or shape-mismatched.

    Examples
    --------
    Scalar thresholds only:

    >>> xarr = [0.0, 1.0, 2.0]
    >>> yarr = [10.0, 20.0, 30.0]
    >>> thresh_crit(1.5, xarr, yarr)
    20.0

    Mixed thresholds and an interpolated interval:

    >>> xarr = [0.0, [1.0, 3.0], 4.0]
    >>> yarr = [5.0, [10.0, 30.0], 40.0]
    >>> thresh_crit(2.0, xarr, yarr)   # inside [1, 3] => interpolate from 10 to 30
    20.0
    >>> thresh_crit(3.5, xarr, yarr)   # beyond interval but before 4.0 => keeps last p
    30.0
    >>> thresh_crit(-1.0, xarr, yarr)  # before first threshold/interval
    1
    """
    p = 1
    for i in range(len(xarr)):
        if not isinstance(xarr[i], list):
            if xarr[i] <= x:
                p = yarr[i]
            else:
                break
        if isinstance(xarr[i], list):
            min_here, max_here = np.min(xarr[i]), np.max(xarr[i])
            if min_here <= x:
                if x >= max_here:
                    p = np.max(yarr[i])
                else:
                    pmin, pmax = np.min(yarr[i]), np.max(yarr[i])
                    p = pmin + (pmax-pmin)/(max_here-min_here) * (x - min_here)
            else:
                break
    return p

def enhanche_jump(t, t1_disk, t2_disk, times_enh, param_to_enh): 
    """
    Map a time ``t`` to an enhanced parameter via placeholder-substituted thresholds,
    ensuring the resulting schedule is ordered.

    Replaces symbolic placeholders ``'t1'`` and ``'t2'`` in ``times_enh`` with the
    concrete times ``t1_disk`` and ``t2_disk``. After substitution, verifies that the
    sequence is **non-decreasing** along the time axis (by comparing each item's
    position: scalars by their value, intervals by their lower bound). If not
    ordered, raises a ``ValueError``. Finally, evaluates the mapping with
    :func:`thresh_crit` to obtain the enhanced value.

    Parameters
    ----------
    t : float
        The query time at which to evaluate the enhancement.
    t1_disk : float
        Value that replaces any ``'t1'`` placeholders in ``times_enh``.
    t2_disk : float
        Value that replaces any ``'t2'`` placeholders in ``times_enh``.
    times_enh : list of (float or list-like or {'t1','t2'})
        Sequence of thresholds/intervals accepted by :func:`thresh_crit`.
        Elements may be scalars, two-element intervals (e.g., ``[t_min, t_max]``),
        or placeholders ``'t1'``/``'t2'``. The list **must be ordered in ascending
        time** after substitution; intervals are compared by their lower bound.
    param_to_enh : list of (float or list-like)
        Values corresponding to ``times_enh`` (same length). Scalars for scalar
        thresholds; two-element sequences (``[pmin, pmax]`` in any order) for
        linearly interpolated intervals.

    Returns
    -------
    current_H_enh : float
        The enhanced/mapped parameter at time ``t`` after substitution and
        evaluation via :func:`thresh_crit`.

    Raises
    ------
    ValueError
        If the substituted schedule (``times_enh`` with placeholders replaced) is
        not non-decreasing by position, or if an interval is empty.
    TypeError
        If an entry cannot be interpreted as a scalar or a two-element interval.

    Notes
    -----
    - This is a thin wrapper around :func:`thresh_crit` with an ordering check.
    - Interval interpolation follows::

          p = pmin + (pmax - pmin) * (t - tmin) / (tmax - tmin)

      with ``tmin = min(interval)`` and ``tmax = max(interval)``.
    - Equal positions are allowed (non-decreasing order). If multiple entries share
      the same position, later entries may overwrite earlier ones in
      :func:`thresh_crit`.

    Examples
    --------
    With placeholders and an interval:

    >>> t = 5.0
    >>> t1_disk, t2_disk = 3.0, 10.0
    >>> times_enh = ['t1', [4.0, 8.0], 't2']
    >>> param_to_enh = [2.0, [2.0, 5.0], 7.0]
    >>> enhanche_jump(t, t1_disk, t2_disk, times_enh, param_to_enh)
    2.75

    Unordered schedule (will raise):

    >>> enhanche_jump(5.0, 3.0, 10.0, [6.0, 2.0, 't2'], [1.0, 2.0, 3.0])
    Traceback (most recent call last):
        ...
    ValueError: times_enh (after substitution) must be non-decreasing by position ...
    """
    # print('unils ', times_enh)
    t_enh_ = [t1_disk if t_ == 't1' else t_ for t_ in times_enh]
    t_enh_final = [t2_disk if t_ == 't2' else t_ for t_ in t_enh_]
    # print(t_enh_final)
    # Determine each item's "position" for ordering:
    #  - scalar -> value
    #  - interval -> lower bound
    def _position(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                raise ValueError("Empty interval in times_enh after substitution.")
            # print(item)
            lo = float(np.min(item))
            # hi = float(np.max(item))
            # zero-length is allowed for ordering, but thresh_crit would divide by zero on interpolation
            return lo
        try:
            return float(item)
        except Exception as exc:
            raise TypeError(f"Cannot interpret entry {item!r} as scalar or interval.") from exc

    positions = [_position(v) for v in t_enh_final]

    # Check non-decreasing order
    if any(positions[i] < positions[i - 1] for i in range(1, len(positions))):
        raise ValueError(
            "times_enh (after substitution) must be non-decreasing by position. "
            f"Computed positions: {positions}"
        )
    current_H_enh = thresh_crit(t, t_enh_final, param_to_enh)
    return current_H_enh

def mydot_novec(a, b):
    """
    Scalar product of vectors a, b.
    """
    xa, ya, za = a
    xb, yb, zb = b
    return xa * xb +  ya * yb + za * zb

mydot = vectorize_func(mydot_novec)

def mycross_novec(a, b):
    """
    Vector product of vectors a, b.
    """
    ax, ay, az = a
    bx, by, bz = b
    return np.array([ay * bz - az * by,
                     az * bx - ax * bz,
                     ax * by - ay * bx])

mycross = vectorize_func(mycross_novec)

def absv_novec(vec):
    """
    Absolute value of a vector vec.
    """
    return (mydot(vec, vec))**0.5

absv = vectorize_func(absv_novec)


def n_from_v_novec(vec):
    """
    Normalized vector vec.
    """
    return vec / absv(vec)

n_from_v = vectorize_func(n_from_v_novec)


def g_from_beta(beta_vel):
    """
    Lorentz factor Gamma from the dimentionless velosity beta_vel.
    """
    return 1. / np.sqrt(1. - beta_vel**2)

def beta_from_g(g_vel):
    """
    Dimentionless velosity beta_vel from Lorentz factor Gamma.
    """
    
    if isinstance(g_vel, np.ndarray):
        res = np.zeros_like(g_vel)
        cond = (g_vel > 1.0+1e-7) 
        res[cond] = ((g_vel[cond]-1.0) * (g_vel[cond]+1.0))**0.5 / g_vel[cond]
    else:
        if g_vel > 1.0+1e-7:
            res =  ((g_vel-1.0) * (g_vel+1.0))**0.5 / g_vel
        else:
            res = 0.0
    return res 

def lor_trans_angle(angle, gamma):
    """
    Lorentz transformed angle (between the direction of the frame
                with lorentz-factor gamma and the direction of interest)

    Parameters
    ----------
    angle : np.ndarray
        angle in radians between some direction and moving frame. 
        Should be non-negative value, between 0 and pi.
    gamma : np.ndarray
        lorentz-factor of the moving frame.

    Returns
    -------
    np.ndarray or float
        Lorentz-transformed angle in radians.

    """
    angle = np.asarray(angle)
    gamma = np.asarray(gamma)
    if isinstance(angle, float):
        if angle < 0 or angle > np.pi:
            raise ValueError("angle should be between 0 and pi")
    if isinstance(angle, np.ndarray):
        if not np.all( (angle >= 0.0) & (angle <= np.pi) ):
            raise ValueError("All angles should be between 0 and pi")
    _betas = beta_from_g(gamma)
    _mu = np.cos(angle)
    _mu_prime = (_mu - _betas) / (1.0 - _mu * _betas)
    _mu_prime = np.clip(_mu_prime, -1.0, 1.0)
    return np.arccos(_mu_prime)


def lor_trans_b_iso(B_iso, gamma):
    """
    Lorentz-transforms a value of a module of an isotropic magnetic 
    field B_iso into a coordinate frame co-moving with a lorentz-factor gamma.
    

    Parameters
    ----------
    B_iso : float | np.ndarray of size gamma
        Absolute value of an isotropic magnetic field strength in a lab frame.
    gamma : float | np.ndarray of size B_iso
        A lorentz factor of a coordinate frame to transform to.

    Returns
    -------
    float | np.ndarray of size gamma/B_iso
        Absolute value of a non-isotropic magnetic field strength in a 
        co-moving frame.

    """
    # bx, by, bz = B_iso / 3**0.5, B_iso / 3**0.5, B_iso / 3**0.5
    # bx_comov = bx
    # by_comov, bz_comov = by * gamma, bz * gamma
    # return (bx_comov**2 + by_comov**2 + bz_comov**2)**0.5
    return B_iso * np.sqrt( (1.0 + 2.0 * gamma**2) / 3. )

def lor_trans_ug_iso(ug_iso, gamma): # Relativistic jets... eq. (2.57)
    """
    Lorentz-transforms a value of a module of an isotropic photon 
    field ug_iso into a coordinate frame co-moving with a lorentz-factor gamma.
    Uses eq. (2.57) from Relativistic Jets from Active Galactic Nuclei Edited
    by Markus Böttcher, Daniel E. Harris, and Henric Krawczynski.

    Parameters
    ----------
    ug_iso : float | np.ndarray of size gamma
        Absolute value of an isotropic photon field energy density in a lab
        frame.
    gamma : float | np.ndarray of size B_iso
        A lorentz factor of a coordinate frame to transform to.

    Returns
    -------
    float | np.ndarray of size gamma/ug_iso
        Absolute value of a non-isotropic photon field energy density in a 
        co-moving frame.

    """
    return ug_iso * gamma**2 * (3 + beta_from_g(gamma)) / 3.

def lor_trans_Teff_iso(Teff_iso, gamma): # assuming u ~ T^4
    """
    Transforms an effective temperature of a photon field into the co-moving 
    system. Fully analogous to lor_trans_ug_iso. Assumes T \propto u^{1/4}
    """
    return Teff_iso * (gamma**2 * (3 + beta_from_g(gamma)) / 3.)**0.25


def lor_trans_vec_novec(vec_n, vec_beta):
    """
    Lorentz tranformes a spatial unit vector vec_n (in lab frame) into \
    a system moving with vec_beta.
    

    Parameters
    ----------
    vec_n : np.array([n_x, n_y, n_z])
        A unit vector in lab frame to transform. Should be a np.array with \
        three coordinates
        x, y, and z.
    vec_beta : np.array([beta_x, beta_y, beta_z])
        A vector beta = v/c of the co-moving system in lab frame. Should be \
        a np.array with three coordinates x, y, and z.

    Returns
    -------
    np.array([n_prime_x, n_prime_y, n_prime_z])
        The unit vector in a co-moving system.

    """
    vec_norm = n_from_v(vec_n) # just making sure that it is, indeed, a unit vector
    beta_norm = n_from_v(vec_beta)
    _gamma = g_from_beta(absv(vec_beta))
    _nbeta = mydot(vec_norm, vec_beta) # scalar product (n * beta)
    vec_norm_parall = _nbeta / absv(vec_beta)  * beta_norm
    vec_norm_perp = vec_norm - vec_norm_parall
    vec_norm_parall_prime = (vec_norm_parall - vec_beta) / (1 - _nbeta)
    vec_norm_perp_prime = vec_norm_perp / _gamma  / (1 - _nbeta)
    return vec_norm_parall_prime + vec_norm_perp_prime

lor_trans_vec = vectorize_func(lor_trans_vec_novec)

def vector_angle_nonvec(n1, n2, vec_beta=np.zeros(3), lor_trans=False):
    """
    Calculates the angle between two UNIT vectors either in lab system, where \
        they are given, or in a system co-moving with velosity vec_beta.

    Parameters
    ----------
    n1 : my vector 
        1st vector in a lab system.
    n2 : my vector 
        2nd vector in a lab system.
    vec_beta : my vector, optional
        Vector of the beta (=v/c) of the other lorentz frame where you want to 
        calculate an angle. The default is np.zeros(3).
    lor_trans : bool, optional
        Whether to perform a lorentz boost (True) or to calculate in a lab
        frame (False). The default is False.

    Returns
    -------
    float
        Angle between vectors (always 0 <= angle <= pi).

    """
    ######### making sure they are indeed unit vectors. If not, well, you should
    ######### have read the documentation.
    n1_ = n_from_v(n1)
    n2_ = n_from_v(n2)
    if np.all(vec_beta == 0) or (not lor_trans):
        return np.arccos( mydot(n1_, n2_) )
    else:    
        n1_prime = lor_trans_vec(n1_, vec_beta)
        n2_prime = lor_trans_vec(n2_, vec_beta)
        return np.arccos( mydot(n1_prime, n2_prime) 
                         / absv(n1_prime) / absv(n2_prime) )
    
vector_angle = vectorize_func(vector_angle_nonvec)    

# def vector_angle(n1, n2, vec_beta=np.zeros(3), lor_trans=False):
#     # If all are vectors, just operate directly
#     if isinstance(n1, np.ndarray) and isinstance(n2, np.ndarray) and isinstance(vec_beta, np.ndarray):
#         return vector_angle_nonvec(n1, n2, vec_beta, lor_trans)
    
#     # If any is a list, broadcast element-wise
#     # Get length of first list found
#     for x in [n1, n2, vec_beta]:
#         if is_list_of_vecs(x, list):
#             n = len(x)
#             break
#     else:
#         raise ValueError("Inputs are not all arrays or lists of arrays!")
    
#     # Build elementwise args
#     result = []
#     for i in range(n):
#         nn1 = n1[i] if isinstance(n1, list) else n1
#         nn2 = n2[i] if isinstance(n2, list) else n2
#         vecvecb = vec_beta[i] if isinstance(vec_beta, list) else vec_beta
#         result.append(vector_angle_nonvec(nn1, nn2, vecvecb, lor_trans))
#     return result

def lor_trans_e_spec_iso(E_lab, dN_dE_lab, gamma, E_comov=None, n_mu=51, 
                         mode=None):
    """
    Returns (E_comov, dN_dE_comov), the angle-averaged spectrum in the cloud frame.

    Steps:
      1. Build an interpolator for the lab spectrum (zero outside input range).
      2. Define a grid of cosines mu' in [-1,1].
      3. For each E' in E_comov, compute the Doppler-shifted lab energies
         E = Γ (E' + β p' c mu'), then sample the lab spectrum there,
         weight by the Jacobian J = 1/[Γ(1+β mu')], and integrate over μ'.
         Currently assumes that all particles are ultra-relativistic.
         
    Parameters
    ----------
    E_lab : np.ndarray
        1D array of lab-frame energies (must be sorted ascending).
    dN_dE_lab : np.ndarray
        1D array of dN/dE in lab frame, same shape as E_lab.
    gamma : float
        bulk Lorentz factor Γ of the cloud.
    E_comov : np.ndarray, optional
        optional 1D array of desired comoving energies; if None, will use a 
        grid spanning from min(E_lab) * Gamma * (1-beta) to 
        max(E_lab) * Gamma * (1+beta). The default is None.
    n_mu : int, optional
        number of μ' samples for angle-average (must be odd for symmetry). The
        real number is calculated as int(n_mu * gamma**2).
        The default is 51.

    Returns
    -------
    E_comov : np.ndarray
        1D array of comoving energies.
    dN_dE_comov : ndarray
        1D array of angle-averaged dN'/dE' in comoving frame.

    """
    beta_v = beta_from_g(gamma)
    n_mu = int(n_mu * gamma)

    if E_comov is None:
        Emi = E_lab.min()
        Ema = E_lab.max()
        Emi_co = Emi * gamma * (1.0 - beta_v)
        Ema_co = Ema * gamma * (1.0 + beta_v)
        needed_len = int(len(E_lab) * np.log10(Ema_co / Emi_co) / 
                         np.log10(Ema / Emi))
        E_comov = np.geomspace(Emi_co, Ema_co, needed_len)

    # set up lab-spectrum interpolator, zero outside
    lab_interp = interp1d(
        E_lab, dN_dE_lab,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    mode = 'g' if gamma > 1.2 else 'mu'

    if mode == 'mu':
        mu_prime = np.linspace(-1, 1, int(n_mu * gamma))
        dN_dE_comov = np.zeros_like(E_comov)
        Ep = E_comov[:,None]
        E_shift = gamma * Ep * (1 + beta_v * mu_prime[None,:])
        J = 1.0 / (gamma * (1.0 + beta_v * mu_prime))[None,:]
        F_lab_at_E = lab_interp(E_shift)
        integrand = J * F_lab_at_E
        dN_dE_comov = 2.0 * np.pi * trapezoid(integrand, mu_prime, axis=1) / 4.0 / np.pi 
    else:
        eta = np.geomspace(gamma*(1-beta_v), gamma*(1+beta_v), int(n_mu))
        dN_dE_comov = np.zeros_like(E_comov)
        Ep = E_comov[:,None]
        E_shift = Ep * eta[None,:]
        J = 1.0 / eta[None,:]
        F_lab_at_E = lab_interp(E_shift)
        integrand = J * F_lab_at_E
        dN_dE_comov = 0.5 * trapz_loglog(integrand, eta, axis=1) / beta_v / gamma
    return E_comov, dN_dE_comov


def loggrid(x1, x2, n_dec):
    """
    Creates a log-spaced grid from x1 to x2 with n_dec points per decade.

    Parameters
    ----------
    x1 : float
        lower bound for a grid.
    x2 : float
        upper bound for a grid.
    n_dec : float
        The number of points per decade. If the number of points turnes out to 
        be 1 (x2 is close to x1 and n_dec is small), an array of [x1, x2] is 
        returned.

    Raises
    ------
    ValueError
        x2 <= x1.

    Returns
    -------
    np.ndarray
        a log-spaced grid from x1 to x2 with n_dec points per decade.

    """
    if x2 <= x1:
        raise ValueError('x2 should be > x1 for loggrid.')
    n_points = max(int( np.log10(x2 / x1) * n_dec) + 1,
                   2)
    return np.geomspace(x1, x2, n_points)

def logrep(xdata, ydata):
    """
    Creates a logarithmic interpolator for the data (xdata, ydata).

    Parameters
    ----------
    xdata : np.ndarray
        1D array of x-coordinates.
    ydata : np.ndarray
        1D array of y-coordinates.

    Returns
    -------
    interp1d-reurned interpolator
        A logarithmic interpolator for the data (xdata, ydata). The xdata should
        be positive and sorted in ascending order. To evaluate the y-values at
        x-values, you should do 
        10**(
        logrep(xdata, ydata)(np.log10(x))
        )

    """
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    return interp1d(np.log10(xdata), np.log10(ydata))

def logev(x, logspl):
    """
    Evaluates a logarithmic interpolator logspl at x.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    logspl : interp1d-returned object.
        interp1d-returned loglog interpolator..

    Returns
    -------
    np.ndarray
        A 1D array of y-coordinates, evaluated at x. The x should be positive
        and sorted in ascending order.

    """
    return 10**( logspl( np.log10(x) ) )

def interplg(x, xdata, ydata, **kwargs):
    """
    Interpolates ydata at x, given xdata/ydata in logarithmic scale.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates to evaluate y at.
    xdata : np.ndarray
        1D array of x-data: x-coordinates at which ydata is given.
    ydata : np.ndarray
        1D array of y-data.

    Returns
    -------
    np.ndarray
        y interpolated at x, using ydata(xdata).

    """
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    spl_ = interp1d(np.log10(xdata), np.log10(ydata), **kwargs)
    return 10**( spl_( np.log10(x) ) )

def interplg_positive(x, xdata, ydata, *, kind="linear",
                        fill_value=0.0, bounds_error=False):
    """
    Log-log interpolation on the region where ydata>0, return fill_value elsewhere.

    If x is outside the positive-support range, returns fill_value (default 0).
    """
    x = np.asarray(x, dtype=float)
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    # keep only strictly positive y points (log domain)
    m = (xdata > 0) & (ydata > 0) & np.isfinite(xdata) & np.isfinite(ydata)
    if np.count_nonzero(m) < 2:
        # not enough points to interpolate
        return np.full_like(x, fill_value, dtype=float)

    xd = xdata[m]
    yd = ydata[m]
    asc = np.argsort(xd)
    xd, yd = xd[asc], yd[asc]

    lx = np.log10(xd)
    ly = np.log10(yd)

    spl = interp1d(lx, ly, kind=kind, bounds_error=bounds_error,
                   fill_value="extrapolate")  # we’ll mask extrapolation ourselves

    out = np.full_like(x, fill_value, dtype=float)

    # define support in x where log-log is valid (positive ydata region)
    x_min, x_max = xd.min(), xd.max()
    inside = (x > 0) & (x >= x_min) & (x <= x_max) & np.isfinite(x)

    out[inside] = 10.0 ** spl(np.log10(x[inside]))
    return out

def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    ---------------------------------------------------------------------------
    ----- Borrowed from Naima utils. I mean, we use Naima anyway, right? ------
    ---------------------------------------------------------------------------

    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret

def rotated_vector(alpha, incl):
    """
    Creates a vector np.array([x, y, z]), which is initially a unit
    vector in z-direction np.array([0, 0, 1]) rotated by `incl` around
    y-axis and then by `alpha` around z-axis. 

    Parameters
    ----------
    alpha : float
        The angle of the second turn around z-axis [rad].
    incl : float
        The angle of the first turn around y-axis [rad].

    Returns
    -------
    np.ndarray of length 3
        The rotated vector.

    """
    return np.array([  cos(alpha) * sin(incl),
                       sin(alpha) * sin(incl),
                       cos(incl)
                       ])

def rotate_z(vec, phi):
    
    """
    rotates vector vec=np.array([x, y, z]) around z-axis at the 
    angle phi in a positive direction
    """
    
    _x, _y, _z = vec
    c_, s_ = cos(phi), sin(phi)
    x_rotated_ = c_ * _x - s_ * _y
    y_rotated_ = s_ * _x + c_ * _y
    return np.array([x_rotated_, y_rotated_, _z])

def rotate_z_xy(x, y, phi):
    """
    Rotates a 2-D vector (x, y) around z-axis at the angle phi in a positive direction.

    Parameters
    ----------
    x : float
        vector x-coordinate.
    y : float
        vector y-coordinate.
    phi : float
        angle [rad] to rotate the vector around z-axis.

    Returns
    -------
    x_rotated_ : float
        x-coordinate of the rotated vector.
        
    y_rotated_ : float
        y-coordinate of the rotated vector.

    """
    c_, s_ = cos(phi), sin(phi)
    x_rotated_ = c_ * x - s_ * y
    y_rotated_ = s_ * x + c_ * y
    return x_rotated_, y_rotated_


def t_avg_func(func, t1, t2, n_t):
    """
    Averages function func(e, t) over a time period t = [t1, t2],
    currently giving an algebraic average.

    Parameters
    ----------
    func : Callable
        A function func = func(e, t)
    t1 : float
        min time for averaging.
    t2 : float
        max time for averaging.
    n_t_points : int
        A number of points to span on the t-array.

    Returns
    -------
    Function \tilde func(e).

    """
    # t_grid = np.linspace(t1, t2, n_t)

    def func_avg(e):
        # # ensure e is array for broadcasting
        # e_arr = np.atleast_1d(e)
        # # evaluate Edot on full (e, t) mesh: shape (len(e), len(t))
        # vals = func(e_arr[:, None], t_grid[None, :])
        # # integrate over t for each e
        # integral = np.trapz(vals, t_grid, axis=1)
        # # normalize by interval length
        # avg = integral / (t2 - t1)
        # # if user passed scalar, return scalar
        # return avg.item() if np.isscalar(e) else avg
        return 0.5 * (func(e, t1) + func(e, t2)) 

    return func_avg

def l2_norm(xarr, yarr):
    """
    An L2-norm of a function yarr(xarr) defined on the grid xarr.

    Parameters
    ----------
    xarr : np.ndarray
        1D array of x-coordinates. Should be sorted in ascending order.
    yarr : np.ndarray
        1D array of y-coordinates. Should be the same length as xarr.

    Returns
    -------
    float
        The L2-norm of the function yarr(xarr), defined as
        \sqrt{ \int y^2 dx }.

    """
    return ( trapezoid(yarr**2, xarr) )**0.5

def wrap_grid(x, frac=0.10, num_points=1000, single_num_points=15):
    """
    Creates a grid of points around the range of x, with a 10% margin
    on both sides. If the range is degenerate (xmin == xmax), it will
    create a grid points evenly spaced between xmin*(1 - frac) and xmax*(1 + frac).


    Parameters
    ----------
    x : float | np.ndarray
        1D array of x-coordinates. Should be sorted in ascending order.
    frac : float, optional
        A margin for the enveloping grid. The default is 0.10.
    num_points : int, optional
        The number of points for the enveloping grid. The default is 1000.
    single_num_points : int, optional
        A number of points for a grid if x is a number. The default is 15.

    Raises
    ------
    ValueError
        If x dimention is > 1.

    Returns
    -------
    np.ndarray
        A 1D array of points, evenly spaced between <xmin and
        >xmax taking sign into account. E.g., if x=np.array([1, ....2, ]),
        the grid will be created between 0.9 and 2.2, with 1000 points.
        If x=np.array([-1, ..., 2]), the grid will be created between
        -1.1 and 2.2, with 1000 points. If x=np.array([1, 1, 1]) or x=1,
        the grid will be created between 0.9 and 1.1, with 15 points.

    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    xmin, xmax = x.min(), x.max()

    # New endpoints per sign-aware 10% rule
    lo = xmin*(1 - frac) if xmin >= 0 else xmin*(1 + frac)
    hi = xmax*(1 + frac) if xmax >= 0 else xmax*(1 - frac)

    if xmax == xmin:
        # Degenerate grid: spread evenly between lo and hi
        t = np.linspace(0.0, 1.0, single_num_points)
        return lo + t*(hi - lo)

    return np.linspace(lo, hi, num_points)

def make_grid(x_grid, grid_kind, n_grid, per_decade=True):
    """
    Create x_grid if not provided. 

    Parameters
    ----------
    x_grid : np.array
        Initial observational grid.
    grid_kind : {'data', 'linear', 'log'}
        If 'data', returns sorted x_grid. 
        If 'linear', then equally spaced grid from min to max of x_grid
        with n_grid points.
        If 'log', then log-spaced grid from min to max of x_grid
        with n_grid points: either in total (per_decade=False) or
        per decade (per_decade=True)
        
    n_grid : int
        Number of nods.
    per_decade : bool, optional
        Whether n_grid is per decate or in total. The default is True.

    Returns
    -------
    np.array
        New x-grid.

    """
    xg = np.asarray(x_grid, dtype=float)
    # Make sure it's sorted ascending
    if np.any(np.diff(xg) <= 0):
        xg = np.sort(xg)

    xmin = np.min(xg)
    xmax = np.max(xg)

    if grid_kind == "data":
        return np.sort(xg.copy())
    elif grid_kind == "linear":
        return np.linspace(xmin, xmax, n_grid)
    elif grid_kind == "log":
        if xmin <= 0:
            raise ValueError("x contains non-positive values; log grid not possible.")
        if not per_decade:
            return np.geomspace(xmin, xmax, n_grid)
        else:
            return loggrid(xmin, xmax, n_grid)
    else:
        raise ValueError(f"Unknown grid_kind: {grid_kind}")
        
def fit_norm(ydata, dy_data, y0_normalized, return_err=False):
    """
    Analytically finds N, assuming y \propto N, (so N is a normalization), 
    given data: ydata, dydata,
    and one model y0 calculated with N0 (y0_normalized = y0/N0)

    Parameters
    ----------
    ydata : np.ndarray
        Data y
    dy_data : np.ndarray
        Data errors dy
    y0_normalized : np.ndarray
        Model calculated with N0, in the same points where ydata was measured,
        divided by this N0

    Returns 
    -------
    float p, np.ndarray y_opt:
        Optimal normalizaion and re-normalized y0

    """
    w = 1.0 / dy_data**2
    denom = np.sum(w * y0_normalized * y0_normalized)
    if denom <= 0:
        raise ValueError('denominator < 0 in fit_norm, cannot find normalization')
    N_best = np.sum(w * y0_normalized * ydata) / denom
    resid = ydata - N_best * y0_normalized
    chi2 = np.sum((resid / dy_data)**2)
    nu = ydata.size - 1
    s2 = chi2 / nu
    sigma_N = np.sqrt(s2 / denom)
    if not return_err:
        return N_best, N_best * y0_normalized
    if return_err:
        return N_best, N_best * y0_normalized, sigma_N
    

def index_simple(dnde, e):
    """
    Finds a powerlaw index: dnde \propto e^\ind. Fitted as a linear function
    analytically, assuming equal weights for all datapoints. 

    Parameters
    ----------
    dnde : np.array (Ne, )
        The y-array.
    e : np.array (Ne, )
        The x-array.

    Raises
    ------
    ValueError
        If the arrays are of different size.

    Returns
    -------
    float
        The index.

    """
    dnde, e = np.asarray(dnde), np.asarray(e)
    if dnde.size != e.size:
        raise ValueError('dnde and e should be the same size')
    _good = np.isfinite(dnde) & (dnde > 0)
    dnde_, e_ = dnde[_good], e[_good]
    if e_.size < 2:
        return np.nan
    elif e_.size == 2:
        return -np.log10(dnde_[-1] / dnde_[0]) / np.log10(e_[-1] / e_[0])
    else:
        return - linear_slope(np.log10(e_), np.log10(dnde_))



def index(dnde, e, e1, e2):
    """
    Electron index of a spectrum dnde. Fits a dnde in a given range 
    [e1, e2] with a powerlaw.

    Parameters
    ----------
    dnde : np.ndarray (Ne, )
        Array to fit
    e : np.ndarray (Ne, )
        Energies at which dnde is calculated
    e1 : float
        Lower energy [eV].
    e2 : float
        Upper energy [eV].

    Returns
    -------
    float | np.nan
        If the fit is successful, the index is returned. If the 
        curve_fit raised an error, np.nan is returned.

    """
        
    _mask = np.logical_and(e >= e1/1.2, e <= e2*1.2)
    _good = _mask & np.isfinite(dnde)
    ind_ = index_simple(dnde[_good],
                        e[_good])
    return ind_



def avg(arr, weights=None, power=None, axis=None):
    """
    Calculates weighted average defined as:
        
        avg^power = \Sum_i arr_i^power * weights_i
    
    Parameters
    ----------
    arr : np.ndarray
        Array to calculate the average of.
    weights : np.ndarray or None, optional
        Weights. If None, all weights=1. The default is None.
    power : np.ndarray, or float, or None, optional
        Power for averaging. If None, power=1. The default is None.
    axis : None or int or tuple of ints, optional
        See np.average. The default is None.

    Returns
    -------
    np.ndarray or float
        Averaged array.

    """
    if power is None:
        power = 1
        
    return (np.average(arr**power, weights=weights, axis=axis))**(1. / power)

def plot_with_gradient(fig, ax, xdata, ydata, some_param, colorbar=False, lw=2,
                       ls='-', colorbar_label='grad', minimum=None, maximum=None,
                       scatter=False, marker='o', s=20, alpha=1.0, cmap='coolwarm'):
    """
    Draw (xdata, ydata) colored by some_param.
    If scatter=True -> per-point scatter; else -> continuous line with gradient.
    """

    xdata = np.asarray(xdata).ravel()
    ydata = np.asarray(ydata).ravel()
    some_param = np.asarray(some_param).ravel()
    m = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(some_param)
    xdata, ydata, some_param = xdata[m], ydata[m], some_param[m]
    if xdata.size == 0:
        return

    # normalization range
    vmin_here = np.min(some_param) if minimum is None else minimum
    vmax_here = np.max(some_param) if maximum is None else maximum
    norm = Normalize(vmin=vmin_here, vmax=vmax_here)

    if scatter:
        sc = ax.scatter(xdata, ydata, c=some_param, cmap=cmap, norm=norm,
                        s=s, marker=marker, linewidths=0, alpha=alpha)
        mappable = sc
    else:
        if xdata.size < 2:
            sc = ax.scatter(xdata, ydata, c=some_param, cmap=cmap, norm=norm,
                            s=s, marker=marker, linewidths=0, alpha=alpha)
            mappable = sc
        else:
            points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(some_param[:-1])  # one color per segment
            lc.set_linewidth(lw)
            lc.set_linestyle(ls)
            lc.set_alpha(alpha)
            line = ax.add_collection(lc)
            mappable = line
    range_x = np.max(xdata) - np.min(xdata)
    range_y = np.max(ydata) - np.min(ydata)
    
    ax.set_xlim(np.min(xdata) - 0.1*range_x, np.max(xdata) + 0.1*range_x)
    ax.set_ylim(np.min(ydata) - 0.1*range_y, np.max(ydata) + 0.1*range_y)

    if colorbar:
        fig.colorbar(mappable, ax=ax, label=colorbar_label)
    return mappable