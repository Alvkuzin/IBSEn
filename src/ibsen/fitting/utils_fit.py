import numpy as np
from ibsen.utils import  interplg
import inspect
from scipy.interpolate import interp1d

DAY = 86400.
T21 = 59254.867359    
Ppsrb=1236.72453
c_light = 3e10

def residuals(xobs, yobs, dyobs, xtheor, ytheor, grid_type='linear',
              add_dy_multi=0.0, add_dy_addit=0.0, cut_at=None):
    if grid_type.lower() in ('lin', 'linear'):
        y_theor_in_obs = np.interp(xobs, xtheor, ytheor)
    elif grid_type.lower() == 'log':
        y_theor_in_obs = interplg(xobs, xtheor, ytheor)
    dyobs_eff = dyobs + yobs * add_dy_multi + add_dy_addit
    res = (yobs - y_theor_in_obs) / dyobs_eff
    if cut_at is not None:
        res = np.where(np.abs(res) > cut_at,
                       cut_at * np.sign(res),
                       res)
        # res[np.abs(res) > cut_at] = 
    return res

def residuals_multi(xobs, yobs, dyobs, xtheor, ytheor, grid_type='linear',
                    add_dy_multi=0.0, add_dy_addit=0.0, cut_at=None):
    
    """
    Uses the 'residuals' above, but xobs, yobs, dyobs, ytheor all can be arrays
    OR lists of arrays.
    """
    if isinstance(xobs, np.ndarray):
        resid_x = residuals(xobs, yobs, dyobs, xtheor, ytheor, grid_type,
                            add_dy_multi, add_dy_addit, cut_at)
        
    elif isinstance(xobs, list) and len(xobs) == 1:
        resid_x = residuals(xobs[0], yobs[0], dyobs[0], xtheor, ytheor[0], grid_type,
                            add_dy_multi, add_dy_addit, cut_at)
    else:
        resid_x = []
        for _t, _f, _df, _fmo in zip(xobs, yobs, dyobs, ytheor):
            _resid = residuals(_t, _f, _df, xtheor, _fmo, grid_type,
                               add_dy_multi, add_dy_addit, cut_at)
            resid_x.append(_resid)
        resid_x = np.concatenate(resid_x)        
    return resid_x


def chi2_multi(xobs, yobs, dyobs, xtheor, ytheor, grid_type='linear',
               add_dy_multi=0.0, add_dy_addit=0.0, cut_at=None):
    """
    Uses the 'residuals' above, but xobs, yobs, dyobs, ytheor all can be arrays
    OR lists of arrays.
    """
    return (residuals_multi(xobs, yobs, dyobs, xtheor, ytheor, grid_type,
                            add_dy_multi, add_dy_addit, cut_at)**2).sum()
    


def build_kwargs(func, params, **overrides):
    sig = inspect.signature(func)

    kwargs = {
        k: v
        for k, v in params.items()
        if k in sig.parameters
    }

    kwargs.update(overrides)

    return kwargs

def yamlify(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: yamlify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [yamlify(v) for v in obj]

    return obj

def min_gap_psrb(t_obs):
    t_obs = np.asarray(t_obs)
    res = np.zeros(t_obs.shape)
    cond_way_before = (t_obs < 120. * DAY)
    cond_before = (t_obs >= 120. * DAY) & (t_obs < -50. * DAY) 
    cond_first_peak = (t_obs >= -50. * DAY) & (t_obs < -10 * DAY)
    cond_between = (t_obs >= -10. * DAY) & (t_obs < 8. * DAY)
    cond_second_peak = (t_obs >= 8. * DAY) & (t_obs < 25. * DAY)
    cond_third_peak = (t_obs >= 25. * DAY) & (t_obs < 50. * DAY)
    cond_after = (t_obs >= 50. * DAY) & (t_obs < 120. * DAY)
    cond_way_after = (t_obs > 120. * DAY)
    res[cond_way_before] = 50. * DAY
    res[cond_before] = 15. * DAY
    res[cond_first_peak] = 1. * DAY
    res[cond_between] = 2.5 * DAY
    res[cond_second_peak] = 1. * DAY
    res[cond_third_peak] = 3. * DAY
    res[cond_after] = 10. * DAY
    res[cond_way_after] = 50. * DAY
    return res.item() if np.ndim(t_obs) == 0 else res
    

def build_timegrid_psrb(t_obs):
    t_obs = np.sort(t_obs)
    res = [t_obs[0]]
    for it, t in enumerate(t_obs[1:]):
        if (t - res[-1] > min_gap_psrb(t)) or (t_obs[it+1] - t > 3. * (t - t_obs[it-1])):
            res.append(t)
    res.append(t_obs[-1])
    res = np.array(res)
    return np.unique(res)

        
def linear_slope(x, y):
    """
    Analytical linear fit of the data x, y by y=kx+b; returns slope k
    """
    _n = x.size
    return ((_n * np.sum(x*y) - np.sum(x) * np.sum(y)) 
            / (_n * np.sum(x**2) - np.sum(x)**2))
        
def _fit_norm_only_multi(ydata, dy_data, y0_normalized, return_err=False):
    """
    Fits a multiplicative coefficient N: 
        y_obs = N * y0_normalized
    """
    if np.any(~np.isfinite(y0_normalized)):
        print("non-finite model!")
        
    w = 1.0 / dy_data**2
    denom = np.sum(w * y0_normalized * y0_normalized)
    # if denom <= 0:
    #     raise ValueError('denominator < 0 in fit_norm, cannot find normalization')
    if not np.isfinite(denom):
        raise ValueError(f"Bad denominator: {denom}")
    
    if denom <= 0:
        raise ValueError(
            f"denom={denom}, "
            f"min(y0)={np.min(y0_normalized)}, "
            f"max(y0)={np.max(y0_normalized)}"
        )
    N_best = np.sum(w * y0_normalized * ydata) / denom
    if not return_err:
        return N_best, N_best * y0_normalized
    resid = ydata - N_best * y0_normalized
    chi2 = np.sum((resid / dy_data)**2)
    nu = ydata.size - 1
    s2 = chi2 / nu
    sigma_N = np.sqrt(s2 / denom)
    return N_best, N_best * y0_normalized, sigma_N
    
def _fit_norm_addit_and_multi(ydata, dy_data, y0_normalized, return_err=False):
    """
    Fits a multiplicative coefficient N and an additive const C: 
        y_obs = C + N * y0_normalized
    """
    f = y0_normalized
    w = 1.0 / dy_data**2

    S0 = np.sum(w)
    Sf = np.sum(w * f)
    Sff = np.sum(w * f * f)
    
    Sy = np.sum(w * ydata)
    Syf = np.sum(w * ydata * f)
    
    D = S0 * Sff - Sf**2
    
    if D <= 0:
        D = np.nan
        # raise ValueError(
        #     'degenerate system in fit_norm(addit_const=True)'
        # )
    
    C_best = (Sy * Sff - Sf * Syf) / D
    N_best = (S0 * Syf - Sf * Sy) / D
    
    model_best = C_best + N_best * f
    
    if not return_err:
        return N_best, C_best, model_best
    
    resid = ydata - model_best
    chi2 = np.sum((resid / dy_data)**2)
    nu = ydata.size - 2
    s2 = chi2 / nu
    
    # covariance matrix = s2 (X^T W X)^(-1)
    sigma_C = np.sqrt(s2 * Sff / D)
    sigma_N = np.sqrt(s2 * S0 / D)
    cov = s2 / D * np.array([
            [Sff, -Sf],
            [-Sf,  S0]
            ])
    
    return N_best, C_best, model_best, sigma_N, sigma_C, cov

def fit_norm(ydata, dy_data, y0_normalized, return_err=False, addit_const=False):
    """
    Analytically finds N and C assuming 
            y = C + N * y0_normalized, 
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
    return_err : bool, optional
        Whether to estimate errors for parameters. Default False
    addit_const : bool, optional
        Whether to add an additive constant C. If False, C === 0. Default False

    Returns 
    -------
    if not addit_const:
        if return_err:
            n_best, y_renorm, dn_best
        else:
            n_best, y_renorm
    if addit_const:
        if return_err:
            n_best, c_best, y_renorm, dn_best, dc_best
        else:
            n_best, c_best, y_renorm
            
    Here optimized normalizaion and addit constant n_best/c_best and their
    errors dn_best/dc_best are floats; re-normalized y_renorm is a np.ndarray

    """
    if addit_const:
        return _fit_norm_addit_and_multi(ydata, dy_data, y0_normalized, return_err)
    return _fit_norm_only_multi(ydata, dy_data, y0_normalized, return_err)

# def df_dx(f, x, eps=1e-5):
#     dx = x * eps
#     return (f(x + dx) - f(x - dx)) / 2. / dx

# def fit_N_x_shift(f, edata, fdata, complicated_params, weights=None, bounds=None):
#     """
#     Fit model = N * f(edata * c, 1.0, **complicated_params)

#     Parameters
#     ----------
#     f : callable
#         f(E, C, **kwargs)
#     edata, fdata : array-like
#         Data x and y.
#     complicated_params : dict
#         Extra parameters passed to f.
#     weights : array-like or None
#         If None, uses ones. Usually this should be 1/dy^2.
#     bounds : tuple or None
#         Bounds for c if using bounded minimization.

#     Returns
#     -------
#     N_opt, c_opt, model_opt
#     """
#     edata = np.asarray(edata, dtype=float)
#     fdata = np.asarray(fdata, dtype=float)

#     if weights is None:
#         weights = np.ones_like(edata, dtype=float)
#     else:
#         weights = np.asarray(weights, dtype=float)

#     def model(c):
#         return f(edata * c, 1.0, **complicated_params)

#     def N_best(c):
#         m = model(c)
#         denom = np.sum(weights * m * m)
#         if denom <= 0:
#             raise ValueError("Degenerate denominator in N fit")
#         return np.sum(weights * fdata * m) / denom

#     def chi2(c):
#         m = model(c)
#         N = N_best(c)
#         resid = fdata - N * m
#         return np.sum(weights * resid * resid)

#     if bounds is None:
#         res = minimize_scalar(chi2)
#     else:
#         res = minimize_scalar(chi2, bounds=bounds, method="bounded")

#     c_opt = res.x
#     m_opt = model(c_opt)
#     N_opt = N_best(c_opt)

#     return N_opt, c_opt, N_opt * m_opt
    

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
        power = 1.
        
    return (np.sum(arr**power * weights, axis=axis) / 
            np.sum(weights, axis=axis))**(1. / power)

def fit_norm_here( x_obs, y_obs, dy_obs,
    x_model, y_model, norm_init, 
    grid_scale='log', return_err=False,
    add_const=False, c_init=None):
    """
    Fits normalization(s) of model to observation set(s).

    x_obs/y_obs/dy_obs can be either:
        - np.ndarray
        - list/tuple of np.ndarray

    If lists are provided, normalization is fitted independently
    for each observational dataset.
    """

    def _fit_single(x_obs_single, y_obs_single, dy_obs_single):

        obs_ok = (
            np.isfinite(x_obs_single)
            & np.isfinite(y_obs_single)
            & np.isfinite(dy_obs_single)
            & (x_obs_single > np.min(x_model))
            & (x_obs_single < np.max(x_model))
        )

        x_obs_ok, y_obs_ok, dy_obs_ok = [
            ar[obs_ok]
            for ar in (x_obs_single, y_obs_single, dy_obs_single)
        ]

        model_ok = np.isfinite(x_model) & np.isfinite(y_model)

        x_model_ok, y_model_ok = [
            ar[model_ok]
            for ar in (x_model, y_model)
        ]
        if not add_const:
            y_model_normalized = y_model_ok / norm_init
        else:
            y_model_normalized = (y_model_ok - c_init) / norm_init

        if grid_scale.lower() in ('linear', 'lin'):
            interp_ = interp1d(
                x=x_model_ok,
                y=y_model_normalized,
                bounds_error=False,
                fill_value=(
                    y_model_normalized[0],
                    y_model_normalized[-1]
                )
            )
            y_model_normalized_in_xobs = interp_(x_obs_ok)

        elif grid_scale.lower() in ('log', 'log10'):
            y_model_normalized_in_xobs = interplg(
                x_obs_ok,
                x_model_ok,
                y_model_normalized,
                bounds_error=False,
                fill_value=(
                    np.log10(y_model_normalized[0]),
                    np.log10(y_model_normalized[-1]),
                )
            )

        else:
            raise ValueError("grid_scale should be one of: 'lin', 'log'")

        if not return_err:
            if not add_const:
                norm_opt, _ = fit_norm(
                    ydata=y_obs_ok,
                    dy_data=dy_obs_ok,
                    y0_normalized=y_model_normalized_in_xobs,
                )
                y_model_renorm = y_model_normalized * norm_opt
                return norm_opt, y_model_renorm
            
            norm_opt, c_opt, _ = fit_norm(
                ydata=y_obs_ok,
                dy_data=dy_obs_ok,
                y0_normalized=y_model_normalized_in_xobs,
                addit_const=add_const,
            )
            y_model_renorm = c_opt + norm_opt * y_model_normalized
            return norm_opt, c_opt, y_model_renorm
        else:
            if not add_const:
                norm_opt, _, dnorm_opt = fit_norm(
                    ydata=y_obs_ok,
                    dy_data=dy_obs_ok,
                    y0_normalized=y_model_normalized_in_xobs,
                    return_err=True,
                )
    
                dn_rel = dnorm_opt / norm_opt
                y_model_renorm = y_model_ok * norm_opt / norm_init
                y_low = y_model_renorm * (1 - dn_rel)
                y_high = y_model_renorm * (1 + dn_rel)
                return (
                    (norm_opt,),
                    y_model_renorm,
                    (dn_rel,),
                    y_low,
                    y_high,
                )
            norm_opt, c_opt, _, dn, dc, cov = fit_norm(
                ydata=y_obs_ok,
                dy_data=dy_obs_ok,
                y0_normalized=y_model_normalized_in_xobs,
                return_err=True,
                addit_const=add_const,
            )
            _f = y_model_normalized
            y_model_renorm  = c_opt + norm_opt * y_model_normalized
            dy_model = np.sqrt(dc**2 + _f**2 * dn**2 + 2. * _f * cov[0, 1])
            y_low, y_high = y_model_renorm - dy_model, y_model_renorm + dy_model
            return (
                (norm_opt, c_opt,),
                y_model_renorm,
                (dn, dc,),
                y_low,
                y_high
                )
            


    is_multiple = isinstance(x_obs, (list, tuple))

    if not is_multiple:
        return _fit_single(x_obs, y_obs, dy_obs)

    if not (len(x_obs) == len(y_obs) == len(dy_obs)):
        raise ValueError("x_obs, y_obs, dy_obs must have same length")

    results = [
        _fit_single(xo, yo, dyo)
        for xo, yo, dyo in zip(x_obs, y_obs, dy_obs)
    ]

    # transpose list of tuples into tuple of lists or whatever
    return tuple(map(list, zip(*results)))