# ibsen/fitting/fit.py
import numpy as np
from scipy.optimize import least_squares
from ibsen.utils import make_grid, fit_norm
import lmfit
from typing import List, Dict, Any, Tuple


def compute_confidence_bands(
    result,
    covar: np.ndarray,
    global_param_names: List[str],
    param_names_per_dataset: List[List[str]],
    models: List,
    x_grids: List[np.ndarray],
    y_grids_best: List[np.ndarray],
    step_rel: float = 1e-4,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Compute 1-sigma confidence bands y_low, y_high on each dataset's x_grid
    from the parameter covariance matrix (linear error propagation).

    Parameters
    ----------
    result : lmfit.MinimizerResult
        The result object from lmfit.minimize (must contain params, and ideally var_names).
    covar : ndarray
        Covariance matrix for the *varying* parameters, shape (n_vary, n_vary).
        Typically `result.covar`.
    global_param_names : list of str
        Names of all global parameters (for convenience when building base_values).
    param_names_per_dataset : list of list of str
        param_names_per_dataset[k] is the list of global parameter names
        used as arguments for models[k], in the order models[k] expects.
    models : list of callables
        models[k] must be callable as models[k](x_grid, *params_for_dataset_k).
    x_grids : list of 1D arrays
        x_grids[k] is the x_grid for dataset k.
    y_grids_best : list of 1D arrays
        y_grids_best[k] is the best-fit model evaluated on x_grids[k] for dataset k.
    step_rel : float, optional
        Relative step size for finite differences in parameter space.
        For parameter p, the step is step_rel * max(|p|, 1.0).

    Returns
    -------
    bands : list of dict
        One entry per dataset:
        {
          "y_best":  y_grid_best,
          "y_low":   y_low,
          "y_high":  y_high,
          "sigma_y": sigma_y,
          "x_grid":  x_grid,
        }

    extra_model_calls : int
        Number of extra model evaluations used for finite differences.
    """
    if covar is None:
        raise RuntimeError(
            "Covariance matrix is None. "
            "Make sure the fit used a least-squares method that computes J and cov."
        )

    cov = np.asarray(covar, dtype=float)

    # Names of varying parameters in the order used by covar
    if getattr(result, "var_names", None) is not None:
        var_names = list(result.var_names)
    else:
        # Fallback: params that vary=True, in the order they appear
        var_names = [name for name, par in result.params.items() if par.vary]

    n_vary = len(var_names)
    if n_vary == 0:
        raise RuntimeError("No varying parameters; cannot compute confidence bands.")

    if cov.shape != (n_vary, n_vary):
        raise RuntimeError(
            f"Covariance shape {cov.shape} does not match number of varying "
            f"parameters ({n_vary})."
        )

    # Best-fit global parameter values as a dict
    base_values = {
        name: result.params[name].value
        for name in global_param_names
    }

    # bands: List[Dict[str, Any]] = []
    extra_model_calls = 0
    y_lows = []
    y_highs = []
    # y_opts = []
    y_sigmas = []

    # Loop over datasets / models
    for k, (xg, model, pnames_ds, y_best) in enumerate(
        zip(x_grids, models, param_names_per_dataset, y_grids_best)
    ):
        xg = np.asarray(xg, dtype=float)
        y_best = np.asarray(y_best, dtype=float)
        n_grid = y_best.size

        # Gradient matrix G for this dataset: shape (n_grid, n_vary)
        G = np.zeros((n_grid, n_vary), dtype=float)

        # For each varying parameter, compute dy/dp via finite differences
        for j, pname in enumerate(var_names):
            vals_pert = dict(base_values)
            p_val = vals_pert[pname]

            step = step_rel * max(abs(p_val), 1.0)
            if step == 0.0:
                step = step_rel

            vals_pert[pname] = p_val + step

            # Build parameter vector for THIS dataset's model
            p_vec_pert = [vals_pert[name] for name in pnames_ds]

            # Evaluate model on grid with perturbed parameters
            y_pert = model(xg, *p_vec_pert)
            extra_model_calls += 1

            # Finite-difference derivative dy/dp
            G[:, j] = (y_pert - y_best) / step

        # Propagate covariance: var_y = diag( G * cov * G^T )
        A = G @ cov                           # shape (n_grid, n_vary)
        var_y = np.sum(A * G, axis=1)         # elementwise multiply, sum over parameters
        var_y = np.maximum(var_y, 0.0)        # avoid tiny negative due to rounding
        sigma_y = np.sqrt(var_y)

        y_low = y_best - sigma_y
        y_high = y_best + sigma_y
        y_lows.append(y_low)
        
        y_highs.append(y_high)
        y_sigmas.append(y_sigmas)

    return y_lows, y_highs, y_sigmas

class BasicModelFitter: # !!!
    """
    Generic 1D fitter with:
    - optional model grid (x_grid) + interpolation to data points
    - optional linear (normalization-like) parameter solved analytically
    """

    def __init__(self, x, y, dy=None, dx=None):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.dy = np.ones_like(self.y) if dy is None else np.asarray(dy, dtype=float)
        self.dx = None if dx is None else np.asarray(dx, dtype=float)

        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        if self.dy.shape != self.y.shape:
            raise ValueError("dy must have the same shape as y")

        self.n_model_calls = 0  # to keep track of expensive model evaluations

    # ------------------------- utilities -------------------------

    def _build_param_vector(self, param_names, base_params):
        """Return param vector in the order expected by model(x, *params)."""
        return [base_params[name] for name in param_names]

    # ------------------------- main fit -------------------------
    
    def _compute_confidence_band(self,
                                model,
                                param_names,
                                params_best,
                                nonlinear_params,
                                cov_nonlin,
                                lin_name,
                                sigma_lin,
                                xg,
                                y_grid_best,
                                step_rel=1e-3,
                                nsigma=2,
                            ):
        """
        Compute 1-sigma confidence band y_low, y_high on x_grid
        from parameter covariance.
    
        - Non-linear params: use finite differences to get dy/dθ_i on the grid.
        - Linear param (if any): derivative is analytic: dy/dA = y / A.
        """
        n_grid = xg.size
        var_y = np.zeros(n_grid)
    
        # --- non-linear parameters contribution ---
        if (cov_nonlin is not None) and (len(nonlinear_params) > 0):
            p_nonlin_best = np.array(
                [params_best[p] for p in nonlinear_params],
                dtype=float,
            )
            n_nonlin = len(nonlinear_params)
    
            # Gradient matrix G_nonlin: shape (n_grid, n_nonlin)
            G_nonlin = np.empty((n_grid, n_nonlin), dtype=float)
    
            # Baseline is y_grid_best already.
            for j, name in enumerate(nonlinear_params):
                p_shift = p_nonlin_best.copy()
                # Relative step for parameter j
                step = step_rel * max(abs(p_nonlin_best[j]), 1.0)
                if step == 0.0:
                    step = step_rel
    
                p_shift[j] += step
    
                # Build shifted parameter dict
                params_shift = dict(params_best)
                params_shift[name] = p_shift[j]
    
                # Build parameter vector in correct order
                theta_shift = self._build_param_vector(param_names, params_shift)
    
                # Model on grid for shifted param
                y_grid_shift = model(xg, *theta_shift)
    
                # Finite-difference derivative dy/dθ_j
                G_nonlin[:, j] = (y_grid_shift - y_grid_best) / step
    
            # var_y_nonlin = diag( G_nonlin @ cov_nonlin @ G_nonlin^T )
            # Efficient computation without forming full (n_grid, n_grid):
            A = G_nonlin @ cov_nonlin                # shape (n_grid, n_nonlin)
            var_y_nonlin = np.sum(A * G_nonlin, axis=1)
            var_y += var_y_nonlin
    
        # --- linear parameter contribution ---
        if (lin_name is not None) and (sigma_lin is not None) and np.isfinite(sigma_lin):
            A_best = params_best[lin_name]
            if A_best != 0.0:
                dy_dA = y_grid_best / A_best       # analytic derivative
                var_y_lin = (dy_dA * sigma_lin) ** 2
                var_y += var_y_lin
    
        # Avoid tiny negative values due to rounding
        var_y = np.maximum(var_y, 0.0)
        sigma_y = np.sqrt(var_y)
    
        y_low = y_grid_best - sigma_y * nsigma
        y_high = y_grid_best + sigma_y * nsigma
    
        return y_low, y_high

    def fit(
        self,
        model,
        param_names,
        initial_params,
        fit_params,
        fixed_params=None,
        *,
        linear_param=None,
        compute_confidence = False,
        nsigma=2,
        x_grid=None,
        grid_kind="linear",
        n_grid=400,
        per_decade=False,
        bounds=None,
    ):
        """
        Fit data with model(x, *params).

        Parameters
        ----------
        model : callable
            Function model(x_grid, *params) -> y_grid.
        param_names : sequence of str
            Names of all model parameters in the order expected by `model`.
        initial_params : dict
            Initial guess for all parameters: {name: value}.
        fit_params : sequence of str
            Names of parameters to optimize (excluding fixed ones).
            If `linear_param` is not None, it may be included or not;
            it will be treated specially anyway.
        fixed_params : dict, optional
            Parameters to keep fixed: {name: value}.
        linear_param : str or None, optional
            Name of a parameter treated as *linear normalization*.
            Must be at most one. It is solved analytically at each step.
        x_grid : array-like or None, optional
            Grid where the model is evaluated. If None, constructed using
            `grid_kind` and `n_grid`.
        grid_kind : {"linear", "log", "data"}, optional
            Type of grid to build if x_grid is None.
        n_grid : int, optional
            Number of grid points if x_grid is None and grid_kind != "data".
        per_decade : bool, optional
            Whether to treat n_grid as number of nods per decate if grid_kind
            is 'log'
        bounds : (dict, dict) or None, optional
            (lower_bounds, upper_bounds) as dicts mapping parameter name -> bound.
            Only applied to *non-linear* fitted parameters.

        Returns
        -------
        result : dict
            {
              "best_params": dict of best-fit values (including fixed and linear),
              "errors": dict of 1σ errors (if estimable),
              "x_grid": x_grid,
              "y_grid": best model on x_grid,
              "chi2": float,
              "red_chi2": float,
              "cov_nonlin": covariance matrix of non-linear params,
              "fitted_nonlin_params": list of non-linear parameter names,
              "linear_param": name or None,
              "n_model_calls": int,
              "opt_result": scipy.optimize.OptimizeResult
            }
        """
        #  --------------  --- Basic setup --- --------------------------------
        n_calls = 0
        param_names = list(param_names)
        init = dict(initial_params)
        fixed_params = dict(fixed_params) if fixed_params is not None else {}

        # Sanity checks
        for name in param_names:
            if name not in init and name not in fixed_params:
                raise ValueError(f"No initial value for parameter '{name}'")

        fit_params = list(fit_params)
        if linear_param is not None:
            if isinstance(linear_param, str):
                lin_name = linear_param
            else:
                raise ValueError("linear_param must be a string or None")
        else:
            lin_name = None

        if lin_name is not None and fit_params.count(lin_name) > 1:
            raise ValueError("linear_param appears multiple times in fit_params")

        # Non-linear fitted params = fit_params excluding the linear one
        nonlinear_params = [p for p in fit_params if p != lin_name]

        # Create x_grid
        
        xg = x_grid if x_grid is not None else make_grid(self.x,
                                            grid_kind, n_grid, per_decade)
        self.x_grid_ = xg

        # Reference value for linear parameter
        if lin_name is not None:
            if lin_name not in init and lin_name not in fixed_params:
                raise ValueError(f"linear_param '{lin_name}' must have an initial or fixed value")
            p_ref = init.get(lin_name, fixed_params.get(lin_name))
            if p_ref == 0.0:
                p_ref = 1.0  # avoid zero; just a reference scaling
        else:
            p_ref = None

        # Build initial vector for non-linear params
        p0_nonlin = np.array([init[p] for p in nonlinear_params], dtype=float)
        # Helper: given non-linear param vector, build full param dict for the model
        def build_params_dict(p_nonlin, lin_value=None):
            params = dict(init)
            params.update(fixed_params)
            for name, val in zip(nonlinear_params, p_nonlin):
                params[name] = float(val)
            if lin_name is not None and lin_value is not None:
                params[lin_name] = float(lin_value)
            return params

        cov_nonlin = None
        opt = None
        y_low = None
        y_high = None
        if len(p0_nonlin) > 0: # if there are non-linear parameters to fit

            # Build bounds for non-linear params
            if bounds is not None:
                lower_dict, upper_dict = bounds
                lower = np.array(
                    [lower_dict.get(p, -np.inf) for p in nonlinear_params],
                    dtype=float
                )
                upper = np.array(
                    [upper_dict.get(p, +np.inf) for p in nonlinear_params],
                    dtype=float
                )
            else:
                lower = -np.inf * np.ones_like(p0_nonlin)
                upper = +np.inf * np.ones_like(p0_nonlin)
    

            # Residual function for least_squares
            def residuals_nonlin(p_nonlin):
                # Put together param values with linear param fixed at reference
                if lin_name is not None:
                    params = build_params_dict(p_nonlin, lin_value=p_ref)
                else:
                    params = build_params_dict(p_nonlin)
    
                theta = self._build_param_vector(param_names, params)
    
                # Call the expensive model ONCE on the grid
                # n_calls += 1
                y_grid = model(xg, *theta)
    
                # Interpolate to data x
                y_model_ref = np.interp(self.x, xg, y_grid)
    
                if lin_name is None:
                    # No linear param: residuals directly
                    res = (y_model_ref - self.y) / self.dy
                    return res
                else:
                    # Linear parameter: model is linear in lin_name.
                    # We computed model at lin_name = p_ref.
                    # So y_model_ref(x) = p_ref * shape(x)
                    p_best, a_normalized = fit_norm(self.y, self.dy,
                                                    y_model_ref / p_ref)
                    # Residuals with best linear param
                    res = (a_normalized - self.y) / self.dy
                    return res
    
            # Run the optimizer
            opt = least_squares(
                residuals_nonlin,
                p0_nonlin,
                bounds=(lower, upper),
            )
            n_calls += (opt.nfev + opt.njev) # ???
    
            # ------------------------------------------------------------------
            # Post-processing: best-fit parameters, errors, grid model, etc.
            # ------------------------------------------------------------------
            p_nonlin_best = opt.x
    
            # Rebuild best-fit full parameter dict, including linear param
            if lin_name is None:
                params_best = build_params_dict(p_nonlin_best)
            else:
                # Need to recompute model once at p_ref and solve for best linear param
                params_for_lin = build_params_dict(p_nonlin_best, lin_value=p_ref)
                theta_ref = self._build_param_vector(param_names, params_for_lin)
    
                # self.n_model_calls += 1
                n_calls += 1
                y_grid_ref = model(xg, *theta_ref)
                y_model_ref = np.interp(self.x, xg, y_grid_ref)
                p_best, _a_renormalized = fit_norm(self.y, self.dy, 
                                                   y_model_ref / p_ref)
                params_best = build_params_dict(p_nonlin_best, lin_value=p_best)
    
            # Best-fit parameter vector in correct order for the model
            theta_best = self._build_param_vector(param_names, params_best)
            # Compute final model on x_grid with best params
            # self.n_model_calls += 1
            n_calls += 1
            y_grid_best = model(xg, *theta_best)
            y_model_best = np.interp(self.x, xg, y_grid_best)
            res_best = (y_model_best - self.y) / self.dy
            chi2 = np.sum(res_best ** 2)

            n_data = self.x.size
            n_free = len(nonlinear_params) + (1 if lin_name is not None else 0)
            dof = max(n_data - n_free, 1)
            red_chi2 = chi2 / dof

            # Covariance matrix for non-linear parameters
            # SciPy's least_squares cost = 0.5 * sum(res^2)
            if opt.jac is not None and opt.jac.size > 0 and len(nonlinear_params) > 0:
                J = opt.jac  # shape (n_data, n_nonlin)
                JTJ = J.T @ J
                s_sq = chi2 / dof  # estimated variance of residuals
                try:
                    cov_nonlin = np.linalg.inv(JTJ) * s_sq
                except np.linalg.LinAlgError:
                    cov_nonlin = np.linalg.pinv(JTJ) * s_sq
            else:
                cov_nonlin = None
            
        else: # if there is only one, linear, parameter to fit
            if lin_name is not None:
                params = build_params_dict(p0_nonlin, lin_value=p_ref)
            else:
                raise ValueError('Nothing to fit')

            theta = self._build_param_vector(param_names, params)

            # Call the expensive model ONCE on the grid
            # self.n_model_calls += 1
            n_calls += 1
            y_grid_ref = model(xg, *theta)
            y_model_ref = np.interp(self.x, xg, y_grid_ref)
            p_best, y_model_best = fit_norm(self.y, self.dy, 
                                               y_model_ref / p_ref)
            params_best = build_params_dict(p0_nonlin, lin_value=p_best)
            y_grid_best = y_grid_ref * p_best / p_ref 
            res_best = (y_model_best - self.y) / self.dy
            chi2 = np.sum(res_best ** 2)

            n_data = self.x.size
            n_free = len(nonlinear_params) + (1 if lin_name is not None else 0)
            dof = max(n_data - n_free, 1)
            red_chi2 = chi2 / dof
            
            

        # Errors
        errors = {}
        # Non-linear params
        if cov_nonlin is not None:
            for i, name in enumerate(nonlinear_params):
                errors[name] = float(np.sqrt(cov_nonlin[i, i]))

        # Linear param error (if present): analytic formula
        if lin_name is not None:
            # We already have a, w for best-fit
            w = 1 / self.dy**2
            denom = np.sum(w * (y_model_best/p_best)**2)
            s_sq = chi2 / dof
            if denom > 0:
                var_lin = s_sq / denom
                errors[lin_name] = float(np.sqrt(var_lin))
            else:
                errors[lin_name] = np.nan

        # Fixed params have error = 0 by definition
        for name in fixed_params.keys():
            errors.setdefault(name, 0.0)

        # Also fill errors for any param that wasn't in fit_params at all
        for name in param_names:
            errors.setdefault(name, np.nan)  # unknown/unused
            
        if len(p0_nonlin) > 0:    
            if compute_confidence and (cov_nonlin is not None):
                sigma_lin = None
                if lin_name is not None:
                    sigma_lin = errors.get(lin_name, None)
                y_low, y_high = self._compute_confidence_band(
                                    model=model,
                                    param_names=param_names,
                                    params_best=params_best,
                                    nonlinear_params=nonlinear_params,
                                    cov_nonlin=cov_nonlin,
                                    lin_name=lin_name,
                                    sigma_lin=sigma_lin,
                                    xg=xg,
                                    y_grid_best=y_grid_best,
                                    nsigma=nsigma,
                                    )
    
                n_calls += len(nonlinear_params)
        else:
            sigma_lin = errors[lin_name]
            y_low, y_high = self._compute_confidence_band(
                                model=model,
                                param_names=param_names,
                                params_best=params_best,
                                nonlinear_params=[],      # none
                                cov_nonlin=None,
                                lin_name=lin_name,
                                sigma_lin=sigma_lin,
                                xg=xg,
                                y_grid_best=y_grid_best,
                                nsigma=nsigma,
                            )
        
        result = {
            "best_params": params_best,
            "errors": errors,
            "x_grid": xg,
            "y_grid": y_grid_best,
            "res": res_best,
            "chi2": chi2,
            "red_chi2": red_chi2,
            "cov_nonlin": cov_nonlin,
            "fitted_nonlin_params": nonlinear_params,
            "linear_param": lin_name,
            "n_model_calls": n_calls,
            "opt_result": opt,
            "y_low": y_low,
            "y_high": y_high,
        }
        return result

# Assumed to exist in your codebase, same signature as before:
# make_grid(x_data, grid_kind, n_grid, per_decade) -> x_grid
# from your_module import make_grid


class MultiDatasetFitter: # !!!
    """
    Global fitter for multiple datasets and models with parameter tying via names.

    - You provide N_d datasets: [(x1, y1, dy1), (x2, y2, dy2, dx2), ...]
      (dx is optional and currently unused, kept for future extensions).
    - You provide N_d models: [model1, model2, ...], each callable as:
        model_i(x_grid, *params_i)
    - You provide parameter names per dataset, e.g.:
        param_names_per_dataset = [
            ["norm_shared", "index1"],    # for dataset 0 / model0
            ["norm_shared", "index2"],    # for dataset 1 / model1
        ]
      Same string name => tied global parameter.

    - You provide initial values and optional bounds for the *global* parameters:
        initial_params = {"norm_shared": 1.0, "index1": 2.0, "index2": 2.5}
        bounds = (
            {"norm_shared": 0.0},   # lower bounds dict
            {"norm_shared": 10.0},  # upper bounds dict
        )

    It uses lmfit to:
    - create a single Parameters() object with one entry per *global* name,
    - run a global least-squares fit over all datasets,
    - give you best-fit values, errors, covariance, etc.
    """

    def __init__(
        self,
        datasets,
        models,
        param_names_per_dataset,
        initial_params,
        bounds=None,
        x_grid=None,
        grid_kind="linear",
        n_grid=400,
        per_decade=False,
    ):
        """
        Parameters
        ----------
        datasets : list
            List of datasets. Each element is either:
              (x, y, dy) or (x, y, dy, dx)
            where x, y, dy, dx are array-like. dx is optional and
            currently not used in the fit.
        models : list of callables
            List of model functions, one per dataset.
            Each model must have signature: model(x_grid, *params).
        param_names_per_dataset : list of list of str
            param_names_per_dataset[k] is a list of parameter names
            for models[k], in the order that model expects them.
            Same name across different datasets => tied global parameter.
        initial_params : dict
            Mapping from global parameter name -> initial value.
            Must cover all names that appear in param_names_per_dataset.
        bounds : (dict, dict) or None, optional
            Tuple (lower_bounds, upper_bounds), where each is a dict:
                lower_bounds[name] = lower_bound_for_name
                upper_bounds[name] = upper_bound_for_name
            Any name not in a dict gets default (-inf, +inf).
        grid_kind : {"linear", "log", "data"}, optional
            How to construct the grid for each dataset if needed.
        n_grid : int, optional
            Number of grid points if grid_kind != "data".
        per_decade : bool, optional
            Whether n_grid is per-decade if grid_kind == "log".
        """
        self.datasets = []
        self.x_grids = []
        # Store grid settings
        self.grid_kind = grid_kind
        self.n_grid = n_grid
        self.per_decade = per_decade
        
        for d in datasets:
            if len(d) == 3:
                x, y, dy = d
                dx = None
            elif len(d) == 4:
                x, y, dy, dx = d
            else:
                raise ValueError("Each dataset must be (x, y, dy) or (x, y, dy, dx).")

            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            dy = np.asarray(dy, dtype=float)
            if x.shape != y.shape or y.shape != dy.shape:
                raise ValueError("x, y, dy must have the same shape.")

            self.datasets.append({"x": x, "y": y, "dy": dy, "dx": dx})
            xg = x_grid if x_grid is not None else make_grid(x, grid_kind=self.grid_kind,
                           n_grid=self.n_grid, per_decade=self.per_decade)
            self.x_grids.append(xg)
    
        self.models = list(models)
        self.param_names_per_dataset = [list(pnames) for pnames in param_names_per_dataset]

        if len(self.datasets) != len(self.models) or \
           len(self.datasets) != len(self.param_names_per_dataset):
            raise ValueError("datasets, models, and param_names_per_dataset must have same length.")

        # Build per-dataset grids once
        # self.x_grids = []
        # for data in self.datasets:
            # x_data = data["x"]
            # xg = make_grid(x_data, grid_kind=self.grid_kind,
            #                n_grid=self.n_grid, per_decade=self.per_decade)
            # self.x_grids.append(xg)

        # Collect all global names from param_names_per_dataset
        all_names = set()
        for names in self.param_names_per_dataset:
            all_names.update(names)
        self.global_param_names = sorted(all_names)

        # Check initial_params covers everything
        missing = [name for name in self.global_param_names if name not in initial_params]
        if missing:
            raise ValueError(f"Missing initial values for parameters: {missing}")
        self.initial_params = dict(initial_params)

        # Bounds
        if bounds is not None:
            lower_dict, upper_dict = bounds
        else:
            lower_dict, upper_dict = {}, {}
        self.lower_bounds = dict(lower_dict)
        self.upper_bounds = dict(upper_dict)

        # Build the lmfit Parameters object
        self.params = lmfit.Parameters()
        for name in self.global_param_names:
            val = float(self.initial_params[name])
            pmin = self.lower_bounds.get(name, -np.inf)
            pmax = self.upper_bounds.get(name, +np.inf)
            self.params.add(name, value=val, min=pmin, max=pmax)

        # Placeholders for fit results
        self.result = None
        self.best_params = None
        self.y_grids_best = None  # list of y_grid arrays per dataset

    # ---------------------------------------------------------
    # Internal helper: build parameter vector for a given dataset
    # ---------------------------------------------------------

    def _get_params_for_dataset(self, params, dataset_index):
        """Return list of parameter values for models[dataset_index]."""
        names = self.param_names_per_dataset[dataset_index]
        return [params[name].value for name in names]

    # ---------------------------------------------------------
    # Residual function for lmfit
    # ---------------------------------------------------------

    def _residuals(self, params):
        """
        Global residual function over all datasets.

        params : lmfit.Parameters
            Current global parameters.
        Returns
        -------
        residuals : 1D numpy array
            Concatenated residuals from all datasets.
        """
        all_res = []
        print({name: params[name].value for name in params if params[name].vary})

        for i, data in enumerate(self.datasets):
            x_data = data["x"]
            y_data = data["y"]
            dy_data = data["dy"]
            xg = self.x_grids[i]
            # model = self.models[i]

            # # Parameter vector for this dataset's model
            # p_vec = self._get_params_for_dataset(params, i)

            # # Evaluate model on grid, interpolate to data x
            # y_grid = model(xg, *p_vec)
            p_vec = [params[name].value for name in self.param_names_per_dataset[i]]
            y_grid = self.models[i](self.x_grids[i], *p_vec)
            y_model = np.interp(x_data, xg, y_grid)

            res = (y_model - y_data) / dy_data
            all_res.append(res.ravel())

        return np.concatenate(all_res)

    # ---------------------------------------------------------
    # Public method: run the fit
    # ---------------------------------------------------------

    def fit(self,  compute_confidence=True,
            **minimize_kws):
        """
        Run lmfit.minimize on the global residual function.

        Parameters
        ----------
        method : str, optional
            Optimization method passed to lmfit.minimize (default: 'leastsq').
        compute_confidence : bool, optional
            Whether to compute confidence bands for fits.
        **minimize_kws :
            Additional keyword arguments forwarded to lmfit.minimize.

        Returns
        -------
        result : dict
            {
              "lmfit_result": lmfit.MinimizerResult,
              "best_params": dict(name -> value),
              "errors": dict(name -> 1-sigma error or None),
              "covar": covariance matrix (or None),
              "y_grids": list of y_grid_best per dataset,
            }
        """
        # Run the minimization
        result = lmfit.minimize(
            self._residuals,
            self.params,
            **minimize_kws,
        )
        self.result = result
        

        # Extract best-fit parameter dict
        best_params = {name: result.params[name].value for name in self.global_param_names}

        # Extract errors (stderr) from lmfit, if available
        errors = {}
        for name in self.global_param_names:
            par = result.params[name]
            errors[name] = par.stderr  # may be None

        # Covariance matrix (if available)
        covar = result.covar  # None if not computed / not applicable

        # Compute best-fit model on each dataset's grid
        y_grids = []
        y_models = []
        ress = []
        for i, data in enumerate(self.datasets):
            xg = self.x_grids[i]
            model = self.models[i]
            p_vec_best = [best_params[name] for name in self.param_names_per_dataset[i]]
            y_grid_best = model(xg, *p_vec_best)
            y_model_best = np.interp(data["x"], xg, y_grid_best)
            res = (y_model_best - data["y"]) / data["dy"]
            y_grids.append(y_grid_best)
            y_models.append(y_model_best)
            ress.append(res)
        self.y_grids_best = y_grids
        
        y_lows, y_highs = None, None
        if compute_confidence:
            y_lows, y_highs, y_sigmas = compute_confidence_bands(
                    result=result,
                    covar=covar,
                    global_param_names=self.global_param_names,
                    param_names_per_dataset=self.param_names_per_dataset,
                    models=self.models,
                    x_grids=self.x_grids,
                    y_grids_best=self.y_grids_best,
                    step_rel=1e-4,
                )

        out = {
            "lmfit_result": result,
            "best_params": best_params,
            "errors": errors,
            "covar": covar,
            "y_grids": y_grids,
            "y_models": y_models,
            "res": ress,
            "y_lows": y_lows,
            "y_highs": y_highs,
            
        }
        return out
