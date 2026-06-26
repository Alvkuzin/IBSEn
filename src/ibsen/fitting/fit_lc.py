import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.optimize import brentq, fsolve, root, least_squares
from pathlib import Path

from ibsen import (Orbit, 
                   # Winds, IBS, IBS3D, SpectrumIBS, 
                   LightCurve)
from ibsen.gui.utils_gui import read_lightcurve_columns
# from ibsen.utils import beta_from_g, vector_angle, doppler_delta, interplg
from ibsen.fitting.utils_fit import (residuals, residuals_multi, chi2_multi, 
                                 build_kwargs, yamlify, min_gap_psrb, 
                                 build_timegrid_psrb, fit_norm_here)

DAY = 86400.
T21 = 59254.867359    
Ppsrb=1236.72453
c_light = 3e10


Norm0 = 3e37
SMAX = 1.0
# ALPHA_DISK = 14.


M0 = 24. * 2e33
incl_minus_180_deg_0 = 23.5

   
g0_test = 1.19
f_d_test = 250.
d0_test = 0.03
default_params = { "sys_name": "psrb",
              
              "puls_b_ref" : 1.0, "puls_r_ref" : 1e13,
              "opt_b_ref": 0.0, "opt_r_ref": 1e13,
              "alpha_disk_deg": 15.,
              "incl_disk_deg": -45.,
              "f_d" : f_d_test, "f_p": 0.03, 
              "delta": d0_test, "np_disk": 3.0, "rad_prof": "pl", "r_trunk":None,
              "height_exp": 0.5,
              
              # "orientation": 'flow',
              "orientation": 'flow_p',
              
              "n_ibs": 15, "n_phi": 23, "gamma_max": g0_test,
              "s_max_g": SMAX, "s_max" : SMAX, "coef_quench": 0.0, "shield_star": 5.0,
              "ibs_ndim": 3,
              
              "cooling": "stat_mimic", "to_inject_e": "secpl", "beta_e": 2.0, "p_e": 2.2,
              "ecut": 3e13, "n_e_cut": 0.5, "eta_syn":1.0, "eta_ic": 1.0, 
              "eta_a": 1.0,  'norm_e': Norm0, 


              "lorentz_boost": True,
              "mechanisms": ['s', 'i'], 
              "method" : "simple",
              # 'method': 'apex', # !!!
              "abs_gg": True,
              "ic_ani": True, "bands": [[3e2, 1e4], [4e11, 1e13]], 
              "to_parall": True, "n_cores": 10,
              
              }

def lc_from_g(g0, fd, d0, times=None, parall=True, ncores=None, **add_kwargs):
    if parall:
        if ncores is None:
            ncores = 10
    else:
        ncores = None
    _params = add_kwargs.copy()
    _params.update({'to_parall':parall, 'n_cores': ncores,
                    "gamma_max": g0,  "f_d": fd, "delta": d0})
    lc = LightCurve(times=times, **_params)
    lc.calculate()
    return lc
    

def lc_xray_fitter(t_xray, f_xray, df_xray, 
                   g0, f0, d0,
                   use_tevs=False, 
                   t_h=None, f_h=None, df_h=None,
                   relative_tev_importance=0.5,
                   eps=1e-3,
                   additive_c_xray=False,
                   additive_c_tev=False,
                   parall=True, ncores=None,
                   **add_kwargs):
  
    if not use_tevs:
        try:
            t_all_x = np.concatenate(t_xray) * DAY
        except ValueError:
            t_all_x = t_xray * DAY
        t_grid = build_timegrid_psrb(t_all_x)
    else:
        try:
            t_all_x = np.concatenate(t_xray) * DAY
        except ValueError:
            t_all_x = t_xray * DAY
        try:
            t_all_h = np.concatenate(t_h) * DAY
        except ValueError:
            t_all_h = t_h * DAY
        
        t_both = np.sort(np.concatenate((t_all_h, t_all_x)))
        t_grid = build_timegrid_psrb(t_both)
        
    if not use_tevs:
        bands = [[3e2, 1e4]]
        mechanisms = ['s']
    else:
        bands = [[3e2, 1e4], [4e11, 1e13]]
        mechanisms = ['s', 'i']
    input_kwargs = add_kwargs.copy()    
    input_kwargs.update({'bands': bands,
                    'mechanisms': mechanisms})
        
    def _residuals(par):
        _logg_m1, _logf, _d = par
        _lc = lc_from_g(g0 = 1.+10**_logg_m1, 
                        fd = 10**_logf,
                        d0=_d,
                        times=t_grid, 
                        parall=parall, ncores=ncores,
                        **input_kwargs,
                        )
        
        xray_flxs = _lc.fluxes[:, 0]
        *_, xrays_normalized = fit_norm_here(x_obs=t_xray, y_obs=f_xray, dy_obs=df_xray, 
                            x_model=t_grid/DAY, y_model=xray_flxs, norm_init=Norm0,
                                grid_scale='lin', add_const=additive_c_xray, c_init=0.0)
        total_resid_x = residuals_multi(t_xray, f_xray, df_xray, t_grid/DAY, xrays_normalized, 'linear',
                                        add_dy_multi=0.1)
        if not use_tevs:
            return total_resid_x
            
        if use_tevs:
            tev_flxs = _lc.fluxes[:, 1]
            *_, tevs_normalized = fit_norm_here(x_obs=t_h, y_obs=f_h, dy_obs=df_h, 
                                x_model=t_grid/DAY, y_model=tev_flxs, norm_init=Norm0,
                                    grid_scale='lin', add_const=additive_c_tev, c_init=0.0)
            total_resid_h = residuals_multi(t_h, f_h, df_h, t_grid/DAY, tevs_normalized, 'linear',
                                            add_dy_multi=0.1)
            return np.concatenate((
                (1. - relative_tev_importance) * total_resid_x,
                relative_tev_importance * total_resid_h,
                ))
        
    
    sol = least_squares(fun = _residuals, x0 = (np.log10(g0-1.), np.log10(f0), d0),
                        bounds = ([-2., 1.0, 0.001], [0.4, 4.5, 0.15]), 
                                                xtol=eps,
                                                ftol=eps,
                                                gtol=eps,
                                                method='trf')
    (log_g_sol_m1, logf_sol, d_sol) = sol.x
    f_sol = 10**logf_sol
    g_sol = 1. + 10**log_g_sol_m1
    _params_output = input_kwargs.copy()
    _params_output.update({"gamma_max": g_sol, "f_d": f_sol, "delta": d_sol,
                           })
    if sol.success:
        return True, g_sol, f_sol, d_sol, _params_output
    else:
        return False, g0, f0, d0, _params_output


def simple_optimizer(init_guess=None, eps=1e-2, **add_kwargs):
    _input_kw = add_kwargs.copy()
    _input_kw.update({'mechanisms': ['s'], 'bands': [[3e2, 1e4]]})
    t_test1 = -12*DAY
    t_test2 = 20*DAY
    t_test_disk1 = 11.5*DAY
    t_test_disk2 = 15.5*DAY
    R_needed_d = 2.7
    R_needed = 1.8
    R_needed_f = 2.5
    if init_guess is None:
        g0, f0, d0 = 1.3, 500., 0.03
    else:
        g0, f0, d0 = init_guess
    def func_to_optimize(params):        
        # print(params)  
        # _g, _f, _d = params
        _logg_m1, _logf, _d = params

        # _ts = np.array([t_test1-1.*DAY,t_test1+1.*DAY, t_test2-1.*DAY, t_test2+1.*DAY,  -1.*DAY, 1.*DAY, ])
        _ts = np.array([t_test1, t_test2, 0., t_test_disk1, t_test_disk2])
        # g0, fd, d0
        # lc_short = lc_from_g(_g, fd=_f, d0=_d, times = _ts, to_parall=True,
        #                      **_input_kw)
        lc_short = lc_from_g(g0 = 1.+10**_logg_m1, 
                        fd = 10**_logf,
                        d0=_d,
                        times=_ts, parall=True, **_input_kw,
                        )
        xray_flxs = lc_short.fluxes[:, 0]
        R_here = (xray_flxs[1]) / (xray_flxs[0])
        R_for_f = (xray_flxs[0]) / (xray_flxs[2])
        R_for_d = (xray_flxs[4]) / (xray_flxs[3])
        res = np.array([R_needed - R_here,
                R_needed_f - R_for_f,
                R_needed_d - R_for_d
                ])
        # print(res)
        return res
    sol = least_squares(fun = func_to_optimize, 
                        x0 = (np.log10(g0-1.), np.log10(f0), d0),
                        bounds = ([-2., 1.0, 0.001], 
                                  [0.4, 4.5, 0.15]), 
                        diff_step=(1e-2, 0.01, 0.001),
                            xtol=eps,
                            ftol=eps,
                            gtol=eps,   
                            method='dogbox')
    (log_g_sol_m1, logf_sol, d_sol) = sol.x
    f_sol = 10**logf_sol
    g_sol = 1. + 10**log_g_sol_m1
    _params_output = _input_kw.copy()
    _params_output.update({"gamma_max": g_sol, "f_d": f_sol, "delta": d_sol,
                           })
    if sol.success:
        return True, g_sol, f_sol, d_sol, _params_output
    else:
        return False, g0, f0, d0, _params_output
    