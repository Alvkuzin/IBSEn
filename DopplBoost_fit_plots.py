import numpy as np
# from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import brentq, root, fsolve, least_squares, minimize
from scipy.optimize import curve_fit
from pathlib import Path
from numpy import pi, sin, cos
from joblib import Parallel, delayed
import multiprocessing
import time
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, interp1d
import LC
import Orbit as Orb
import GetObsData
# import pandas as pd
import pickle

# from LC import LC.Xray_Light_curve
# import Plots as plts
start_time = time.time()
#import matplotlibs
#matplotlib.use('NbAgg')
G = 6.67e-8
c_light = 3e10
sigma_b = 5.67e-5
h_planck_red = 1.05e-27
Rsun = 7e10
AU = 1.5e13
DAY = 86400.
PSRB_par = Orb.Get_PSRB_params()
P, a, e, M, GM, D, Ropt = [PSRB_par[str(x)] for x in ('P', 'a', 'e', 'M', 'GM', 'D', 'Ropt')]
r_periastron = a * (1 - e)
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

def unpack(query, dictat):
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def DelChi2(Npar, conf):
    if conf == 1:
        if Npar == 1:
            return 1.00
        if Npar == 2:
            return 2.30
        if Npar >= 3:
            return 3.50
    if conf == 2:
        if Npar == 1:
            return 2.71
        if Npar == 2:
            return 4.61
        if Npar >= 3:
            return 6.25
    if conf == 3:
        if Npar == 1:
            return 6.63
        if Npar == 2:
            return 9.21
        if Npar >= 3:
            return 11.30

def residuals(xobs, yobs, dyobs, xtheor, ytheor):
    sort_args = np.argsort(xtheor)
    xtheor_, ytheor_ = [ar[sort_args] for ar in (xtheor, ytheor)]
    spline_ = interp1d(xtheor_, ytheor_)
    # yth_in_obs = splev(xobs, spline_)
    res = (yobs - spline_(xobs)) / dyobs
    return res

def residuals_simple(yobs, dyobs, ytheor):
    res = (yobs - ytheor) / dyobs
    return res

from math import ceil

def make_custom_xgrid(xdata, N, Nbefore=2, Nafter=4):
    xdata = xdata / DAY
    x_min = np.min(xdata) - 5
    x_max = np.max(xdata) + 5
    parts = []

    # Segment 1
    if x_min < -30:
        seg_len = -30 - x_min
        Mmin = max(2, int(ceil(seg_len * (Nbefore/10))))
        parts.append(np.linspace(x_min, -30, Mmin))
        start2 = -30
    else:
        start2 = x_min

    # Segment 2
    end2 =  40 if x_max > 40 else x_max
    parts.append(np.linspace(start2, end2, int(N * (end2-start2) / 70.)))

    # Segment 3
    if x_max > 40:
        seg_len = x_max - 40
        M = max(2, int(ceil(seg_len * (Nafter/10))))
        parts.append(np.linspace(40, x_max, M))

    # Concatenate, sort, unique
    grid = np.unique(np.concatenate(parts))
    return grid * DAY
    


def Normalization_byhands(xdata, ydata, dydata, params_noN):
    xgrid = make_custom_xgrid(xdata=xdata, N=65, Nbefore=0.5, Nafter=2)
    params_dummy = params_noN.copy()
    params_dummy['logNorm'] = 0
    Fsx_fake = LC.Xray_Light_curve(tplot = xgrid, params = params_dummy, 
                              ifparallel = True, swift_only = True)   

    fspl = interp1d(xgrid, Fsx_fake)
    Fsx_fake = fspl(xdata)
    def to_fit(xda, norma):              
        return (norma * Fsx_fake - ydata)
    
    Ampl_e, dNsq = curve_fit(to_fit, xdata = xdata, 
                    ydata = np.zeros(xdata.size), p0 = (1e48,))   
    logAmpl_e = np.log10(Ampl_e[0])
    dNsq = dNsq[0][0]**0.5 / 10**logAmpl_e / np.log(10)
    return logAmpl_e, dNsq, Fsx_fake * 10**logAmpl_e

    
def int_an(g0, sm):
    return sm / (g0 - 1) * (g0**2 - 1)**0.5 
    
def Fit_MClike(xdata, ydata, dydata, params0, sigmas_for_params,
               Delta_for_impr = 1.5, Niter=10, tplot=None, cond_fit=None,
               full_output = True):  
    # the first attempt: calculate the first model from params0
    # and then start making the butterfly. If, with new parameters, we meet the chi2
    # better by DELTA (= 0.5 ???) then we change the best model to the 
    # new parameters and start baking the butterfly again
    # # xdata, ydata, dydata should all be full data
    if cond_fit is None:
        xfit, yfit, dyfit = xdata, ydata, dydata
    if cond_fit is not None:
        xfit, yfit, dyfit = [arr[cond_fit] for arr in (xdata, ydata, dydata)]
        
    Norm_opt, dNorm_opt, Fsx_opt  = Normalization_byhands(xfit, yfit, dyfit, params0)
    # print(Norm_opt)
    chi2_opt = np.sum(residuals_simple(yfit, dyfit, Fsx_opt)**2)
    print('initial chi2 = ', chi2_opt)
    Npars = 3
    dChi2 = DelChi2(Npars, 1)
    iteration = 0
    F_var = []
    F_var.append(Fsx_opt)
    params_fitted = params0.copy()
    params_fitted['logNorm'] = Norm_opt
    params_1sig = []
    params_1sig.append([params0['f_d'], params0['delta'], params0['Gamma']])

    while iteration < Niter:
        print(iteration)
        f_d0, delta0, Gamma0 = unpack("f_d delta Gamma", params_fitted)
        df, ddel, dG = sigmas_for_params
        f_d, delta, Gamma = np.random.normal(np.array([f_d0, delta0, Gamma0]),
                                             np.array([df, ddel, dG]))
        if f_d < 1. or delta < 5e-3 or Gamma < 1.:
            continue
        else:
            params_iter = params_fitted.copy()
            params_iter['f_d'] = f_d
            params_iter['delta'] = delta
            params_iter['Gamma'] = Gamma
            Norm_iter, dNorm_iter, Fsx_iter  = Normalization_byhands(xfit, yfit, dyfit, params_iter)
            params_iter['logNorm'] = Norm_iter
            
            chi2_iter = np.sum(residuals_simple(yfit, dyfit, Fsx_iter)**2)
            if chi2_opt - chi2_iter > Delta_for_impr:
                # if chi2 improvement > Delta_for_impr, we renew all the data
                # and start ``MCMC'' again
                params_fitted = params_iter.copy()
                chi2_opt = chi2_iter
                Fsx_opt = Fsx_iter
                # out_opt = out_iter
                F_var = []
                F_var.append(Fsx_opt)
                params_1sig = []
                params_1sig.append([f_d, delta, Gamma])
                print(f'--- found new params: chi2 = {chi2_opt:.3e} after {iteration} iterations')
                print(f_d, delta, Gamma)
                iteration = 0
                continue
            else:
                if chi2_iter - chi2_opt < dChi2:
                    # if no improvement but chi2 is only dChi2 worse, save this
                    # model as the one in 1-sigma range
                    print('found model in 1-sigma range')
                    F_var.append(Fsx_iter)
                    params_1sig.append([f_d, delta, Gamma])
                    iteration += 1
    print(iteration)
    params_1sig = np.array(params_1sig)
    # we obtained the parameters, now let's calculate the full model on a 
    # good grid of tplot that was provided
    if  len(F_var) in (0, 1): # if empty
        F_low, F_high = np.zeros(Fsx_opt.size), np.zeros(Fsx_opt.size)
        if full_output:
            out_opt = LC.Xray_Light_curve(tplot, params_fitted, True, False)
        else:
            out_opt = LC.Xray_Light_curve(tplot, params_fitted, True, True)
        par_fitted = np.array([params_fitted['f_d'], params_fitted['delta'], params_fitted['Gamma']])
        dpar_fitted = np.array(sigmas_for_params)
    else:
        F_var = np.array(F_var)
        F_low, F_high = np.zeros(Fsx_opt.size), np.zeros(Fsx_opt.size)
        F_low = np.array([ np.min(F_var[:, i]) for i in range(Fsx_opt.size) ])
        F_high = np.array([ np.max(F_var[:, i]) for i in range(Fsx_opt.size) ])
        par_fitted = np.mean(params_1sig, axis=0)
        dpar_fitted = np.std(params_1sig, axis=0)
        params_fitted['f_d'] = par_fitted[0]
        params_fitted['delta'] = par_fitted[1]
        params_fitted['Gamma'] = par_fitted[2]
        if full_output:
            out_opt = LC.Xray_Light_curve(tplot, params_fitted, True, False)            
        if not full_output:
            out_opt = LC.Xray_Light_curve(tplot, params_fitted, True, True)        
    

    return params_fitted, out_opt, Fsx_opt, F_low, F_high, par_fitted, dpar_fitted, params_1sig

    
# def Fit_PHN(xdata, ydata, dydata, params_noPd_noN, full_output = False):
#     # params_noN = params_noPd_noN.copy()
#     def to_fit(xdata, delta, enh_p):
#         params_noN_fake = params_noPd_noN.copy()
#         params_noN_fake['delta'] = delta
#         params_noN_fake['enh_p'] = [enh_p, ]
#         Norm, dNorm, Fsx_fake, out_  = Normalization_byhands(xdata, ydata, dydata, params_noN_fake, True)
#         # params_full = params_noN.copy()
#         # params_full['logNorm'] = Norm
#         # Fsx_fake = LC.Xray_Light_curve(xdata, params_full, True, 'f')
#         return (Fsx_fake - ydata)/dydata
#     popt, pcov = curve_fit(to_fit, p0 = (7e-2, 3), xdata = xdata,
#                               ydata = np.zeros(xdata.size),
#                               bounds = ([0, 0], [np.inf, np.inf]))
#     delta, enh_p = popt
#     ddelta, denh_p = np.diag(pcov)**0.5
#     params_noN = params_noPd_noN.copy()
#     params_noN['delta'] = delta
#     params_noN['enh_p'] = [enh_p, ]
#     Norm, dNorm, Fsx, output = Normalization_byhands(xdata, ydata, dydata, params_noN, True)
#     if full_output:
#         return delta, enh_p, Norm, ddelta, denh_p, dNorm, Fsx, output
#     if not full_output:
#         return delta, enh_p, Norm, ddelta, denh_p, dNorm

"""!!!!!! HERE THE YEAR IS DEFINED, ALL HAIL TO THE YEAR !!!!! """ #!!!

year = '101417_full'
# year = '101417'
# year = '2021'
# year = '2024'


""" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! """

da = GetObsData.getSingleLC(year, )
da_xray, da_tev = da['xray'], da['tev']

(tdata, dt, f, dfminus, dfplus,
 df, ind, dind_minus, dind_plus) = [da_xray[ind_] for ind_ in ('t', 'dt',
    'f', 'df_minus', 'df_plus', 'df', 'ind', 'dind_minus', 'dind_plus')]
                                                               
hd, hdd, hfl, hdfl = [da_tev[ind_] for ind_ in ('t', 'dt', 'f', 'df')]                                                               


# def Fit_GHN(xdata, ydata, dydata, params_noGH_noN, full_output = False):
#     # params_noN = params_noPd_noN.copy()
#     def to_fit(xdata, delta, Gamma):
#         params_noN_fake = params_noGH_noN.copy()
#         params_noN_fake['delta'] = delta
#         params_noN_fake['Gamma'] = Gamma
#         Norm, dNorm, Fsx_fake, out_  = Normalization_byhands(xdata, ydata, dydata, params_noN_fake, True)
#         # params_full = params_noN.copy()
#         # params_full['logNorm'] = Norm
#         # Fsx_fake = LC.Xray_Light_curve(xdata, params_full, True, 'f')
#         return (Fsx_fake - ydata)/dydata
#     popt, pcov = curve_fit(to_fit, p0 = (7e-2, 1.2), xdata = xdata,
#                               ydata = np.zeros(xdata.size),
#                               bounds = ([0, 0], [np.inf, np.inf]))
#     delta, Gamma = popt
#     ddelta, dGamma = np.diag(pcov)**0.5
#     params_noN = params_noGH_noN.copy()
#     params_noN['delta'] = delta
#     params_noN['Gamma'] = Gamma
#     Norm, dNorm, Fsx, output = Normalization_byhands(xdata, ydata, dydata, params_noN, True)
#     if full_output:
#         return delta, Gamma, Norm, ddelta, dGamma, dNorm, Fsx, output
#     if not full_output:
#         return delta, Gamma, Norm, ddelta, dGamma, dNorm    
    
# def Fit_fGHN(xdata, ydata, dydata, params_nofGH_noN, full_output = False):
#     # params_noN = params_noPd_noN.copy()
#     def to_fit(xdata, delta, Gamma, f_d):
#         params_noN_fake = params_nofGH_noN.copy()
#         params_noN_fake['delta'] = delta
#         params_noN_fake['Gamma'] = Gamma
#         params_noN_fake['f_d'] = f_d
        
#         Norm, dNorm, Fsx_fake, out_  = Normalization_byhands(xdata, ydata, dydata, params_noN_fake, True)
#         # params_full = params_noN.copy()
#         # params_full['logNorm'] = Norm
#         # Fsx_fake = LC.Xray_Light_curve(xdata, params_full, True, 'f')
#         return (Fsx_fake - ydata)/dydata
#     # p0_init = (2e-2, 1.2, 20)
#     p0_init = (2e-2, 5, 1e4)
#     popt, pcov = curve_fit(to_fit, p0 = p0_init, xdata = xdata,
#                               ydata = np.zeros(xdata.size),
#                               bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]))
#     delta, Gamma, f_d = popt
#     ddelta, dGamma, df_d = np.diag(pcov)**0.5
#     params_noN = params_nofGH_noN.copy()
#     params_noN['delta'] = delta
#     params_noN['Gamma'] = Gamma
#     params_noN['f_d'] = f_d
    
#     Norm, dNorm, Fsx, output = Normalization_byhands(xdata, ydata, dydata, params_noN, True)
#     if full_output:
#         return delta, Gamma, f_d,  Norm, ddelta, dGamma, df_d, dNorm, Fsx, output
#     if not full_output:
#         return delta, Gamma, f_d, Norm, ddelta, dGamma, df_d, dNorm    
    
    
f0 = 0   
Nplot = 100
tplot = make_custom_xgrid(tdata*DAY, Nplot, 1, 2) 

" this is with the cooling simple"
# par_templ101417 = {'f_w': 1., 'f_d': 46, 'f_p': 0.1,  't_enh_p': [0,],
#              't_enh_H': ['t2', ],  'enh_H':[1,],
#            'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
#           'beta': 1., 'np_disk': 2.7, 'Bx': 1e11, 'Bopt': 0., 'E0_e': 1.,
#           'Ecut_e': 5., 'p_e': 1.7, 'Topt': 3.3e4,
#           'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
#           'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
#           'if_boost': True, 'Gamma': 1.9, 'LoS_to_orb': 2.3, 'delta_pow': 3,
#           'MF_boost': False,  'delta':1.3e-2, 'enh_p': [1,],
#           'cooling': 'stat_mimic', 
#           # 'cooling': 'no', 
#           'eta_flow': 1,
#           'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
#           'simple': False, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}



par_1 = {'f_w': 1., 'f_d': 25, 'f_p': 0.1,  't_enh_p': [0,],
             't_enh_H': ['t2', ],  'enh_H':[1,],
           'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
          'beta': 1., 'np_disk': 2.7, 'Bx': 5e11, 'Bopt': 0., 'E0_e': 1.,
          'Ecut_e': 10., 'p_e': 1.7, 'Topt': 3.3e4,
          'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
          'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
          'if_boost': True, 'Gamma': 1.9, 'LoS_to_orb': 2.3, 'delta_pow': 3,
          'MF_boost': False,  'delta':1.4e-2, 'enh_p': [1,],
          'cooling': 'stat_mimic', 
          # 'cooling': 'no', 
          'eta_flow': 1,
          'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
          'simple': False, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}

par_2 = {'f_w': 1., 'f_d': 25, 'f_p': 0.1,  't_enh_p': [0,],
             't_enh_H': ['t2', ],  'enh_H':[1,],
           'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
          'beta': 1., 'np_disk': 2.7, 'Bx': 5e11, 'Bopt': 0., 'E0_e': 1.,
          'Ecut_e': 10., 'p_e': 1.7, 'Topt': 3.3e4,
          'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
          'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
          'if_boost': True, 'Gamma': 2.0, 'LoS_to_orb': 2.3, 'delta_pow': 3,
          'MF_boost': False,  'delta':1.3e-2, 'enh_p': [1,],
          'cooling': 'stat_mimic', 
          # 'cooling': 'no', 
          'eta_flow': 1,
          'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
          'simple': False, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}

par_3 = {'f_w': 1., 'f_d': 100, 'f_p': 0.1,  't_enh_p': [0,],
             't_enh_H': ['t2', ],  'enh_H':[1,],
           'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
          'beta': 1., 'np_disk': 2.7, 'Bx': 5e11, 'Bopt': 0., 'E0_e': 1.,
          'Ecut_e': 10., 'p_e': 1.7, 'Topt': 3.3e4,
          'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
          'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
          'if_boost': True, 'Gamma': 1.68, 'LoS_to_orb': 2.3, 'delta_pow': 3,
          'MF_boost': False,  'delta':1.5e-2, 'enh_p': [1,],
          'cooling': 'stat_mimic', 
          # 'cooling': 'no', 
          'eta_flow': 1,
          'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
          'simple': False, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}

par_4 = {'f_w': 1., 'f_d': 40, 'f_p': 0.1,  't_enh_p': [0,],
             't_enh_H': ['t2', ],  'enh_H':[1,],
           'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
          'beta': 1., 'np_disk': 2.7, 'Bx': 5e11, 'Bopt': 0., 'E0_e': 1.,
          'Ecut_e': 1., 'p_e': 1.7, 'Topt': 3.3e4,
          'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
          'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
          'if_boost': True, 'Gamma': 2.6, 'LoS_to_orb': 2.3, 'delta_pow': 3,
          'MF_boost': False,  'delta':1.4e-2, 'enh_p': [1,],
          'cooling': 'no', 
          # 'cooling': 'no', 
          'eta_flow': 1,
          'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
          'simple': False, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}

" this is with the cooling advection"
# par_templ101417 = {'f_w': 1., 'f_d': 20, 'f_p': 0.1,  't_enh_p': [0,],
#              't_enh_H': ['t2', ],  'enh_H':[1,],
#            'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
#           'beta': 1., 'np_disk': 2.7, 'Bx': 5e12, 'Bopt': 0., 'E0_e': 1.,
#           'Ecut_e': 50., 'p_e': 1.7, 'Topt': 3e4,
#           'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
#           'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
#           'if_boost': True, 'Gamma': 1.18, 'LoS_to_orb': 2.3, 'delta_pow': 3,
#           'MF_boost': False,  'delta':1e-2, 'enh_p': [1,],
#           'cooling': True, 'eta_flow': 1e20,
#           'eta_syn': 1, 'eta_IC': 1., 's_adv': True, 'lorentz_boost': True,
#           'simple': False}

" this is with without"
####### we can set enh_H = infty and then Gamma = 1.22 (approx)
# par_templ101417 = {'f_w': 1., 'f_d': 258, 'f_p': 0.1,  't_enh_p': [0,],
#              't_enh_H': ['t2', ],  'enh_H':[3,],
#            'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
#           'beta': 1., 'np_disk': 2.7, 'Bx': 1.5e12, 'Bopt': 0., 'E0_e': 1.,
#           'Ecut_e': 5., 'p_e': 1.7, 'Topt': 3e4,
#           'char_r': r_periastron, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
#           'r_pr': 'broken_pl', 'R_t': 5 * Ropt,
#           'if_boost': True, 'Gamma': 1.33, 'LoS_to_orb': 2.1, 'delta_pow': 3,
#           'MF_boost': False,  'delta':1e-2, 'enh_p': [1,],
#           'cooling': False, 'eta_flow': 0.1,
#           'eta_syn': 1, 'eta_IC': 1., 's_adv': False, 'lorentz_boost': False,
#                     'simple': True}

if year in ('101417', '101417_full'): #!!! to find it in the text easily do not delete dumbass


    """
    Monte-Carlo like fit
    """
    # conds = [np.where(np.logical_and(tdata < 40, tdata > -32)),
    #          np.where(np.logical_and(tdata < 110, tdata > -32)),
    #         np.where(np.logical_and(tdata < 110, tdata > -320)),
    #                            ]
    # filenames = ['fit_lc_all_m32_40_normMF_bigCut.pkl', 'fit_lc_all_m32_110_normMF_bigCut.pkl', 
    #              'fit_lc_all_m320_110_normMF_bigCut.pkl']  
    # par_inits = [par_1, par_2, par_3]
    
    conds = [np.where(np.logical_and(tdata < 33, tdata > -32)),
                               ]
    filenames = ['fit_lc_all_noCool_normMF_smallCut.pkl']
    par_inits = [par_4]
    
    for cond, filename, par_init in zip( conds, filenames, par_inits):
        params_early = par_init.copy()
        alpha, incl = unpack('alpha0 incl', params_early)
        err_tot = 0.5 * (dfminus + dfplus)
        # cond = np.where(np.logical_and(tdata < 40, tdata > -32))
        # cond = np.logical_or(tdata < -25, tdata > 80)
        # cond = np.logical_or(cond, np.logical_and(tdata > -7, tdata < 8))
        params = params_early.copy()
        params = params_early.copy()
    
        params, Fsx, Fsx_opt, F_low, F_high, pa, dpa, pa1s = Fit_MClike(xdata =  tdata*DAY,
        ydata=(f-f0), dydata=df, params0 = params_early,
        sigmas_for_params = [4., 1e-3, 7e-2], Delta_for_impr=0.5, Niter=51, tplot=tplot,
        cond_fit=cond, full_output = False)
        # print(Fsx.size)
        # print(F_low.size)
        
        f_d, delta, Gamma = pa
        df_d, ddelta, dGamma = dpa
        tdata_fit = tdata[cond]*DAY
        # print(tdata_fit.size)
    
        # (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,
        #     Bsstar_scal, Bspuls_scal, Bstot_scal, Ftev, 
        #     Fi, G_ind) = out_opt
        # Fsx = 
        Fsx += f0
        f_spl = interp1d(tplot, Fsx)
        ress = residuals_simple(f, df, f_spl(tdata*DAY))
        dind = 0.5 * (dind_minus + dind_plus)
        # ind_ress = residuals_simple(ind, dind, G_ind)
        chisq = np.sum(ress**2) / (tdata.size - 4)
        tosave = {'params': params, 'Fsx': Fsx, 'Fsx_opt': Fsx_opt, 'F_low': F_low,
              'F_high': F_high, 'pa': pa, 'dpa': dpa, 'tdata': tdata,
              'tdata_fit': tdata_fit, 'ress': ress, 'par_1sigma': pa1s, 'tplot': tplot}
        
        with open(Path(Path.cwd(), 'Outputs', filename), 'wb') as file_tosave:
            pickle.dump(tosave, file_tosave)
        # tplot = tdata * DAY #!!!
        # Ftev = Ftev #* 10**Norm1
    """
    only norm-fit
    """
    # cond = conds[0]
    # params_early = par_4
    # params = params_early.copy()
    # logNorm, dlogNorm, Fsx = Normalization_byhands(xdata = tdata[cond]*DAY, 
    #                 ydata=f[cond]-f0, dydata=df[cond], params_noN=params_early)
    # alpha, incl = unpack('alpha0 incl', params_early)
    # print('N = ', logNorm, ' dN = ', dlogNorm)
    # params['logNorm'] = logNorm
    # delta, Gamma, Norm1, ddelta, dGamma, dNorm, f_d, df_d = (params['delta'],  params['Gamma'],
    #             logNorm, 0, 0, dlogNorm, params['f_d'], 0)
    # Norm, dNorm = Norm1, dNorm
    # params['abs_gg'] = True

    """
     dP, delta, N - fit
    """
    # delta1, enh_p1, Norm1, ddelta1, denh_p1, dNorm1, Fsx1, output = Fit_PHN(tdata[cond]*DAY,
    #                 f[cond]-f0, df[cond], params_early, True)
    # params = params_early.copy()
    # Norm1, dNorm1 = Norm1[0], dNorm1[0][0]
    # print('N = ', Norm1, ' dN = ', dNorm1)
    # print('delta = ', delta1, ' d delta = ', ddelta1)
    # print('enh_p = ', enh_p1, ' d enh_p = ', denh_p1)
    # params['logNorm'] = Norm1
    # params['delta'] = delta1
    # params['enh_p'] = [enh_p1, ]
    
    # (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,
    #         Bsstar_scal, Bspuls_scal, Bstot_scal, Ftev, 
    #         Fi, G_ind) = LC.Xray_Light_curve(tplot, params, True, False)
    # Fsx = LC.Xray_Light_curve(tplot, params, True, True)
    # Ftev, G_ind = np.zeros(Fsx.size), np.zeros(Fsx.size)
    # # print(G_ind)
    # F_low = Fsx; F_high= Fsx
    # Fsx += f0
    # Ftev = Ftev*1e11
    # spl_ = interp1d(tdata[cond]*DAY, Fsx)
    # # ress = residuals(tdata[cond], f[cond], df[cond], tplot[cond]/DAY, spl_(tplot[cond]))
    # ress = residuals_simple(f[cond], df[cond], Fsx)
    # dind = 0.5 * (dind_minus + dind_plus)
    # tplot = tdata[cond] * DAY#!!!
    # # ind_ress = residuals(tdata, ind, dind, tplot/DAY, G_ind)
    # chisq = np.sum(ress**2) / (tdata.size - 3)
           
t_disk1, t_disk2 = Orb.times_of_disk_passage(alpha, incl)

fontsize = 14
if year in ('2021', '2024'):
    rows = 3
    cols = 1
    fig,ax = plt.subplots(rows, cols, sharex='col', sharey='row',
                          gridspec_kw={'height_ratios': [3, 1, 3]})
else:
    rows = 2
    cols = 1
    fig,ax = plt.subplots(rows, cols, sharex='col', sharey='row',
                          gridspec_kw={'height_ratios': [3, 1]})

for row in range(rows):
    for col in range(cols):
        ax_ = ax[row]
        # if row == 2:
        #     ax_.plot(tplot / DAY, Ftev, c = 'r')
        #     ax_.errorbar(hd, hfl, xerr = hdd, yerr = hdfl, c='k', fmt='o')
        if row == 0:
            ax_.set_title('year '+ year, fontsize=fontsize)

            if year in ('101417_full', '101417', '2021', '2024'):
                label1 = r'$f_d = %s \pm %s, \Gamma = %s \pm %s, 100~\delta = %s \pm %s, \chi^2=%s$'%(round(f_d, 2),
                round(df_d, 2), round(Gamma, 2), round(dGamma, 2),
                round(100*delta, 2), round(100*ddelta, 3), round(chisq, 2))

                ax_.plot(tplot / DAY, Fsx,  label = label1, c='b')
                # ax_.fill_between(tdata_fit / DAY, F_low, F_high, color='b', alpha=0.2)

            if year in ('101417',):

                label1=None
                ax_.plot(tplot / DAY, Fsx, c='b', label = label1)

            ax_.errorbar(tdata, f, yerr=(dfminus, dfplus), fmt='o', c='k',
                         xerr = dt)
            ax_.set_yscale('linear')
            # ax_.set_yscale('log')
            ax_.set_ylabel(r'$F_{\rm X-ray}$, CGS', fontsize = fontsize)
            ax_.legend()
                
        if row == 1:

            if year in ('101417_full', '101417'):
                ax_.errorbar(tdata, ress, yerr = 1, fmt='o', c='k')
            ax_.axhline(y=0, c='lime', alpha=0.5)
            ax_.axhline(y=2, c='g', alpha=0.5)
            ax_.axhline(y=-2, c='g', alpha=0.5)
            ax_.axhline(y=5, c='r', alpha=1)
            ax_.axhline(y=-5, c='r', alpha=1)
            ax_.set_ylabel(r'$\chi$')
            # ax_.set_ylabel(r'(Obs - Calc)/Err', fontsize = fontsize)
        # if row == 2:
        #     ax_.errorbar(hd)
        # if row == 3:
        #     ax_.errorbar(tdata, ind, yerr=(dind_minus, dind_plus), fmt='o', 
        #                  color='k')
        #     # print(G_ind)
        #     ax_.plot(tplot/DAY, G_ind, c='b')
        #     ax_.set_ylim(1., 2.6)

        if row == rows - 1:
            ax_.set_xlabel(r'$t - t_p$, Days', fontsize = fontsize)
        ax_.axvline(x=t_disk1/DAY)
        ax_.axvline(x=t_disk2/DAY)
        ax_.set_xlim(left=np.min(tdata)-5, right=np.max(tdata)+5)
        
# fig.show()         
fig.subplots_adjust(hspace=0)

print('----- took %s sec ----- '%(round(time.time() - start_time, 2)))
    # plt.show()