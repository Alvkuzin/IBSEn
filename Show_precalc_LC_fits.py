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
from brokenaxes import brokenaxes
import matplotlib.gridspec as gridspec

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
Mopt = 24
Ropt = 10 * Rsun
Mx = 1.4
GM = G * (Mopt + Mx) * 2e33
P = 1236.724526
Torb = P * DAY
a = (Torb**2 * GM / 4 / pi**2)**(1/3)
e = 0.87
r_periastron = a * (1 - e)
D_system = 2.4e3 * 206265 * AU
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0


def unpack(query, dictat):
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def residuals_simple(yobs, dyobs, ytheor):
    res = (yobs - ytheor) / dyobs
    return res


def residuals(xobs, yobs, dyobs, xtheor, ytheor):
    sort_args = np.argsort(xtheor)
    xtheor_, ytheor_ = [ar[sort_args] for ar in (xtheor, ytheor)]
    spline_ = interp1d(xtheor_, ytheor_)
    # yth_in_obs = splev(xobs, spline_)
    res = (yobs - spline_(xobs)) / dyobs
    return res

year = '101417_full'
da = GetObsData.getSingleLC(year, )
da_xray, da_tev = da['xray'], da['tev']

(tdata, dt, f, dfminus, dfplus,
 df, ind, dind_minus, dind_plus) = [da_xray[ind_] for ind_ in ('t', 'dt',
    'f', 'df_minus', 'df_plus', 'df', 'ind', 'dind_minus', 'dind_plus')]
                                                               
hd, hdd, hfl, hdfl = [da_tev[ind_] for ind_ in ('t', 'dt', 'f', 'df')]       

fontsize = 14

rows = 2
cols = 1
# fig,ax = plt.subplots(rows, cols, sharex='col', sharey='row',
#                       gridspec_kw={'height_ratios': [3, 1]})      
fig = plt.figure()
gs = gridspec.GridSpec(nrows=rows, ncols=cols, height_ratios=[3, 1], hspace=0.2) 

ax0 = brokenaxes(
    xlims=((np.min(tdata)-5, -115), (-35, np.max(tdata)+5)),
    subplot_spec=gs[0], wspace=3e-2
)

ax1 = brokenaxes(
    xlims=((np.min(tdata)-5, -115), (-35, np.max(tdata)+5)),
    subplot_spec=gs[1], wspace=3e-2
)
to_show = 'normMF_normCut'
pref = 'fit_lc_all_'
#   
# filenames = ['fit_lc_all_m32_40_normMF_bigCut.pkl', 'fit_lc_all_m32_110_normMF_bigCut.pkl', 
#              'fit_lc_all_m320_110_normMF_bigCut.pkl']                         
# filenames = ['fit_lc_all_m32_40_normMF_smallCut.pkl', 'fit_lc_all_m32_110_normMF_smallCut.pkl', 
#              'fit_lc_all_m320_110_normMF_smallCut.pkl']
# filenames = ['fit_lc_all_m32_40_normMF.pkl', 'fit_lc_all_m32_110_normMF.pkl', 
#              'fit_lc_all_m320_110_normMF.pkl']    
# filenames = [pref + x + to_show + '.pkl' for x in ('m32_40_', 'm32_110_', 'm320_110_')]
filenames = [pref + x + to_show + '.pkl' for x in ( 'm320_110_',)]

filenames.append(pref + 'noCool_'+ to_show + '.pkl')    
# colors = ['r', 'g', 'b', 'm']
colors = [ 'b', 'm']

for  filename, color in  zip(filenames, colors):
    with open(Path(Path.cwd(), 'Outputs', filename), 'rb') as file_load:
        loaded_results = pickle.load(file_load)
        
    (params, Fsx, Fsx_opt, F_low, F_high, pa, dpa, tdata, tdata_fit, 
     ress, pa1s, tplot) =  [loaded_results[i_] for i_ in ( 'params', 
        'Fsx', 'Fsx_opt', 'F_low',
          'F_high', 'pa', 'dpa', 'tdata',
          'tdata_fit', 'ress', 'par_1sigma', 'tplot') ]
        
    alpha, incl = unpack('alpha0 incl', params)
    err_tot = 0.5 * (dfminus + dfplus)


    f_d, delta, Gamma = pa
    df_d, ddelta, dGamma = dpa
    # tdata_fit = tdata[cond]*DAY
    # print(tdata_fit.size)

    # (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,
    #     Bsstar_scal, Bspuls_scal, Bstot_scal, Ftev, 
    #     Fi, G_ind) = out_opt
    # Fsx = 
    # Fsx += f0
    f_spl = interp1d(tplot, Fsx)
    ress = residuals_simple(f, df, f_spl(tdata))
    dind = 0.5 * (dind_minus + dind_plus)
    # ind_ress = residuals_simple(ind, dind, G_ind)
    # chisq = np.sum(ress[np.logical_and(tdata>-30, tdata<100)]**2) / (tdata.size - 4)

    # tplot = tdata * DAY #!!!
    # Ftev = Ftev #* 10**Norm1
 
    ress = residuals(tdata, f, df, tplot/DAY, Fsx)
    dind = 0.5 * (dind_minus + dind_plus)
    # ind_ress = residuals(tdata, ind, dind, tplot/DAY, G_ind)
    chisq = np.sum(ress[np.logical_and(tdata>-30, tdata<100)]**2) / (tdata.size - 3)
       
    t_disk1, t_disk2 = Orb.times_of_disk_passage(alpha, incl)
    
    # for row in range(rows):
    #     for col in range(cols):

    # if row == 0:
    # ax0.set_title('year '+ year, fontsize=fontsize)
    ax0.set_title(to_show, fontsize=fontsize)

    if year in ('101417_full', '101417', '2021', '2024'):
        label1 = r'$f_d = %s \pm %s, \Gamma = %s \pm %s, 100~\delta = %s \pm %s, \chi^2=%s$'%(round(f_d, 2),
        round(df_d, 2), round(Gamma, 2), round(dGamma, 2),
        round(100*delta, 2), round(100*ddelta, 3), round(chisq, 2))

        ax0.plot(tplot / DAY, Fsx,  label = label1, c=color)
        ax0.fill_between(tdata_fit / DAY, F_low, F_high, color=color, alpha=0.3)

        # if year in ('101417',):

        #     label1=None
        #     ax_.plot(tplot / DAY, Fsx, c=color, label = label1)

        ax0.errorbar(tdata, f, yerr=(dfminus, dfplus), fmt='o', c='k',
                     xerr = dt)
        # ax_.set_yscale('linear')
        # # ax_.set_yscale('log')
        ax0.set_ylabel(r'$F_{\rm X-ray}$, CGS', fontsize = fontsize)
        ax0.legend(loc = 'upper left')
        ax0.set_ylim(-1e-12, 6e-11)
            
    # if row == 1:

    if year in ('101417_full', '101417'):
        ax1.errorbar(tdata, ress, yerr = 1, fmt='o', c=color)
    ax1.axhline(y=0, c='lime', alpha=0.5)
    ax1.axhline(y=2, c='g', alpha=0.5)
    ax1.axhline(y=-2, c='g', alpha=0.5)
    ax1.axhline(y=5, c='r', alpha=1)
    ax1.axhline(y=-5, c='r', alpha=1)
    ax1.set_ylabel(r'$\chi$')
        # ax_.set_ylabel(r'(Obs - Calc)/Err', fontsize = fontsize)
    # if row == 2:
    #     ax_.errorbar(hd)
    # if row == 3:
    #     ax_.errorbar(tdata, ind, yerr=(dind_minus, dind_plus), fmt='o', 
    #                  color='k')
    #     # print(G_ind)
    #     ax_.plot(tplot/DAY, G_ind, c='b')
    #     ax_.set_ylim(1., 2.6)

    # if row == rows - 1:
    ax1.set_xlabel(r'$t - t_p$, Days', fontsize = fontsize)
    ax0.axvline(x=t_disk1/DAY)
    ax0.axvline(x=t_disk2/DAY)
    ax1.axvline(x=t_disk1/DAY)
    ax1.axvline(x=t_disk2/DAY)
            # ax_.set_xlim(left=np.min(tdata)-5, right=np.max(tdata)+5)
            
    # fig.show()         
    # fig.subplots_adjust(hspace=0)
    
    print('----- took %s sec ----- '%(round(time.time() - start_time, 2)))
    plt.show()