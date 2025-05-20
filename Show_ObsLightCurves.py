import numpy as np
# from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
# import astropy.units as u
import matplotlib.pyplot as plt
# from scipy.integrate import trapezoid
# from scipy.optimize import brentq, root, fsolve, least_squares, minimize
# from scipy.optimize import curve_fit
# from pathlib import Path

# import Orbit as Orb
import GetObsData    



year = '101417_full'
# year = '2017'

da = GetObsData.getSingleLC(year)
da_xray, da_tev = da['xray'], da['tev']

(tdata, dt, f, dfminus, dfplus,
 df, ind, dind_minus, dind_plus) = [da_xray[ind_] for ind_ in ('t', 'dt',
    'f', 'df_minus', 'df_plus', 'df', 'ind', 'dind_minus', 'dind_plus')]
                                                               
hd, hdd, hfl, hdfl = [da_tev[ind_] for ind_ in ('t', 'dt', 'f', 'df')] 

fig, ax = plt.subplots(nrows = 3, ncols=1, sharex=True)

ax[0].errorbar(tdata, f, xerr=dt, yerr=(dfminus, dfplus), fmt='o', label = 'Swift 0.3-10 keV flux')
ax[1].errorbar(tdata, ind, xerr=dt, yerr=(dind_minus, dind_plus), fmt='o', label = 'Swift 0.3-10 keV index')
ax[2].errorbar(hd, hfl, xerr=hdd, yerr=hdfl, fmt='o', label = 'HESS 0.4-100 TeV flux')
