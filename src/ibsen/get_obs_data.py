# import astropy.units as u
import numpy as np
# from numpy import pi, sin, cos
# import matplotlib.pyplot as plt
# from scipy.interpolate import splev, splrep, make_smoothing_spline, interp1d
# from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import curve_fit
# import xarray as xr
from pathlib import Path
from astropy import constants as const

# import Orbit as Orb

G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

def get_parameters(sys_name):
    """
    Quickly access some PSRB orbital parameters: 
    orbital period T [s], eccentricity e, total mass M [g],  
    distance to the system D [cm], star radius Ropt [cm].

    Returns : dictionary
    -------
    'e', 'M' [g], 'D' [cm], 'Ropt' [cm], 'T' [s]

    """

    if sys_name == 'psrb': # Negueruela et al 2011
        Torb_here = 1236.724526*DAY; e_here = 0.874; Topt_here = 3.3e4
        Mopt = 31 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 9.2 * R_SOLAR
        nu_los = 2.3; incl_los= 22 / 180 * np.pi
    elif sys_name == 'rb':
        Torb_here = 0.5*DAY; e_here = 0; Topt_here = 3e3 
        Mopt = 0.5  * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1e3 * PARSEC; Ropt_here = 0.3 * R_SOLAR
        nu_los = 0; incl_los=np.pi/4
    elif sys_name == 'bw':
        Torb_here = 0.1*DAY; e_here = 0; Topt_here = 1e3 
        Mopt = 0.1  * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1e3 * PARSEC; Ropt_here = 0.01 * R_SOLAR
        nu_los = 0.;  incl_los=np.pi/4
    elif sys_name == 'test':
        Torb_here = 100.0*DAY; e_here = 0.5; Topt_here = 3.0e4
        Mopt = 30 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 10.0 * R_SOLAR
        nu_los = 135/180*np.pi; incl_los= 45 / 180 * np.pi
    elif sys_name == 'ls5039': # Casares et al  2005 # !!! dist?
        Torb_here = 3.906*DAY; e_here = 0.35; Topt_here = 3.9e4
        Mopt = 22.9 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 9.3 * R_SOLAR
        nu_los = (270-45.8)/180*np.pi; incl_los= 60 / 180 * np.pi
    elif sys_name == 'psrj2032': # Ho et al 2017, Lyne et al 2015 # !!! dist?
        Torb_here = 16500*DAY; e_here = 0.95; Topt_here = 2e4
        Mopt = 15 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 10 * R_SOLAR
        nu_los = (270-40)/180*np.pi; incl_los= 30 / 180 * np.pi
    elif sys_name == 'ls61': # Chernyakova et al 2020 # !!! dist? nu_los? Ropt?
        Torb_here = 26.5*DAY; e_here = 0.537; Topt_here = 2e4
        Mopt = 12 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 10 * R_SOLAR
        nu_los = (270-40)/180*np.pi; incl_los= 30 / 180 * np.pi
        
        
    else:
        raise ValueError(f'Unknown name: {sys_name}')
    

    res = { 'e': e_here, 'M': M_here, 'D': D_here, 'Ropt': Ropt_here, 
           'T': Torb_here, 'Topt': Topt_here, 'Mopt': Mopt, 'M_ns': M_ns,
           'nu_los': nu_los, 'incl_los': incl_los}
    return res


_here = Path(__file__).parent          
_tabdata = _here / "absorb_tab" 

# obs_fold = Path(Path.cwd(), 'ObsData')
# exposure_S = np.array([576.8878,2157.656, 2135.181, 2614.655, 1278.609, 1917.999])
# exposure_I = np.array([47857.7861796112,34018.4876335561, 43982.9039376908, 46504.9046225975,
#               41991.5653142337, 43802.487989002])
# # print(np.sum(exposure_I))
# eff_S = exposure_S / np.mean(exposure_S)
# eff_I = exposure_I / np.mean(exposure_I)
# # print(eff_S)
# # print(eff_I)

# def const(x, C):
#     return C

# def Extr_sp(path, num = 5):
#     with open(path, "r") as file:
#         raw_data = file.read().strip().split("#")  # Split by "#"
#     datasets = [block.strip() for block in raw_data if block.strip()]
#     data_arrays = [np.loadtxt(dataset.splitlines()) for dataset in datasets]
#     Da = {}
#     if num == 5:
#         for i in range(1, 6):
#             Da[f"Sp{i}"] = np.concatenate((data_arrays[2*(i-1)], data_arrays[2*i-1]))
#         S1, S2, S3, S4, S5 = Da['Sp1'], Da['Sp2'], Da['Sp3'], Da['Sp4'], Da['Sp5']
#         return S1, S2, S3, S4, S5
#     if num == 6:
#         for i in range(1, 7):
#             Da[f"Sp{i}"] = np.concatenate((data_arrays[2*(i-1)], data_arrays[2*i-1]))
#         S1, S2, S3, S4, S5, S6 = Da['Sp1'], Da['Sp2'], Da['Sp3'], Da['Sp4'], Da['Sp5'], Da['Sp6']
#         return S1, S2, S3, S4, S5, S6

# def Stack_int(S1, S2, S3, S4, S5, S6):
#     i1, i2, i3, i4, i5, i6 = (S1[np.where(S1[:, 0] > 20.), :][0],
#                               S2[np.where(S2[:, 0] > 20.), :][0],
#                               S3[np.where(S3[:, 0] > 20.), :][0],
#                               S4[np.where(S4[:, 0] > 20.), :][0],
#                               S5[np.where(S5[:, 0] > 20.), :][0],
#                               S6[np.where(S6[:, 0] > 20.), :][0])
#     ys = np.zeros((5, 6))
#     dys = np.zeros((5, 6))
#     xs = i1[:, 0]
#     dxs= i1[:, 1]
#     for i_ch in range(5):
#         ys[i_ch, :] = np.array([i1[i_ch, 2], i2[i_ch, 2], i3[i_ch, 2], i4[i_ch, 2], i5[i_ch, 2], i6[i_ch, 2] ])
#         dys[i_ch, :] = np.array([i1[i_ch, 3], i2[i_ch, 3], i3[i_ch, 3], i4[i_ch, 3], i5[i_ch, 3], i6[i_ch, 3] ])
#     ys_fit = np.zeros(5)
#     dys_fit = np.zeros(5)
#     for i_ch in range(5):
#         c, dc = curve_fit(f=const, xdata=1, ydata=ys[i_ch, :], sigma=dys[i_ch, :]/eff_S,
#                           absolute_sigma=False)
#         # print(c, dc)
#         ys_fit[i_ch] = c[0]
#         dys_fit[i_ch] = dc[0][0]
#     return xs, dxs, ys_fit, dys_fit**0.5

# S1, S2, S3, S4, S5, S6 = Extr_sp(Path(obs_fold, 'six_sp.qdp'), 6)

# # Integral data
# eI, deI, sedI, dsedI = Stack_int(S1, S2, S3, S4, S5, S6)
# data_Integral24 = {'e': eI, 'de': deI, 'sed': sedI, 'dsed': dsedI,
#                    'de_minus': deI, 'de_plus': deI, 'dsed_minus': dsedI,
#                    'dsed_plus': dsedI}

# da_S_st = np.genfromtxt(Path(obs_fold, "stack_mine.qdp"))

# # Swift data
# eS, deS, sedS, dsedS, thS = da_S_st[:-1, 0], da_S_st[:-1, 1], da_S_st[:-1, 2], da_S_st[:-1, 3], da_S_st[:-1, 4]
# data_Swift24 = {'e': eS, 'de': deS, 'sed': sedS, 'dsed': dsedS,
#                 'de_minus': deS, 'de_plus': deS, 'dsed_minus': dsedS,
#                 'dsed_plus': dsedS}


# """
# ###############################################################################
# TeV data from 2021
# ###############################################################################
# """


# da_H = np.genfromtxt(Path(obs_fold, "Hess_spec21.txt"), delimiter=', ')
# eH, sedH = da_H[:, 0] * 1e9, da_H[:, 1] /1.6e-9

# # print(eH)
# deH = np.zeros(eH.size)
# deH[0] = 0.5 * (eH[1] - eH[0])
# for i in range(1, eH.size):
#     deH[i] = np.abs(eH[i] - eH[i-1]) - deH[i-1]
# dsedH = sedH/3
# data_HESS21_full = {'e': eH, 'de': deH, 'sed': sedH, 'dsed': dsedH,
#                 'de_minus': deH, 'de_plus': deH, 'dsed_minus': dsedH,
#                 'dsed_plus': dsedH}

# """
# ###############################################################################
# TeV data from 2021 by me for days 25-36 
# ###############################################################################
# """


# da_H1 = np.genfromtxt(Path(obs_fold, "spec_21_18d_35d.spec"))
# eH1, FlH1 = da_H1[:, 0] * 1e9, da_H1[:, 3] / 1e9
# sedH1 = FlH1 * eH1**2
# # print(eH)
# deH1 = np.zeros(eH1.size)
# deH1[0] = 0.5 * (eH1[1] - eH1[0])
# for i in range(1, eH1.size):
#     deH1[i] = np.abs(eH1[i] - eH1[i-1]) - deH1[i-1]
# dFl_min, dFl_pl = da_H1[:, 4] / 1e9, da_H1[:, 5] / 1e9
# dsedH1 = (dFl_min + dFl_pl) / 2 * eH1**2

# data_HESS21_21_36_me = {'e': eH1, 'de': deH1, 'sed': sedH1, 'dsed': dsedH1,
#                 'de_minus': deH1, 'de_plus': deH1, 'dsed_minus': dsedH1,
#                 'dsed_plus': dsedH1}

# """
# ###############################################################################
# TeV data from 2024 ((((((NEU NEU NEU NEU NEU NEU ACHTUNG ACHTUNG))))))
# ###############################################################################
# """

# # da_H4 = np.genfromtxt(Path(obs_fold, "out.spec"))
# # eH4, FlH4 = da_H4[:, 0] * 1e9, da_H4[:, 3] / 1e9
# # sedH4 = FlH4 * eH4**2
# # # print(eH)
# # deH4 = np.zeros(eH4.size)
# # deH4[0] = 0.5 * (eH4[1] - eH4[0])
# # for i in range(1, eH4.size):
# #     deH4[i] = np.abs(eH4[i] - eH4[i-1]) - deH4[i-1]
# # dFl_min, dFl_pl = da_H4[:, 4] / 1e9, da_H4[:, 5] / 1e9
# # dsedH4 = (dFl_min + dFl_pl) / 2 * eH4**2

# # data_HESS24_full = {'e': eH4, 'de': deH4, 'sed': sedH4, 'dsed': dsedH4,
# #                 'de_minus': deH4, 'de_plus': deH4, 'dsed_minus': dsedH4,
# #                 'dsed_plus': dsedH4}


# """
# ###############################################################################
# Red Fermi region
# ###############################################################################
# """


# eF = np.logspace(np.log10(7e4), np.log10(2e6), 100)
# sedF_low = 2e-11 * (eF / 1e5)**(-0.7) /1.6e-9
# sedF_high = 4.2e-11 * (eF / 1e5)**(-0.7) /1.6e-9
# sedF = 0.5 * (sedF_low + sedF_high)

# data_RedFermi = {'e': eF, 'de': np.zeros(eF.size), 'sed': sedF, 
#                  'dsed': 0.5*(sedF_high - sedF_low),
#                 'de_minus': np.zeros(eF.size), 'de_plus': np.zeros(eF.size), 
#                 'dsed_minus': sedF - sedF_low,
#                 'dsed_plus': sedF_high - sedF}


# """
# ###############################################################################
# Blue Fermi region
# ###############################################################################
# """

# data_blusedrmi = np.genfromtxt(Path(obs_fold, "Fermi_spec_ch.txt"),
#                                delimiter = ', ', skip_header=0)
# N = int(np.shape(data_blusedrmi)[0] / 2)
# eFblue, sedFblue_low, sedFblue_high = np.zeros((3, N))
# for i in range(N):
#     eFblue[i] = data_blusedrmi[int(2 * i + 1), 0]/1e3
    
#     sedFblue_low[i] = data_blusedrmi[int(2 * i), 1]/1.6e-9
#     sedFblue_high[i] = data_blusedrmi[int(2 * i + 1), 1]/1.6e-9
    
# sedFblue = (sedFblue_low + sedFblue_high)/2

# data_BlueFermi = {'e': eFblue, 'de': np.zeros(eFblue.size), 'sed': sedFblue, 
#                  'dsed': 0.5*(sedFblue_high - sedFblue_low),
#                 'de_minus': np.zeros(eFblue.size), 'de_plus': np.zeros(eFblue.size), 
#                 'dsed_minus': sedFblue - sedFblue_low,
#                 'dsed_plus': sedFblue_high - sedFblue}


# """
# ###############################################################################
# Green Fermi (flares)
# ###############################################################################
# """

        

# data_greenFermi = np.genfromtxt(Path(obs_fold, "Fermi_specGreen_ch.txt"),
#                                delimiter = ', ', skip_header=0)
# N = int(np.shape(data_greenFermi)[0] / 5)
# eFgr, sedFgr, sedFgr_low, sedFgr_high, eFgr_left, eFgr_right = np.zeros((6, N))
# for i in range(N):
#     eFgr[i] = data_greenFermi[int(5 * i + 1), 0]/1e3
#     eFgr_left[i] = data_greenFermi[int(5 * i + 3), 0]/1e3
#     eFgr_right[i] = data_greenFermi[int(5 * i + 4), 0]/1e3
    
#     sedFgr[i] = data_greenFermi[int(5 * i+1), 1]/1.6e-9
#     sedFgr_low[i] = data_greenFermi[int(5 * i), 1]/1.6e-9
#     sedFgr_high[i] = data_greenFermi[int(5 * i + 2), 1]/1.6e-9
# deFgr_left = eFgr-eFgr_left
# deFgr_right = eFgr_right-eFgr
# deFgr = 0.5 * (deFgr_left + deFgr_right)
# dsedFgr_low = sedFgr-sedFgr_low
# dsedFgr_high = sedFgr_high-sedFgr
# arg_norm = np.where(sedFgr_low < sedFgr)
# arg_upper = np.where(sedFgr_low > sedFgr)
# dsedFgr_low[arg_upper] = sedFgr[arg_upper]/2
# dsedFgr_high[arg_upper] = sedFgr[arg_upper]/2
# uplims = np.zeros(eFgr.size)
# uplims[arg_upper] = False
# uplims[arg_upper] = True

# data_GreenFermi = {'e': eFgr, 'de': deFgr, 'sed': sedFgr, 
#                  'dsed': 0.5*(sedFgr_high - sedFgr_low),
#                 'de_minus': deFgr_left, 'de_plus': deFgr_right, 
#                 'dsed_minus': dsedFgr_low,
#                 'dsed_plus': dsedFgr_high, 'uplimits': uplims}




# def getSpecData():
#     res_spec = {
#         'spec_I24': data_Integral24,
#         'spec_S24': data_Swift24,
#         'spec_Hess21': data_HESS21_full, 
#         # 'spec_Hess24': data_HESS24_full,
#         'spec_Hess21_25_36': data_HESS21_21_36_me,
#         'spec_Fermi24_m20_0': data_BlueFermi,
#         'spec_Fermi24_0_20': data_RedFermi, 
#         'spec_Fermi24_flares': data_GreenFermi
#         }
#     return res_spec


# def getSingleLC(year, cond  = None):
#     P = 1236.724526

#     prefix = '_' + year + ''
#     # if year != 'all':
#     name_xray_file = 'light_curve%s.txt'%(prefix,)
#     # if year == 'all':
#     #     name_xray_file = 'Stacked_Xray.dat'
#     path = Path(Path.cwd(), 'ObsData')
#     data = np.genfromtxt(Path(path, name_xray_file), delimiter = ' ',
#                         skip_header = False, names=True)

#     t_p21 = 59254.867359
#     t_p24 = t_p21 + P
#     if year == '2024':
#         t00 = t_p24    
#     if year == '2021':
#         t00 = t_p21    
#     if year == '2017':
#         t00 = t_p21 - P
#     if year == '2014':
#         t00 = t_p21 - 2 * P
#     if year == '2010':
#         t00 = t_p21 - 3 * P
#     if year == '2007':
#         t00 = t_p21 - 4 * P

#     if year in ('all', '101417', '2124', '101417_full'):
#         tdata = data['t'] 

        
#     # if year in ('', '2024'):
#     #     hd1, hd2, hfl, hdfl = np.loadtxt(Path(path, 'lc_ct14.night'),
#     #                                      usecols=[0,1,2,6], unpack=True)
#     #     hd1 += - (t_p21 + P)
#     #     hd2 += - (t_p21 + P)
#     #     hd = 0.5*(hd1 + hd2)
#     #     hdd = 0.5*np.abs(hd1 - hd2)
        
#     if year == '2021':
#         # Nt1, Nt2,Nf, Nf1, Nf2 = np.loadtxt(Path(path, 'Nicer2021.txt'),
#         #                                    usecols=[0,1,8,9,10], unpack=True)
#         # Nt = 0.5*(Nt1+Nt2)
#         # Nt += -t_p21
#         # Nf = 10**Nf
#         # Nf1 = 10**Nf1
#         # Nf2 = 10**Nf2
#         # Ndf = 0.5*(Nf2-Nf1)
#         # Ndt = 0.5 * (Nt2 - Nt1)
#         ###########################################################################
#         hd1, hd2, hfl, hdfl = np.loadtxt(Path(path, 'lc_ct14_2021.night'),
#                                          usecols=[0,1,2,6], unpack=True)
#         hd1 += -t_p21
#         hd2 += -t_p21
#         hd = 0.5*(hd1 + hd2)
#         hdd = 0.5*np.abs(hd1 - hd2)
        
#     if year in ('101417', '101417_full', '2007', '2010', '2014', '2017'):
#         data_hessLC = np.genfromtxt(Path(path, 'StackedHessLC.txt'),
#                                        delimiter = ', ', skip_header=1)
#         N = int(np.shape(data_hessLC)[0] / 3)
#         tStack, FStack, FStack_high, FStack_low  = np.zeros((4, N))
#         for i in range(N):
#             tStack[i] = data_hessLC[int(3 * i + 1), 0]
#             FStack_low[i] = data_hessLC[int(3 * i), 1]
#             FStack[i] = data_hessLC[int(3 * i + 1), 1]
#             FStack_high[i] = data_hessLC[int(3 * i + 2), 1]
#         dFStack_low = np.abs(FStack - FStack_low)
#         dFStack_high = np.abs(FStack_high - FStack)
#         hd, hdd, hfl, hdfl = tStack, np.zeros(tStack.size), FStack/1e11, 0.5*(dFStack_high+dFStack_low)/1e11
        
#     # plt.scatter(tdata, f)
#     if year in ('2007', '2010', '2014', '2017', '2021', '2024'):
#         tdata = data['t'] - t00
#         f = 10**data['flux']
#         fmin = 10**(data['flux'] - data['dflux_minus'])
#         fmax = 10**(data['flux'] + data['dflux_plus'])
#         dfminus = f - fmin
#         dfplus = fmax - f
#         df = (dfminus + dfplus) / 2
#         ind = data['ind']
#         dind_minus = data['dind_minus']
#         dind_plus = data['dind_plus']
#         # NH = data['NH'][plot_inds]
#         #dNH_minus = data['dNH_minus'][plot_inds]
#         #dNH_plus = data['dNH_plus'][plot_inds]
#         # if year == '2021':
#         #     tdata = np.concatenate((tdata, Nt))
#         #     dt = np.concatenate((dt, Ndt))
#         #     f = np.concatenate((f, Nf))
#         #     fmin = np.concatenate((fmin, Nf-Ndf))
#         #     fmax = np.concatenate((fmax, Nf+Ndf))
#         #     dfminus = np.concatenate((dfminus, Ndf))
#         #     dfplus = np.concatenate((dfplus, Ndf))
#         #     df = np.concatenate((df, Ndf))
#         # print(tdata.size)
#     if year in ('all', '2124', '101417','101417_full' ):
#         f = data['flux']
#         fmin = (data['flux'] - data['dflux_minus'])
#         fmax = (data['flux'] + data['dflux_plus'])
#         dfminus = f - fmin
#         dfplus = fmax - f
#         df = (dfminus + dfplus) / 2
#         ind = data['ind']
#         dind_minus = data['dind_minus']
#         dind_plus = data['dind_plus']
        
        
#     if year == '2024':
#         data1 = np.genfromtxt(Path(path, 'light_curve_2024_SwIn_new.txt'),
#                               names = True)
#         tdataI = data1['tIn'] - t_p24
#         # tdataS = data1['tSw'] - t_p24
#         plot_indsI = np.where((tdataI) > -500) 
#         # plot_indsS = np.where((tdataS) > -500) 
#         # tdataS = tdataS[plot_indsS]
#         tdataI = tdataI[plot_indsI]
#         # dtS = data1['dt_minusSw'][plot_indsS]
#         dtI = data1['dt_minusIn'][plot_indsI]
#         # fS = 10**data1['fluxSw'][plot_indsS]
#         fI = 10**data1['fluxIn'][plot_indsI]
#         # plt.scatter(tdata, f)
#         # fminS = 10**(data1['fluxSw'] - data1['dflux_minusSw'])[plot_indsS]
#         # fmaxS = 10**(data1['fluxSw'] + data1['dflux_plusSw'])[plot_indsS]
#         fminI = 10**(data1['fluxIn'] - data1['dflux_minusIn'])[plot_indsI]
#         fmaxI = 10**(data1['fluxIn'] + data1['dflux_plusIn'])[plot_indsI]
#         # dfminusS = fS - fminS
#         # dfplusS = fmaxS - fS
#         # dfS = (dfminusS + dfplusS) / 2
#         dfminusI = fI - fminI
#         dfplusI = fmaxI - fI
#         dfI = (dfminusI + dfplusI) / 2
#         perI = np.argsort(tdataI)
#         tdataI, dtI, fI, dfminusI, dfplusI, dfI = [arr[perI] for arr in (tdataI, dtI,
#                                                 fI, dfminusI, dfplusI, dfI)]
    
    
#     # plot_inds = np.where((tdata) > -33)   
#     # plot_inds = [int(i_) for i_ in np.linspace(0, tdata.size-1, tdata.size-1)]
#     dt = data['dt_minus']
#     if cond != None:
#         what_inds = np.where(tdata > cond)
#         (tdata, dt, f, fmin, fmax, dfminus, dfplus,
#          df, ind, dind_minus, dind_plus) = [arr[what_inds] for arr in (tdata, dt,
#                                                 f, fmin, fmax, dfminus, dfplus, df,
#                                                 ind, dind_minus, dind_plus)]    
#     per = np.argsort(tdata)
#     (tdata, dt, f, fmin, fmax, dfminus, dfplus,
#      df, ind, dind_minus, dind_plus) = [arr[per] for arr in (tdata, dt,
#                                             f, fmin, fmax, dfminus, dfplus, df,
#                                             ind, dind_minus, dind_plus)]
#     res_lc_xray = {'t': tdata, 'dt': dt, 
#                    'f': f, 'df': df, 'df_minus': dfminus, 'df_plus': dfplus,
#                    'ind': ind, 'dind_minus':dind_minus, 'dind_plus':dind_plus}
    
#     res_lc_tev = {'t': hd, 'f': hfl, 'df': hdfl, 'dt': hdd}
#     res = {'xray': res_lc_xray, 'tev': res_lc_tev}
#     return res
    


