import astropy.units as u
import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, make_smoothing_spline, interp1d
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
# import xarray as xr
from pathlib import Path
import Absorbtion
import Orbit as Orb
import SpecIBS

# start = time.time()
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
# e = 0
r_periastron = a * (1 - e)
D_system = 2.4e3 * 206265 * AU
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

def const(x, C):
    return C

def fit_n(x_obs, y_obs, xth, yth):
    spline_theor = interp1d(xth, yth)
    y_th = spline_theor(x_obs)
    def th_(xwhatever, N):
        return y_th * N
    n_, dn_ = curve_fit(f = th_, xdata = x_obs,
                   ydata = y_obs, p0=(1e25,))
    return n_[0], dn_[0][0]**0.5



def anis_err(dx, dy, phi):
    return dx * cos(phi)**2 + dy * sin(phi)**2

def res_diag_single(xth, yth, x0, y0, dx, dy):
    # spl = splrep(x = xth, y = yth, k=1)
    # df_num = np.gradient(yth, xth, edge_order=2)
    # dspl = splrep(x = xth, y = df_num, k=1)
    # func = lambda x: splev(x, spl)
    # dfunc = lambda x: splev(x, dspl)
    # x_, y_, d, phi = dist(f=func, df=dfunc, x0=x0, y0=y0)
    i_ = np.argmin((x0-xth)**2 + (y0-yth)**2)
    x_, y_ = xth[i_], yth[i_]
    d, phi = ((x_-x0)**2 + (y_-y0)**2)**0.5, np.arctan2(y_-y0, x_-x0)
    err_ = anis_err(dx, dy, phi)
    # print(err_)
    return d / err_*np.sign(np.sin(phi))

def residuals_diag(xobs, yobs, dxobs, dyobs, xtheor, ytheor):
    # ress = np.zeros(xobs.size)-1
    # for i in range(xobs.size):
    #     ress[i] = res_diag_single(xtheor/dxobs[i], ytheor/dyobs[i], xobs[i]/dxobs[i], yobs[i]/dyobs[i], 1, 1)
    # return ress
    spl = interp1d(xtheor, ytheor)
    ress = (spl(xobs) - yobs) / dyobs
    return ress

obs_fold = Path(Path.cwd(), 'ObsData')
exposure_S = np.array([576.8878,2157.656, 2135.181, 2614.655, 1278.609, 1917.999])
exposure_I = np.array([47857.7861796112,34018.4876335561, 43982.9039376908, 46504.9046225975,
              41991.5653142337, 43802.487989002])
# print(np.sum(exposure_I))
eff_S = exposure_S / np.mean(exposure_S)
eff_I = exposure_I / np.mean(exposure_I)
# print(eff_S)
# print(eff_I)

def Extr_sp(path, num = 5):
    with open(path, "r") as file:
        raw_data = file.read().strip().split("#")  # Split by "#"
    datasets = [block.strip() for block in raw_data if block.strip()]
    data_arrays = [np.loadtxt(dataset.splitlines()) for dataset in datasets]
    Da = {}
    if num == 5:
        for i in range(1, 6):
            Da[f"Sp{i}"] = np.concatenate((data_arrays[2*(i-1)], data_arrays[2*i-1]))
        S1, S2, S3, S4, S5 = Da['Sp1'], Da['Sp2'], Da['Sp3'], Da['Sp4'], Da['Sp5']
        return S1, S2, S3, S4, S5
    if num == 6:
        for i in range(1, 7):
            Da[f"Sp{i}"] = np.concatenate((data_arrays[2*(i-1)], data_arrays[2*i-1]))
        S1, S2, S3, S4, S5, S6 = Da['Sp1'], Da['Sp2'], Da['Sp3'], Da['Sp4'], Da['Sp5'], Da['Sp6']
        return S1, S2, S3, S4, S5, S6

def Stack_int(S1, S2, S3, S4, S5, S6):
    i1, i2, i3, i4, i5, i6 = (S1[np.where(S1[:, 0] > 20.), :][0],
                              S2[np.where(S2[:, 0] > 20.), :][0],
                              S3[np.where(S3[:, 0] > 20.), :][0],
                              S4[np.where(S4[:, 0] > 20.), :][0],
                              S5[np.where(S5[:, 0] > 20.), :][0],
                              S6[np.where(S6[:, 0] > 20.), :][0])
    ys = np.zeros((5, 6))
    dys = np.zeros((5, 6))
    xs = i1[:, 0]
    dxs= i1[:, 1]
    for i_ch in range(5):
        ys[i_ch, :] = np.array([i1[i_ch, 2], i2[i_ch, 2], i3[i_ch, 2], i4[i_ch, 2], i5[i_ch, 2], i6[i_ch, 2] ])
        dys[i_ch, :] = np.array([i1[i_ch, 3], i2[i_ch, 3], i3[i_ch, 3], i4[i_ch, 3], i5[i_ch, 3], i6[i_ch, 3] ])
    ys_fit = np.zeros(5)
    dys_fit = np.zeros(5)
    for i_ch in range(5):
        c, dc = curve_fit(f=const, xdata=1, ydata=ys[i_ch, :], sigma=dys[i_ch, :]/eff_S,
                          absolute_sigma=False)
        # print(c, dc)
        ys_fit[i_ch] = c[0]
        dys_fit[i_ch] = dc[0][0]
    return xs, dxs, ys_fit, dys_fit**0.5

S1, S2, S3, S4, S5, S6 = Extr_sp(Path(obs_fold, 'six_sp.qdp'), 6)

# Integral data
eI, deI, sedI, dsedI = Stack_int(S1, S2, S3, S4, S5, S6)

da_S_st = np.genfromtxt(Path(obs_fold, "stack_mine.qdp"))

# Swift data
eS, deS, sedS, dsedS, thS = da_S_st[:-1, 0], da_S_st[:-1, 1], da_S_st[:-1, 2], da_S_st[:-1, 3], da_S_st[:-1, 4]


"""
###############################################################################
TeV data from 2021
###############################################################################
"""


da_H = np.genfromtxt(Path(obs_fold, "Hess_spec21.txt"), delimiter=', ')
eH, sedH = da_H[:, 0] * 1e9, da_H[:, 1] /1.6e-9

# print(eH)
deH = np.zeros(eH.size)
deH[0] = 0.5 * (eH[1] - eH[0])
for i in range(1, eH.size):
    deH[i] = np.abs(eH[i] - eH[i-1]) - deH[i-1]
dsedH = sedH/3

"""
###############################################################################
TeV data from 2021 by me for days 25-36 
###############################################################################
"""


da_H1 = np.genfromtxt(Path(obs_fold, "spec_21_18d_35d.spec"))
eH1, FlH1 = da_H1[:, 0] * 1e9, da_H1[:, 3] / 1e9
sedH1 = FlH1 * eH1**2
# print(eH)
deH1 = np.zeros(eH1.size)
deH1[0] = 0.5 * (eH1[1] - eH1[0])
for i in range(1, eH1.size):
    deH1[i] = np.abs(eH1[i] - eH1[i-1]) - deH1[i-1]
dFl_min, dFl_pl = da_H1[:, 4] / 1e9, da_H1[:, 5] / 1e9
dsedH1 = (dFl_min + dFl_pl) / 2 * eH1**2

"""
###############################################################################
TeV data from 2024 ((((((NEU NEU NEU NEU NEU NEU ACHTUNG ACHTUNG))))))
###############################################################################
"""

da_H4 = np.genfromtxt(Path(obs_fold, "out.spec"))
eH4, FlH4 = da_H4[:, 0] * 1e9, da_H4[:, 3] / 1e9
sedH4 = FlH4 * eH4**2
# print(eH)
deH4 = np.zeros(eH4.size)
deH4[0] = 0.5 * (eH4[1] - eH4[0])
for i in range(1, eH4.size):
    deH4[i] = np.abs(eH4[i] - eH4[i-1]) - deH4[i-1]
dFl_min, dFl_pl = da_H4[:, 4] / 1e9, da_H4[:, 5] / 1e9
dsedH4 = (dFl_min + dFl_pl) / 2 * eH4**2


"""
###############################################################################
Red Fermi region
###############################################################################
"""


eF = np.logspace(np.log10(7e4), np.log10(2e6), 100)
sedF_low = 2e-11 * (eF / 1e5)**(-0.7) /1.6e-9
sedF_high = 4.2e-11 * (eF / 1e5)**(-0.7) /1.6e-9
sedF = 0.5 * (sedF_low + sedF_high)


"""
###############################################################################
Blue Fermi region
###############################################################################
"""

data_blusedrmi = np.genfromtxt(Path(obs_fold, "Fermi_spec_ch.txt"),
                               delimiter = ', ', skip_header=0)
N = int(np.shape(data_blusedrmi)[0] / 2)
eFblue, sedFblue_low, sedFblue_high = np.zeros((3, N))
for i in range(N):
    eFblue[i] = data_blusedrmi[int(2 * i + 1), 0]/1e3
    
    sedFblue_low[i] = data_blusedrmi[int(2 * i), 1]/1.6e-9
    sedFblue_high[i] = data_blusedrmi[int(2 * i + 1), 1]/1.6e-9
    
sedFblue = (sedFblue_low + sedFblue_high)/2


"""
###############################################################################
Green Fermi (flares)
###############################################################################
"""

        

data_greenFermi = np.genfromtxt(Path(obs_fold, "Fermi_specGreen_ch.txt"),
                               delimiter = ', ', skip_header=0)
N = int(np.shape(data_greenFermi)[0] / 5)
eFgr, sedFgr, sedFgr_low, sedFgr_high, eFgr_left, eFgr_right = np.zeros((6, N))
for i in range(N):
    eFgr[i] = data_greenFermi[int(5 * i + 1), 0]/1e3
    eFgr_left[i] = data_greenFermi[int(5 * i + 3), 0]/1e3
    eFgr_right[i] = data_greenFermi[int(5 * i + 4), 0]/1e3
    
    sedFgr[i] = data_greenFermi[int(5 * i+1), 1]/1.6e-9
    sedFgr_low[i] = data_greenFermi[int(5 * i), 1]/1.6e-9
    sedFgr_high[i] = data_greenFermi[int(5 * i + 2), 1]/1.6e-9
deFgr_left = eFgr-eFgr_left
deFgr_right = eFgr_right-eFgr
dsedFgr_low = sedFgr-sedFgr_low
dsedFgr_high = sedFgr_high-sedFgr
arg_norm = np.where(sedFgr_low < sedFgr)
arg_upper = np.where(sedFgr_low > sedFgr)
dsedFgr_low[arg_upper] = sedFgr[arg_upper]/2
dsedFgr_high[arg_upper] = sedFgr[arg_upper]/2
uplims = np.zeros(eFgr.size)
uplims[arg_upper] = False
uplims[arg_upper] = True












'''
Bs = np.logspace(12.7, 14.2, 33)
cuts = np.logspace(0.8, 2, 40)
chisq = np.zeros((Bs.size, cuts.size))
for ib, B in enumerate(Bs):
    print(ib)
    def func_par(ic):
    # for ic, Ecut_e in enumerate(cuts):
        Ecut_e = cuts[ic]
        t = 27 * DAY
        nu_true = Orb.True_an(t)
        LoS_to_orb = 2.3
        LoS_to_shock = - (pi - (LoS_to_orb - nu_true))
        pe = 1.7
        r_SP = Orb.Radius(t)
        beta_eff = 0.1
        r_SE = r_SP / (1 + beta_eff**0.5)
        Topt = 3e4
        # Bx = 2e13
        Bx = B
        Bopt = 0
        Gamma = 1
        
        # phi = pi/2
        Bp_apex, Bs_apex, uapex = SpecIBS.B_and_u_test(Bx = Bx, Bopt = Bopt, 
                                               r_SE = r_SE, r_PE = (r_SP - r_SE),
                                    T_opt = Topt)
        B_field =  Bp_apex+Bs_apex
        smax = 3
        Nibs = 23
        Es = np.logspace(1, 14, 203)
        cond_syn = (Es < 1e17)
        E0_e = 1; Ampl_e = 1e25
        eta_a = 1e-6
        sed = SpecIBS.SED_from_IBS(E = Es, B_apex = B_field,
                              u_g_apex = uapex, Topt = Topt, 
                           r_SP = r_SP, E0_e = E0_e, Ecut_e = Ecut_e, Ampl_e = Ampl_e,
                           p_e = pe,  beta_IBS = beta_eff, Gamma = Gamma, s_max = smax,
                           N_shock = Nibs, bopt2bpuls_ap = Bs_apex/Bp_apex, phi = LoS_to_shock,
                           s_adv = False, lorentz_boost=True, simple = False, eta_a = eta_a)
        Abs_tbabs = Absorbtion.abs_photoel(Es*1.6e-12, Nh = 0.78)
        Abs_gg = Absorbtion.abs_gg_tab(Es*1.6e-12, nu_los = 2.4, t = t)
        Ab = Abs_tbabs * Abs_gg
        sed = sed * Ab / 1.6e-9
        # spl_sed = make_smoothing_spline(x = Es/1e3, y = sed,
        #     lam=1e-5)
        norm, dnorm = fit_n(eS, sedS, Es/1e3, sed)
        # norm, dnorm = 1, 1
        sed = sed * norm
        eTot = np.concatenate((eI, eS, eH))
        sedTot = np.concatenate((sedI, sedS, sedH))
        deTot = np.concatenate((deI, deS, deH))
        dsedTot = np.concatenate((dsedI, dsedS, dsedH))
        
        res = residuals_diag(xobs = eTot, yobs = sedTot,
                         dxobs= deTot, dyobs = dsedTot, 
                                xtheor = Es/1e3, ytheor = sed)
        # chisq[ib, ic] = np.sum(res**2)
        return np.sum(res**2)
    
    res = Parallel(n_jobs = 10)(delayed(func_par)(ic) for ic in range(cuts.size))
    chisq[ib, :] = np.array(res)
        
chisq[chisq > 2*np.min(chisq)] = np.nan
BB, EE = np.meshgrid(Bs, cuts, indexing='ij')        
cs = plt.contourf(BB, EE, chisq, levels=10)
plt.title('Chi2')
plt.xlabel('x'); plt.ylabel('y')
plt.xscale('log')
plt.yscale('log')


plt.colorbar(cs)
'''
"""
###############################################################################
Plot: data on ax1
###############################################################################
"""


Lspin = 8e35
Fspin = Lspin / 4 / pi / (2.4e3 * 3e18)**2
Fspin_kev = Fspin /1.6e-9
# fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]},
#                        sharex=True)
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                       sharex=True)
ax1, ax2 = ax[0], ax[1]
# fig, ax = plt.subplots(1, 1)
# ax1 = ax
# ax1.errorbar(eI, sedI, xerr=deI, yerr=dsedI, fmt='none', c='orange',
#              label = 'INTEGRAL 2024 stacked')
ax1.errorbar(eS, sedS, xerr=deS, yerr=dsedS, fmt='none', c='k',
             label = 'Swift/XRT 2024 stacked')

ax1.errorbar(eH, sedH, xerr=deH, yerr=dsedH, fmt='none', c='m',
             label = 'HESS 2021: 25-36 days')
ax1.errorbar(eH4, sedH4, xerr=deH4, yerr=dsedH4, fmt='none', c='r',
             label = 'HESS 2024 total')
# ax1.errorbar(eH1, sedH1, xerr=deH1, yerr=dsedH1, fmt='none', c='pink',
#              label = 'HESS 2021 by me')

ax1.fill_between(eF, sedF_low, sedF_high, alpha=0.3, color='r',
                 label = 'Fermi 2024 0-20 days')

ax1.fill_between(eFblue, sedFblue_low, sedFblue_high, alpha=0.3, color='b',
                 label = 'Fermi 2024 -20...0 days')
# ax1.errorbar(eFgr, sedFgr, 
#              xerr=(deFgr_left, deFgr_right),
#              yerr=(dsedFgr_low, dsedFgr_high),
#              c='g', fmt='none',
#              uplims=uplims, 
#              label = 'Fermi 2024 \'flares\' (19-77 days)')

ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.axhline(y=Fspin_kev, alpha=0.2)
ax1.set_ylabel(r'$E F_E, \mathrm{keV} ~\mathrm{cm}^{-2} ~\mathrm{s}^{-1}$', fontsize=14)
ax1.set_ylim(bottom=8e-5, top=1)
ax1.tick_params(direction = 'in', which = 'both')
# ax3.tick_params(direction = 'in', which = 'both')

# ax2.tick_params(direction = 'in', which = 'both')

# ax3.axhline(y=0, color='lime', lw=1)

# Es = np.logspace(-0.3, 10, 1000)
# ax3.set_xscale('log')


ax2.axhline(y=0, color='lime', lw=1)
ax2.set_xscale('log')
ax2.set_ylim(top=5, bottom=-5)
# ax3.set_ylim(top=5, bottom=-5)

ax2.set_xlabel(r'$E, \mathrm{keV}$', fontsize=14)
ax2.set_ylabel(r'Resid., $\sigma$', fontsize=14)

fig.subplots_adjust(hspace=0)

for beta in (1, ):

    
    """
    ###############################################################################
    theor calculation of spectrum and norm
    ###############################################################################
    """
    t = 27 * DAY
    nu_true = Orb.True_an(t)
    LoS_to_orb = 2.3
    LoS_to_shock = - (pi - (LoS_to_orb - nu_true))
    pe = 1.7
    r_SP = Orb.Radius(t)
    beta_eff = 0.001
    r_SE = r_SP / (1 + beta_eff**0.5)
    Topt = 3.3e4
    Bx = 6e11
    Bopt = 0
    Gamma = 1
    
    # phi = pi/2
    Bp_apex, Bs_apex, uapex = SpecIBS.B_and_u_test(Bx = Bx, Bopt = Bopt, 
                                           r_SE = r_SE, r_PE = (r_SP - r_SE),
                                T_opt = Topt)
    B_field =  Bp_apex+Bs_apex
    # smax = 3
    Nibs = 23
    Es = np.logspace(1, 14, 203)
    cond_syn = (Es < 1e17)
    E0_e = 1; Ecut_e = 15; Ampl_e = 1e25
    eta_a = 1e-1
    sed = SpecIBS.SED_from_IBS(E = Es, B_apex = B_field,
                          u_g_apex = uapex, Topt = Topt, 
                       r_SP = r_SP, E0_e = E0_e, Ecut_e = Ecut_e, Ampl_e = Ampl_e,
                       p_e = pe,  beta_IBS = beta_eff, Gamma = Gamma, s_max_em = 'bow',
                       s_max_g = 4.0,
                       N_shock = Nibs, bopt2bpuls_ap = Bs_apex/Bp_apex, phi = LoS_to_shock,
                       s_adv = False, lorentz_boost=True, simple = False, eta_a = eta_a,
                       abs_photoel=True, abs_gg=True, t_ggabs=t)
    
    # sed = sed[Es>1e6]
    # Es = Es[Es>1e6]
    norm = trapezoid(sed[cond_syn]/Es[cond_syn]**2, Es[cond_syn])
    
    
    # Abs_tbabs = Absorbtion.abs_photoel(Es*1.6e-12, Nh = 0.78)
    # Abs_gg = Absorbtion.abs_gg_tab(Es*1.6e-12, nu_los = 2.4, t = t)
    # Ab = Abs_tbabs * Abs_gg
    Ab = 1
    # sed, sed_SY, sed_IC = [arr* Ab / 1.6e-9 for arr in (sed, sed_SY, sed_IC)]
    sed = sed * Ab / 1.6e-9
    # spl_sed = make_smoothing_spline(x = Es/1e3, y = sed,
    #     lam=1e-5)
    # norm, dnorm = fit_n(np.concatenate((eS, eI)), 
    #                     np.concatenate((sedS, sedI)), Es/1e3, sed)
    norm, dnorm = fit_n(eS, sedS, Es/1e3, sed)
    # print('log norm, d log norm = ', np.log10(norm), dnorm/norm)
    # print('magn', par['MF'])
    # print('r_SP, au', par['r_SP'] / AU)
    # print('r_PE, au', par['r_PE'] / AU)
    
    sed = sed * norm
    # sed, sed_SY, sed_IC = sed/sed.unit * norm, sed_SY/sed_SY.unit * norm, sed_IC/sed_IC.unit * norm
    spl_sed = make_smoothing_spline(x = Es/1e3, y = sed,
        lam=1e-5)
    # where_fit = np.where(np.logical_and(Es > 5, Es < 10))
    # # e_fit = np.logspace(np.log10(5), 1, 10)
    # popt, pcov = curve_fit(f = po, xdata = Es[where_fit], ydata = sed[where_fit],
    #                        p0=(1, 0.3))
    # ax1.plot(Es[where_fit], po(Es[where_fit], *popt))
    # G_ind = popt[0] + 2
    # dG_ind = (np.diag(pcov)**0.5)[0]
    # print('Photon index 5-10 kev is ', G_ind)
    # print('With the error  ', dG_ind)
    
    # Ab = Anal_abs(Es)
    
    """
    ###############################################################################
    ax1
    ###############################################################################
    """
    
    if beta == 1:
        ls= '-'
    else:
        ls='--'
    label = 'Theor syn + IC'
    ax1.plot(Es/1e3, sed, c='k', ls=ls, lw=1, label=label)

    
    if beta == 1:
        ax__ = ax2
        
    # if beta == 3:
    #     ax__ = ax3
        ax__.errorbar(x=eS, y=residuals_diag(xobs = eS, yobs = sedS,
                                                  dxobs= deS,dyobs = dsedS, 
                                xtheor = Es/1e3, ytheor = sed), xerr=deS, yerr=1, 
                     fmt='none', color='k')
        
        # ax__.errorbar(x=eI, y=residuals_diag(xobs = eI, yobs = sedI,
        #                                           dxobs= deI, dyobs = dsedI, 
        #                         xtheor = Es/1e3, ytheor = sed), xerr=deI, yerr=1, 
        #              fmt='none', color='orange')
        
        ax__.errorbar(x=eH, y=residuals_diag(xobs = eH, yobs = sedH, dyobs = dsedH, 
                                dxobs = deH, xtheor = Es/1e3, ytheor = sed), xerr=deH, 
                     yerr = 1, c='m', fmt='none')
        
        ax__.errorbar(x=eH4, y=residuals_diag(xobs = eH4, yobs = sedH4, dyobs = dsedH4, 
                                dxobs = deH4, xtheor = Es, ytheor = sed), xerr=deH4, 
                     yerr = 1, c='r', fmt='none')
        
        th_re = -(sedF - spl_sed(eF)) / (sedF_high - sedF_low)*2
        
        ax__.fill_between(eF, th_re + 1, th_re - 1,
                         alpha=0.3, color='r')  
    
        th_bl = -(sedFblue - spl_sed(eFblue)) / (sedFblue_high - sedFblue_low)*2
        ax__.fill_between(eFblue, th_bl + 1, th_bl - 1,
                         alpha=0.3, color='b')
        ax1.legend(fontsize=12, loc='lower center', framealpha=1)

    # uplims = np.array(, dtype=bool)
    
    # eFgrN, sedFgrN, sedFgr_lowN, sedFgr_highN, eFgr_leftN, eFgr_rightN = [arr[arg_norm]
    #     for arr in (eFgr, sedFgr, sedFgr_low, sedFgr_high, eFgr_left, eFgr_right)]
    
    # eFgrU, sedFgrU, sedFgr_lowU, sedFgr_highU, eFgr_leftU, eFgr_rightU = [arr[arg_upper]
    #     for arr in (eFgr, sedFgr, sedFgr_low, sedFgr_high, eFgr_left, eFgr_right)]
    

plt.show()
