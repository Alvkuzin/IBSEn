import numpy as np
from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import brentq#, root, fsolve, least_squares, minimize
from scipy.optimize import curve_fit
# from pathlib import Path
from numpy import pi, sin, cos
from joblib import Parallel, delayed
import multiprocessing
# import time
# import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, interp1d
# import Boost_from_shockwave as Boost
# from My_e_evol import Evolved_ECPL
import Orbit as Orb
import SpecIBS
import GetObsData

# start_time = time.time()
#import matplotlib
#matplotlib.use('NbAgg')
G = 6.67e-8
c_light = 3e10
sigma_b = 5.67e-5
h_planck_red = 1.05e-27
Rsun = 7e10
AU = 1.5e13
DAY = 86400.
# Mopt = 24
# Ropt = 10 * Rsun
# Mx = 1.4
# GM = G * (Mopt + Mx) * 2e33
# P = 1236.724526
# Torb = P * DAY
# a = (Torb**2 * GM / 4 / pi**2)**(1/3)
# e = 0.87

# r_periastron = a * (1 - e)
# D_system = 2.4e3 * 206265 * AU
orb_p_psrb = Orb.Get_PSRB_params()
# Torb, e, Mtot, Ropt, P, GM = [orb_p_psrb[key] for key in ('T', 'e', 'M', 'Ropt', 'P', 'GM')]
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0

def logrep(xdata, ydata):
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    return interp1d(np.log10(xdata), np.log10(ydata))

def logev(x, logspl):
    return 10**( logspl( np.log10(x) ) )

def interplg(x, xdata, ydata):
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    spl_ = interp1d(np.log10(xdata), np.log10(ydata))
    return 10**( spl_( np.log10(x) ) )

def fit_norm(xdata, ydata, xtheor, ytheor):
    fspl = interp1d(xtheor, ytheor)
    ydata_fake = fspl(xdata)
    def to_fit(xda, norma):              
        return (norma * ydata_fake - ydata)
    
    norm, dnorm = curve_fit(to_fit, xdata = xdata, 
                    ydata = np.zeros(xdata.size), p0 = (1e44,))
    norm = norm[0]
    return ytheor * norm

def po(E, g, norm):
    return norm * E**(-g)

def unpack(query, dictat):
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def c_sound(delta, r_inD, Ropt, orb_p = orb_p_psrb):
    GM = orb_p['GM']
    mu_gas = 0.6
    if isinstance(r_inD, np.ndarray):
        res = delta * (GM / Ropt / mu_gas)**0.5 * (Ropt / r_inD)**0.25
        res[r_inD < 0] = 0
    else:
        if r_inD < 0: res = 0
        else:
            res = delta * (GM / Ropt / mu_gas)**0.5 * (Ropt / r_inD)**0.25
    return res

def Pressure_wind(R_from_star, Norm, R_reference):    
    return Norm * (R_reference / R_from_star)**2

def Pressure_pulsar(R_from_pulsar, Norm, R_reference):
    return Norm * (R_reference / R_from_pulsar)**2 

def Pressure_disk(Vec_R_from_star, alpha, incl, Norm, delta, np_disk,
            R_reference, radial_profile = 'pl', R_trunk = None):
    ndisk = Orb.N_disk(alpha, incl, orb_p='psrb')
    r_fromdisk = Orb.mydot(Vec_R_from_star, ndisk) * ndisk
    r_indisk = Vec_R_from_star - r_fromdisk
    r_toD = Orb.ABSV(r_fromdisk) 
    r_indisk = Vec_R_from_star - r_fromdisk
    r_inD = Orb.ABSV(r_indisk)
    z0 = delta * r_inD * (r_inD / R_reference)**0.25 
    vert = np.exp(-r_toD**2 / 2 / z0**2)
    if radial_profile == 'pl':
        rad = (R_reference / r_inD)**np_disk
    if radial_profile == 'broken_pl':
        if r_inD < R_trunk:
            rad = (R_reference / r_inD)**np_disk
        if r_inD >= R_trunk:
            rad = (R_reference / R_trunk)**np_disk * (R_trunk / r_inD)**2
    return Norm * rad * vert 

def Dist_SE_1d(t, alpha, incl, f_w, f_d, f_p, delta, np_disk, r_pr, R_t, Ropt,
               hyst = False, SE_prev = None, t_prev = None):   
    r_sp_vec = Orb.Vector_S_P(t, orb_p='psrb')
    nwind = Orb.N_from_V(r_sp_vec) # unit vector from S to P
    r_sp = Orb.ABSV(r_sp_vec)
    pres_w = lambda r_se: Pressure_wind(R_from_star=r_se, Norm=f_w, 
                                        R_reference=Ropt)
    pres_d = lambda r_se: Pressure_disk(Vec_R_from_star=r_se*nwind,
                        alpha=alpha, incl=incl, Norm=f_d, delta=delta,
                        np_disk=np_disk, radial_profile = r_pr, R_trunk = R_t,
                        R_reference=Ropt)
    pres_p = lambda r_se: Pressure_pulsar(R_from_pulsar=np.abs(r_sp - r_se),
                                          Norm=f_p, R_reference=Ropt)
    to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
    rse = brentq(to_solve, Ropt, r_sp*(1-1e-6))
    ### ---------------- test if the solution is good -------------------------
    p_ref = pres_p(rse)
    max_rel_err = np.max(to_solve(rse) / p_ref)
    if max_rel_err > 1e-3:
        print('t = %s error is huge: %s'%(t/DAY, max_rel_err))  
    ### -----------------------------------------------------------------------
    # if hyst = True (hysteresis), then compare if the shock wave had enough
    # time to evolve since previous t. If yes, then return rse. If no, 
    # return rse_prev - v * delta t
    # Hysteresis only works when rse decreases, not increases
    rse_to_return = rse

    if hyst:
        rpe = r_sp - rse
        rsp_prev = Orb.Radius(t_prev, orb_p='psrb')
        rpe_prev = rsp_prev - SE_prev
                # if SE_prev < 0 or (rse/Orb.Radius(t) < SE_prev/Orb.Radius(t_prev) ): # when r_se decreases, hysreresis works
        # if SE_prev < 0 or (rpe/r_sp > rpe_prev/rsp_prev): # if r_sp increases, hysteresis works
        csound = c_sound(delta=delta, r_inD = SE_prev)
        
        deltaT = abs(t - t_prev)
        dot_r = np.abs(Orb.Radius(t, orb_p='psrb') -
        Orb.Radius(t_prev, orb_p='psrb')) / deltaT# * SE_prev / Orb.Radius(t_prev)
        # v_shock = max(csound, dot_r) #!!!
        v_shock = csound + dot_r
        # print(csound)
        # print(SE_prev)
        # print(deltaT)
        rse_to_return = max(rse, SE_prev - v_shock * deltaT)
        # rse_to_return = 1e13
        # else: # when r_se increases, turn off the hysreresis
        #     rse_to_return = rse
            
    return rse_to_return

def Thresh_crit(x, xarr, yarr):
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

def P_and_h(t, params): 
    que_va = 'delta f_d enh_p enh_H t_enh_p t_enh_H alpha0 incl'
    delta, f_d, enh_p, enh_H, t_enh_p, t_enh_H, alpha, incl = unpack(que_va, params)
    tdisk1, tdisk2 = Orb.times_of_disk_passage(alpha, incl, orb_p='psrb')
    t_enh_H_ = [tdisk1 if t_ == 't1' else t_ for t_ in t_enh_H]
    t_enh_H_ = [tdisk2 if t_ == 't2' else t_ for t_ in t_enh_H_]
    t_enh_p_ = [tdisk1 if t_ == 't1' else t_ for t_ in t_enh_p]
    t_enh_p_ = [tdisk2 if t_ == 't2' else t_ for t_ in t_enh_p_]
    current_p_enh = Thresh_crit(t, t_enh_p_, enh_p)
    current_H_enh = Thresh_crit(t, t_enh_H_, enh_H)
    return current_p_enh * f_d, current_H_enh * delta

def B_and_u(Bx, Bopt, r_SE, r_PE, T_opt, Ropt):      
    L_spindown, sigma_magn = 8e35 * (Bx / 3e11)**2, 1e-2
    B_puls = (L_spindown * sigma_magn / c_light / r_PE**2)**0.5
    B_star = Bopt * (Ropt / r_SE)
    factor = 2. * (1. - (1. - (Ropt / r_SE)**2)**0.5 )
    u_dens = sigma_b * T_opt**4 / c_light * factor
    return B_puls, B_star, u_dens

def SED_tot_PSRB(t, E, params, syn_only, hyst = False, SE_prev = None, t_prev = None):
    que_va = """delta f_w f_d f_p np_disk Bx Bopt alpha0 incl enh_p enh_H 
    E0_e Ecut_e logNorm p_e Topt r_pr R_t if_boost Gamma LoS_to_orb 
    delta_pow MF_boost beta cooling eta_flow eta_syn eta_IC
     lorentz_boost simple abs_photoel abs_gg s_max Ropt"""
    (delta, f_w, f_d, f_p, np_disk, Bx, Bopt, alpha, incl, enhance_p,
     enhance_h, E0_e, Ecut_e, logAmpl_e, p_e, T_opt, 
     r_pr, R_t, if_boost, Gamma, LoS_to_orb, delta_pow, 
     MF_boost, beta, cooling, eta_flow,
     eta_syn, eta_IC,
      lorentz_boost, simple, abs_photoel, abs_gg, s_max, Ropt) = unpack(que_va, params)
    
    f_d_mod, delta_mod = P_and_h(t, params)
    SP_vec = Orb.Vector_S_P(t, orb_p='psrb')
    nu_true = Orb.True_an(t, orb_p='psrb')
    r_SP = Orb.ABSV(SP_vec)
    r_SE = Dist_SE_1d(t, alpha, incl, f_w, f_d_mod, f_p, delta_mod, 
                         np_disk, r_pr, R_t, hyst, SE_prev, t_prev)
    Bp, Bs, u = B_and_u(Bx = Bx, Bopt = Bopt, r_SE = r_SE, r_PE = r_SP - r_SE,
                        T_opt = T_opt, Ropt=Ropt)
    r_PE = r_SP - r_SE
    beta_eff = (r_PE / r_SE)**2
    sed = SpecIBS.SED_from_IBS(E=E, B_apex=Bp+Bs, u_g_apex=u, Topt=T_opt,
        r_SP = r_SP, E0_e = E0_e, Ecut_e = Ecut_e, Ampl_e=10**logAmpl_e, p_e=p_e,
        beta_IBS=beta_eff, Gamma=Gamma, s_max_em=s_max, s_max_g = 4.0, N_shock=11,
        bopt2bpuls_ap=Bs/Bp, phi = -(pi - LoS_to_orb + nu_true),
        delta_power=delta_pow, lorentz_boost = lorentz_boost,
        simple = simple, eta_a = eta_flow, cooling=cooling,
        abs_photoel = abs_photoel, abs_gg = abs_gg, Nh_tbabs = 0.8,
        nu_los_ggabs = LoS_to_orb, t_ggabs = t, syn_only=syn_only)

    some_params = {'MF': Bp + Bs, 'Bp': Bp, 'Bs': Bs, 'r_SP': r_SP,
                   'r_PE': r_PE, 'r_SE': r_SE, 'u_rad': u}
    # return sed, sed_SY, sed_IC, some_params
    return sed, some_params      

def Flux(t, params, bands, syn_only = False, hyst = False, 
         SE_prev = None, t_prev = None):
    N_per_dec = 51
    if syn_only:
        e1, e2 = 3e2, 1e4
        E_integr = np.logspace(np.log10(e1), np.log10(e2), int(N_per_dec*np.log10(e2/e1)))
        sed_in_range, some_params = SED_tot_PSRB(t, E_integr, params, syn_only,
                                                 hyst, SE_prev, t_prev)
        Fluxes = [trapezoid(x = E_integr, y = sed_in_range/E_integr), -1, -1]
        G_ind = -1
    if not syn_only:
        Fluxes = []
        for band in bands:
            e1, e2 = band
            E_integr = np.logspace(np.log10(e1), np.log10(e2), int(N_per_dec*np.log10(e2/e1)))
            sed_in_range, some_params = SED_tot_PSRB(t, E_integr, params, syn_only,
                                                     hyst, SE_prev, t_prev)
            Fluxes.append(trapezoid(x = E_integr, y = sed_in_range/E_integr))
            if e1 == 3e2:
                where_fit = np.where(np.logical_and(E_integr > 3e3, E_integr < 1e4))
                popt, pcov = curve_fit(f = po, xdata = E_integr[where_fit],
                                       ydata = sed_in_range[where_fit],
                                       p0=(1, 0.5))
                G_ind = popt[0] + 2 

    return np.array(Fluxes), G_ind, some_params        

def Xray_flux_full_output(t, params,  swift_only=False, hyst = False, 
         SE_prev = None, t_prev = None):
    # if not swift_only:
    bands = np.array([np.array([3e2, 1e4]), 
                      np.array([3e4, 5e4]),
                      np.array([4e11, 1e14])])
    Fluxes, G_ind, some_params = Flux(t, params, bands, swift_only,
                                      hyst, SE_prev, t_prev)
    Fx, FI, Ftev = Fluxes
    Bp, Bs, us_rad,  r_SP, r_SE, r_PE = [some_params[n_] 
        for n_ in ('Bp', 'Bs', 'u_rad', 'r_SP', 'r_SE', 'r_PE')]
    Btot = Bs + Bp
    beta_an = params['f_p'] / params['f_w']
    r_StoE_an = r_SP / (1 + beta_an**0.5)
    return (r_SE, r_StoE_an, r_SP, r_PE, us_rad, Fx,
             Bs, Bp, Btot, Ftev, FI, G_ind)
    # if swift_only:
    #     Fx, G_ind, some_params = Flux(t, params, swift_only, hyst, SE_prev, t_prev)
    #     return Fx, some_params['r_SE']
    
def Xray_Light_curve(tplot, params, ifparallel = False,  swift_only=False):
    hyst = params['hyst']
    if hyst: ifparallel = False # if hysteresis if on, no paral !

    if ifparallel == False:
        (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,  Bsstar_scal, 
         Bspuls_scal, Bstot_scal, Ftev, FI, G_ind) = np.zeros((12, tplot.size))
        if not hyst:
            for it in range(tplot.size):
                t_ = tplot[it]
                (r_StoE[it], r_StoE_an[it], r_StoP[it], r_PtoE[it], us_rad[it],
                 Fsx[it],  Bsstar_scal[it],  Bspuls_scal[it], 
                 Bstot_scal[it], Ftev[it],  FI[it], G_ind[it]) = Xray_flux_full_output(t_, params,  swift_only)
        if hyst:
            
            rse_prev = -1e100 # first ``previous`` r_se is very small so that
            tprev = np.min(tplot) - 1.*DAY # first ``previous`` t
            t_disk1, t_disk2 = Orb.times_of_disk_passage(params['alpha0'], params['incl'], orb_p='psrb')
            for it in range(tplot.size):
                t_ = tplot[it]
                if (t_ > -10.3*DAY and t_ < 5*DAY) or (t_ > 20*DAY): 
                # if t_ > t_disk1:
                    hyst_here = True
                else:
                    hyst_here = False
                (r_StoE[it], r_StoE_an[it], r_StoP[it], r_PtoE[it], us_rad[it],
                 Fsx[it],  Bsstar_scal[it],  Bspuls_scal[it], 
                 Bstot_scal[it], Ftev[it],  FI[it], G_ind[it]) = Xray_flux_full_output(t_, params,  swift_only,
                                  hyst=hyst_here, SE_prev=rse_prev, t_prev= tprev)
                rse_prev = r_StoE[it] # re-define r_se on ``previous`` step
                tprev = t_ # re-define t on ``previous`` step
            
    if ifparallel == True:
        def func_(i_):
            return Xray_flux_full_output(tplot[i_], params, swift_only)
        if params['cooling'] == 'adv':
            n_cores = 3
        else:
            n_cores = multiprocessing.cpu_count()
        res= Parallel(n_jobs=n_cores)(delayed(func_)(i) for i in range(0, len(tplot)))
        res=np.array(res)
        ##    print(np.array(res).size)
        (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,  Bsstar_scal, 
        Bspuls_scal, Bstot_scal, Ftev, FI, G_ind) = [res[:, ii] for ii in range(12)]
                
    return (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,  Bsstar_scal, 
                            Bspuls_scal, Bstot_scal, Ftev, FI, G_ind)

        
if __name__=='__main__':
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0, ax1 = ax[0], ax[1]
    da = GetObsData.getSingleLC('101417_full', )
    da_xray = da['xray']

    (tdata, dt, f, dfminus, dfplus,
     df, ind, dind_minus, dind_plus) = [da_xray[ind_] for ind_ in ('t', 'dt',
        'f', 'df_minus', 'df_plus', 'df', 'ind', 'dind_minus', 'dind_plus')]
    cond =  np.logical_and((tdata>-320),  (tdata<165))
    tdata, f, dfminus, dfplus = [ar[cond] for ar in (tdata, f, dfminus, dfplus)]
    ax0.errorbar(tdata, f, yerr=(dfminus, dfplus), c='k', fmt='o')                                                             

    import time
    start = time.time()
    for hyst, ls in zip([False, True], ['-', '--']):
    # for hyst, ls in zip([ True, ], ['--', ]):
        Ropt_ = 10 * 7e10
        par_templ101417 = {'f_w': 1., 'f_d':58, 'f_p': 0.1,  't_enh_p': [0,],
                     't_enh_H': ['t2', ],  'enh_H':[1,],
                   'alpha0': -8. * RAD_IN_DEG, 'incl': 25. * RAD_IN_DEG,
                  'beta': 1., 'np_disk': 2.7, 'Bx': 5e11, 'Bopt': 0., 'E0_e': 1.,
                  'Ecut_e': 1., 'p_e': 1.7, 'Topt': 3.3e4,
                  'char_r': 1.5e13, 'wobble_ampl':0, 'phi_offset': 0 * pi/180,
                  'r_pr': 'broken_pl', 'R_t': 5 * Ropt_, 'Ropt': Ropt_, 
                  'if_boost': True, 'Gamma': 1.4, 'LoS_to_orb': 2.3, 'delta_pow': 3,
                  'MF_boost': False,  'delta':3e-2, 'enh_p': [1,],
                  'cooling': 'stat_mimic',  'hyst': hyst, 'logNorm': 43.7,
                  'eta_flow': 1,
                  'eta_syn': 1, 'eta_IC': 1.,  'lorentz_boost': True,
                  'simple': True, 'abs_photoel': True,  'abs_gg': False, 's_max': 'bow'}
        tpl = np.linspace(-210, 120, 361) * DAY
        (r_StoE, r_StoE_an, r_StoP, r_PtoE, us_rad, Fsx,  Bsstar_scal, 
        Bspuls_scal, Bstot_scal, Ftev, FI, G_ind) = Xray_Light_curve(tpl, par_templ101417, ifparallel = False, 
                                                                     swift_only=True)
        Fsx = fit_norm(tdata, f, tpl/DAY, Fsx)
        ax0.plot(tpl/DAY, Fsx, label = 'swift', ls=ls)
        ax0.set_yscale('log')
        # ax0.plot(tpl/DAY, Ftev*1e-9, label = 'hess')
        # ax0.plot(tpl/DAY, (FI-0.1)*1e-9, label = 'integral')
        # ax0.plot(tpl/DAY, (G_ind+0.1)*1e-9, label = 'ind')
        ax1.plot(tpl/DAY, r_StoE,ls=ls)
        ax1.plot(tpl/DAY, r_PtoE,ls=ls)
        ax1.plot(tpl/DAY, r_StoP,ls=ls, color='k')
        # ax1.plot(tpl/DAY, np.gradient(r_StoP, tpl), ls=ls, color='r')
        # ax1.plot(tpl/DAY, np.gradient(-r_PtoE, tpl), ls=ls, color='m')
        
        # ax1.plot(tpl/DAY, c_sound(1.4e-2, r_StoE), ls=ls, color='b')
        # ax1.plot(tpl/DAY, c_sound(1.4e-2, r_StoE), ls=ls, color='b')
        
        
        
        # print(c_sound(0.01, r_))
    
    # ax0.plot(tpl/DAY, Fsx, label = 'swift')
    # ax0.plot(tpl/DAY, Ftev, label = 'hess')
    # ax0.plot(tpl/DAY, FI, label = 'integral')
    # ax0.plot(tpl/DAY, (G_ind)*1e-9, label = 'ind')
    ax0.legend()
    print(time.time() - start)
    