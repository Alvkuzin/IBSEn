# pulsar/lightcurve.py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
# from pathlib import Path
from numpy import pi, sin, cos
from joblib import Parallel, delayed
import multiprocessing
# import time
# import matplotlib.animation as animation
# from scipy.interpolate import splev, splrep, interp1d
from astropy import constants as const
from ibsen.get_obs_data import get_parameters
from ibsen.utils import loggrid#, trapz_loglog
# import ibsen
from ibsen.orbit import Orbit
from ibsen.winds import Winds
from ibsen.ibs import IBS
from ibsen.spec import SpectrumIBS
from ibsen.el_ev import ElectronsOnIBS

import pickle

G = float(const.G.cgs.value)
K_BOLTZ = float(const.k_B.cgs.value)
HBAR = float(const.hbar.cgs.value)
C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)
M_E = float(const.m_e.cgs.value)
E_ELECTRON = 4.803204e-10
MC2E = M_E * C_LIGHT**2
R_ELECTRON = E_ELECTRON**2 / MC2E

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)

ERG_TO_EV = 6.24E11 # 1 erg = 6.24E11 eV
DAY = 86400.
sed_unit = u.erg / u.s / u.cm**2
RAD_IN_DEG = pi / 180.0


def unpack_orbit(orb_type=None, T=None, e=None, M=None, nu_los=None, 
                 Topt=None, Ropt=None, Mopt=None,
                 dist=None):
    """
    Unpack orbital parameters with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        T, e, M, nu_los: float or None
            Explicit values that override defaults.

    Returns:
        Tuple of T, e, M, nu_los
    """
    # Step 1: Determine the source of defaults
    if isinstance(orb_type, str):
        known_types = ['psrb', 'rb', 'bw']
        if orb_type not in known_types:
            raise ValueError(f"Unknown orbit type: {orb_type}")
        defaults = get_parameters(orb_type)
    elif isinstance(orb_type, dict):
        defaults = orb_type
    else:
        defaults = {}

    # Step 2: Build final values, giving priority to explicit arguments
    T_final = T if T is not None else defaults.get('T')
    e_final = e if e is not None else defaults.get('e')
    M_final = M if M is not None else defaults.get('M')
    nu_los_final = nu_los if nu_los is not None else defaults.get('nu_los')
    Topt_final = Topt if Topt is not None else defaults.get('Topt')
    Ropt_final = Ropt if Ropt is not None else defaults.get('Ropt')
    Mopt_final = Mopt if Mopt is not None else defaults.get('Mopt')
    # Handle distance, if provided
    dist_final = dist if dist is not None else defaults.get('D')


    # # Add any additional parameters you expect
    # # For example, if you want to also unpack `omega`, `i`, etc.:
    result = [T_final, e_final, M_final, nu_los_final, Topt_final,
              Ropt_final, Mopt_final, dist_final]
    # T_, e_, mtot_, nu_los_, Ropt_, Topt_, Mopt_, distance_
    # for key in kwargs:
    #     value = kwargs[key] if kwargs[key] is not None else defaults.get(key)
    #     result.append(value)

    return tuple(result)

class LightCurve:
    def __init__(self,
                 
                 times, bands = ( [3e2, 1e4], ), bands_ind = ( [3e3, 1e4], ),
                    to_parall=False, # lc itself
                 full_spec = False,
                 
                 sys_name=None, period=None, e=None, tot_mass=None, nu_los=None,
                 Ropt=None, Topt=None, Mopt=None,  distance = None,
                 
                 M_ns = 1.4*M_SOLAR, f_p = 0.1, 
                 alpha_deg=0, incl_deg=30.,   
                 f_d=10., np_disk = 3., delta=0.01, 
                 height_exp = 0.5,
                 rad_prof = 'pl', r_trunk = None,
                 
                 s_max=1., gamma_max=3., s_max_g=4., n_ibs=31,   # ibs
                             
                              
                cooling='stat_mimic', to_inject_e = 'ecpl',   # el_ev
                to_inject_theta = '3d', ecut = 1.e12, p_e = 2., norm_e = 1.e37,
                eta_a = 1.,
                eta_syn = 1., eta_ic = 1.,
                emin = 1e9, emax = 5.1e14, to_cut_e = True, 
                emin_grid=1e8, emax_grid=5.1e14,
                to_cut_theta =  False, 
                where_cut_theta = pi/2,

                ns_field_model = 'linear', ns_field_surf = 1, ns_r_scale = 1e13,
                ns_L_spindown = None, ns_sigma_magn = None,
                opt_field_model = 'linear', opt_field_surf = 0, opt_r_scale = 1e12,
                             
                             
                delta_power=4, lorentz_boost=True, simple=False,          # spec
                abs_photoel=True, abs_gg=False, nh_tbabs=0.8,
                ic_ani=False, apex_only=False, mechanisms=['syn', 'ic'],
                 
                ):
        ####################################################################
        self.times = times
        self.bands = bands
        self.bands_ind = bands_ind
        self.to_parall = to_parall
        self.full_spec = full_spec # if to calculate spec just in bands=bands 
                                   # or across all energies
        #####################################################################
        self.sys_name = sys_name
        T_, e_, mtot_, nu_los_, Topt_, Ropt_, Mopt_, distance_ = unpack_orbit(
            orb_type=sys_name, T=period, e=e, M=tot_mass, nu_los=nu_los,
                Ropt=Ropt, Topt=Topt, Mopt=Mopt, dist=distance) 
        self.period = T_
        self.e = e_  
        self.tot_mass = mtot_
        self.nu_los = nu_los_
        self.Ropt = Ropt_
        self.Topt = Topt_
        self.Mopt = Mopt_
        self.distance = distance_
        self.M_ns = M_ns
        ####################################################################
        self.f_p = f_p
        self.alpha_deg = alpha_deg
        self.incl_deg = incl_deg
        self.f_d = f_d
        self.np_disk = np_disk
        self.delta = delta
        self.height_exp = height_exp
        self.rad_prof = rad_prof
        self.r_trunk = r_trunk
        ####################################################################
        self.s_max = s_max
        self.gamma_max = gamma_max
        self.s_max_g = s_max_g
        self.n_ibs = n_ibs
        ####################################################################
        self.cooling = cooling
        self.to_inject_e = to_inject_e
        self.to_inject_theta = to_inject_theta
        self.ecut = ecut
        self.p_e = p_e
        self.norm_e = norm_e
        self.eta_a = eta_a
        self.eta_syn = eta_syn
        self.eta_ic = eta_ic
        self.emin = emin
        self.emax = emax
        self.emin_grid = emin_grid
        self.emax_grid = emax_grid
        
        self.to_cut_e = to_cut_e
        self.to_cut_theta = to_cut_theta
        self.where_cut_theta = where_cut_theta
        ####################################################################
        self.ns_field_model = ns_field_model
        self.ns_field_surf = ns_field_surf
        self.ns_r_scale = ns_r_scale
        self.ns_L_spindown = ns_L_spindown
        self.ns_sigma_magn = ns_sigma_magn
        self.opt_field_model = opt_field_model
        self.opt_field_surf = opt_field_surf
        self.opt_r_scale = opt_r_scale
        ####################################################################
        self.delta_power = delta_power
        self.lorentz_boost = lorentz_boost
        self.simple = simple
        self.abs_photoel = abs_photoel
        self.abs_gg = abs_gg
        self.nh_tbabs = nh_tbabs
        self.ic_ani = ic_ani
        self.apex_only = apex_only
        self.mechanisms = mechanisms
        # self.syn_only = syn_only
        ####################################################################
        self.orbit = None
        self.winds = None
        self.set_orbit()
        self.set_winds()

    ############################################################
    def set_orbit(self):
        """Set the orbit object based on the system parameters.
        """
        if self.orbit is None:

            orb = Orbit(sys_name = self.sys_name,
                        period=self.period,
                        e=self.e,
                        tot_mass=self.tot_mass,
                        nu_los=self.nu_los, n=None)
            
            self.orbit = orb
        
    ############################################################    
    def set_winds(self):
        """
        Set the winds object based on the orbit and other parameters.
        """
        if self.orbit is None:
            self.set_orbit()
        orb = self.orbit
        
        winds = Winds(orbit=orb, 
                       sys_name = self.sys_name,
                       alpha=self.alpha_deg/180*pi, 
                       incl=self.incl_deg*pi/180,
                       f_d=self.f_d,
                       # f_d=100,
                       f_p=self.f_p, 
                       delta=self.delta,
                       np_disk=self.np_disk,
                       rad_prof=self.rad_prof,
                       height_exp=self.height_exp,
                       r_trunk=self.r_trunk,

                       Ropt = self.Ropt,
                       Topt=self.Topt, 
                       Mopt=self.Mopt,

                       ns_field_model = self.ns_field_model,
                       ns_field_surf = self.ns_field_surf,
                       ns_r_scale = self.ns_r_scale,
                       ns_L_spindown = self.ns_L_spindown,
                       ns_sigma_magn = self.ns_sigma_magn,
                       opt_field_model = self.opt_field_model,
                       opt_field_surf = self.opt_field_surf,
                       opt_r_scale = self.opt_r_scale,
                       )
        self.winds = winds

    def calculate_at_time(self, t):
        """
        return: r_sp, r_pe, r_se, Bp_apex, Bopt_apex, ibs, els, dNe_de_IBS, e_vals,
        spec, E_ph, sed_tot, sed_s, fluxes, indexes, emissiv
        """
        ibs_now = IBS(winds=self.winds, 
                    gamma_max=self.gamma_max, 
                    s_max=self.s_max, 
                    s_max_g=self.s_max_g, 
                    n=self.n_ibs, 
                    t_to_calculate_beta_eff=t,
                )
        # ibs_now = ibs_now_nonscaled.rescale_to_position()
        r_sp_now = self.orbit.r(t=t)
        r_se_now = self.winds.dist_se_1d(t=t)
        r_pe_now = r_sp_now - r_se_now
        # nu_true = self.orbit.true_an(t=t)
        Bp_apex_now, Bopt_apex_now = self.winds.magn_fields_apex(t)
        # print(Bp_apex_now, Bopt_apex_now)
        # dopl_factors_now = ibs_now.dopl

        # print('a!')
# 
        els_now = ElectronsOnIBS(ibs = ibs_now,
                            Bp_apex = Bp_apex_now,
                            Bs_apex = Bopt_apex_now,   
                            cooling=self.cooling,
                            eta_a = self.eta_a,
                            eta_syn = self.eta_syn,
                            eta_ic = self.eta_ic,
                            to_inject_e = self.to_inject_e,
                            to_inject_theta = self.to_inject_theta,
                            ecut = self.ecut,
                            p_e=self.p_e,
                            norm_e = self.norm_e,
                            emin = self.emin,
                            emax = self.emax,
                            emin_grid = self.emin_grid,
                            emax_grid = self.emax_grid,
                            to_cut_e = self.to_cut_e,
                            to_cut_theta = self.to_cut_theta,
                            where_cut_theta = self.where_cut_theta,
                            ) 

        dNe_de_IBS_now, e_vals_now = els_now.calculate(to_return=True)
        spec_now = SpectrumIBS(els=els_now,
                                delta_power = self.delta_power,
                                lorentz_boost = self.lorentz_boost,
                                simple = self.simple,
                                abs_photoel = self.abs_photoel,
                                abs_gg = self.abs_gg,
                                nh_tbabs = self.nh_tbabs,
                                mechanisms=self.mechanisms,
                                apex_only=self.apex_only,
                                ic_ani=self.ic_ani,
                                distance = self.distance,
                                )
        if self.full_spec:
            E_ = np.logspace(2, 14, 1000)
        else:
            E_ = []
            for band in self.bands:
                E_in_band = loggrid(band[0]/1.1, band[1]*1.1, 67)
                E_.append(E_in_band)
            E_ = np.concatenate(E_)

        E_ph_now, sed_tot_now, sed_s_now = spec_now.calculate_sed_on_ibs(E =  E_,                                         
                                        to_set_onto_ibs=True,
                                        to_return=True)
        
        emissiv_now = trapezoid(sed_s_now/E_ph_now, E_ph_now, axis=1)
        fluxes_now = spec_now.fluxes(bands=self.bands)
        indexes_now = spec_now.indexes(bands=self.bands_ind)
        return (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                emissiv_now,
                )

    ############################################################
    def calculate(self):

        fluxes = np.zeros((self.times.size, len(self.bands)))
        indexes = np.zeros((self.times.size, len(self.bands_ind)))
        r_sps = np.zeros(self.times.size)
        r_ses = np.zeros(self.times.size)
        r_pes = np.zeros(self.times.size)
        B_p_apexs = np.zeros(self.times.size)
        B_opt_apexs = np.zeros(self.times.size)
        nu_trues = np.zeros(self.times.size)
        ibs_classes = []
        els_classes = []
        spec_classes = []
        dNe_des = []
        e_es = []
        seds = []
        seds_s = []
        e_phots = []
        emiss_s = []
        
        if not self.to_parall:
            for i_t, t in enumerate(self.times):
                (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                    ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    ) = self.calculate_at_time(t)

                ###################################
                r_sps[i_t] = r_sp_now
                r_pes[i_t] = r_pe_now
                r_ses[i_t] = r_se_now
                B_p_apexs[i_t] = Bp_apex_now
                B_opt_apexs[i_t] = Bopt_apex_now 
                ###################################
                els_classes.append(els_now)
                ibs_classes.append(ibs_now)
                dNe_des.append(dNe_de_IBS_now)
                e_es.append(e_vals_now)
                ####################################
                spec_classes.append(spec_now)
                e_phots.append(E_ph_now)
                seds.append(sed_tot_now)
                seds_s.append(sed_s_now)
                fluxes[i_t, :] = spec_now.fluxes(bands=self.bands)
                indexes[i_t, :] = spec_now.indexes(bands=self.bands_ind)
                emiss_s.append(emissiv_now)
                ####################################
                
        if self.to_parall:
            def func_to_parall(i_t):
                (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                    ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    ) = self.calculate_at_time(self.times[i_t])
                
                return (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                    ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    )
            n_jobs = max(multiprocessing.cpu_count() - 5, 1)
            res= Parallel(n_jobs=n_jobs)(delayed(func_to_parall)(i_t)
                                 for i_t in range(0, len(self.times)))

            r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, ibs_classes, els_classes, \
            dNe_des, e_es, spec_classes, e_phots, seds, seds_s, \
            fluxes, indexes, emiss_s = zip(*res)

            r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, fluxes, indexes = [np.array(ar) 
                for ar in (r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, fluxes, indexes)]
            ibs_classes = list(ibs_classes)
            els_classes = list(els_classes)
            dNe_des = list(dNe_des)
            e_es = list(e_es)
            spec_classes = list(spec_classes)
            e_phots = list(e_phots)
            seds = [np.array(ar) for ar in seds]
            seds_s = [np.array(ar) for ar in seds_s]
            emiss_s = [np.array(ar) for ar in emiss_s]
        

        self.r_sps = r_sps
        self.r_ses = r_ses
        self.r_pes = r_pes
        self.B_p_apexs = B_p_apexs
        self.B_opt_apexs = B_opt_apexs
        self.nu_trues = nu_trues
        self.ibs_classes = ibs_classes
        self.els_classes = els_classes
        self.spec_classes = spec_classes
        self.dNe_des = dNe_des
        self.e_es = e_es
        self.seds = seds
        self.seds_s = seds_s
        self.e_phots = e_phots
        self.emiss_s = emiss_s
        self.fluxes = fluxes
        self.indexes = indexes    

    def peek(self,
                ax=None, 
                **kwargs):
        

        """
        Plot the light curve.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=1, ncols=4,
                                    figsize=(16, 4))
        
        ax_first = ax[0]


        for i, band in enumerate(self.bands):
            log_lo = np.log10(band[0])
            log_hi = np.log10(band[1])
            ax_first.plot(self.times/DAY, 
                        self.fluxes[:, i], 
                        label=f'logE = {log_lo:.2}-{log_hi:.2} eV', **kwargs)
            
        for i, band in enumerate(self.bands_ind):
            log_lo = np.log10(band[0])
            log_hi = np.log10(band[1])
            
            ax[1].plot(self.times/DAY, 
                        self.indexes[:, i], 
                        label=f'logE = {log_lo:.2f}-{log_hi:.2f} eV', **kwargs)
        
        _nt = self.times.size
        for i_t in (int(0.2*_nt), int(0.5*_nt), int(0.8*_nt)):
            t_now_days = self.times[i_t] / DAY
            ax[2].scatter(self.e_phots[i_t], self.seds[i_t],
                        label=f't = {t_now_days:.2f} days',  **kwargs,
                        )
            ax[3].plot(self.ibs_classes[i_t].s_mid, self.emiss_s[i_t],
                        label=f't = {t_now_days:.2f} days', **kwargs)

        
        ax_first.grid()
        for i in range(4):   
            ax[i].legend()
            ax[i].grid()
            if i == 2: 
                ax[i].set_xscale('log')
                ax[i].set_yscale('log')
            if i == 0:
                ax[i].set_yscale('log')

        ax_first.set_title('Light Curve')
        ax[1].set_title('Index')
        ax[2].set_title('SED')
        ax[3].set_title('Emissivity')
        
        ax_first.set_xlabel('t, days')
        ax_first.set_ylabel(r'$F$ erg s^-1 cm^-2')
        
        ax[1].set_xlabel('t, days')
        ax[2].set_xlabel('E, eV')
        ax[3].set_xlabel(r'$s$')

        maxsed = np.nanmax(self.seds)
        ax[2].set_ylim(1e-3*maxsed, maxsed*1.5)

        plt.show()

if __name__ == "__main__":

# from ibsen.lc import LightCurve
    t1 = np.linspace(-650, -40, 40) * DAY
    t2 = np.linspace(-40, 90, 70) * DAY
    # t2 = np.linspace(-15, -15, 1) * DAY
    t3 = np.linspace(100, 650, 40) * DAY
    #ts = np.concatenate((t1, t2, t3))
    ts=t2
    lc = LightCurve(sys_name = 'psrb',
                    n_ibs = 16,
                    p_e = 1.7, 
                    times = ts,
                    bands = ([3e2, 1e4], [4e11, 1e13],
                             ),
                    bands_ind = ([3e3, 1e4],),
                    full_spec = False,
                    to_parall = True, 
                    f_d = 150,
                    mechanisms=['syn', 'ic'],
                    apex_only=False,
                    ic_ani=False,
                    simple = True,
                    alpha_deg = -8.,
                    s_max = 1,
                    gamma_max=1.2,
                    delta=0.01,
                    cooling='stat_mimic',
                    eta_a=1, # you may want to set that to 1e20 if cooling='adv'
                    ns_field_surf=1,
                    
                    
                    abs_gg=False
                   )
    import time
    start = time.time()
    lc.calculate()
    print(f'LC done in {time.time() - start}')
    lc.peek()
    # fig, ax = plt.subplots(nrows=3, sharex=True)
    # # DAY=86400
    # Ne_e = []
    # Ninj = []
    # edots = []
    # i_show = 0
    # for i_t in range(lc.times.size):
    #     e_cl_ =  lc.els_classes[i_t]
    #     n_ = e_cl_.ibs.n
    #     Ne_e.append( np.sum(trapezoid(lc.dNe_des[i_t], lc.e_es[i_t], axis=1)) )
    #     # Ne_e.append( lc.dNe_des[i_t][16, 101] )
    #     # print(e_cl_.ibs.s.size)
    #     smesh, emesh = np.meshgrid(e_cl_.ibs.s[n_:2*n_], e_cl_.e_vals, indexing='ij')
    #     f_inj_now = e_cl_.f_inject(smesh, emesh)
    #     edot_a_now = e_cl_.edot(smesh, emesh)[n_-1, :]
        
    #     Ninj.append(np.sum(trapezoid(f_inj_now, e_cl_.e_vals)))
    #     edots.append((e_cl_.e_vals / edot_a_now)[0])
    # # Ninjected = np.array([ np.sum(trapezoid(lc.dNe_des[i], lc.e_es[i], axis=1))
    #                       # 
    #                       # for i in range(lc.times.size)])
    # # Ne_e = np.array([ (trapezoid((trapezoid(lc.dNe_des[i], lc.e_es[i], axis=1)), lc.ibs_classes[i].s ) /
    # #                    trapezoid(np.ones(lc.ibs_classes[i].n), lc.ibs_classes[i].s[lc.ibs_classes[i].n:2*lc.ibs_classes[i].n] )) 
    # #                  for i in range(lc.times.size)])
    # Ne_e = np.array(Ne_e)
    # Ninj = np.array(Ninj)
    # edots = np.array(edots)
    # ax[0].plot(lc.times/DAY, lc.fluxes)
    # # print(lc.fluxes)
    # ax[1].plot(lc.times/DAY, Ne_e)
    # # if i_t == 1:
    # print(lc.times[i_show]/DAY)
    # ax[2].plot(lc.times/DAY, edots)    
    # # ax[2].plot(lc.spec_classes[i_show].e_ph, lc.spec_classes[i_show].sed)
    # # ax[2].set_xscale('log')
    # # ax[2].set_yscale('log')
        
    # # ax[1].plot(lc.times/DAY, lc.r_pes/lc.orbit.r_periastr)
    # # ax[1].plot(lc.times/DAY, lc.r_ses/lc.orbit.r_periastr)
    
    # # ax[1].plot(lc.times/DAY, lc.B_p_apexs)
    # # ax[1].plot(lc.times/DAY, lc.B_opt_apexs)
    
    
    # ax[0].set_yscale('log')
    





        