# pulsar/lightcurve.py
import numpy as np
# from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import astropy.units as u
# import matplotlib.pyplot as plt
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
from astropy import constants as const
from ibsen.get_obs_data import get_parameters
from .utils import loggrid, trapz_loglog
# import ibsen
from ibsen.orbit import Orbit
from ibsen.winds import Winds
from ibsen.ibs import IBS
from ibsen.spec import SpectrumIBS
from ibsen.el_ev import ElectronsOnIBS

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
                to_cut_theta =  False, 
                where_cut_theta = pi/2,
                             # 1 = B_p = B_p_surf * (ns_r_scale / r_pe)

                ns_field_model = 'linear', ns_field_surf = 1, ns_r_scale = 1e13,
                ns_L_spindown = None, ns_sigma_magn = None,
                opt_field_model = 'linear', opt_field_surf = 0, opt_r_scale = 1e12,
                             
                             
                delta_power=4, lorentz_boost=True, simple=False,          # spec
                abs_photoel=True, abs_gg=False, nh_tbabs=0.8, syn_only=False,

                 
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
        self.syn_only = syn_only
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
        ibs_now = IBS( beta=None,
                    winds=self.winds, 
                    gamma_max=self.gamma_max, 
                    s_max=self.s_max, 
                    s_max_g=self.s_max_g, 
                    n=self.n_ibs, 
                    one_horn=False, 
                    t_to_calculate_beta_eff=t,
                )
        r_sp_now = self.orbit.r(t=t)
        r_se_now = self.winds.dist_se_1d(t=t)
        r_pe_now = r_sp_now - r_se_now
        nu_true = self.orbit.true_an(t=t)
        Bp_apex_now, Bopt_apex_now = self.winds.magn_fields_apex(t)
        dopl_factors_now = ibs_now.dopl(nu_true = nu_true)


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
                            to_cut_e = self.to_cut_e,
                            to_cut_theta = self.to_cut_theta,
                            where_cut_theta = self.where_cut_theta,
                            ) 

        dNe_de_IBS_now, e_vals_now = els_now.calculate(to_return=True, to_set_onto_ibs=True)
        spec_now = SpectrumIBS(els=els_now,
                                delta_power = self.delta_power,
                                lorentz_boost = self.lorentz_boost,
                                simple = self.simple,
                                abs_photoel = self.abs_photoel,
                                abs_gg = self.abs_gg,
                                nh_tbabs = self.nh_tbabs,
                                syn_only = self.syn_only,
                                distance = self.distance,
                                )
        if self.full_spec:
            E_ = np.logspace(2, 14, 1000),
        else:
            E_ = []
            for band in self.bands:
                E_in_band = loggrid(band[0], band[1], 30)
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
            n_jobs = multiprocessing.cpu_count() - 5
            res= Parallel(n_jobs=n_jobs)(delayed(func_to_parall)(i_t)
                                 for i_t in range(0, len(self.times)))

            r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, ibs_classes, els_classes, \
            dNe_des, e_es, spec_classes, e_phots, seds, seds_s, \
            fluxes, indexes, emiss_s = zip(*res)

            r_sps, r_pes, r_ses, nu_trues, B_p_apexs, B_opt_apexs, fluxes, indexes = [np.array(ar) 
                for ar in (r_sps, r_pes, r_ses, nu_trues, B_p_apexs, B_opt_apexs, fluxes, indexes)]
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
                 show_index=False, 
                     to_save=None, 
                     save_format='png', 
                     to_show_legend=True, 
                     to_show_grid=True,
                     to_show_title=True,
                     fontsize=12,
                     ylog=False,
                     **kwargs):
        """
        Plot the light curve.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            if  show_index:
                fig, ax = plt.subplots(nrows=2, ncols=1)
            if not show_index:
                fig, ax = plt.subplots(figsize=(10, 6))
        
        if show_index:
            ax_first = ax[0]
        else:
            ax_first = ax

        for i, band in enumerate(self.bands):
            log_lo = np.log10(band[0])
            log_hi = np.log10(band[1])
            ax_first.plot(self.times/DAY, 
                        self.fluxes[:, i], 
                        label=f'logE = {log_lo:.2}-{log_hi:.2} eV', **kwargs)
            
        if show_index:
            for i, band in enumerate(self.bands_ind):
                log_lo = np.log10(band[0])
                log_hi = np.log10(band[1])
                ax[1].plot(self.times/DAY, 
                            self.indexes[:, i], 
                            label=f'logE = {log_lo:.2}-{log_hi:.2} eV', **kwargs)
            

        if to_show_legend:
            ax_first.legend()
            if show_index:
                ax[1].legend()
        
        if to_show_grid:
            ax_first.grid()
            if  show_index:
                ax[1].grid()

        if to_show_title:
            ax_first.set_title('Light Curve')
            if show_index:
                ax[1].set_title('Index')
        
        if not show_index:
            ax_first.set_xlabel('t, days')
            ax_first.set_ylabel(r'$F$ erg s^-1 cm^-2')
            ax_first.tick_params(labelsize=fontsize)
        
        if show_index:
            ax[1].set_xlabel('t, days')
            ax_first.set_ylabel(r'$F$ erg s^-1 cm^-2')
            ax[1].set_ylabel(r'$\Gamma$')
            ax_first.tick_params(labelsize=fontsize)
            ax[1].tick_params(labelsize=fontsize)
        
        if ylog:
            ax_first.set_yscale('log')
        
        if to_save:
            plt.savefig(to_save, format=save_format)
        
        plt.show()







        