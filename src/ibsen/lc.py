# ibsen/lc.py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.integrate import trapezoid
from numpy import pi
from joblib import Parallel, delayed
import multiprocessing
from scipy.interpolate import interp1d
from astropy import constants as const
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params, loggrid, fill_nans
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

docstr_lc = f"""
Broadband light curve builder for an intrabinary shock (IBS) system.

For a grid of times, this class assembles the full pipeline —
:class:`Orbit` → :class:`Winds` → :class:`IBS` → :class:`ElectronsOnIBS`
→ :class:`SpectrumIBS` — and tabulates band fluxes, photon indices,
SEDs, and auxiliary physical quantities (separations, apex fields, etc.).
Energies are in eV, distances in cm, times in seconds, and SEDs in
erg s⁻¹ cm⁻².

Parameters
----------
times : array_like, shape (Nt,)
    Time grid (relative to periastron, seconds) at which to compute the LC.
bands : iterable of (float, float), optional
    Energy bands [e1, e2] (eV) for band-integrated fluxes. Default ``([3e2, 1e4],)``.
bands_ind : iterable of (float, float), optional
    Energy bands (eV) for photon-index fits. Default ``([3e3, 1e4],)``.
epows : None | scalar | iterable, optional
    Moment order(s) for band flux integrals (see :meth:`SpectrumIBS.fluxes`).
    If None, epow=1 is used for all bands. If scalar, the same epow is used
    for all; if iterable, must match ``len(bands)``.
to_parall : bool, optional
    If True, compute times in parallel with joblib. Default False.
n_cores : int, or 'all', or None, optional
    Number of CPU cores to use in paralleling. If 'all' or None (default),
    all cores are used. 
full_spec : bool, optional
    If True, compute SEDs on a global grid (``logspace(2,14,1000)``);
    otherwise build per-band grids expanded by ±20% and concatenate.
    Default False.

# Orbit / system parameters (forwarded via ``unpack_orbit``)
    sys_name : {known_names}, or None, optional
        If provided, load default systep parameters via
        ``ibsen.get_obs_data.get_parameters(sys_name)``; explicit arguments
        below override those defaults. 
        If None, all parameters must be contained in `sys_params` dictionary or
        given explicitly.
    sys_params : dict or None, optional
          If provided, a dictionary of orbital parameters to use instead of
          default values from ``get_parameters``. Keys are: 
            `T`, `e`, `M`, `nu_los`, `incl_los`, 'Ropt', 'Mopt', 'Topt', 'D'.
          Explicit arguments below override those in
          this dictionary. If None, all parameters must be given explicitly.  
    T, e, M, nu_los, incl_los : float, optional
        Orbital period (s), eccentricity, total mass (g), LoS positional 
        angle (rad), LoS inclination (rad).


# Winds (external media & fields)
    Ropt, Topt, Mopt : float, optional
        Optical star radius (cm), temperature (K), and mass (g).
    M_ns : float, optional
        Neutron-star mass (g). Default ``1.4*M_SOLAR``.
    f_p : float, optional
        Pulsar-wind pressure normalization. Default 0.1.
    alpha_deg, incl_deg : float, optional
        Disk orientation angles in degrees (see :class:`Winds`). Defaults 0 and 30.
    f_d : float, optional
        Disk pressure normalization. Default 10.
    p_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk pressure. Default [1,].
    h_enh : list of floats and lists of length 2, optional
        A list of multipliers for the disk height. Default [1,].
    p_enh_times : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of times for pressure enhancement. Default [0,].
    h_enh_times : list of floats, or lists of length 2, or {'t1', 't2'}, optional
        A list of times for pressure enhancement. Default [0,].
    np_disk : float, optional
        Disk radial power-law index. Default 3.
    delta : float, optional
        Disk opening half-angle at the stellar surface. Default 0.01.
    height_exp : float, optional
        Exponent in the disk opening law. Default 0.5.
    rad_prof : {'pl','bkpl'}, optional
        Disk radial profile (power law or broken power law). Default 'pl'.
    r_trunk : float or None, optional
        Disk truncation radius (cm) for broken-profile models.
    ns_b_model, ns_b_ref, ns_r_ref, ns_L_spindown, ns_sigma_magn
        Pulsar magnetic-field model and parameters (see :class:`Winds`).
    opt_b_model, opt_b_ref, opt_r_ref
        Stellar magnetic-field model and parameters.

# IBS geometry
    s_max : float, optional
        Dimensionless arclength cutoff passed to :class:`IBS_norm`. Default 1.0.
    gamma_max : float, optional
        Max bulk Lorentz factor at ``s_max_g``. Default 3.0.
    s_max_g : float, optional
        Arclength at which ``gamma==gamma_max`` (dimensionless). Default 4.0.
    n_ibs : int, optional
        Sampling points (per horn) for IBS construction. Default 31.
    
    # Electrons on IBS
    cooling : {'no','stat_apex','stat_ibs','stat_mimic',
               'leak_apex','leak_ibs','leak_mimic','adv'}, optional
        Cooling/evolution mode for :class:`ElectronsOnIBS`. Default 'stat_mimic'.
    to_inject_e : {'pl','ecpl','secpl'}, optional
        Injection law in energy. Default 'ecpl'.
    to_inject_theta : {'2d','3d'}, optional
        Spatial weighting along IBS. Default '3d'.
    ecut : float, optional
        Cutoff energy for ECPL/SECPL (eV). Default 1e12.
    p_e : float, optional
        Injection spectral index. Default 2.0.
    norm_e : float, optional
        Injection normalization (s⁻¹). Default 1e37.
    eta_a, eta_syn, eta_ic : float, optional
        Multipliers for adiabatic, synchrotron, and IC terms. Defaults 1, 1, 1.
    emin, emax : float, optional
        Injection energy band (eV). Defaults 1e9, 5.1e14.
    emin_grid, emax_grid : float, optional
        Solver grid bounds (eV). Defaults 1e8, 5.1e14.
    to_cut_e : bool, optional
        Zero injection outside [emin, emax]. Default True.
    to_cut_theta : bool, optional
        Cut injection to |θ| < ``where_cut_theta``. Default False.
    where_cut_theta : float, optional
        Angular cut (rad) if ``to_cut_theta`` is True. Default π/2.

# Spectrum / radiation
    distance : float or None, optional
        Source distance (cm); if None and a named system is used, taken from
        system defaults.
    delta_power : float, optional
        Doppler weight exponent (segment integration). Default 4.
    lorentz_boost : bool, optional
        Apply comoving-frame treatment/boosting. Default True.
    simple : bool, optional
        Use apex SED + scaling instead of per-segment radiation. Default False.
    abs_photoel : bool, optional
        Apply photoelectric absorption. Default True.
    abs_gg : bool, optional
        Apply γγ absorption along LoS (system-dependent). Default False.
    abs_gg_filename: str or None, optional
        Name of the file with tabulated gamma-gamma absorbtion. If None,
        (default), IBSEn tries to find a tabulated
        gg-abs file if `sys_name` is a known system.
    nh_tbabs : float, optional
        Column density for photoelectric absorption (10²² cm⁻² units used by helper).
        Default 0.8.
    ic_ani : bool, optional
        Use anisotropic IC (requires IBS angles). Default False.
    apex_only : bool, optional
        Compute only the apex contribution (no curve integration). Default False.
    mechanisms : list of {'syn','ic'}, optional
        Emission mechanisms to include. Default ``['syn','ic']``.

Attributes
----------
orbit : Orbit
    Initialized orbit object built from the provided/system parameters.
winds : Winds
    Wind/disk/field model bound to the orbit.
r_sps : ndarray, shape (Nt,)
    Star–pulsar separation r_sp(t) [cm].
r_pes : ndarray, shape (Nt,)
    Pulsar→apex distance r_pe(t) [cm].
r_ses : ndarray, shape (Nt,)
    Star→apex distance r_se(t) [cm].
B_p_apexs, B_opt_apexs : ndarray, shape (Nt,)
    Pulsar/stellar magnetic fields at the apex [G].
winds_classes : list of Winds 
    Winds objects at each time.
ibs_classes : list of IBS
    IBS objects at each time.
els_classes : list of ElectronsOnIBS
    Electron-population objects at each time.
spec_classes : list of SpectrumIBS
    Spectrum calculators at each time.
dNe_des : list of ndarray, length Nt
    Electron distributions on the IBS, each with shape (Ns, Ne) [1 s⁻¹ cm⁻¹ eV⁻¹].
e_es : list of ndarray, length Nt
    Energy grids corresponding to ``dNe_des`` (eV).
seds : list of ndarray, length Nt
    Total SEDs per time (erg s⁻¹ cm⁻²).
seds_s : list of ndarray, length Nt
    Per-segment SEDs with shape (n_segments, Ne_ph) (erg s⁻¹ cm⁻²).
e_phots : list of ndarray, length Nt
    Photon-energy grids used for the SEDs (eV).
emiss_s : list of ndarray, length Nt
    Emissivity integrated over photon energy along arclength [erg s⁻¹ cm⁻¹].
fluxes : ndarray, shape (Nt, Nb)
    Band fluxes in ``bands`` (erg s⁻¹ cm⁻²).
indexes : ndarray, shape (Nt, Ni)
    Photon indices fitted in ``bands_ind``.

Methods
-------
set_orbit()
    Build and assign the :class:`Orbit` object from inputs/defaults.
set_winds()
    Build and assign the :class:`Winds` object tied to the orbit.
calculate_at_time(t)
    Compute IBS, electrons, spectrum, and derived quantities at one time;
    returns a tuple of all per-epoch results.
calculate()
    Loop over all times (optionally in parallel), fill attributes listed above.
peek(ax=None, **kwargs)
    Quick-look plot with four panels: band fluxes, photon indices, SED at
    three representative times, and emissivity along the IBS.

Notes
-----
* When ``full_spec=False``, the SED energy grid is assembled from all bands
  expanded to [e/1.2, e*1.2] and concatenated; this accelerates LC runs focused
  on specific bands.
* Parallel execution uses ``joblib.Parallel`` with up to ``cpu_count()-5``
  workers (minimum 1).

"""

class LightCurve:
    __doc__ = docstr_lc
    def __init__(self,
                 
                 times, bands = ( [3e2, 1e4], ), bands_ind = ( [3e3, 1e4], ),
                 epows=None,
                    to_parall=False, # lc itself
                    n_cores=None,
                 full_spec = False,
                 sys_name=None, sys_params=None,
                 T=None, e=None, M=None, nu_los=None,
                 incl_los=None,
                 Ropt=None, Topt=None, Mopt=None,  distance = None,
                 allow_missing=False,
                 
                 M_ns = 1.4*M_SOLAR, f_p = 0.1, 
                 alpha_deg=0, incl_deg=30.,   
                 f_d=10., np_disk = 3., delta=0.01, 
                 p_enh = [1, ], p_enh_times = [0, ], 
                 h_enh = [1, ], h_enh_times = [0, ],
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

                ns_b_model = 'linear', ns_b_ref = 1, ns_r_ref = 1e13,
                ns_L_spindown = None, ns_sigma_magn = None,
                opt_b_model = 'linear', opt_b_ref = 0, opt_r_ref = 1e12,
                             
                             
                delta_power=4, lorentz_boost=True, method='full',          # spec
                abs_photoel=True, abs_gg=False, abs_gg_filename=None, nh_tbabs=0.8,
                ic_ani=False, mechanisms=['syn', 'ic'],
                mode='int',
                ne_mult=0.7,
                nEed_syn = None,
                nEed_ic = None,
                 
                ):
        ############## ---------- LC own arguments ---------- #################
        self.t = times # array of t to calc LC on
        self.bands = bands # tuple of bands ([e1, e2], [e3, e4]) to calc flux in
        self.epows = epows # moment order for flux calc 
        self.bands_ind = bands_ind # tuple of bands to estimate phot index in
        self.to_parall = to_parall # whether to parall with Parallel
        self.n_cores = n_cores
        self.full_spec = full_spec # if to calculate spec just in bands=bands 
                                   # or across all energies
        ################ ---- arguments from orbit ---- #######################
        self.calculated = False
        self.sys_name = sys_name
        self.sys_params=sys_params
        self.allow_missing = allow_missing
        (T_, e_, M_, nu_los_, incl_los_, Topt_, Ropt_,
         Mopt_, distance_) = unpack_params(('T', 'e', 'M', 'nu_los',
            'incl_los', 'Topt', 'Ropt', 'Mopt', 'D'),
            orb_type=sys_name, 
            sys_params=sys_params,
            known_types=known_names,
            get_defaults_func=get_parameters,
            T=T, e=e, M=M, nu_los=nu_los, incl_los=incl_los,
            Topt=Topt, Ropt=Ropt, Mopt=Mopt, D=distance,
            allow_missing=allow_missing)
        self.T = T_
        self.e = e_  
        self.M = M_
        self.nu_los = nu_los_
        self.incl_los = incl_los_
        self.Ropt = Ropt_
        self.Topt = Topt_
        self.Mopt = Mopt_
        self.distance = distance_
        self.M_ns = M_ns
        ################ ---- arguments from winds ---- #######################
        self.f_p = f_p
        self.alpha_deg = alpha_deg
        self.incl_deg = incl_deg
        self.f_d = f_d
        self.p_enh = p_enh
        self.p_enh_times = p_enh_times
        self.h_enh = h_enh
        self.h_enh_times = h_enh_times
        self.np_disk = np_disk
        self.delta = delta
        self.height_exp = height_exp
        self.rad_prof = rad_prof
        self.r_trunk = r_trunk
        ####### --------- also from winds, about magn fields --------- ########
        self.ns_b_model = ns_b_model
        self.ns_b_ref = ns_b_ref
        self.ns_r_ref = ns_r_ref
        self.ns_L_spindown = ns_L_spindown
        self.ns_sigma_magn = ns_sigma_magn
        self.opt_b_model = opt_b_model
        self.opt_b_ref = opt_b_ref
        self.opt_r_ref = opt_r_ref
        ################ ----- arguments from ibs ----- #######################
        self.s_max = s_max
        self.gamma_max = gamma_max
        self.s_max_g = s_max_g
        self.n_ibs = n_ibs
        ################ ---- arguments from el_ev ---- #######################
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
        ################ ---- arguments from spec ----- #######################
        self.delta_power = delta_power
        self.lorentz_boost = lorentz_boost
        self.method = method
        self.abs_photoel = abs_photoel
        self.abs_gg = abs_gg
        self.abs_gg_filename = abs_gg_filename
        
        self.nh_tbabs = nh_tbabs
        self.ic_ani = ic_ani
        self.mechanisms = mechanisms
        self.mode = mode
        self.ne_mult = ne_mult
        self.nEed_syn = nEed_syn
        self.nEed_ic = nEed_ic
        
        ####################################################################
        self.orbit = None
        self.set_orbit()

    ############################################################
    def set_orbit(self):
        """Set the orbit object based on the system parameters.
        """
        if self.orbit is None:

            orb = Orbit(sys_name = self.sys_name,
                        sys_params=self.sys_params,
                        T=self.T,
                        e=self.e,
                        M=self.M,
                        nu_los=self.nu_los,
                        incl_los=self.incl_los,
                        allow_missing=self.allow_missing,
                        n=1001)
            
            self.orbit = orb

    def calculate_at_time(self, t_):
        """
        Auxillary function to calculate a lot of stuff at one moment of time.
        Initializes IBS, then ElectronsOnIBS, then SpectrumIBS, and calculates
        a bunch of stuff.

        Parameters
        ----------
        t : float
            Time to calculate verything at [s].

        Returns
        -------
        tuple of (
                  r_sp (float): distance from the pulsar to the star [cm], \\
                  r_pe (float): distance from the pulsar to the apex of IBS [cm], \\
                  r_se (float): distance from the star to the apex of IBS [cm], \\
                  Bp_apex (float): magnetic field at the apex of IBS due to pulsar [G], \\
                  Bopt_apex (float): magnetic field at the apex of IBS due to star [G], \\
                  winds (Winds object): the Winds object at time t, \\
                  ibs (IBS object): the IBS object at time t, \\
                  els (ElectronsOnIBS object): the ElectronsOnIBS object at time t, \\
                  dNe_de_IBS (2d np.array): electron distribution on IBS [1/s/cm/eV], \\
                  e_vals (1d np.array): electron energies grid [eV], \\
                  spec (SpectrumIBS object): the SpectrumIBS object at time t, \\
                  E_ph (1d np.array): photon energies grid [eV], \\
                  sed_tot (1d np.array): total SED on IBS [erg/s/cm2], \\ 
                  sed_s (2d np.array): SED in every segment of IBS [erg/s/cm2], \\
                  fluxes (1d np.array): fluxes in self.bands [erg/s/cm2], \\
                  indexes (1d np.array): photon indexes in self.bands_ind, \\
                  emissiv (1d np.array): emissivity along the IBS [erg/s/cm]. \\
                  )

        """
        # print('lc ', self.h_enh_times)
        winds_now = Winds(orbit=self.orbit, 
                sys_name = self.sys_name,
                alpha=self.alpha_deg/180*pi, 
                incl=self.incl_deg*pi/180,
                f_d=self.f_d,
                t_forwinds = t_,
                p_enh = self.p_enh,
                p_enh_times = self.p_enh_times,
                h_enh = self.h_enh,
                h_enh_times = self.h_enh_times,
                f_p=self.f_p, 
                delta=self.delta,
                np_disk=self.np_disk,
                rad_prof=self.rad_prof,
                height_exp=self.height_exp,
                r_trunk=self.r_trunk,

                Ropt = self.Ropt,
                Topt=self.Topt, 
                Mopt=self.Mopt,

                ### no options ns_b_apex/opt_b_apex here, cause wtf would that mean??
                ns_b_model = self.ns_b_model,
                ns_b_ref = self.ns_b_ref,
                ns_r_ref = self.ns_r_ref,
                ns_L_spindown = self.ns_L_spindown,
                ns_sigma_magn = self.ns_sigma_magn,
                
                opt_b_model = self.opt_b_model,
                opt_b_ref = self.opt_b_ref,
                opt_r_ref = self.opt_r_ref,
                allow_missing=self.allow_missing,
                )

        ibs_now = IBS(winds=winds_now, 
                    gamma_max=self.gamma_max, 
                    s_max=self.s_max, 
                    s_max_g=self.s_max_g, 
                    n=self.n_ibs, 
                    t_to_calculate_beta_eff=t_,
                    abs_gg_filename=self.abs_gg_filename,
                )
        r_sp_now = self.orbit.r(t=t_)
        r_se_now = winds_now.dist_se_1d(t=t_)
        r_pe_now = r_sp_now - r_se_now
        Bp_apex_now, Bopt_apex_now = winds_now.magn_fields_apex(t_)

        els_now = ElectronsOnIBS(ibs = ibs_now,
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

        e_vals_now, dNe_de_IBS_now = els_now.calculate(to_return=True)
        spec_now = SpectrumIBS(els=els_now,
                                delta_power = self.delta_power,
                                lorentz_boost = self.lorentz_boost,
                                method = self.method,
                                abs_photoel = self.abs_photoel,
                                abs_gg = self.abs_gg,
                                nh_tbabs = self.nh_tbabs,
                                mechanisms=self.mechanisms,
                                ic_ani=self.ic_ani,
                                distance = self.distance,
                                ne_mult=self.ne_mult,
                                mode=self.mode,
                                nEed_ic=self.nEed_ic,
                                nEed_syn=self.nEed_syn,
                                )
        if self.full_spec:
            E_ = loggrid(1e2, 1e14, 25)
        else:
            E_ = []
            for band in self.bands:
                # ndec=20-25 is enough for accuracy >1%
                E_in_band = loggrid(band[0]/1.2, band[1]*1.2, 25)
                E_.append(E_in_band)
            E_ = np.concatenate(E_)

        E_ph_now, sed_tot_now, sed_s_now = spec_now.calculate(e_ph =  E_,                                         
                                        to_return=True)
        
        emissiv_now = trapezoid(sed_s_now/E_ph_now, E_ph_now, axis=1)
        fluxes_now = spec_now.fluxes(bands=self.bands, epows=self.epows)
        indexes_now = spec_now.indexes(bands=self.bands_ind)
        return (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                winds_now, ibs_now, els_now, 
                dNe_de_IBS_now, e_vals_now, spec_now,
                E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                emissiv_now,
                )

    ############################################################
    def calculate(self):
        """
        Calculates a bunch of stuff on the grid self.t of times (relative
        to the periastron passage) and sets some attributes:
            
        self.r_sps (np.array of size of self.t): distance from the pulsar to the star [cm], \\
        self.r_ses (np.array of size of self.t): distance from the pulsar to the IBS apex [cm], \\
        self.r_pes (np.array of size of self.t): distance from the star to the IBS apex [cm], \\
        self.B_p_apexs (np.array of size of self.t): pulsar magn field the IBS apex [cm], \\
        self.B_opt_apexs (np.array of size of self.t): star magn field the IBS apex [cm], \\
        self.winds_classes (list of Winds objects of size of self.t): Winds objects at every time, \\        
        self.ibs_classes (list of IBS objects of size of self.t): IBS objects at every time, \\
        self.els_classes (list of ElectronsOnIBS objects of size of self.t): 
                            ElectronsOnIBS objects at every time, \\
        self.spec_classes (list of SpectrumIBS objects of size of self.t):
                                 SpectrumIBS objects at every time, \\
        self.dNe_des = (list of 2d np.arrays of size of self.t): electron distribution on IBS [1/s/cm/eV], \\
        self.e_es = (list of 1d np.arrays of size of self.t): electron energies grid [eV], \\
        self.seds = (list of 1d np.arrays of size of self.t): total SED on IBS [erg/s/cm2], \\
        self.seds_s = (list of 2d np.arrays of size of self.t): SED in every segment of IBS [erg/s/cm2], \\
        self.e_phots = (list of 1d np.arrays of size of self.t): photon energies grid [eV], \\
        self.emiss_s = (list of 1d np.arrays of size of self.t): emissivity along the IBS [erg/s/cm], \\
        self.fluxes = (np.array of size of self.t x len(self.bands)): fluxes in self.bands [erg/s/cm2], \\
        self.indexes = (np.array of size of self.t x len(self.bands_ind)): photon indexes in self.bands_ind. \\
        self.f_ds_eff = (np.array of size of self.t): effective f_d at every time, \\
        self.deltas_eff = (np.array of size of self.t): effective disk z/r at every time. \\

        """

        fluxes = np.zeros((self.t.size, len(self.bands)))
        indexes = np.zeros((self.t.size, len(self.bands_ind)))
        r_sps = np.zeros(self.t.size)
        r_ses = np.zeros(self.t.size)
        r_pes = np.zeros(self.t.size)
        B_p_apexs = np.zeros(self.t.size)
        B_opt_apexs = np.zeros(self.t.size)
        
        winds_classes = []
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
            for i_t, t_ in enumerate(self.t):
                (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                    winds_now, ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    ) = self.calculate_at_time(t_)
                ###################################
                r_sps[i_t] = r_sp_now
                r_pes[i_t] = r_pe_now
                r_ses[i_t] = r_se_now
                B_p_apexs[i_t] = Bp_apex_now
                B_opt_apexs[i_t] = Bopt_apex_now 
                ###################################
                winds_classes.append(winds_now)
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
                    winds_now, ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    ) = self.calculate_at_time(self.t[i_t])
                
                return (r_sp_now, r_pe_now, r_se_now, Bp_apex_now, Bopt_apex_now,
                    winds_now, ibs_now, els_now, dNe_de_IBS_now, e_vals_now, spec_now,
                    E_ph_now, sed_tot_now, sed_s_now, fluxes_now, indexes_now, 
                    emissiv_now,
                    )
            if self.n_cores is None or self.n_cores == 'all':
                n_jobs = multiprocessing.cpu_count()
            elif isinstance(self.n_cores, int): 
                n_jobs = self.n_cores
            else:
                raise ValueError('n_cores should be an int, or None, or \'all\'.')
            n_jobs = max(1, min(20, n_jobs - 5) )
            res= Parallel(n_jobs=n_jobs)(delayed(func_to_parall)(i_t)
                                 for i_t in range(0, len(self.t)))

            r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, winds_classes, ibs_classes, els_classes, \
            dNe_des, e_es, spec_classes, e_phots, seds, seds_s, \
            fluxes, indexes, emiss_s = zip(*res)

            r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, fluxes, indexes, dNe_des, e_es = [np.array(ar) 
                for ar in (r_sps, r_pes, r_ses, B_p_apexs, B_opt_apexs, fluxes,
                           indexes, dNe_des, e_es)]
            winds_classes = list(winds_classes)
            ibs_classes = list(ibs_classes)
            els_classes = list(els_classes)
            spec_classes = list(spec_classes)
            e_phots = list(e_phots)
        seds, seds_s, emiss_s, dNe_des, e_es = [np.array(ar) for ar in (seds, 
                                        seds_s, emiss_s, dNe_des, e_es)]
        
        dne_des_tot = np.sum(dNe_des, axis=1)
        sed_e_tot = np.array([ dne_des_tot[it_, :] * e_es[it_, :]**2 for 
                              it_ in range(self.t.size)])
        self.r_sps = r_sps
        self.r_ses = r_ses
        self.r_pes = r_pes
        self.B_p_apexs = B_p_apexs
        self.B_opt_apexs = B_opt_apexs
        
        self.winds_classes = winds_classes
        self.ibs_classes = ibs_classes
        self.els_classes = els_classes
        self.spec_classes = spec_classes
        self.dne_des = dNe_des
        self.e_es = e_es
        self.dne_des_tot = dne_des_tot
        self.seds_e_tot = sed_e_tot
        
        self.seds = seds
        self.seds_s = seds_s
        self.ss = np.array([ibs_classes[i_t].s_mid for i_t in range(self.t.size)])
        self.e_phots = e_phots
        self.emiss_s = emiss_s
        self.fluxes = fluxes
        self.indexes = indexes    
        
        self.f_ds_eff = np.array([winds_.f_d for winds_ in winds_classes])
        self.deltas_eff = np.array([winds_.delta for winds_ in winds_classes])
        self.calculated = True
    
    def sed(self, t):
        """
        SED at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size, Ne)
            SED.

        """
        seds_ok = fill_nans(self.seds)
        spl_ = interp1d(self.t, seds_ok, axis=0)
        return spl_(t)

    def sed_s(self, t):
        """
        SED as function of s at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size, Ns_mid, Ne)
            SED_s.

        """
        seds_s_ok = fill_nans(self.seds_s)
        spl_ = interp1d(self.t, seds_s_ok, axis=0)
        return spl_(t)
    
    def s(self, t):
        """
        Arclength s on IBS at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size, Ns)
            s.

        """
        ss_ok = fill_nans(self.ss)
        spl_ = interp1d(self.t, ss_ok, axis=0)
        return spl_(t)
    
    
    def sed_e_tot(self, t):
        """
        Electron SED at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size, Ne_el)
            Electron SED.

        """
        sed_e_ok = fill_nans(self.seds_e_tot)
        spl_ = interp1d(self.t, sed_e_ok, axis=0)
        return spl_(t)
    
    def flux(self, t, which=None):
        """
        Flux at the time t.

        Parameters
        ----------
        t : np.array
            Time [s].
        which : int or None, optional
            Which flux to use (used as a numpy index: which==0 means the first
                    flux, etc.). If None, which=0 is set. Defult None

        Returns
        -------
        np.ndarray (t.size, )
            Flux.

        """
        if which is None:
            which = 0
        ok = np.isfinite(self.fluxes[:, which])
        spl_ = interp1d(self.t[ok], self.fluxes[ok, which], kind='cubic')
        return spl_(t)
    
    def index(self, t, which=None):
        """
        Index at the time t.

        Parameters
        ----------
        t : np.array
            Time [s].
        which : int or None, optional
            Which index to use (used as a numpy index: which==0 means the first
                    index, etc.). If None, which=0 is set. Defult None

        Returns
        -------
        np.ndarray (t.size, )
            Index.

        """
        if which is None:
            which = 0
        ok = np.isfinite(self.indexes[:, which])
        spl_ = interp1d(self.t[ok], self.indexes[ok, which], kind='cubic')
        return spl_(t)
    
    def emiss(self, t):
        """
        Emission on IBS at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size, Ns)
            Emission.

        """
        ok = np.isfinite(self.emiss_s)
        spl_ = interp1d(self.t[ok], self.emiss_s[ok], axis=0)
        return spl_(t)
    
    def r_sp(self, t):
        """
        Distance from the optical star (s) to the neutron star (p)
         at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size,)
            r_sp.

        """
        ok = np.isfinite(self.r_sps)
        spl_ = interp1d(self.t, self.r_sps[ok])
        return spl_(t)
    
    def r_se(self, t):
        """
        Distance from the optical star (s) to the IBS apex (e, emission zone)
         at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size,)
            r_se.

        """
        ok = np.isfinite(self.r_ses)
        spl_ = interp1d(self.t, self.r_ses[ok])
        return spl_(t)
    
    def r_pe(self, t):
        """
        Distance from the neutron star (p) to the IBS apex (e, emission zone)
         at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size,)
            r_pe.

        """
        ok = np.isfinite(self.r_pes)
        spl_ = interp1d(self.t, self.r_pes[ok])
        return spl_(t)
    
    def b_p_apex(self, t):
        """
        Pulsar magnetic field in the IBS apex
         at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size,)
            b_p.

        """
        ok = np.isfinite(self.B_p_apexs)
        spl_ = interp1d(self.t, self.B_p_apexs[ok])
        return spl_(t)
    
    def b_opt_apex(self, t):
        """
        Optical magnetic field in the IBS apex
         at the time t

        Parameters
        ----------
        t : np.array
            Time [s].

        Returns
        -------
        np.ndarray (t.size,)
            b_opt.

        """
        ok = np.isfinite(self.B_opt_apexs)
        spl_ = interp1d(self.t, self.B_opt_apexs[ok])
        return spl_(t)
    

    def peek(self,
                ax=None, to_label=True,
                **kwargs):
        """
        Quick look at the results. If ax is None, creates a new figure with
        4 subplots: fluxes(t), indexes(t),
          SED at 3 times, emissivity at 3 times.

        Parameters
        ----------
        ax : pyplot ax object, optional
            Axes to plot on. None or axes with at least 1 row and 
             4 columns. If None, the object ax is created. The default is None.
        to_label : bool, optional
            Whether to add legends to figures. Default True
        **kwargs : .
            extra arguments to pass to plot() function. Passed to all plots.
        """
        
        if ax is None:
            # import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=1, ncols=4,
                                    figsize=(16, 4))
        
        ax_first = ax[0]


        for i, band in enumerate(self.bands):
            log_lo = np.log10(band[0])
            log_hi = np.log10(band[1])
            ax_first.plot(self.t/DAY, 
                        self.fluxes[:, i], 
                        label=f'logE = {log_lo:.2}-{log_hi:.2} eV', **kwargs)
            
        for i, band in enumerate(self.bands_ind):
            log_lo = np.log10(band[0])
            log_hi = np.log10(band[1])
            
            ax[1].plot(self.t/DAY, 
                        self.indexes[:, i], 
                        label=f'logE = {log_lo:.2f}-{log_hi:.2f} eV', **kwargs)
        
        _nt = self.t.size
        for i_t in (int(0.2*_nt), int(0.5*_nt), int(0.8*_nt)):
            t_now_days = self.t[i_t] / DAY
            ax[2].scatter(self.e_phots[i_t], self.seds[i_t],
                        label=f't = {t_now_days:.2f} days',  **kwargs,
                        )
            ax[3].plot(self.ibs_classes[i_t].s_mid, self.emiss_s[i_t],
                        label=f't = {t_now_days:.2f} days', **kwargs)

        
        ax_first.grid()
        for i in range(4):   
            if to_label:
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

        maxsed = np.nanmax(self.seds[np.isfinite(self.seds)])
        ax[2].set_ylim(1e-3*maxsed, maxsed*1.5)
        
        
        
    def peek_animation(self,
                       ibs_color='doppler',
                       filename="animation.mp4",
                       fps=None,
                       duration=None):
        """
        Save a movie corresponding to the light curve.

        Parameters
        ----------
        ibs_color : str, optional
            Keyword for the IBS coloring. The default is 'doppler'.
        filename : str, optional
            Name of the movie to save. The default is "animation.mp4".
        fps : int, optional
            Frames per second. The default is None.
        duration : float, optional
            Duration of the movie in seconds. If provided, 
            `fps` is ignored. The default is None.

        Returns
        -------
        None.

        """
    
        colors = ['b', 'r', 'g', 'k', 'm']
        Nt = len(self.t)
    
        if duration is not None:
            fps_final = Nt / duration
        else:
            fps_final = fps if fps is not None else 30
    
        writer = imageio.get_writer(filename, fps=fps_final, codec="libx264")
    
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        ax00, ax01 = ax[0]
        ax10, ax11 = ax[1]
    
        winds_static = (
            self.winds_classes[0].p_enh == [1,]
            and self.winds_classes[0].h_enh == [1,]
        )

        if winds_static:
            self.winds_classes[0].peek(
                ax=ax00,
                showtime=(np.min(self.t), np.max(self.t)),
                plot_rs=False
            )
        ax00.set_title("System scheme")
    
        # [1,0] light curve background (curves only, no points yet)
        lc_points = []
        for if_, flux_ in enumerate(self.fluxes.T):
            ax10.plot(self.t / DAY, flux_, lw=1, color=colors[if_])
        ax10.set_xlabel(r"$t-t_\mathrm{p}$, days")
        ax10.set_title("Light curves")
    
        # [0,1] electron SED axes
        ax01.set_xscale("log")
        ax01.set_yscale("log")
        ax01.set_title("Electron SED")
        ax01.set_xlabel(r"$E_e$, eV")
        ax01.set_xlim(np.min(self.e_es), np.max(self.e_es))
    
        sed_e_good = np.isfinite(self.seds_e_tot)
        ax01.set_ylim(
            np.nanmax(self.seds_e_tot[sed_e_good]) / 1e3,
            np.nanmax(self.seds_e_tot[sed_e_good]) * 1.1,
        )
    
        ax11.set_xscale("log")
        ax11.set_yscale("log")
        ax11.set_title("Photon SED")
        ax11.set_xlabel(r"$E_\gamma$, eV")
        ax11.set_xlim(np.min(self.e_phots), np.max(self.e_phots))
    
        sed_good = np.isfinite(self.seds)
        ax11.set_ylim(
            np.nanmax(self.seds[sed_good]) / 1e3,
            np.nanmax(self.seds[sed_good]) * 1.1
        )
    
        fig.tight_layout()
        
        star_scatter = ax00.scatter(0.0, 0.0, color='gold', s=60, zorder=5)


        fig.canvas.draw()
        if winds_static:
            background = fig.canvas.copy_from_bbox(fig.bbox)

        p0 = self.orbit.vector_sp(self.t[0])
        xp0, yp0 = p0[0], p0[1]
        
        pulsar_scatter = ax00.scatter(xp0, yp0, color='k', s=30, zorder=6)
        
        line_sp, = ax00.plot([0.0, xp0], [0.0, yp0],
                             color='gray', ls='--', lw=1.0, zorder=4)
        ibs_lines = self.ibs_classes[0].peek(
            fig=fig, ax=ax00, show_winds=False,
            showtime=(np.min(self.t), np.max(self.t)),
            ibs_color=ibs_color,
            to_label=False
        )
        if not isinstance(ibs_lines, (list, tuple)):
            ibs_lines = [ibs_lines]
    
        for if_, flux_ in enumerate(self.fluxes.T):
            pt = ax10.scatter(self.t[0] / DAY, flux_[0],
                              color=colors[if_], zorder=3)
            lc_points.append(pt)
        ax00.scatter(0, 0, color='r', s=20)
        (line_electron,) = ax01.plot(self.e_es[0], self.seds_e_tot[0], color="C0")
    
        line_photon = ax11.scatter(self.e_phots[0], self.seds[0], color="C3")


        for i_t in range(Nt):
            if winds_static:

                fig.canvas.restore_region(background)

                for artist in ibs_lines:
                    artist.remove()
    
                ibs_lines = self.ibs_classes[i_t].peek(
                    ax=ax00, fig=fig, show_winds=False,
                    showtime=(np.min(self.t), np.max(self.t)),
                    ibs_color=ibs_color,
                    to_label=False
                )
                if not isinstance(ibs_lines, (list, tuple)):
                    ibs_lines = [ibs_lines]
    
                for artist in ibs_lines:
                    ax00.draw_artist(artist)
    
            else:

                ax00.cla()
                self.winds_classes[i_t].peek(
                    ax=ax00,
                    showtime=(np.min(self.t), np.max(self.t)),
                    plot_rs=False
                )
                ibs_lines = self.ibs_classes[i_t].peek(
                    ax=ax00, fig=fig, show_winds=False,
                    showtime=(np.min(self.t), np.max(self.t)),
                    ibs_color=ibs_color,
                    to_label=False
                )
                if not isinstance(ibs_lines, (list, tuple)):
                    ibs_lines = [ibs_lines]
                ax00.set_title("System scheme")

            if winds_static and pulsar_scatter in ax00.collections:
                pulsar_scatter.remove()
            if winds_static and line_sp in ax00.lines:
                line_sp.remove()
            
            puls_vector = self.orbit.vector_sp(self.t[i_t])
            xp, yp = puls_vector[0], puls_vector[1]
            
            pulsar_scatter = ax00.scatter(xp, yp, color='k', s=30, zorder=6)
            line_sp, = ax00.plot([0.0, xp], [0.0, yp], color='gray', ls='--', lw=1.0, zorder=4)
            
            ax00.draw_artist(pulsar_scatter)
            ax00.draw_artist(line_sp)

            pulsar_scatter.set_offsets([[xp, yp]])
            ax00.draw_artist(pulsar_scatter)
            
            line_sp.set_data([0.0, xp], [0.0, yp])
            ax00.draw_artist(line_sp)
    

            for if_, flux_ in enumerate(self.fluxes.T):
                lc_points[if_].set_offsets([self.t[i_t] / DAY, flux_[i_t]])
                if winds_static:
                    ax10.draw_artist(lc_points[if_])

            line_electron.set_xdata(self.e_es[i_t])
            line_electron.set_ydata(self.seds_e_tot[i_t])
            if winds_static:
                ax01.draw_artist(line_electron)
    

            line_photon.set_offsets(
                np.column_stack([self.e_phots[i_t], self.seds[i_t]])
            )

            if winds_static:
                ax11.draw_artist(line_photon)

            if winds_static:
                fig.canvas.blit(fig.bbox)
            else:
                fig.canvas.draw()

            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf, dtype=np.uint8)[:, :, :3]
            writer.append_data(frame)
    
        writer.close()
        plt.close(fig)
