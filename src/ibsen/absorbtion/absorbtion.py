import numpy as np
from numpy import pi, sin, cos
# import matplotlib.pyplot as plt
from scipy.interpolate import  interp1d
from scipy.integrate import  tplquad
import xarray as xr
from pathlib import Path
# from joblib import Parallel, delayed
# import multiprocessing
# import Orbit as Orb

c_light = 2.998e10
sigma_b = 5.67e-5
h_planck_red = 1.055e-27
k_b = 1.381e-16
Rsun = 7e10
Ropt = 10 * Rsun
DAY = 86400.

m_e = 9.109e-28
h_planck = h_planck_red * 2 * pi
e_char = 4.803e-10
MC2E = m_e * c_light**2
sigma_t = 8/3 * pi * (e_char**2 / MC2E)**2



_here = Path(__file__).parent          # points to pulsar/
_tabdata = _here / "absorb_tab" 
### ----------------- For photoelectric absorbtion ------------------------ ###
# file_phel = Path(Path.cwd(), 'TabData', 'sgm_tbabs_new.txt')
file_phel = _tabdata / 'sgm_tbabs_new.txt'
da_phel = np.genfromtxt(file_phel, skip_header=1, delimiter = ', ')
_e, _sgm = da_phel[:, 0], 10**da_phel[:, 1]
per_ = np.argsort(_e)
_e, _sgm = _e[per_], _sgm[per_]
spl_phel = interp1d(np.log10(_e), np.log10(_sgm), 'linear')

### ----------------------- For gg- absorbtion ---------------------------- ###
# name_gg = Path(Path.cwd(), 'TabData', 'taus.nc')
# name_gg = Path(Path.cwd(), 'TabData', 'taus_gg_new.nc')
name_gg = _tabdata / 'taus_gg_new.nc'
ds_gg = xr.open_dataset(name_gg)
ds_gg = ds_gg.rename({'__xarray_dataarray_variable__': 'data'})



def abs_photoel(E, Nh): 
    """    
    Photoelectric absorbtion TBabs
    Parameters
    ----------
    E : np.ndarray or float
        Photon energy [eV].
    Nh : float
        Hydrogen column densiry as in XSPEC [10^22 g cm^-2].

    Returns
    -------
    Dimentionless multiplicative absorbtion coef = exp(-tau):  0 < coef < 1.

    """
    # file1 = "/home/alvkuzin/Downloads/sgm_tbabs.txt"

    # e = e + 0.001 * np.random.normal(loc = 0, scale = 1, size=e.size)
    e_min, e_max = np.min(_e), np.max(_e)
    # spl_sgm = make_smoothing_spline(x = np.log10(e), y = np.log10(sgm),
    # lam=1e-5)
    E_kev = E / 1e3 #/ 1.6e-9
    if isinstance(E_kev, np.ndarray):
        E_low = E_kev[E_kev <= e_min]
        E_good = E_kev[np.logical_and(E_kev < e_max, E_kev > e_min)]
        E_high = E_kev[E_kev > e_max]
        # sgm_sm = 10**spl_sgm(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
        sgm_sm = 10**spl_phel(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
        
    
        a_low = np.zeros(E_low.size)
        a_good = np.exp(-sgm_sm * Nh * 1e22)
        a_high = np.zeros(E_high.size) + 1
        absorb = np.concatenate((a_low, a_good, a_high))
    else:
        if E_kev < e_min:
            absorb = 0
        elif E_kev > e_max:
            absorb = 1
        else:
            sgm_sm = 10**spl_phel(np.log10(E_good)) * 1e6 * 1e-24 # in cm^2
            absorb = np.exp(-sgm_sm * Nh * 1e22)
    return absorb

def abs_gg_tab(E, nu_los, t, Teff): # gamma-gamma absorbtion
    """
    Tabulated gamma-gamma absorbtion of a target photon on a seed optical photons of the 
    star. For PSRB only. The star is a blackbody with T = 33 000 K.
    The line of sight inclination is fixed at 22.2 deg to the orbit normal.
    The photon is assumed to be emitted at the pulsar position at time t.
    Temporal. Only ONE of the arguments may be an
    array: inputs of multi-dimentional meshgrid-arrays were not tested.

    Parameters
    ----------
    E : np.ndarray or float
        Energy of a photon [eV].
    nu_los : np.ndarray or float
        The angle in the pulsar plane between the direction from
        optical star  to periastron and
        a projection of the LoS onto the orbit [rad]. Mind: the longtitude of 
        periastron w (for PSRB, w=138 deg) = 3pi/2 - nu_los. So for PSRB,
        nu_los = 132 deg = 2.30 rad.
    t : np.ndarray or float
        Time relative to periastron passage [sec].
    Teff : np.ndarray or float
        Effective temperature of the star [K]

    Returns
    -------
    np.ndarray or float
        Dimentionless multiplicative absorbtion coef = e^(-tau):  0 < coef < 1.

    """
    gammas = E * 1.6e-12 / MC2E
    taus = ds_gg['data'].interp(eg=gammas,
             nu_los = nu_los, t=t, Teff=Teff, method = 'linear').values
    taus[taus != taus] = 0
    return np.exp(-taus)


def sigma_gg(e_star, e_g, mu):
    """
    Cross-section of anisotropic gamma gamma --> e+ e- conversion.
    Simple analytic expression from 1703.00680 (they cite 
    **(Jauch & Rohrlich 1976)**) that should work for any multidimentional
    e_star, e_g, mu

    Parameters
    ----------
    e_star : np.ndarray
        Star seed photon energy in units of m_e c^2.
    e_g : np.ndarray
        Target photon energy in units of m_e c^2.
    mu : np.ndarray
        The scattering angle in [rad] but I'm slightly confused in which frame.
        I hope, in the lab frame...

    Returns
    -------
    np.ndarray
        gamma-gamma cross-section [cm^2].

    """
    b =  (1 - 2 / e_g / e_star / (1 - mu))**0.5
    return 3/16 * (1 - b**2) * ( (3 - b**4) * np.log( (1 + b) / (1 - b) ) - 2 * b * (2 - b**2) ) * sigma_t

def n_ph(e, dist, R_star = Ropt, T_star = 3.3e4):
    """
    Planckian photon number density from a star.

    Parameters
    ----------
    e : TYPE
        Photon energy [erg].
    dist : TYPE
        Distance from the star [cm].
    R_star : TYPE
        Star radius [cm]. The default is the radius of LS 2883, 10 R_sol.
    R_star : TYPE
        Star effective temperature [K]. The default is the Teff of LS 2883,
        which is 33.000 K here.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    kappa = Ropt**2 / 4 / dist**2
    w = e / h_planck_red
    exp_ = np.exp(- e / k_b / T_star)
    # d n / d e (e -- dimensionless) :
    num_coef = 1 / 4 / np.pi**3 / c_light**3 / h_planck_red * MC2E
    return num_coef * w**2 * exp_ / (1 - exp_) * kappa 

def dist_to_star(l, dt, cos_):
    # My brother in Christ, why do you have a funciton for the cosine theorem?
    # But mind, the sign `+` is the correct one here.
    return (l**2 + dt**2 + 2 * l * dt * cos_)**0.5

# def tau_gg(t, eg, nu_los = 2.3, incl = 22.2/180.*pi, T_star = 3.3e4):
#     """
#     Optical depth due to gamma-gamma pair production for a photon of energy
#     eg (in units of electron rest-energy) emitted from the position of a pulsar
#     at the time t from periastron.

#     Parameters
#     ----------
#     t : float
#         Time from periastron [s].
#     eg : float
#         Energy of a photon divided by m_e c^2 [dimless].
#     nu_los : float, optional
#         The angle between the direction from the opt. star to the orbit 
#         periastron and the projection of the LoS onto the orbit plane.
#         The default is 2.3.
#     incl : float, optional
#         Inclination of the LoS to the normal to the pulsar plane.
#         The default is 22.2/180.*pi.
#     T_star : float, optional
#         The effective temperature of a star. The default is 33.000 K.

#     Returns
#     -------
#     float
#         DESCRIPTION.

#     """
#     dist_here = Orb.Radius(t)
#     vec_sp = Orb.N_from_V(Orb.Vector_S_P(t))
#     vec_obs = Orb.N_from_V(Orb.N_disk(alpha = nu_los, incl = incl)) 
#     cos_ = Orb.mydot(vec_sp, vec_obs)
#     n_ph_here = lambda e_, mu_, l_: ( n_ph(e = e_ * MC2E,
#         dist = dist_to_star(l = l_, dt=dist_here, cos_ = cos_),T_star = T_star)
#                                      )
    
#     sigma_here = lambda e_, mu_, l_: (
#         sigma_gg(e_star = e_ , e_g = eg, mu = mu_)
#                                       )
    
#     under_int = lambda e_, mu_, l_: ( n_ph_here(e_, mu_, l_) *
#                                      sigma_here(e_, mu_, l_) * (1 - mu_)
#                                      )

#     low_inner = lambda l_, mu_: 2 / eg / (1 - mu_) * (1 + 1e-6)
#     hi_inner = lambda l_, mu_: low_inner(l_, mu_) * 1e3

#     return tplquad( under_int, 0, 50 * dist_here,
#         -1, 1, low_inner, hi_inner, epsrel = 1e-3)[0] * 2 * np.pi 


# if __name__=='__main__':
    ### ------- compare tabulated tau-gg with just-calculated ------------ ####
    # import time
    # start = time.time()
    # tplot = np.linspace(-300, 300, 500) * DAY
    # # tplot = 10 * DAY
    # nu_los = 2.3
    # Teff = 3.1e4
    # for tpl_ in tplot:
    #     E = np.logspace(9, 13, 500)*1.6e-12 # erg
    #     tau = abs_gg_tab(E, nu_los, tpl_, Teff)
    #     # plt.scatter(tplot / DAY, tau, s=1)
    # plt.scatter(E/MC2E, tau, s=1)
    # print(time.time() - start)
    # tnum = np.linspace(-11, 11, 13) * DAY
    # tnum = tplot
    # Enum = np.logspace(10, 13, 17)*1.6e-12 # erg
    # tau_num = np.zeros(Enum.size)
    # for i in range(Enum.size):
    #     tau_num[i] = tau_gg(t = tnum, eg = Enum[i]/MC2E, nu_los = 2.3, T_star=Teff)
    # plt.plot(tnum/DAY, np.exp(-tau_num))
    # plt.plot(Enum/MC2E, np.exp(-tau_num), color='r')
    
    # plt.xscale('log')
    
    ###  ----------------- Tabulate tau_gg opacities --------------------- ####
    
    # ts = np.concatenate(( np.linspace(-300, -70, 10),
    #                       np.linspace(-60, -15, 10),
    #                       np.linspace(-14, 14, 20),
    #                       np.linspace(15, 60, 10),
    #                       np.linspace(70, 300, 10))) * DAY
    # nu_loss = np.array([1.3, 2.0, 2.3,  2.7, 3.1])
    # Teffs = np.array([2.7e4, 3e4, 3.4e4])
    # # ts = np.linspace(-300, -70, 3) * DAY
    # # Teffs = np.array([2.7e4, 3e4, 3.4e4])
    # # nu_loss = np.array([1.7, 2.4, 2.7])
    # egs = np.logspace(4, 8.7, 80)   
    # taus = np.zeros((len(Teffs), len(nu_loss), len(ts), len(egs)))
    # for i_tef in range(Teffs.size):
    #     for i_nu in range(nu_loss.size):
    #         for i_ts in range(ts.size):
    #             print(i_tef, i_nu, i_ts)
    #             def func_to_parallel(eg):
    #                 return tau_gg(t = ts[i_ts], eg = egs[eg], nu_los = nu_loss[i_nu], T_star = Teffs[i_tef])
    #             n_cores = multiprocessing.cpu_count()
    #             print('n_cores', n_cores)
    #             res = Parallel(n_jobs = n_cores)(delayed(func_to_parallel)(i_eg) for i_eg in range(egs.size))
    #             taus[i_tef, i_nu, i_ts, :] = np.array(res)   
                  
    # da = xr.DataArray(taus, coords=[('Teff', Teffs), ('nu_los', nu_loss), ('t', ts), ('eg', egs)],)

    # # Save to NetCDF (or HDF5 if preferred)
    # da.to_netcdf(Path(Path.cwd(), 'TabData', "taus_gg_new.nc"))