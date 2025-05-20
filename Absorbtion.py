import astropy.units as u
import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, make_smoothing_spline, interp1d
from scipy.integrate import trapezoid, quad, dblquad, tplquad
from scipy.optimize import curve_fit
import xarray as xr
from pathlib import Path
import Orbit as Orb

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
Topt = 3.3e4

# We don't want to read the files each time functions are called, so we read 
# them once
### ----------------- For photoelectric absorbtion ------------------------ ###
file_phel = Path(Path.cwd(), 'TabData', 'sgm_tbabs_new.txt')
da_phel = np.genfromtxt(file_phel, skip_header=1,delimiter = ', ')
e, sgm = da_phel[:, 0], 10**da_phel[:, 1]
per_ = np.argsort(e)
e, sgm = e[per_], sgm[per_]
spl_phel = interp1d(np.log10(e), np.log10(sgm), 'linear')

### ----------------------- For gg- absorbtion ---------------------------- ###
name_gg = Path(Path.cwd(), 'TabData', 'taus.nc')
ds_gg = xr.open_dataset(name_gg)
ds_gg = ds_gg.rename({'__xarray_dataarray_variable__': 'data'})



def abs_photoel(E, Nh): # Nh as in XSPEC in 10^22
    """    
    Photoelectric absorbtion TBabs
    Parameters
    ----------
    E : np.ndarray or float
        Photon energy [erg].
    Nh : float
        Hydrogen column densiry as in XSPEC [10^22 g cm^-2].

    Returns
    -------
    Dimentionless multiplicative absorbtion coef = e^(-tau):  0 < coef < 1.

    """
    # file1 = "/home/alvkuzin/Downloads/sgm_tbabs.txt"

    # e = e + 0.001 * np.random.normal(loc = 0, scale = 1, size=e.size)
    e_min, e_max = np.min(e), np.max(e)
    # spl_sgm = make_smoothing_spline(x = np.log10(e), y = np.log10(sgm),
    # lam=1e-5)
    E_kev = E / 1.6e-9
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

def abs_gg_tab(E, nu_los, t): # gamma-gamma absorbtion
    """
    Tabulated gamma-gamma absorbtion of a target photon on a seed optical photons of the 
    star. For PSRB only, the star is a blackbody with T = 33 000 K.
    The line of sight inclination is fixed at 22.2 deg to the orbit normal.
    The photon is assumed to be emitted at the pulsar position at time t.
    Temporal. Only ONE of the arguments may be an
    array. Imputs of multi-dimentional meshgrid-arrays were NOT tested.

    Parameters
    ----------
    E : np.ndarray or float
        Energy of a photon [erg].
    nu_los : np.ndarray or float
        The angle in the pulsar plane between the direction to periastron and
        a projection of the LoS onto the orbit [rad]. Mind: the longtitude of 
        periastron w (for PSRB, w=138 deg) = 3pi/2 - nu_los. So for PSRB,
        nu_los = 132 deg = 2.30 rad.
    t : np.ndarray or float
        Time relative to periastron passage [sec].

    Returns
    -------
    np.ndarray or float
        Dimentionless multiplicative absorbtion coef = e^(-tau):  0 < coef < 1.

    """
    gammas = E / MC2E
    taus = ds_gg['data'].interp(eg=gammas,
             nu_los = nu_los, t=t, method = 'cubic').values
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

def n_ph(e, dist, R_star = Ropt):
    """
    Planckian photon number density from a star.

    Parameters
    ----------
    e : TYPE
        Photon energy [erg].
    dist : TYPE
        Distance from the star [cm].
    R_star : TYPE
        Star radius [cm]. The default is the radius of LS 2883.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    kappa = Ropt**2 / 4 / dist**2
    w = e / h_planck_red
    exp_ = np.exp(- e / k_b / Topt)
    # d n / d e (e -- dimensionless) :
    num_coef = 1 / 4 / np.pi**3 / c_light**3 / h_planck_red * MC2E
    return num_coef * w**2 * exp_ / (1 - exp_) * kappa 

def dist_to_star(l, dt, cos_):
    # nu = LC.True_an(t)
    # dt = LC.Radius(t)
    return (l**2 + dt**2 + 2 * l * dt * cos_)**0.5
    # return (l**2 + dt**2 + 2 * l * dt * cos(nu_los - nu))**0.5

def tau_gg(t, eg, nu_los, incl = 22.2/180.*pi):
    dist_here = Orb.Radius(t)
    # nu_ = LC.True_an(t)
    vec_sp = Orb.N_from_V(Orb.Vector_S_P(t))
    vec_obs = Orb.N_from_V(Orb.N_disk(alpha = nu_los, incl = incl)) #!!!
    cos_ = Orb.mydot(vec_sp, vec_obs)
    # print('nu ', nu_ - nu_los)
    # print('dist ', dist_here / r_periastron)
    n_ph_here = lambda e_, mu_, l_: ( n_ph(e = e_ * MC2E,
        dist = dist_to_star(l = l_, dt=dist_here, cos_ = cos_))
                                     )
    
    sigma_here = lambda e_, mu_, l_: (
        sigma_gg(e_star = e_ , e_g = eg, mu = mu_)
                                      )
    
    under_int = lambda e_, mu_, l_: ( n_ph_here(e_, mu_, l_) *
                                     sigma_here(e_, mu_, l_) * (1 - mu_)
                                     )
    # print(t / DAY)

    # print('int ', under_int(1e-5, 0.5, 2*dist_here) * 10*dist_here * 4*pi * eg)

    low_inner = lambda l_, mu_: 2 / eg / (1 - mu_) * (1 + 1e-6)
    hi_inner = lambda l_, mu_: low_inner(l_, mu_) * 1e3
    
    # es = np.logspace(-10, 10, 10000)
    # plt.plot(es, under_int(es, 0.3, 2 * dist_here) * es)
    # plt.axvline(x = low_inner(2, 0.3))
    return tplquad( under_int, 0, 50 * dist_here,
        -1, 1, low_inner, hi_inner, epsrel = 1e-3)[0] * 2 * np.pi 


if __name__=='__main__':
    # tplot = np.linspace(-30, 40, 500) * DAY
    tplot = 10 * DAY
    nu_los = 2.3
    E = np.logspace(9, 13, 500)*1.6e-12 # erg
    tau = abs_gg_tab(E, nu_los, tplot)
    # plt.scatter(tplot / DAY, tau, s=1)
    plt.scatter(E, tau, s=1)
    
    # tnum = np.linspace(-11, 11, 13) * DAY
    tnum = 10 * DAY
    Enum = np.logspace(10, 13, 17)*1.6e-12 # erg
    tau_num = np.zeros(Enum.size)
    for i in range(Enum.size):
        tau_num[i] = tau_gg(t = tnum, eg = Enum[i]/MC2E, nu_los = 2.3)
    # plt.plot(tnum/DAY, np.exp(-tau_num))
    plt.plot(Enum, np.exp(-tau_num), color='r')
    
    plt.xscale('log')