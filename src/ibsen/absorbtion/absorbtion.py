import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from scipy.interpolate import  interp1d
from scipy.integrate import  tplquad, dblquad
import xarray as xr
from pathlib import Path
from ibsen.utils import n_from_v, rotated_vector, absv, vector_angle, mydot
from joblib import Parallel, delayed
import multiprocessing

c_light = 2.998e10
sigma_b = 5.67e-5
h_planck_red = 1.055e-27
k_b = 1.381e-16
Rsun = 7e10
Msun = 2e33
Ropt = 10 * Rsun
DAY = 86400.

m_e = 9.109e-28
h_planck = h_planck_red * 2 * pi
e_char = 4.803e-10
MC2E = m_e * c_light**2
sigma_t = 8/3 * pi * (e_char**2 / MC2E)**2



_here = Path(__file__).parent          
_tabdata = _here / "absorb_tab" 

### ----------------- For photoelectric absorbtion ------------------------ ###
# file_phel = Path(Path.cwd(), 'TabData', 'sgm_tbabs_new.txt')
file_phel = _tabdata / 'sgm_tbabs_new.txt'
da_phel = np.genfromtxt(file_phel, skip_header=1, delimiter = ', ')
_e, _sgm = da_phel[:, 0], 10**da_phel[:, 1]
per_ = np.argsort(_e)
_e, _sgm = _e[per_], _sgm[per_]
spl_phel = interp1d(np.log10(_e), np.log10(_sgm), 'linear')


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

def f_helper(mu0, d0):
    """
    Phase-dependent helper for a Suchch & van Soelen analytical gg-absorbtion
    """
    sqrt_ = np.sqrt(1-mu0**2)
    return 1/d0 * (
        1/sqrt_ * (pi/2 - np.arctan(mu0/sqrt_) ) - 1
        )

def phi_helper(eg, R_star, T_star):
    """
    Phase-independent helper for a Suchch & van Soelen analytical gg-absorbtion
    """
    overall_coef = 64*pi / 3 / (h_planck * c_light)**3 * sigma_t * R_star**2
    exp_ = np.exp(-2 * MC2E / eg / k_b / T_star)
    return (  overall_coef * 
              (MC2E / eg)**3 * 
              exp_ / (1 - exp_)
              )

def gg_analyt(eg, x, y, R_star, T_star, nu_los, incl_los):
    """
    Gamma-gamma absorbtion coefficient (e^-tau) on the seed photon field  of 
    the star.

    Parameters
    ----------
    eg : np.ndarray (Ne, )
        The photon energy E_gamma / (m_e c^2).
    x : float
        X-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    y : float
        Y-coordinate in the pulsar orbit [cm] of the VHE photon emission.
    R_star : float
        Optical star radius [cm].
    T_star : float
        Optical star effective temperature.
    nu_los : float
        The angle in the pulsar plane between the direction from
        optical star  to periastron and
        a projection of the LoS onto the orbit [rad]. Connected to the longtitude of 
        periastron w as: w = 3pi/2 - nu_los. So for PSRB w=138 deg and
        nu_los = 132 deg = 2.30 rad.
    incl_los : float
        The orbit inclination: the angle between the line of sight and the 
        angular velosity of the pulsar.

    Returns
    -------
    np.ndarray (Ne, )
        Coeffifient e^tau <1 for the gamma-gamma absorbtion.
        
    Notes:
        There is a pseuso-vectorization for the x- and y-coordinates. It works 
        only for a scalar energy `eg`(e.g., eg=1e6), then you can pass x and y
        coordinates as np.arrays or lists ot tuples of the same length (Nx,).
        In this case, the return is np.ndarray (Nx,).
        Currently you cannot pass both `eg` and `x`/`y` as vectors. 

    """
    if isinstance(x, np.ndarray) or isinstance(x, list) or (isinstance(x, tuple)):
        x, y = np.asarray(x), np.asarray(y)
        r_init=[]
        mu_init=[]
        for x_, y_ in zip(x, y):
            vec_init = np.array([x_, y_, 0])
            r_init.append(absv(vec_init))
            n_init = n_from_v(vec_init)
            n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
            mu_init.append( mydot(n_init, n_los) )
        r_init, mu_init = [np.array(ar) for ar in (r_init, mu_init)]
    else:
        vec_init = np.array([x, y, 0])
        r_init = absv(vec_init)
        n_init = n_from_v(vec_init)
        n_los = n_from_v(rotated_vector(alpha=nu_los, incl=incl_los))
        mu_init = mydot(n_init, n_los) 
    return np.exp(-2 * pi * 
                  phi_helper(eg, R_star, T_star) *
                  f_helper(mu_init, r_init)
                  )
    

if __name__=='__main__':
    from ibsen.orbit import Orbit
    T = 100 * DAY
    orb = Orbit(sys_name='psrb')
    T = orb.T
    # orb.peek()
    print(orb.nu_los)
    print(orb.incl_los*180/pi)
    print(orb.T/DAY)
    print(orb.e)
    
    # orb1 = Orbit(na)
    N = 80
    tplot = np.linspace(-35*DAY, 35*DAY, 50)
    xplot, yplot = orb.x(tplot), orb.y(tplot)
    tau = []
    # i = 0
    # incls = np.linspace(0, 90, 20)
    # ws = np.linspace(0, 360, 20) 
    # egs = np.geomspace(1e3, 1e8, N)
    abs_ = gg_analyt(eg = 1e11/5.11e5, x=xplot, y=yplot, R_star=9.2*Rsun,
                                 T_star=3.3e4, incl_los=22*pi/180, nu_los=2.3)
    # plt.plot(tplot/DAY, abs_)
    plt.plot(tplot/DAY, -np.log(abs_))
    
    plt.yscale('log')
    # for ix in range(tplot.size):
    #     def func_par(i):
    #     # for x_, y_ in zip(xplot, yplot):
    #         # print(i); i+= 1
    #         # tau_ = tau_gg_iso_2d(eg = 1e12/5.11e5, x=xplot[i], y=yplot[i], R_star=10*Rsun,
    #         #                      T_star=3e4, incl_los=45*pi/180, nu_los=(270-w)*pi/180)
    #         tau_ = gg_analyt(eg = egs[i], x=xplot[ix], y=yplot[ix], R_star=10*Rsun,
    #                              T_star=3.3e4, incl_los=22*pi/180, nu_los=2.3)
    #         return tau_
    #     n_cores = multiprocessing.cpu_count()
    #     print('n_cores', n_cores)
    #     res = Parallel(n_jobs = n_cores)(delayed(func_par)(i_eg) for i_eg in range(egs.size))
    #     #             taus[i_tef, i_nu, i_ts, :] = np.array(res)   
    #     tau = np.array(res)
    #     c_ = (tplot[ix] + 100*DAY)/200/DAY
    #     color = [1-c_, 0, c_]
    #     # plt.plot(180+orb.true_an(tplot)/pi*180, tau)
    #     plt.plot(egs*5.11e5, np.exp(-tau), color=color)
    #     plt.xscale('log')
        
    
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