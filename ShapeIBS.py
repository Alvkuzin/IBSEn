import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
from numpy import pi, sin, cos, tan
from scipy.interpolate import interp1d
import xarray as xr


# We don't want to read the file each time functions are called, so we read 
# it once
### --------------------- For shock front shape --------------------------- ###
file_sh = Path(Path.cwd(), 'TabData', "Shocks4.nc")
ds_sh = xr.load_dataset(file_sh)

# def lin(x, k, b):
#     return k*x + b


# def lin_appr(arr1, arr2, x1, x2, xn):
#     return (arr1*(xn-x2) + arr2*(x1-xn))/(x1-x2)

def Theta_inf(beta):
    to_solve1 = lambda tinf: tinf - tan(tinf) - pi / (1. - beta)
    th_inf = brentq(to_solve1, pi/2 + 1e-5, pi - 1e-5)
    return th_inf

def Theta1_CRW(theta, beta):
    if theta == 0:
        return 0
    else:
        th1_inf = pi - Theta_inf(beta) 
        to_solve2 = lambda t1: t1 / tan(t1) - 1. - beta * (theta / tan(theta) - 1)
        th1 = brentq(to_solve2, 1e-10, th1_inf)
        return th1
    
# def d_boost(Gamma, angle_beta_obs):
#     b_ = angle_beta_obs
#     beta_ = (1 - 1/Gamma**2)**0.5
#     return 1 / Gamma / (1 + beta_ * cos(b_))

def Shock_front(beta, s_max, N, full_return = False):
    """
    Calculates the IBS shape in the model of Canto, Raga, Wilkin (1996)
    https://ui.adsabs.harvard.edu/abs/1996ApJ...469..729C/abstract

    Parameters
    ----------
    beta : float
        The winds momenta relation [dimless].
    s_max : float
        The dimentionless arclength of the IBS at which it should be cut.
    N : int
        The number of points on the one horn of IBS.
    full_return : bool, optional
        Whether to return less or more. The default is False.

    Returns
    -------
    Tuple
        If full_return=True, returnes 7 arrays of length N: x, y, theta,
        r, s, theta1, r1. If full_return=False, returnes 5 arrays of length N:
        x, y, theta, r, s.

    """
    th_inf = Theta_inf(beta)
    if beta > 1e-3:
        thetas = np.linspace(1e-3, th_inf-1e-3, N)
        theta1s = np.zeros(thetas.size)
        theta1s = np.array([Theta1_CRW(thetas[i], beta) for i in range(theta1s.size)])
    if beta <= 1e-3:
        thetas = np.linspace(1e-3, th_inf*(1-beta**2), N)
        theta1s = (7.5 * (-1. + (1. + 0.8 * beta * (1 - thetas / tan(thetas)) )**0.5) )**0.5
        # theta1s = np.array([Theta1_CRW(thetas[i], beta) for i in range(theta1s.size)])
    rs = sin(theta1s) / sin(thetas + theta1s)
    ys = rs * sin(thetas)
    xs = rs * cos(thetas)
    r1s = ( (1-xs)**2 + ys**2)**0.5
    ds2 = np.zeros(rs.size)
    ss = np.zeros(rs.size)
    
    ds2[1:] = np.array([ (xs[i] - xs[i-1])**2 + 
                    (ys[i] - ys[i-1])**2 for i in range(1, rs.size)])
    ss[1:] = np.array([np.sum(ds2[0:i+1]**0.5) for i in range(1, rs.size)])
    inds = np.where(ss < 1.5 * s_max) #!!!
    thetas, theta1s, rs, xs, ys, ss, r1s = [arr[inds] for arr in (thetas, theta1s, rs, xs, ys, ss, r1s)]
    if not full_return:    
        return xs, ys, thetas, rs, ss
    if full_return:
        return xs, ys, thetas, rs, ss, theta1s, r1s


def approx_IBS(b, Na, s_max, full_output = False):
    """
    The tabulated shape of the IBS in the model of Canto, Raga, Wilkin (1996)
    https://ui.adsabs.harvard.edu/abs/1996ApJ...469..729C/abstract. It reads
    the pre-calculated file TabData/Shocks4.nc. 
    
    Parameters
    ----------
    b : float
        b = | log10 (beta_eff) |. Should be > 0.1.
    Na : int
        The number of nods in the grid.
    s_max : float or str
        Describes where the IBS should be cut. If float, then it is treated as
        the dimentionless arclength of the IBS at which it should be cut 
        (should be less than 5.0). If 'bow', then the part of the IBS with
        theta < 90 deg is left. If 'incl', these parts of the shock left for
        which the angle between the radius-vector from the pulsar and the 
        tanential is < 90 + 10 deg
        
    full_output : bool, optional
        Whether to return less or more. The default is False.

    Returns
    -------
    tuple
        The shape of the IBS: the tuple of its characteristics. If full_output
        = False, then the tuple is (x, y, theta, r, s). If full_output
        = True, then the tuple is (x, y, theta, r, s, theta1, r1, 
        theta_tangent, theta_inf (float), r_in_apex (float)). 
        All quantities are dimentionless, so that the distance between the 
        star and the pulsar = 1.

    """
    # first, find the shape in given b by interpolation
    intpl = ds_sh.interp(abs_logbeta=b)
    # and get its x, y, theta, r, s, theta1, r1 as np.arrays
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = (intpl.x, intpl.y, intpl.theta,
                    intpl.r, intpl.s, intpl.theta1, intpl.r1, )
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [np.array(arr) for arr in (xs_, ys_,
                                    ts_, rs_, ss_, t1s_, r1s_)]
    tang = np.arctan(np.gradient(ys_, xs_, edge_order=2))
    if isinstance(s_max, float) or isinstance(s_max, int):
        s_max = float(s_max)
        # leave only the points where arclength < s_max. This may not work good
        # when the b is super big, like > 8-9, as many points will be cut from
        # already sparse arrays
        ok = np.where(ss_ < s_max)
    if isinstance(s_max, str):
        if s_max == 'bow':
            # leave only the part of the shock in forward half-sphere from the 
            # pulsar
            ok = np.where(xs_ >= 0)
        if s_max == 'incl':
            # leave only the parts where the angle between the flow and the line
            # from pulsar is < 90 + 10
            ok = np.where(ts_ + np.abs(tang) <= pi/2 + 10/180*pi)
    xs_, ys_, ts_, rs_, ss_, t1s_, r1s_, tang = [arr[ok] for arr in (xs_, ys_,
                                        ts_, rs_, ss_, t1s_, r1s_, tang)]
    # now we interpolate the values onto the equally-spaced grid over y with
    # Na nods, since this is the best way to interpolate (not over s)
    intx, ints, intth, intr, intth1, intr1, inttan = (interp1d(ys_, xs_),
            interp1d(ys_, ss_),
            interp1d(ys_, ts_), interp1d(ys_, rs_), interp1d(ys_, t1s_), 
            interp1d(ys_, r1s_), interp1d(ys_, tang))    
    yplot = np.linspace(np.min(ys_)*1.001, np.max(ys_)*0.999, int(Na))
    xp, tp, rp, sp, t1p, r1p, tanp = (intx(yplot), intth(yplot), intr(yplot),
                                      ints(yplot),
            intth1(yplot), intr1(yplot), inttan(yplot))
    yp = yplot
    if full_output:
        return (xp, yp, tp, rp, sp, t1p, r1p, tanp, intpl.theta_inf.item(),
    intpl.r_apex.item())
    if not full_output:
        return xp, yp, tp, rp, sp
    
    
    

if __name__=='__main__':
    ### ----------------------------------------------------------------- #####
    ###  ----- Tabulates the shapes of the intrabinary shocks ----------- #####
    ### ----------------------------------------------------------------- #####
    
    # betas = np.concatenate((np.logspace(-12, -3, 31),
    #                         np.logspace(-2.9, -1e-3, 33),
    #                         # np.logspace(-12, -6.3, 2)
    #                         )
    #                        )
    # tinfs = np.zeros(betas.size)
    # s_max = 5
    # N = 10000
    # N_towrite = 1003
    # xs, ys, thetas, rs, ss, r1s, theta1s = (np.zeros((betas.size, N_towrite)), 
    #                           np.zeros((betas.size, N_towrite)),
    #                           np.zeros((betas.size, N_towrite)), 
    #                           np.zeros((betas.size, N_towrite)),
    #                           np.zeros((betas.size, N_towrite)),
    #                           np.zeros((betas.size, N_towrite)),
    #                           np.zeros((betas.size, N_towrite)),
    #                           )
    # # r1s, theta1s = np.zeros((betas.size, N)), np.zeros((betas.size, N))
    # # r_topts = np.zeros((betas.size, N))
    # log_abs_betas = np.zeros(betas.size)
    # r_apexs = np.zeros(betas.size)
    # theta_infs = np.zeros(betas.size)
    # for i in range(betas.size):
    #     beta_ = betas[i]
    #     abs_log_beta = round(abs(np.log10(beta_)), 3)
    #     print(abs_log_beta)
    #     log_abs_betas[i] = abs_log_beta

    #     xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = Shock_front(beta_, s_max=s_max, N=N, full_return=True)
    #     intx, ints, intth, intr, intth1, intr1= (interp1d(ys_, xs_), interp1d(ys_, ss_),
    #             interp1d(ys_, ts_), interp1d(ys_, rs_), interp1d(ys_, t1s_), 
    #             interp1d(ys_, r1s_))
    #     ok = np.where(ss_ < s_max)
    #     xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [arr[ok] for arr in (xs_, ys_, ts_, rs_, ss_, t1s_, r1s_)]
    #     yplot = np.linspace(np.min(ys_)*1.001, np.max(ys_)*0.999, N_towrite)
    #     xp, tp, rp, sp, t1p, r1p = (intx(yplot), intth(yplot), intr(yplot), ints(yplot),
    #             intth1(yplot), intr1(yplot))
    #     xs[i, :xp.size] = xp; ys[i, :xp.size] = yplot; thetas[i, :xp.size] = tp; rs[i, :xp.size] = rp; ss[i, :xp.size] = sp 
    #     theta1s[i, :xp.size] = t1p; r1s[i, :xp.size] = r1p

    #     if i%3 == 0:
    #         plt.plot(xs_, ys_, label = abs_log_beta)
    #     r_apexs[i] = beta_**0.5 / (1 + beta_**0.5)
    #     theta_infs[i] = Theta_inf(beta_)
        
    
    # ds = xr.Dataset(
    #     {
    #      "x": (("abs_logbeta", "point") , xs),
    #      "y": (("abs_logbeta", "point") , ys),
    #      "theta": (("abs_logbeta", "point") , thetas),
    #      "r": (("abs_logbeta", "point") , rs),
    #      "s": (("abs_logbeta", "point") , ss),
    #      "theta1": (("abs_logbeta", "point") , theta1s),
    #      "r1": (("abs_logbeta", "point") , r1s),
    #      "theta_inf": (("abs_logbeta",), theta_infs),
    #      "r_apex": (("abs_logbeta",), r_apexs),
    #      },
    #     coords = {
    #         "abs_logbeta": log_abs_betas,
    #         "point": np.arange(N_towrite) 
    #         }
    #     )    
    # ds.to_netcdf("Shocks4.nc")

    ### ----------------------------------------------------------------- #####
    ###  ---------- Tests if the tabulated data is all right ------------ #####
    ### ----------------------------------------------------------------- #####
    
    ds = xr.load_dataset(Path(Path.cwd(), 'TabData', "Shocks4.nc"))
    
    # Interpolate to a new b value (say b=5.3)
    b_target = 5.5
    beta = 10**(-b_target)
    interpolated = ds.interp(abs_logbeta=b_target)
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 3.5, True)
    plt.plot(x, y, label = f'b = {b_target}')
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 'bow', True)
    plt.plot(x, y, label = None, lw=4)
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 'incl', True)
    plt.plot(x, y, label = None, lw=6)

    
    b_target = 1.75
    beta = 10**(-b_target)
    interpolated = ds.interp(abs_logbeta=b_target)
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 3.5, True)
    plt.plot(x, y, label = f'b = {b_target}')
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 'bow', True)
    plt.plot(x, y, label = None, lw=4)
    x, y, t, r, s, t1, s1, tt, tinf, ra = approx_IBS(b_target, 100, 'incl', True)
    plt.plot(x, y, label = None, lw=6)
    plt.legend()

    
    
    
    plt.scatter(0, 0, c='k')
    plt.scatter(1, 0, c='k')
    plt.axhline(y=0, c='k', alpha=0.2)
    plt.axvline(x=0, c='k', alpha=0.2)



