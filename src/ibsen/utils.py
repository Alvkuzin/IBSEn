# pulsar/orbit_utis.py
import numpy as np
# from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from numpy import pi, sin, cos
import warnings
import astropy.units as u


G = 6.67e-8
DAY = 86400.
AU = 1.5e13
def is_list_of_vecs(x):
    return isinstance(x, list) and all(isinstance(xx, np.ndarray) for xx in x)

def vectorize_func(func_simple):
    def wrapper(*args):
        if all(isinstance(x, np.ndarray) for x in args):
            return func_simple(*args)
        for x in args:
            if isinstance(x, list):
                n = len(x)
                break
        else:
            raise ValueError("No list argument found!")
        result = []
        for i in range(n):
            new_args = [x[i] if isinstance(x, list) else x for x in args]
            result.append(func_simple(*new_args))
        return result
    return wrapper    

def unpack(query, dictat):
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def mydot_novec(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return xa * xb +  ya * yb + za * zb

mydot = vectorize_func(mydot_novec)

def mycross_novec(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return np.array([xa * zb - za * yb, za * xb - xa * zb, xa * yb - ya * xb])

mycross = vectorize_func(mycross_novec)

def absv_novec(Vec):
    return (mydot(Vec, Vec))**0.5

absv = vectorize_func(absv_novec)


def n_from_v_novec(some_vector):
    return some_vector / absv(some_vector)

n_from_v = vectorize_func(n_from_v_novec)


def g_from_beta(beta_vel):
    return 1 / np.sqrt(1 - beta_vel**2)

def beta_from_g(g_vel):
    if isinstance(g_vel, np.ndarray):
        res = np.zeros_like(g_vel)
        cond = (g_vel > 1.0+1e-7) 
        res[cond] = ((g_vel[cond]-1.0) * (g_vel[cond]+1.0))**0.5 / g_vel[cond]
    else:
        if g_vel > 1.0+1e-7:
            res =  ((g_vel-1.0) * (g_vel+1.0))**0.5 / g_vel
        else:
            res = 0.0
    return res 

def lor_trans_angle(angle, gamma):
    """
    Lorentz transformed angle (between the direction of the frame
                with lorentz-factor gamma and the direction of interest)

    Parameters
    ----------
    angle : np.ndarray
        angle in radians between some direction and moving frame. 
        Should be non-negative value, between 0 and pi.
    gamma : np.ndarray
        lorentz-factor of the moving frame.

    Returns
    -------
    np.ndarray or float
        Lorentz-transformed angle in radians.

    """
    angle = np.asarray(angle)
    gamma = np.asarray(gamma)
    if isinstance(angle, float):
        if angle < 0 or angle > np.pi:
            raise ValueError("angle should be between 0 and pi")
    if isinstance(angle, np.ndarray):
        if not np.all( (angle >= 0.0) & (angle <= np.pi) ):
            raise ValueError("All angles should be between 0 and pi")
    _betas = beta_from_g(gamma)
    _mu = np.cos(angle)
    _mu_prime = (_mu - _betas) / (1.0 - _mu * _betas)
    _mu_prime = np.clip(_mu_prime, -1.0, 1.0)
    return np.arccos(_mu_prime)


def lor_trans_b_iso(B_iso, gamma):
    bx, by, bz = B_iso / 3**0.5, B_iso / 3**0.5, B_iso / 3**0.5
    bx_comov = bx
    by_comov, bz_comov = by * gamma, bz * gamma
    return (bx_comov**2 + by_comov**2 + bz_comov**2)**0.5

def lor_trans_ug_iso(ug_iso, gamma): # Relativistic jets... eq. (2.57)
    # delta_doppl = d_boost(gamma, ang_beta_obs)
    return ug_iso * gamma**2 * (3 + beta_from_g(gamma)) / 3.

def lor_trans_Teff_iso(Teff_iso, gamma): # assuming u ~ T^4
    # delta_doppl = d_boost(gamma, ang_beta_obs)
    return Teff_iso * (gamma**2 * (3 + beta_from_g(gamma)) / 3.)**0.25


def lor_trans_vec_novec(vec_n, vec_beta):
    """
    Lorentz tranformes a spatial unit vector vec_n (in lab frame) into \
    a system moving with vec_beta.
    

    Parameters
    ----------
    vec_n : np.array([n_x, n_y, n_z])
        A unit vector in lab frame to transform. Should be a np.array with \
        three coordinates
        x, y, and z.
    vec_beta : np.array([beta_x, beta_y, beta_z])
        A vector beta = v/c of the co-moving system in lab frame. Should be \
        a np.array with three coordinates x, y, and z.

    Returns
    -------
    np.array([n_prime_x, n_prime_y, n_prime_z])
        The unit vector in a co-moving system.

    """
    vec_norm = n_from_v(vec_n) # just making sure that it is, indeed, a unit vector
    beta_norm = n_from_v(vec_beta)
    _gamma = g_from_beta(absv(vec_beta))
    _nbeta = mydot(vec_norm, vec_beta) # scalar product (n * beta)
    vec_norm_parall = _nbeta / absv(vec_beta)  * beta_norm
    vec_norm_perp = vec_norm - vec_norm_parall
    vec_norm_parall_prime = (vec_norm_parall - vec_beta) / (1 - _nbeta)
    vec_norm_perp_prime = vec_norm_perp / _gamma  / (1 - _nbeta)
    return vec_norm_parall_prime + vec_norm_perp_prime

lor_trans_vec = vectorize_func(lor_trans_vec_novec)

def vector_angle_nonvec(n1, n2, vec_beta=np.zeros(3), lor_trans=False):
    """
    Calculates the angle between two UNIT vectors either in lab system, where \
        they are given, or in a system co-moving with velosity vec_beta.

    Parameters
    ----------
    n1 : my vector 
        1st vector in a lab system.
    n2 : my vector 
        2nd vector in a lab system.
    vec_beta : my vector, optional
        Vector of the beta (=v/c) of the other lorentz frame where you want to 
        calculate an angle. The default is np.zeros(3).
    lor_trans : bool, optional
        Whether to perform a lorentz boost (True) or to calculate in a lab
        frame (False). The default is False.

    Returns
    -------
    float
        Angle between vectors (always 0 <= angle <= pi).

    """
    ######### making sure they are indeed unit vectors. If not, well, you should
    ######### have read the documentation.
    n1_ = n_from_v(n1)
    n2_ = n_from_v(n2)
    if np.all(vec_beta == 0) or (not lor_trans):
        return np.arccos( mydot(n1_, n2_) )
    else:    
        n1_prime = lor_trans_vec(n1_, vec_beta)
        n2_prime = lor_trans_vec(n2_, vec_beta)
        return np.arccos( mydot(n1_prime, n2_prime) 
                         / absv(n1_prime) / absv(n2_prime) )
    
vector_angle = vectorize_func(vector_angle_nonvec)    

# def vector_angle(n1, n2, vec_beta=np.zeros(3), lor_trans=False):
#     # If all are vectors, just operate directly
#     if isinstance(n1, np.ndarray) and isinstance(n2, np.ndarray) and isinstance(vec_beta, np.ndarray):
#         return vector_angle_nonvec(n1, n2, vec_beta, lor_trans)
    
#     # If any is a list, broadcast element-wise
#     # Get length of first list found
#     for x in [n1, n2, vec_beta]:
#         if is_list_of_vecs(x, list):
#             n = len(x)
#             break
#     else:
#         raise ValueError("Inputs are not all arrays or lists of arrays!")
    
#     # Build elementwise args
#     result = []
#     for i in range(n):
#         nn1 = n1[i] if isinstance(n1, list) else n1
#         nn2 = n2[i] if isinstance(n2, list) else n2
#         vecvecb = vec_beta[i] if isinstance(vec_beta, list) else vec_beta
#         result.append(vector_angle_nonvec(nn1, nn2, vecvecb, lor_trans))
#     return result


def lor_trans_e_spec_iso(E_lab, dN_dE_lab, gamma, E_comov=None, n_mu=101):
    """
    Returns (E_comov, dN_dE_comov), the angle-averaged spectrum in the cloud frame.

    Steps:
      1. Build an interpolator for the lab spectrum (zero outside input range).
      2. Define a grid of cosines mu' in [-1,1].
      3. For each E' in E_comov, compute the Doppler-shifted lab energies
         E = Γ (E' + β p' c mu'), then sample the lab spectrum there,
         weight by the Jacobian J = 1/[Γ(1+β mu')], and integrate over μ'.
         Currently assumes that all particles are ultra-relativistic.
         
    Parameters
    ----------
    E_lab : np.ndarray
        1D array of lab-frame energies (must be sorted ascending).
    dN_dE_lab : np.ndarray
        1D array of dN/dE in lab frame, same shape as E_lab.
    gamma : float
        bulk Lorentz factor Γ of the cloud.
    E_comov : np.ndarray, optional
        optional 1D array of desired comoving energies; if None, will use a 
        grid spanning from min(E_lab) * Gamma * (1-beta) to 
        max(E_lab) * Gamma * (1+beta). The default is None.
    n_mu : int, optional
        number of μ' samples for angle-average (must be odd for symmetry).
        The default is 101.

    Returns
    -------
    E_comov : np.ndarray
        1D array of comoving energies.
    dN_dE_comov : ndarray
        1D array of angle-averaged dN'/dE' in comoving frame.

    """
    # derived boost quantities
    beta_v = beta_from_g(gamma)

    # if user did not supply E_comov, take same dynamic range scaled down by Γ
    if E_comov is None:
        Emi = E_lab.min()
        Ema = E_lab.max()
        Emi_co = Emi * gamma * (1.0 - beta_v)
        Ema_co = Ema * gamma * (1.0 + beta_v)
        needed_len = int(len(E_lab) * np.log10(Ema_co / Emi_co) / 
                         np.log10(Ema / Emi))
        E_comov = np.logspace(np.log10(Emi_co), np.log10(Ema_co), needed_len)

    # set up lab-spectrum interpolator, zero outside
    lab_interp = interp1d(
        E_lab, dN_dE_lab,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # cosθ' grid for angle-averaging
    u_even = np.linspace(0, 1, int(n_mu * gamma))
    mu_prime = np.tanh(gamma * (2*u_even**2 - 1)) # denser grid near mu'=1

    # prepare output array
    dN_dE_comov = np.zeros_like(E_comov)

    # now loop (vectorized over mu')
    # Here we assume E_lab and E_comov are kinetic energies >> rest mass, so p'c≈E'.
    # If you need exact, include rest-mass term.

    # for each E', compute E_lab grid for all mu'
    # shape will be (n_E', n_mu)
    Ep = E_comov[:,None]
    E_shift = gamma * Ep * (1 + beta_v * mu_prime[None,:])

    # Jacobian factor J = 1 / [Γ (1 + β μ')]
    # J = d^3 p' / d^3 p = (E/E')^2 * dE/dE' * dOmega / dOmega'
    J = 1.0 / (gamma * (1.0 + beta_v * mu_prime))[None,:]

    # sample lab spectrum at each shifted energy
    F_lab_at_E = lab_interp(E_shift)

    # integrand = J * F_lab
    integrand = J * F_lab_at_E

    # integrate over μ' and multiply by 2π. Then divide by 4π for averaging
    dN_dE_comov = 2.0 * np.pi * trapezoid(integrand, mu_prime, axis=1) / 4.0 / np.pi 

    return E_comov, dN_dE_comov


def loggrid(x1, x2, n_dec):
    n_points = int( np.log10(x2 / x1) * n_dec) + 1
    return np.logspace(np.log10(x1), np.log10(x2), n_points)

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


def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Borrowed from Naima utils. I mean, we use Naima anyway, right?

    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret

def rotated_vector(alpha, incl):
    return np.array([  cos(alpha) * sin(incl),
                     - sin(alpha) * sin(incl),
                       cos(incl)
                       ])

def rotate_z(vec, phi):
    """
    rotates vector vec=np.array([x, y, z]) around z-axis in a positive direction
    """
    
    _x, _y, _z = vec
    c_, s_ = cos(phi), sin(phi)
    x_rotated_ = c_ * _x - s_ * _y
    y_rotated_ = s_ * _x + c_ * _y
    return np.array([x_rotated_, y_rotated_, _z])

def t_avg_func(func, t1, t2, n_t):
    """
    Averages function func(e, t) over a time period t = [t1, t2],

    Parameters
    ----------
    func : Callable
        A function func = func(e, t)
    t1 : float
        min time for averaging.
    t2 : float
        max time for averaging.
    n_t_points : int
        A number of points to span on the t-array.

    Returns
    -------
    Function \tilde func(e).

    """
    # t_grid = np.linspace(t1, t2, n_t)

    def func_avg(e):
        # # ensure e is array for broadcasting
        # e_arr = np.atleast_1d(e)
        # # evaluate Edot on full (e, t) mesh: shape (len(e), len(t))
        # vals = func(e_arr[:, None], t_grid[None, :])
        # # integrate over t for each e
        # integral = np.trapz(vals, t_grid, axis=1)
        # # normalize by interval length
        # avg = integral / (t2 - t1)
        # # if user passed scalar, return scalar
        # return avg.item() if np.isscalar(e) else avg
        return 0.5 * (func(e, t1) + func(e, t2)) 

    return func_avg

def l2_norm(xarr, yarr):
    return ( trapezoid(yarr**2, xarr) )**0.5

# def Get_PSRB_params(orb_p = 'psrb'):
#     """
#     Quickly access some PSRB orbital parameters: orbital period P [days],
#     orbital period T [s], major half-axis a [cm], e, M [g], GM, 
#     distance to the system D [cm], star radius Ropt [cm].

#     Returns : dictionary
#     -------
#     'P' [days], 'a' [cm], 'e', 'M' [g], 'GM': cgs, 'D' [cm], 'Ropt' [cm],
#     'T' [s]

#     """
#     MoptPSRB = 24
#     MxNStyp = 1.4
#     GMPSRB = G * (MoptPSRB + MxNStyp) * 2e33
#     PPSRB = 1236.724526
#     TorbPSRB = PPSRB * DAY
#     aPSRB = (TorbPSRB**2 * GMPSRB / 4 / pi**2)**(1/3)
#     ePSRB = 0.874
#     MPSRB_cgs = (MoptPSRB + MxNStyp) * 2e33
#     DPSRB = 2.4e3 * 206265 * AU
    
#     if orb_p == 'psrb':
#         P_here = PPSRB; Torb_here = TorbPSRB; a_here = aPSRB; e_here = ePSRB 
#         M_here = MPSRB_cgs; GM_here = GMPSRB; D_here = DPSRB; Ropt_here = 10 * 7e10
#     elif orb_p == 'circ':
#         P_here = PPSRB; Torb_here = TorbPSRB; a_here = aPSRB; e_here = 0 
#         M_here = MPSRB_cgs; GM_here = GMPSRB; D_here = DPSRB; Ropt_here = 10 * 7e10

#     res = {'P': P_here, 'a': a_here, 'e': e_here, 'M': M_here, 'GM': GM_here,
#            'D': D_here, 'Ropt': Ropt_here, 'T': Torb_here}
#     return res

# def unpack_orbit(orb_p, Torb=None, e=None, Mtot=None, to_return = None):
#     """
#     Unpack the orbital parameters from a dictionary or a string.
#     If orb_p is None, Torb, e, and Mtot should be provided.
#     """
#     if orb_p is None:
#         markes = to_return.split()
#         returns = []
#         for name in markes:
#             if name == 'T':
#                 returns.append(Torb)
#             elif name == 'e':
#                 returns.append(e)
#             elif name == 'M':
#                 returns.append(Mtot)
#             else:
#                 print('Unknown parameter:', name)
#         return returns
#     else:
#         if isinstance(orb_p, str):
#             orb_par = Get_PSRB_params(orb_p)
#         else:
#             orb_par = orb_p
#         return unpack(query=to_return, dictat=orb_par)
    
# # print(unpack_orbit('psrb', e=4, to_return= '  e '))

# def a_axis(orb_p = None, Torb=None, Mtot=None):
#     """
#     Calculate the semi-major axis of the orbit.
#     """
#     Torb_, M_ = unpack_orbit(orb_p, Torb, Mtot=Mtot, to_return='T M')
#     GM_ = G * M_
#     return (Torb_**2 * GM_ / 4 / pi**2)**(1/3)

# def r_peri(orb_p = None, Torb=None, Mtot=None, e=None):
#     a_ = a_axis(orb_p, Torb, Mtot)
#     e_, = unpack_orbit(orb_p, e=e, to_return='e') 
#     return a_ * (1 - e_)

# def Mean_motion(t, Torb):    
#     return 2 * np.pi * t / Torb

# def Ecc_an(t, Torb_, e_): 
#     """
#     Eccentric anomaly as a function of time. t [s] (float or array),
#     Torb_ [s] (float), e_ (float).
#     This function is considered useless outside  of this module, so
#     Torb and e should always be provided explicitly.
#     """
#     if isinstance(t, float):
#         func_to_solve = lambda E: E - e_ * np.sin(E) - Mean_motion(t, Torb_)
#         try:
#             E = brentq(func_to_solve, -1e3, 1e3)
#             return E
#         except:
#             print('fuck smth wrong with Ecc(t): float')
#             return -1
#     else:
#         E_ = np.zeros(t.size)
#         for i in range(t.size):
#             func_to_solve = lambda E: E - e_ * np.sin(E) - Mean_motion(t[i], Torb_)
#             try:
#                 E_[i] = brentq(func_to_solve, -1e3, 1e3)
#             except:
#                 print('fuck smth wrong with Ecc(t): array')
#                 E_[i] = np.nan
#         return E_

# def Radius(t, orb_p=None, Torb=None, e=None, Mtot=None):
#     a_ = a_axis(orb_p, Torb, Mtot)
#     Torb_, e_ = unpack_orbit(orb_p, Torb, e=e, to_return='T e')   
#     return a_ * (1 - e_ * np.cos(Ecc_an(t, Torb_, e_)))

# def True_an(t, orb_p=None, Torb=None, e=None):
#     Torb_, e_ = unpack_orbit(orb_p, Torb, e=e, to_return='T e')   
#     Ecc_ = Ecc_an(t, Torb_, e_)
#     b_ = e_ / (1 + (1 - e_**2)**0.5)
#     return Ecc_ + 2 * np.arctan(b_ * sin(Ecc_) / (1 - b_ * cos(Ecc_)))

# def X_coord(t, Torb, e, Mtot):
#     a_ = a_axis(None, Torb, Mtot)
#     return a_ * (np.cos(Ecc_an(t, Torb, e)) - e)

# def Y_coord(t, Torb, e, Mtot):
#     a_ = a_axis(None, Torb, Mtot)
#     return a_ * (1 - e**2)**0.5 * sin(Ecc_an(t, Torb, e))

# def Z_coord(t, Torb, e, Mtot):
#     if isinstance(t, np.ndarray):
#         return np.zeros(t.size)
#     else:
#         return 0.

# def Vector_S_P(t, orb_p=None, Torb=None, e=None, Mtot=None):
#     Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
#     x_, y_, z_ = (X_coord(t, Torb_, e_, Mtot_),
#                   Y_coord(t, Torb_, e_, Mtot_),
#                   Z_coord(t, Torb_, e_, Mtot_))
#     return np.array([x_, y_, z_])


# def Dist_to_disk(rvec, alpha, incl):
#     return mydot(rvec, rotated_vector(alpha, incl))

# def times_of_disk_passage(alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
#     Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
#     Dist_to_disk_time = lambda t: mydot(Vector_S_P(t, orb_p, Torb_, e_, Mtot_),
#                              rotated_vector(alpha, incl))
#     t1 = brentq(Dist_to_disk_time, -Torb_/2., 0)
#     t2 = brentq(Dist_to_disk_time, 0, Torb_/2.)
#     return t1, t2


# def r_to_DP(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
#     Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
#     radius = Vector_S_P(t, Torb_, e_, Mtot_)
#     ndisk = rotated_vector(alpha, incl)
#     return mydot(radius, ndisk) * ndisk

# def r_in_DP(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
#     Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
#     radius = Vector_S_P(t, Torb_, e_, Mtot_)
#     ndisk = rotated_vector(alpha, incl)
#     d_to_disk = mydot(radius, ndisk)
#     return radius - ndisk * d_to_disk

# def r_in_DP_fromV(Vx, Vy, normal):
#     nx_, ny_, nz_ = normal
#     dot_prod = Vx * nx_ + Vy * ny_
#     V_toD_x, V_toD_y, V_toD_z = nx_ * dot_prod, ny_ * dot_prod, nz_ * dot_prod 
#     V_inD_x, V_inD_y, V_inD_z = Vx - V_toD_x, Vy - V_toD_y, -V_toD_z
#     return V_inD_x, V_inD_y, V_inD_z

# def n_DiskMatter(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
#     Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
#     n_indisk = n_from_v(r_in_DP(t, alpha, incl, Torb_, e_, Mtot_))
#     ndisk = rotated_vector(alpha, incl)
#     return mycross(ndisk, n_indisk)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    # n1 = [np.array([0, -1, 0.0]),  np.array([1, 0, 0.0])]
    # n1 = n_from_v(n1)
    # n2 = [np.array([1, -1, 0.0]), np.array([1, -1, 0.0])]
    # n2 = n_from_v(n2)
    # gamma = 2.23
    # b_ = beta_from_g(gamma)
    # gamma1 = 3.23
    # b1_ = beta_from_g(gamma1)
    
    # beta = [np.array([1, 0, 0]) * b_, np.array([1, 0, 0]) * b1_]
    # print('n1 = ', n1)
    # print('n2 = ', n2)
    
    # print(b_)
    # print(lor_trans_vec( n1, beta))
    # # print([ (1/2**0.5-b_)/(1-b_/2**0.5), 1/2**0.5/gamma/(1-b_/2**0.5), 0 ])
    
    

    # print(lor_trans_vec( n2, beta))
    
    # print(np.array(vector_angle(n1, n2, beta, True))/np.pi)
    # print('between 0 and 90', np.arccos(-b_)/np.pi) # between 0 and 90
    # print('between 0 and 45', np.arccos( (1/2**0.5-b_)/(1-b_/2**0.5))/np.pi) # between 0 and 45
    # print('between 45 and 90', np.arccos(-b_)/np.pi-np.arccos( (1/2**0.5-b_)/(1-b_/2**0.5))/np.pi) # between 45 and 90
    
    # print('between 0 and 90 new', np.arccos(-b1_)/np.pi) # between 0 and 90
    # print('between 0 and 45 new', np.arccos( (1/2**0.5-b1_)/(1-b1_/2**0.5))/np.pi) # between 0 and 45
    # print('between 45 and 90 new', np.arccos(-b1_)/np.pi-np.arccos( (1/2**0.5-b1_)/(1-b1_/2**0.5))/np.pi) # between 45 and 90
    
    for ang1 in (0, 30, 45, 60, 90, 120):
        diff = np.linspace(0, 180-ang1, 100) / 180 * np.pi
        ang1 = ang1  / 180 * np.pi

        vec0 = np.array([cos(ang1), sin(ang1), 0])
        beta_vec = 0.86 * np.array([1, 0, 0])
        vecs = []
        for d_ in diff:
            vecs.append(np.array([cos(ang1 + d_), sin(ang1 + d_), 0 ]))
        angs = np.array(vector_angle(vecs, vec0, beta_vec, True))
        plt.plot(diff*180/pi, diff*180/pi, color='k', ls='--')
        plt.plot(diff*180/pi, angs*180/pi, label = f'diff = {ang1 * 180 / np.pi}')
        plt.legend()
        
    