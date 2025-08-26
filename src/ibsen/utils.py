# pulsar/orbit_utis.py
import numpy as np
# from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from numpy import pi, sin, cos
import warnings
import astropy.units as u


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

G = 6.67e-8
DAY = 86400.
AU = 1.5e13
# def is_list_of_vecs(x):
#     return isinstance(x, list) and all(isinstance(xx, np.ndarray) for xx in x)

def vectorize_func(func_simple):
    """
    Vectorises a function func_simple of *args, so that each can be either
    a single vector or a tuple of vectors. In the second case, the lengths
    of tuples should be the same.

    Parameters
    ----------
    func_simple : callable
        Function of some vector arguments.

    Raises
    ------
    ValueError
        If any argument is not a vector or a list of vectors.

    Returns
    -------
    callable
        Vectorized function.

    """
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
    """
    Unpacking values from a string of names.

    Parameters
    ----------
    query : str
        A string of names from the dictionary to unpack, e.g. 
        "a b c".
    dictat : dict
        A dictionary from which to get values.

    Returns
    -------
    list_ : list
        A list of values [dictat['a'], ...].

    """
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def mydot_novec(a, b):
    """
    Scalar product of vectors a, b.
    """
    xa, ya, za = a
    xb, yb, zb = b
    return xa * xb +  ya * yb + za * zb

mydot = vectorize_func(mydot_novec)

def mycross_novec(a, b):
    """
    Vector product of vectors a, b.
    """
    ax, ay, az = a
    bx, by, bz = b
    return np.array([ay * bz - az * by,
                     az * bx - ax * bz,
                     ax * by - ay * bx])

mycross = vectorize_func(mycross_novec)

def absv_novec(vec):
    """
    Absolute value of a vector vec.
    """
    return (mydot(vec, vec))**0.5

absv = vectorize_func(absv_novec)


def n_from_v_novec(vec):
    """
    Normalized vector vec.
    """
    return vec / absv(vec)

n_from_v = vectorize_func(n_from_v_novec)


def g_from_beta(beta_vel):
    """
    Lorentz factor Gamma from the dimentionless velosity beta_vel.
    """
    return 1. / np.sqrt(1. - beta_vel**2)

def beta_from_g(g_vel):
    """
    Dimentionless velosity beta_vel from Lorentz factor Gamma.
    """
    
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
    """
    Lorentz-transforms a value of a module of an isotropic magnetic 
    field B_iso into a coordinate frame co-moving with a lorentz-factor gamma.
    

    Parameters
    ----------
    B_iso : float | np.ndarray of size gamma
        Absolute value of an isotropic magnetic field strength in a lab frame.
    gamma : float | np.ndarray of size B_iso
        A lorentz factor of a coordinate frame to transform to.

    Returns
    -------
    float | np.ndarray of size gamma/B_iso
        Absolute value of a non-isotropic magnetic field strength in a 
        co-moving frame.

    """
    bx, by, bz = B_iso / 3**0.5, B_iso / 3**0.5, B_iso / 3**0.5
    bx_comov = bx
    by_comov, bz_comov = by * gamma, bz * gamma
    return (bx_comov**2 + by_comov**2 + bz_comov**2)**0.5

def lor_trans_ug_iso(ug_iso, gamma): # Relativistic jets... eq. (2.57)
    """
    Lorentz-transforms a value of a module of an isotropic photon 
    field ug_iso into a coordinate frame co-moving with a lorentz-factor gamma.
    Uses eq. (2.57) from Relativistic Jets from Active Galactic Nuclei Edited
    by Markus Böttcher, Daniel E. Harris, and Henric Krawczynski.

    Parameters
    ----------
    ug_iso : float | np.ndarray of size gamma
        Absolute value of an isotropic photon field energy density in a lab
        frame.
    gamma : float | np.ndarray of size B_iso
        A lorentz factor of a coordinate frame to transform to.

    Returns
    -------
    float | np.ndarray of size gamma/ug_iso
        Absolute value of a non-isotropic photon field energy density in a 
        co-moving frame.

    """
    return ug_iso * gamma**2 * (3 + beta_from_g(gamma)) / 3.

def lor_trans_Teff_iso(Teff_iso, gamma): # assuming u ~ T^4
    """
    Transforms an effective temperature of a photon field into the co-moving 
    system. Fully analogous to lor_trans_ug_iso. Assumes T \propto u^{1/4}
    """
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
    """
    Creates a log-spaced grid from x1 to x2 with n_dec points per decade.

    Parameters
    ----------
    x1 : float
        lower bound for a grid.
    x2 : float
        upper bound for a grid.
    n_dec : float
        The number of points per decade. If the number of points turnes out to 
        be 1 (x2 is close to x1 and n_dec is small), an array of [x1, x2] is 
        returned.

    Raises
    ------
    ValueError
        x2 <= x1.

    Returns
    -------
    np.ndarray
        a log-spaced grid from x1 to x2 with n_dec points per decade.

    """
    if x2 <= x1:
        raise ValueError('x2 should be > x1 for loggrid.')
    n_points = max(int( np.log10(x2 / x1) * n_dec) + 1,
                   2)
    return np.logspace(np.log10(x1), np.log10(x2), n_points)

def logrep(xdata, ydata):
    """
    Creates a logarithmic interpolator for the data (xdata, ydata).

    Parameters
    ----------
    xdata : np.ndarray
        1D array of x-coordinates.
    ydata : np.ndarray
        1D array of y-coordinates.

    Returns
    -------
    interp1d-reurned interpolator
        A logarithmic interpolator for the data (xdata, ydata). The xdata should
        be positive and sorted in ascending order. To evaluate the y-values at
        x-values, you should do 
        10**(
        logrep(xdata, ydata)(np.log10(x))
        )

    """
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    return interp1d(np.log10(xdata), np.log10(ydata))

def logev(x, logspl):
    """
    Evaluates a logarithmic interpolator logspl at x.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    logspl : interp1d-returned object.
        interp1d-returned loglog interpolator..

    Returns
    -------
    np.ndarray
        A 1D array of y-coordinates, evaluated at x. The x should be positive
        and sorted in ascending order.

    """
    return 10**( logspl( np.log10(x) ) )

def interplg(x, xdata, ydata):
    """
    Interpolates ydata at x, given xdata/ydata in logarithmic scale.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates to evaluate y at.
    xdata : np.ndarray
        1D array of x-data: x-coordinates at which ydata is given.
    ydata : np.ndarray
        1D array of y-data.

    Returns
    -------
    np.ndarray
        y interpolated at x, using ydata(xdata).

    """
    asc = np.argsort(xdata)
    xdata, ydata = xdata[asc], ydata[asc] 
    spl_ = interp1d(np.log10(xdata), np.log10(ydata))
    return 10**( spl_( np.log10(x) ) )


def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    ---------------------------------------------------------------------------
    ----- Borrowed from Naima utils. I mean, we use Naima anyway, right? ------
    ---------------------------------------------------------------------------

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
    """
    Creates a vector np.array([x, y, z]), which is initially a unit
    vector in z-direction np.array([0, 0, 1]) rotated by `incl` around
    y-axis and then by `alpha` around z-axis. 

    Parameters
    ----------
    alpha : float
        The angle of the second turn around z-axis [rad].
    incl : float
        The angle of the first turn around y-axis [rad].

    Returns
    -------
    np.ndarray of length 3
        The rotated vector.

    """
    return np.array([  cos(alpha) * sin(incl),
                       sin(alpha) * sin(incl),
                       cos(incl)
                       ])

def rotate_z(vec, phi):
    
    """
    rotates vector vec=np.array([x, y, z]) around z-axis at the 
    angle phi in a positive direction
    """
    
    _x, _y, _z = vec
    c_, s_ = cos(phi), sin(phi)
    x_rotated_ = c_ * _x - s_ * _y
    y_rotated_ = s_ * _x + c_ * _y
    return np.array([x_rotated_, y_rotated_, _z])

def rotate_z_xy(x, y, phi):
    """
    Rotates a 2-D vector (x, y) around z-axis at the angle phi in a positive direction.

    Parameters
    ----------
    x : float
        vector x-coordinate.
    y : float
        vector y-coordinate.
    phi : float
        angle [rad] to rotate the vector around z-axis.

    Returns
    -------
    x_rotated_ : float
        x-coordinate of the rotated vector.
        
    y_rotated_ : float
        y-coordinate of the rotated vector.

    """
    c_, s_ = cos(phi), sin(phi)
    x_rotated_ = c_ * x - s_ * y
    y_rotated_ = s_ * x + c_ * y
    return x_rotated_, y_rotated_


def t_avg_func(func, t1, t2, n_t):
    """
    Averages function func(e, t) over a time period t = [t1, t2],
    currently giving an algebraic average.

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
    """
    An L2-norm of a function yarr(xarr) defined on the grid xarr.

    Parameters
    ----------
    xarr : np.ndarray
        1D array of x-coordinates. Should be sorted in ascending order.
    yarr : np.ndarray
        1D array of y-coordinates. Should be the same length as xarr.

    Returns
    -------
    float
        The L2-norm of the function yarr(xarr), defined as
        \sqrt{ \int y^2 dx }.

    """
    return ( trapezoid(yarr**2, xarr) )**0.5

def wrap_grid(x, frac=0.10, num_points=1000, single_num_points=15):
    """
    Creates a grid of points around the range of x, with a 10% margin
    on both sides. If the range is degenerate (xmin == xmax), it will
    create a grid points evenly spaced between xmin*(1 - frac) and xmax*(1 + frac).


    Parameters
    ----------
    x : float | np.ndarray
        1D array of x-coordinates. Should be sorted in ascending order.
    frac : float, optional
        A margin for the enveloping grid. The default is 0.10.
    num_points : int, optional
        The number of points for the enveloping grid. The default is 1000.
    single_num_points : int, optional
        A number of points for a grid if x is a number. The default is 15.

    Raises
    ------
    ValueError
        If x dimention is > 1.

    Returns
    -------
    np.ndarray
        A 1D array of points, evenly spaced between <xmin and
        >xmax taking sign into account. E.g., if x=np.array([1, ....2, ]),
        the grid will be created between 0.9 and 2.2, with 1000 points.
        If x=np.array([-1, ..., 2]), the grid will be created between
        -1.1 and 2.2, with 1000 points. If x=np.array([1, 1, 1]) or x=1,
        the grid will be created between 0.9 and 1.1, with 15 points.

    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    xmin, xmax = x.min(), x.max()

    # New endpoints per sign-aware 10% rule
    lo = xmin*(1 - frac) if xmin >= 0 else xmin*(1 + frac)
    hi = xmax*(1 + frac) if xmax >= 0 else xmax*(1 - frac)

    if xmax == xmin:
        # Degenerate grid: spread evenly between lo and hi
        t = np.linspace(0.0, 1.0, single_num_points)
        return lo + t*(hi - lo)

    return np.linspace(lo, hi, num_points)

########## some helper function for drawing, whatev ############################ 
####################################################################################
####################################################################################
def plot_with_gradient(fig, ax, xdata, ydata, some_param, colorbar=False, lw=2,
                       ls='-', colorbar_label='grad', minimum=None, maximum=None):
    """
    
    Draw the plot (xdata, ydata) on the axis ax with color along the curve
    marking some_param. The color changes from blue to red as some_param increases.
    You may provide your own min and max values for some_param:
    minimum and maximum, then the color will be scaled according to them.
    

    Parameters
    ----------
    fig : pyplot figure
        The figure on which to draw the plot.
    ax : pyplot axis
        The axis on which to draw the plot. 
    xdata : np.ndarray
        1D array of x-coordinates.
    ydata : np.ndarray
        1D array of y-coordinates. Should be the same length as xdata.
    some_param : np.ndarray
        1D array of values to color the line by. Should be the same length as xdata.
    colorbar : bool, optional
        Whether to draw a colorbar. The default is False.
    lw : float, optional
        A linewidth keyword for pyplot.plot. The default is 2.
    ls : string, optional
        A linestyle keyword for pyplot.plot.. The default is '-'.
    colorbar_label : string | float, optional
        Label for a colorbar. The default is 'grad'.
    minimum : float, optional
        If provided, it is used as a minimum value for the 
        colorcoding of some_param. The default is None.
    maximum : float, optional
        If provided, it is used as a maximum value for the 
        colorcoding of some_param. The default is None.

    Returns
    -------
    None.

    """
    # Prepare line segments
    points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize some_p values to the range [0, 1] for colormap
    vmin_here = minimum if minimum is not None else np.min(some_param)
    vmax_here = maximum if maximum is not None else np.max(some_param)
    
    norm = Normalize(vmin=vmin_here, vmax=vmax_here)
    
    # Create the LineCollection with colormap
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(some_param[:-1])  # color per segment; same length as segments
    lc.set_linewidth(lw)
    
    # Plot

    line = ax.add_collection(lc)
    
    if colorbar:
        fig.colorbar(line, ax=ax, label=colorbar_label)  # optional colorbar
        
    ax.set_xlim(xdata.min(), xdata.max())
    ax.set_ylim(ydata.min(), ydata.max())

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

# if __name__ == "__main__":
    ### А ты сюдой зачем смотришь? Что ты хочешь тут увидеть? Вот то-то и оно.
#     import matplotlib.pyplot as plt 
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
    
    # for ang1 in (0, 30, 45, 60, 90, 120):
    #     diff = np.linspace(0, 180-ang1, 100) / 180 * np.pi
    #     ang1 = ang1  / 180 * np.pi

    #     vec0 = np.array([cos(ang1), sin(ang1), 0])
    #     beta_vec = 0.86 * np.array([1, 0, 0])
    #     vecs = []
    #     for d_ in diff:
    #         vecs.append(np.array([cos(ang1 + d_), sin(ang1 + d_), 0 ]))
    #     angs = np.array(vector_angle(vecs, vec0, beta_vec, True))
    #     plt.plot(diff*180/pi, diff*180/pi, color='k', ls='--')
    #     plt.plot(diff*180/pi, angs*180/pi, label = f'diff = {ang1 * 180 / np.pi}')
    #     plt.legend()
    
    # x = np.linspace(-1.1, 3.4, 17)
    # x = 4
    # print(wrap_grid(x, frac=0.1, num_points=21, single_num_points=12))
        
    
# from skimage.measure import marching_cubes
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# def plot_isosurface_parametric(ax,
#                                h_forp, r_forp, phi,
#                                u_be, v, w,
#                                values,                  # disk_ps3, shape (Nh, Nr, Nphi)
#                                levels,
#                                color=(1.0, 0.5, 0.0),   # orange
#                                alpha=0.30,
#                                step_size=1,
#                                allow_degenerate=True):
#     """
#     Plot one or more isosurfaces of 'values' defined on a parametric grid (h, r, phi)
#     with axisymmetric embedding x = h*u + r*(cosφ v + sinφ w).

#     Parameters
#     ----------
#     ax : 3D axes
#     h_forp, r_forp, phi : 1D arrays (uniformly spaced recommended)
#         h = coordinate along the axis u_be
#         r = radius from the axis in the slice plane
#         φ = rotation angle around u_be (radians)
#     u_be, v, w : (3,) arrays, orthonormal basis (u_be is the symmetry axis)
#     values : ndarray, shape (Nh, Nr, Nphi)
#         Scalar field on the (h,r,φ) grid (e.g., log10 pressure)
#     levels : float or sequence of float
#         Isosurface level(s) in the same units as 'values' (e.g., np.nanmax(values)-k)
#     color : RGB tuple
#     alpha : float
#     step_size : marching cubes step size (increase for speed, decrease for quality)
#     """

#     # Input checks
#     values = np.asarray(values, float)
#     assert values.shape == (h_forp.size, r_forp.size, phi.size), \
#         f"'values' must have shape (Nh, Nr, Nphi) = {(h_forp.size, r_forp.size, phi.size)}, got {values.shape}"

#     # Replace NaNs with very low values so they won't appear at high iso levels
#     vmin_finite = np.nanmin(values[np.isfinite(values)])
#     vol = np.where(np.isfinite(values), values, vmin_finite - 1e6)

#     # Precompute linear spacing info (we assume uniform spacing here)
#     # marching_cubes returns verts in index order (z,y,x) == (ih, ir, iphi)
#     Nh, Nr, Np = values.shape
#     h0, r0, p0 = h_forp[0], r_forp[0], phi[0]
#     dh = (h_forp[-1] - h_forp[0]) / (Nh - 1) if Nh > 1 else 1.0
#     dr = (r_forp[-1] - r_forp[0]) / (Nr - 1) if Nr > 1 else 1.0
#     dp = (phi[-1] - phi[0]) / (Np - 1) if Np > 1 else 1.0

#     # Ensure numpy arrays
#     u_be = np.asarray(u_be, float); v = np.asarray(v, float); w = np.asarray(w, float)

#     # Multiple or single level
#     if np.isscalar(levels):
#         levels = [levels]

#     artists = []
#     for lvl in levels:
#         verts, faces, _, _ = marching_cubes(
#             vol, level=lvl, step_size=step_size, allow_degenerate=allow_degenerate
#         )
#         if verts.size == 0 or faces.size == 0:
#             continue  # nothing at this level

#         # verts[:,0]=ih (along h), verts[:,1]=ir (along r), verts[:,2]=ip (along phi)
#         hv = h0 + dh * verts[:, 0]
#         rv = r0 + dr * verts[:, 1]
#         pv = p0 + dp * verts[:, 2]

#         # Map to physical 3D: x = h*u + r*cosφ*v + r*sinφ*w
#         cos_p = np.cos(pv)[:, None]
#         sin_p = np.sin(pv)[:, None]
#         verts_xyz = (hv[:, None] * u_be[None, :]
#                      + rv[:, None] * cos_p * v[None, :]
#                      + rv[:, None] * sin_p * w[None, :])

#         tri = ax.plot_trisurf(verts_xyz[:, 0],
#                               verts_xyz[:, 1],
#                               faces,
#                               verts_xyz[:, 2],
#                               color=color, alpha=alpha, linewidth=0)
#         artists.append(tri)
#     return artists