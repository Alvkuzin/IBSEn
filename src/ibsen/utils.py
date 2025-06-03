# pulsar/orbit_utis.py
import numpy as np
# from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from numpy import pi, sin, cos

G = 6.67e-8
DAY = 86400.
AU = 1.5e13

def unpack(query, dictat):
    markers = query.split()
    list_ = []
    for name in markers:
        list_.append(dictat[name])
    return list_

def mydot(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return xa * xb +  ya * yb + za * zb

def mycross(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return np.array([xa * zb - za * yb, za * xb - xa * zb, xa * yb - ya * xb])

def absv(Vec):
    return (mydot(Vec, Vec))**0.5

def n_from_v(some_vector):
    return some_vector / absv(some_vector)

def beta_from_g(g_vel):
    if isinstance(g_vel, np.ndarray):
        res = np.zeros_like(g_vel)
        cond = (g_vel > 1.0) 
        res[cond] = ((g_vel[cond]-1.0) * (g_vel[cond]+1.0))**0.5 / g_vel[cond]
    else:
        if g_vel > 1.0:
            res =  ((g_vel-1.0) * (g_vel+1.0))**0.5 / g_vel
        else:
            res = 0.0
    return res 


def lor_trans_b_iso(B_iso, gamma):
    bx, by, bz = B_iso / 3**0.5, B_iso / 3**0.5, B_iso / 3**0.5
    bx_comov = bx
    by_comov, bz_comov = by * gamma, bz * gamma
    return (bx_comov**2 + by_comov**2 + bz_comov**2)**0.5

def lor_trans_ug_iso(ug_iso, gamma): # Relativistic jets... eq. (2.57)
    # delta_doppl = d_boost(gamma, ang_beta_obs)
    return ug_iso * gamma**2 * (3 + beta_from_g(gamma)) / 3.

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

def rotated_vector(alpha, incl):
    return np.array([  cos(alpha) * sin(incl),
                     - sin(alpha) * sin(incl),
                       cos(incl)
                       ])

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


