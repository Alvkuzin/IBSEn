# pulsar/orbit_utis.py
import numpy as np
from scipy.optimize import brentq
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

def Get_PSRB_params(orb_p = 'psrb'):
    """
    Quickly access some PSRB orbital parameters: orbital period P [days],
    orbital period T [s], major half-axis a [cm], e, M [g], GM, 
    distance to the system D [cm], star radius Ropt [cm].

    Returns : dictionary
    -------
    'P' [days], 'a' [cm], 'e', 'M' [g], 'GM': cgs, 'D' [cm], 'Ropt' [cm],
    'T' [s]

    """
    MoptPSRB = 24
    MxNStyp = 1.4
    GMPSRB = G * (MoptPSRB + MxNStyp) * 2e33
    PPSRB = 1236.724526
    TorbPSRB = PPSRB * DAY
    aPSRB = (TorbPSRB**2 * GMPSRB / 4 / pi**2)**(1/3)
    ePSRB = 0.874
    MPSRB_cgs = (MoptPSRB + MxNStyp) * 2e33
    DPSRB = 2.4e3 * 206265 * AU
    
    if orb_p == 'psrb':
        P_here = PPSRB; Torb_here = TorbPSRB; a_here = aPSRB; e_here = ePSRB 
        M_here = MPSRB_cgs; GM_here = GMPSRB; D_here = DPSRB; Ropt_here = 10 * 7e10
    elif orb_p == 'circ':
        P_here = PPSRB; Torb_here = TorbPSRB; a_here = aPSRB; e_here = 0 
        M_here = MPSRB_cgs; GM_here = GMPSRB; D_here = DPSRB; Ropt_here = 10 * 7e10

    res = {'P': P_here, 'a': a_here, 'e': e_here, 'M': M_here, 'GM': GM_here,
           'D': D_here, 'Ropt': Ropt_here, 'T': Torb_here}
    return res

def unpack_orbit(orb_p, Torb=None, e=None, Mtot=None, to_return = None):
    """
    Unpack the orbital parameters from a dictionary or a string.
    If orb_p is None, Torb, e, and Mtot should be provided.
    """
    if orb_p is None:
        markes = to_return.split()
        returns = []
        for name in markes:
            if name == 'T':
                returns.append(Torb)
            elif name == 'e':
                returns.append(e)
            elif name == 'M':
                returns.append(Mtot)
            else:
                print('Unknown parameter:', name)
        return returns
    else:
        if isinstance(orb_p, str):
            orb_par = Get_PSRB_params(orb_p)
        else:
            orb_par = orb_p
        return unpack(query=to_return, dictat=orb_par)
    
# print(unpack_orbit('psrb', e=4, to_return= '  e '))

def a_axis(orb_p = None, Torb=None, Mtot=None):
    """
    Calculate the semi-major axis of the orbit.
    """
    Torb_, M_ = unpack_orbit(orb_p, Torb, Mtot=Mtot, to_return='T M')
    GM_ = G * M_
    return (Torb_**2 * GM_ / 4 / pi**2)**(1/3)

def r_peri(orb_p = None, Torb=None, Mtot=None, e=None):
    a_ = a_axis(orb_p, Torb, Mtot)
    e_, = unpack_orbit(orb_p, e=e, to_return='e') 
    return a_ * (1 - e_)

def Mean_motion(t, Torb):    
    return 2 * np.pi * t / Torb

def Ecc_an(t, Torb_, e_): 
    """
    Eccentric anomaly as a function of time. t [s] (float or array),
    Torb_ [s] (float), e_ (float).
    This function is considered useless outside  of this module, so
    Torb and e should always be provided explicitly.
    """
    if isinstance(t, float):
        func_to_solve = lambda E: E - e_ * np.sin(E) - Mean_motion(t, Torb_)
        try:
            E = brentq(func_to_solve, -1e3, 1e3)
            return E
        except:
            print('fuck smth wrong with Ecc(t): float')
            return -1
    else:
        E_ = np.zeros(t.size)
        for i in range(t.size):
            func_to_solve = lambda E: E - e_ * np.sin(E) - Mean_motion(t[i], Torb_)
            try:
                E_[i] = brentq(func_to_solve, -1e3, 1e3)
            except:
                print('fuck smth wrong with Ecc(t): array')
                E_[i] = np.nan
        return E_

def Radius(t, orb_p=None, Torb=None, e=None, Mtot=None):
    a_ = a_axis(orb_p, Torb, Mtot)
    Torb_, e_ = unpack_orbit(orb_p, Torb, e=e, to_return='T e')   
    return a_ * (1 - e_ * np.cos(Ecc_an(t, Torb_, e_)))

def True_an(t, orb_p=None, Torb=None, e=None):
    Torb_, e_ = unpack_orbit(orb_p, Torb, e=e, to_return='T e')   
    Ecc_ = Ecc_an(t, Torb_, e_)
    b_ = e_ / (1 + (1 - e_**2)**0.5)
    return Ecc_ + 2 * np.arctan(b_ * sin(Ecc_) / (1 - b_ * cos(Ecc_)))

def X_coord(t, Torb, e, Mtot):
    a_ = a_axis(None, Torb, Mtot)
    return a_ * (np.cos(Ecc_an(t, Torb, e)) - e)

def Y_coord(t, Torb, e, Mtot):
    a_ = a_axis(None, Torb, Mtot)
    return a_ * (1 - e**2)**0.5 * sin(Ecc_an(t, Torb, e))

def Z_coord(t, Torb, e, Mtot):
    if isinstance(t, np.ndarray):
        return np.zeros(t.size)
    else:
        return 0.

def Vector_S_P(t, orb_p=None, Torb=None, e=None, Mtot=None):
    Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
    x_, y_, z_ = (X_coord(t, Torb_, e_, Mtot_),
                  Y_coord(t, Torb_, e_, Mtot_),
                  Z_coord(t, Torb_, e_, Mtot_))
    return np.array([x_, y_, z_])

def rotated_vector(alpha, incl):
    return np.array([  cos(alpha) * sin(incl),
                     - sin(alpha) * sin(incl),
                       cos(incl)
                       ])

def Dist_to_disk(rvec, alpha, incl):
    return mydot(rvec, rotated_vector(alpha, incl))

def times_of_disk_passage(alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
    Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
    Dist_to_disk_time = lambda t: mydot(Vector_S_P(t, orb_p, Torb_, e_, Mtot_),
                             rotated_vector(alpha, incl))
    t1 = brentq(Dist_to_disk_time, -Torb_/2., 0)
    t2 = brentq(Dist_to_disk_time, 0, Torb_/2.)
    return t1, t2


def r_to_DP(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
    Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
    radius = Vector_S_P(t, Torb_, e_, Mtot_)
    ndisk = rotated_vector(alpha, incl)
    return mydot(radius, ndisk) * ndisk

def r_in_DP(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
    Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
    radius = Vector_S_P(t, Torb_, e_, Mtot_)
    ndisk = rotated_vector(alpha, incl)
    d_to_disk = mydot(radius, ndisk)
    return radius - ndisk * d_to_disk

def r_in_DP_fromV(Vx, Vy, normal):
    nx_, ny_, nz_ = normal
    dot_prod = Vx * nx_ + Vy * ny_
    V_toD_x, V_toD_y, V_toD_z = nx_ * dot_prod, ny_ * dot_prod, nz_ * dot_prod 
    V_inD_x, V_inD_y, V_inD_z = Vx - V_toD_x, Vy - V_toD_y, -V_toD_z
    return V_inD_x, V_inD_y, V_inD_z

def n_DiskMatter(t, alpha, incl, orb_p=None, Torb=None, e=None, Mtot=None):
    Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
    n_indisk = n_from_v(r_in_DP(t, alpha, incl, Torb_, e_, Mtot_))
    ndisk = rotated_vector(alpha, incl)
    return mycross(ndisk, n_indisk)


