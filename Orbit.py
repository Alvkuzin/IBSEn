import numpy as np
from scipy.optimize import brentq
from numpy import pi, sin, cos

G = 6.67e-8
DAY = 86400.
AU = 1.5e13

MoptPSRB = 24
MxNStyp = 1.4
GMPSRB = G * (MoptPSRB + MxNStyp) * 2e33
PPSRB = 1236.724526
TorbPSRB = PPSRB * DAY
aPSRB = (TorbPSRB**2 * GMPSRB / 4 / pi**2)**(1/3)
ePSRB = 0.874
MPSRB_cgs = (MoptPSRB + MxNStyp) * 2e33
DPSRB = 2.4e3 * 206265 * AU

def Get_PSRB_params():
    """
    Quickly access some PSRB orbital parameters: orbital period P [days],
    orbital period T [s], major half-axis a [cm], e, M [g], GM, 
    distance to the system D [cm], star radius Ropt [cm].

    Returns : dictionary
    -------
    'P' [days], 'a' [cm], 'e', 'M' [g], 'GM': cgs, 'D' [cm], 'Ropt' [cm],
    'T' [s]

    """
    res = {'P': PPSRB, 'a': aPSRB, 'e': ePSRB, 'M': MPSRB_cgs, 'GM': GMPSRB,
           'D': DPSRB, 'Ropt': 10. * 7.e10, 'T': TorbPSRB}
    return res

def a_axis(Torb=TorbPSRB, Mtot=MPSRB_cgs):
    GM_ = G * Mtot
    return (Torb**2 * GM_ / 4 / pi**2)**(1/3)

def r_peri(Torb=TorbPSRB, Mtot=MPSRB_cgs, e=ePSRB):
    a_ = a_axis(Torb, Mtot)
    return a_ * (1 - e)

def Mean_motion(t, Torb):
    return 2 * np.pi * t / Torb

def Ecc_an(t, Torb, e):
    if isinstance(t, float):
        func_to_solve = lambda E: E - e * np.sin(E) - Mean_motion(t, Torb)
        try:
            E = brentq(func_to_solve, -1e3, 1e3)
            return E
        except:
            print('fuck smth wrong with Ecc(t): float')
            return -1
    else:
        E_ = np.zeros(t.size)
        for i in range(t.size):
            func_to_solve = lambda E: E - e * np.sin(E) - Mean_motion(t[i], Torb)
            try:
                E_[i] = brentq(func_to_solve, -1e3, 1e3)
            except:
                print('fuck smth wrong with Ecc(t): array')
                E_[i] = np.nan
        return E_

def Radius(t, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    a_ = a_axis(Torb, Mtot)
    return a_ * (1 - e * np.cos(Ecc_an(t, Torb, e)))

def True_an(t, Torb=TorbPSRB, e=ePSRB):
    Ecc = Ecc_an(t, Torb, e)
    b__ = e / (1 + (1 - e**2)**0.5)
    return Ecc + 2 * np.arctan(b__ * sin(Ecc) / (1 - b__ * cos(Ecc)))

def mydot(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return xa * xb +  ya * yb + za * zb

def mycross(a, b):
    xa, ya, za = a
    xb, yb, zb = b
    return np.array([xa * zb - za * yb, za * xb - xa * zb, xa * yb - ya * xb])


def ABSV(Vec):
    return (mydot(Vec, Vec))**0.5

def X_coord(t, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    a_ = a_axis(Torb, Mtot)
    return a_ * (np.cos(Ecc_an(t, Torb, e)) - e)

def Y_coord(t, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    a_ = a_axis(Torb, Mtot)
    return a_ * (1 - e**2)**0.5 * sin(Ecc_an(t, Torb, e))

def Z_coord(t, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    if isinstance(t, np.ndarray):
        return np.zeros(t.size)
    else:
        return 0.

def Vector_S_P(t, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    x_, y_, z_ = (X_coord(t, Torb, e, Mtot),
                  Y_coord(t, Torb, e, Mtot),
                  Z_coord(t, Torb, e, Mtot))
    return np.array([x_, y_, z_])

def N_from_V(some_vector):
    return some_vector / ABSV(some_vector)

def N_disk(alpha, incl):
    return np.array([  cos(alpha) * sin(incl),
                     - sin(alpha) * sin(incl),
                       cos(incl)
                       ])

def Dist_to_disk(rvec, alpha, incl):
    return mydot(rvec, N_disk(alpha, incl))

def times_of_disk_passage(alpha, incl, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    # vec_SP = lambda t: Vector_S_P(t, Torb, e, Mtot)
    Dist_to_disk_time = lambda t: mydot(Vector_S_P(t, Torb, e, Mtot),
                             N_disk(alpha, incl))
    t1 = brentq(Dist_to_disk_time, -100 * DAY, 0)
    t2 = brentq(Dist_to_disk_time, 0, 100 * DAY)
    return t1, t2


def r_to_DP(t, alpha, incl, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    radius = Vector_S_P(t, Torb, e, Mtot)
    ndisk = N_disk(alpha, incl)
    return mydot(radius, ndisk) * ndisk

def r_in_DP(t, alpha, incl, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    radius = Vector_S_P(t, Torb, e, Mtot)
    ndisk = N_disk(alpha, incl)
    d_to_disk = mydot(radius, ndisk)
    return radius - ndisk * d_to_disk

def r_in_DP_fromV(Vx, Vy, normal):
    nx_, ny_, nz_ = normal
    dot_prod = Vx * nx_ + Vy * ny_
    V_toD_x, V_toD_y, V_toD_z = nx_ * dot_prod, ny_ * dot_prod, nz_ * dot_prod 
    V_inD_x, V_inD_y, V_inD_z = Vx - V_toD_x, Vy - V_toD_y, -V_toD_z
    return V_inD_x, V_inD_y, V_inD_z

def n_DiskMatter(t, alpha, incl, Torb=TorbPSRB, e=ePSRB, Mtot=MPSRB_cgs):
    n_indisk = N_from_V(r_in_DP(t, alpha, incl, Torb, e, Mtot))
    ndisk = N_disk(alpha, incl)
    return mycross(ndisk, n_indisk)


