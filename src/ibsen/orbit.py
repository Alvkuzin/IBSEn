# pulsar/orbit.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq

from astropy import constants as const
from astropy import units as u
from ibsen.get_obs_data import get_parameters
G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

def unpack_orbit(orb_type=None, T=None, e=None, M=None, nu_los=None, **kwargs):
    """
    Unpack orbital parameters with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        T, e, M, nu_los: float or None
            Explicit values that override defaults.

    Returns:
        Tuple of T, e, M, nu_los
    """
    # Step 1: Determine the source of defaults
    if isinstance(orb_type, str):
        known_types = ['psrb', 'rb', 'bw']
        if orb_type not in known_types:
            raise ValueError(f"Unknown orbit type: {orb_type}")
        defaults = get_parameters(orb_type)
    elif isinstance(orb_type, dict):
        defaults = orb_type
    else:
        defaults = {}

    # Step 2: Build final values, giving priority to explicit arguments
    T_final = T if T is not None else defaults.get('T')
    e_final = e if e is not None else defaults.get('e')
    M_final = M if M is not None else defaults.get('M')
    nu_los_final = nu_los if nu_los is not None else defaults.get('nu_los')


    # # Add any additional parameters you expect
    # # For example, if you want to also unpack `omega`, `i`, etc.:
    result = [T_final, e_final, M_final, nu_los_final]
    # for key in kwargs:
    #     value = kwargs[key] if kwargs[key] is not None else defaults.get(key)
    #     result.append(value)

    return tuple(result)

# print(unpack_orbit('psrb', kwargs='Ropt'))

class Orbit:
    def __init__(self, sys_name=None, period=None, e=None, tot_mass=None, nu_los=None, n=1000):
        T_, e_, mtot_, nu_los_ = unpack_orbit(orb_type=sys_name, 
                                T=period, e=e, M=tot_mass, nu_los=nu_los)
        self.e = e_
        self.T = T_
        self.mtot = mtot_
        self.nu_los = nu_los_
        self.GM = G * mtot_
        self.name = sys_name
        self.xtab = None
        self.ytab = None
        self.ztab = None
        self.n = n
        
        if n is not None:
            self.calculate()
        
    @property    
    def a(self):
        """
        Calculate the semi-major axis of the orbit.
        """
        return (self.T**2 * self.GM / 4. / pi**2)**(1/3)

    
    @property
    def b(self):
        return self.a * np.sqrt(1 - self.e**2)
    
    @property
    def r_periastr(self):
        return self.a * (1 - self.e)
    
    @property
    def r_apoastr(self):
        return self.a * (1 + self.e)
    
    def mean_motion(self, t):    
        return 2 * pi * t / self.T

    def ecc_an(self, t): 
        """
        Eccentric anomaly as a function of time. t [s] (float or array).
        """
        if isinstance(t, float):
            func_to_solve = lambda E: E - self.e * np.sin(E) - Orbit.mean_motion(self, t)
            try:
                E = brentq(func_to_solve, -1e3, 1e3)
                return E
            except:
                print('fuck smth wrong with Ecc(t): float')
                return -1
        else:
            E_ = np.zeros(t.size)
            for i in range(t.size):
                func_to_solve = lambda E: E - self.e * np.sin(E) - Orbit.mean_motion(self, t[i])
                try:
                    E_[i] = brentq(func_to_solve, -1e3, 1e3)
                except:
                    print('fuck smth wrong with Ecc(t): array')
                    E_[i] = np.nan
            return E_
        
        
    def r(self, t):
        return self.a * (1 - self.e * np.cos(Orbit.ecc_an(self, t)))
       
        
    def true_an(self, t):
        ecc_ = Orbit.ecc_an(self, t)
        b_ = self.e / (1 + (1 - self.e**2)**0.5)
        return ecc_ + 2 * np.arctan(b_ * sin(ecc_) / (1 - b_ * cos(ecc_))) 
    
    def x(self, t):
        return self.a * (cos(Orbit.ecc_an(self, t)) - self.e)

    def y(self, t):
        return self.a * (1 - self.e**2)**0.5 * sin(Orbit.ecc_an(self, t))

    def z(self, t):
        if isinstance(t, np.ndarray):
            return np.zeros(t.size)
        else:
            return 0.

    def vector_sp(self, t):
        return np.array([Orbit.x(self, t), Orbit.y(self, t), Orbit.z(self, t)])   

    def calculate(self):
        t_tab = np.linspace(0, self.T, int(self.n))
        self.xtab = Orbit.x(self, t_tab)
        self.ytab = Orbit.y(self, t_tab)
        self.ztab = Orbit.y(self, t_tab)    
    
    
    
    