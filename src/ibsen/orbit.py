# pulsar/orbit.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq
import matplotlib.pyplot as plt

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

        if isinstance(t, np.ndarray):
            E_ = np.zeros(t.size)
            for i in range(t.size):
                func_to_solve = lambda E: E - self.e * np.sin(E) - Orbit.mean_motion(self, t[i])
                try:
                    E_[i] = brentq(func_to_solve, -1e3, 1e3)
                except:
                    print('fuck smth wrong with Ecc(t): array')
                    E_[i] = np.nan
            return E_
        else:
            func_to_solve = lambda E: E - self.e * np.sin(E) - Orbit.mean_motion(self, t)
            try:
                E = brentq(func_to_solve, -1e3, 1e3)
                return E
            except:
                print('fuck smth wrong with Ecc(t): float')
                return np.nan
        
        
    def r(self, t):
        return self.a * (1 - self.e * np.cos(Orbit.ecc_an(self, t)))
       
        
    def true_an(self, t):
        ecc_ = Orbit.ecc_an(self, t)
        b_ = self.e / (1 + (1 - self.e**2)**0.5)
        return ecc_ + 2 * np.arctan(b_ * sin(ecc_) / (1 - b_ * cos(ecc_))) 
    
    @property
    def t_los(self):
        if abs(Orbit.true_an(self, self.T/2) - self.nu_los) < 1e-6:
            return self.T/2
        else:
            to_solve = lambda t_: Orbit.true_an(self, t_) - self.nu_los
            t_to_obs = brentq(to_solve, -self.T/2, self.T/2)
            return t_to_obs
    
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
        t_tab = np.linspace(-self.T * 1.1, self.T * 1.1, int(self.n))
        self.xtab = Orbit.x(self, t_tab)
        self.ytab = Orbit.y(self, t_tab)
        self.ztab = Orbit.y(self, t_tab)    
        self.ttab = t_tab
        self.rtab = Orbit.r(self, t_tab)
        self.nu_truetab = Orbit.true_an(self, t_tab)

    def peek(self, ax=None,
             showtime = None,
             times_pos = (),
             color='k',
             xplot='time'):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=3,
                                   figsize=(12, 4))
            
        if showtime is None:
            showtime = [-self.T/2, self.T/2]
        show_cond  = np.logical_and(self.ttab > showtime[0], 
                                    self.ttab < showtime[1])
            
        # ax[0].set_aspect('equal')
        ax[0].plot(self.xtab[show_cond], self.ytab[show_cond], color=color) # plot the orbit
        ax[0].scatter(x=0, y=0, color='r') # place an optical star in the center of coordinates
        ax[0].plot([0, 3 * self.b * cos(self.nu_los)],
                [0, 3 * self.b * sin(self.nu_los)],
                color=color, ls='--') # plot a line from the optical star to the direction of an observer

        if xplot=='time':
            x_norma = DAY
            xlabel_ = 't, days'
        if xplot=='phase':
            x_norma = self.T
            xlabel_ = r'$t/T$'

        ax[0].set_title('Orbit')
        ax[1].set_title('r(t)')
        ax[2].set_title(r'$\nu_\mathrm{true}(t)$')
        
        ax[1].plot(self.ttab[show_cond]/x_norma, self.rtab[show_cond], color=color)
        ax[2].plot(self.ttab[show_cond]/x_norma, self.nu_truetab[show_cond] * 180. / pi, color=color)
        ax[1].axvline(x=self.t_los/x_norma, color=color, alpha=0.3)
        ax[2].axvline(x=self.t_los/x_norma, color=color, alpha=0.3)

        ax[1].set_ylabel(r'$r_\mathrm{sp}$, cm')
        ax[2].set_ylabel(r'$\nu_\mathrm{true}$, deg')
        ax[1].set_xlabel(xlabel_)
        ax[2].set_xlabel(xlabel_)
        
        
        for t_pos in times_pos:
            ax[0].scatter(x=Orbit.x(self, t_pos),
                          y=Orbit.y(self, t_pos), color=color) # draw a point at time t_pos
            
            ax[1].scatter(x=t_pos/x_norma, y=Orbit.r(self, t_pos), color=color)
            ax[2].scatter(x=t_pos/x_norma,
                          y=Orbit.true_an(self, t_pos) * 180. / pi, color=color)
            
        

        for ax_ in ax[1:]:
            pos = ax_.get_position()        # get [left, bottom, width, height]
            size = min(pos.width, pos.height)
            # Make the axes square, preserving center
            new_pos = [
                pos.x0 + (pos.width - size) / 2,
                pos.y0 + (pos.height - size) / 2,
                size,
                size,
            ]
            ax_.set_position(new_pos)
    
    plt.show()
            
    
    