# pulsar/orbit.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from astropy import constants as const
# from astropy import units as u
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params
G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

orbit_docstring = """
    Keplerian binary orbit in the orbital plane (CGS units).
    
    This class represents a two-body elliptical orbit with the periastron
    aligned with the +X axis and motion confined to the XY-plane (Z=0).
    It provides geometry and anomalies as functions of time measured from
    periastron passage and pre-tabulates the orbit for quick
    plotting/evaluation.
    
    Parameters
    ----------
    sys_name : %s, or None, optional
        If provided, load default orbital parameters via
        ``ibsen.get_obs_data.get_parameters(sys_name)``; explicit arguments
        below override those defaults. 
        If None, all parameters must be contained in `sys_params` dictionary or
        given explicitly.
    sys_params : dict or None, optional
          If provided, a dictionary of orbital parameters to use instead of
          default values from ``get_parameters``. Keys are: `T`, `e`, `M`,
          `nu_los`, `incl_los`. Explicit arguments below override those in
          this dictionary. If None, all parameters must be given explicitly.  
    T : float, optional
        Orbital period `T` in seconds.
    e : float, optional
        Orbital eccentricity (`0 <= e < 1`).
    M : float, optional
        Total system mass `M` in grams. Used to form ``GM = G*M``.
    nu_los : float, optional
        Line-of-sight true anomaly (radians) in the orbital plane. Used to
        locate the time of line-of-sight passage.
    incl_los : float, optional
        Line-of-sight inclination.
    n : int or None, optional
        If not None (default 1000), pre-compute and store tabulated arrays of
        `x(t), y(t), z(t), r(t), \\nu_\\mathrm{{true}}(t)` and ``t`` over
        one period for quick access/plotting. If None, no tabulation is
        performed at initialization.
    allow_missing : bool, optional
        Fill the missing parameters (not explicitly provided, no keyword 
        recognized, and not found in `sys_params`) with None. Default False
    
    Attributes
    ----------
    e : float
        Eccentricity.
    T : float
        Orbital period (s).
    M : float
        Total system mass (g).
    GM : float
        Gravitational parameter :math:`G M` (cgs).
    nu_los : float
        Line-of-sight true anomaly (rad).
    incl_los : float
        Line-of-sight inclination (rad).      
    name : str or None
        Value of ``sys_name`` used to initialize the orbit (if any).
    n : int or None
        Requested number of tabulation points.
    a : float
        Semi-major axis (cm), computed from Kepler's third law.
    b : float
        Semi-minor axis (cm).
    r_periastr : float
        Periastron distance (cm).
    r_apoastr : float
        Apoastron distance (cm).
    t_los: float
        Time of the line of sight crossing [s].
    xtab, ytab, ztab : ndarray or None
        Tabulated coordinates (cm) if ``n`` is not None. ``ztab`` is identically
        zero for the planar orbit.
    ttab : ndarray or None
        Tabulated times (s) relative to periastron.
    rtab : ndarray or None
        Tabulated separation :math:`r(t)` (cm).
    nu_truetab : ndarray or None
        Tabulated true anomaly :math:`\\nu_\\mathrm{{true}}(t)` (rad).
        
    Methods
    ----------
    mean_motion(t)
        Mean motion at time(s) `t`.
    ecc_an(t)
        Eccentric anomaly at time(s) `t`.
    r(t)
        Separation at time(s) `t`.
    true_an(t)
        True anomaly at time(s) `t`.
    t_from_true_an(nu)
        Time(s) since periastron for given true anomaly(ies) `nu`.
    x(t), y(t), z(t)
        Coordinates at time(s) `t`.
    vector_sp(t)
        3D position vector at time(s) `t`.
    peek(ax=None, showtime=None, times_pos=(), color='k', xplot='time')
        Quick look at the orbit.
    
    
    Notes
    -----
    Times ``t`` are interpreted as offsets from periastron passage (``t=0``).
    All distances are returned in centimeters (cgs).

  """%(known_names,)

class Orbit:
    __doc__ = orbit_docstring
    def __init__(self, sys_name=None, sys_params=None, T=None, e=None, M=None,
                 nu_los=None, incl_los=None, n=1000, allow_missing=False):
        T_, e_, M_, nu_los_, incl_los_ = unpack_params(('T', 'e', 'M',
                                                        'nu_los', 'incl_los'),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names, get_defaults_func=get_parameters,
                                T=T, e=e, M=M, nu_los=nu_los,
                                incl_los=incl_los, allow_missing=allow_missing)
        self.e = e_
        self.T = T_
        self.M = M_
        self.nu_los = nu_los_
        self.incl_los = incl_los_
        self.GM = G * M_
        self.name = sys_name
        self.xtab = None
        self.ytab = None
        self.ztab = None
        self.n = n
        
        if n is not None:
            self._calculate()
        
    @property    
    def a(self):
        """
        Calculate the semi-major axis of the orbit.

        Returns
        -------
        float
            Semi-major axis of the orbit.

        """
        return np.cbrt(self.T**2 * self.GM / 4. / pi**2)

    
    @property
    def b(self):
        """
        Calculate the semi-minor axis of the orbit.

        Returns
        -------
        float
            Semi-minor axis of the orbit.

        """
        return self.a * np.sqrt(1 - self.e**2)
    
    @property
    def r_periastr(self):
        """
        Calculate the periastron distance.

        Returns
        -------
        float
            Periastron distance.

        """
        return self.a * (1 - self.e)
    
    @property
    def r_apoastr(self):
        """
        Calculate the apoastron distance.

        Returns
        -------
        float
            Apoastron distance.

        """
        return self.a * (1 + self.e)
    
    def mean_motion(self, t): 
        """
        Mean motion M = 2 pi t / T_orb

        Parameters
        ----------
        t : np.ndarray
            time from periastron passage [s].

        Returns
        -------
        np.ndaray
            Mean motion at time t.

        """
        return 2 * pi * t / self.T
    
    def _ecc_an_novec(self, t):
        """
        Calculates the eccentric anomaly at the time t.

        Parameters
        ----------
        t : float
            Time after the periastron passage [s].

        Returns
        -------
        float
            E(t).

        """
        func_to_solve = lambda E: E - self.e * sin(E) - Orbit.mean_motion(self, t)
        try:
            E = brentq(func_to_solve, -1e3, 1e3)
            return E
        except:
            print('fuck smth wrong with Ecc_novec(t)')
            return np.nan

    def ecc_an(self, t): 
        """
        Calculates the eccentric anomaly at the time(s) t.

        Parameters
        ----------
        t : float | np.ndarray
            Times relative to the periastron passage [s].

        Returns
        -------
        float | np.ndarray
            E(t).

        """
        t_ = np.asarray(t)
        if t_.ndim == 0:
            return float(Orbit._ecc_an_novec(self, float(t_)))
        
        E_ = np.array([
            Orbit._ecc_an_novec(self, t_now) for t_now in t_
            ])
        return E_
        
        
    def r(self, t):
        """
        Distance between the components at time t.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            r(t).

        """
        return self.a * (1 - self.e * cos(Orbit.ecc_an(self, t)))
       
        
    def true_an(self, t):
        """
        True anomaly at time t. The angle between the direction of
          periastron and the current position of the body.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            nu_true(t).

        """
        ecc_ = Orbit.ecc_an(self, t)
        b_ = self.e / (1 + (1 - self.e**2)**0.5)
        return ecc_ + 2 * np.arctan(b_ * sin(ecc_) / (1 - b_ * cos(ecc_))) 
    

    def t_from_true_an(self, nu):
        """
        Time since periastron for a given true anomaly nu.
        Inverts true_an(t): solves t such that true_an(t) == nu.
    
        Parameters
        ----------
        nu : float or array_like
            True anomaly [rad]. May be a scalar or 1D array.
    
        Returns
        -------
        np.ndarray
            Time(s) since periastron passage [s], same shape as `nu`.
        """
        nu = np.asarray(nu, dtype=float)
        twopi = 2 * pi
    
        # Normalize ν to (-π, π] for a consistent branch
        nu_norm = (nu + pi) % twopi - pi
    
        # Convert true anomaly -> eccentric anomaly using a quadrant-safe formula
        # E = 2 * atan2( sqrt(1-e) * sin(ν/2), sqrt(1+e) * cos(ν/2) )
        s = sin(0.5 * nu_norm)
        c = cos(0.5 * nu_norm)
        E = 2.0 * np.arctan2(np.sqrt(1.0 - self.e) * s, np.sqrt(1.0 + self.e) * c)
    
        # Wrap E to (-π, π] to match the chosen branch
        E = (E + pi) % twopi - pi
        M = E - self.e * sin(E)
        # Add the correct number of full revolutions from the *unwrapped* ν
        # k is the integer number of 2π turns implied by ν with t=0 at ν=0.
        # For ν >= 0: k = floor(ν / 2π); for ν < 0: k = ceil(ν / 2π)
        # This makes small negative ν map to small negative t (not near -T).
        turns = np.where(nu >= 0.0, np.floor(nu / twopi+0.5),
                         np.ceil(nu / twopi-0.5)).astype(float)
        M_ext = M + turns * twopi
        n = twopi / self.T
        t = M_ext / float(n)
        return np.asarray(t)

    
    @property
    def t_los(self):
        """
        Time of the line-of-sight passage.

        Returns
        -------
        float
            t_los.

        """
        if abs(Orbit.true_an(self, self.T/2) - self.nu_los) < 1e-6:
            return self.T/2
        else:
            to_solve = lambda t_: Orbit.true_an(self, t_) - self.nu_los
            t_to_obs = brentq(to_solve, -self.T/2, self.T/2)
            return t_to_obs
    
    def x(self, t):
        """
        X-coordinate of the body at time t. The direction of the periastron is along the X-axis.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            x(t).

        """
        return self.a * (cos(Orbit.ecc_an(self, t)) - self.e)

    def y(self, t):
        """
        Y-coordinate of the body at time t. The direction of the periastron is along the X-axis.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            y(t).

        """
        return self.a * (1 - self.e**2)**0.5 * sin(Orbit.ecc_an(self, t))

    def z(self, t):
        """
        Z-coordinate of the body at time t. The direction of the periastron is along the X-axis.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            z(t).

        """
        return Orbit.x(self, t) * 0

    def vector_sp(self, t):
        """
        Position vector of the body at time t. The direction of the periastron is along the X-axis.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray of shape (3, len(t))
            3D position vector at time t.

        """
        return np.array([Orbit.x(self, t), Orbit.y(self, t), Orbit.z(self, t)])   

    def _calculate(self):
        """
        Tabulate the orbit, set xtab, ytab, ztab, ttab, rtab, nu_truetab.

        Returns
        -------
        None.

        """
        _E_tab = np.linspace(-2.5 * pi, 2.5 * pi, int(self.n))
        t_tab = self.T/ (2 * pi) * (_E_tab - self.e * sin(_E_tab))
        # t_tab = np.linspace(-self.T * 1.1, self.T * 1.1, int(self.n))
        self.xtab = Orbit.x(self, t_tab)
        self.ytab = Orbit.y(self, t_tab)
        self.ztab = Orbit.z(self, t_tab)    
        self.ttab = t_tab
        self.rtab = Orbit.r(self, t_tab)
        self.nu_truetab = Orbit.true_an(self, t_tab)

    def peek(self, ax=None,
             showtime = None,
             times_pos = (),
             color='k',
             xplot='time'):
        """
        Quick look at the orbit.

        Parameters
        ----------
        ax : axis from pyplot, optional
            The axis to draw an orbit on. Should be at least
              with 1 row and 3 columns.  The default is None.
        showtime : tuple (tmin, tmax), optional
            A tuple of min anf max times [s] for displaying the orbit.
            If None, then show from -T/2 to T/2.
              The default is None.
        times_pos : tuple, optional
            Times [s] at which to draw points on the orbit.
             The default is ().
        color : str, optional
            Pyplot-recognized color keyword. The default is 'k'.
        xplot : str, optional
            What should the x-axis be on the plots.
            Either 'time' or 'phase'.
               The default is 'time'.

        Returns
        -------
        None.

        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=3,
                                   figsize=(8, 8))
            
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
            

        # for ax_ in ax[1:]:
        #     pos = ax_.get_position()        # get [left, bottom, width, height]
        #     size = min(pos.width, pos.height)
        #     # Make the axes square, preserving center
        #     new_pos = [
        #         pos.x0 + (pos.width - size) / 2,
        #         pos.y0 + (pos.height - size) / 2,
        #         size,
        #         size,
        #     ]
        #     ax_.set_position(new_pos)