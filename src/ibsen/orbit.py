# ibsen/orbit.py
import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from astropy import constants as const
from ibsen.get_obs_data import get_parameters, known_names
from ibsen.utils import unpack_params, n_from_v, rotated_vector
G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

orbit_docstring = """
    Newtonian two-body Keplerian orbit in cgs units.

    The orbit is planar: the relative periastron direction is +X, the orbital
    angular-momentum direction is +Z, and ``t=0`` is periastron.  The primary
    coordinate convention is the *relative* vector
    ``vector_sp(t) = r_p(t) - r_s(t)`` from the optical star to the pulsar.
    Thus ``x``, ``y``, ``z``, ``r`` and their derivatives describe the
    star-to-pulsar separation, not a barycentric position.

    Barycentric positions are available separately: ``vector_s`` is the
    optical-star position and ``vector_p`` the pulsar position.  They satisfy
    ``M_s * vector_s + M_p * vector_p = 0`` and
    ``vector_p - vector_s = vector_sp``.  The scalar methods ending in
    ``_s`` and ``_p`` use the same barycentric convention.
    
    Parameters
    ----------
    sys_name : %s, or None, optional
        If provided, load default orbital parameters; explicit arguments
        below override those defaults. 
        If None, all parameters must be contained in `sys_params` dictionary or
        given explicitly.
    sys_params : dict or None, optional
        Parameter dictionary used instead of system defaults.  Relevant keys
        are ``T``, ``e``, ``M_s``, ``M_p``, ``nu_los`` and ``incl_los``.
        Explicit keyword arguments take precedence.
    T : float, optional
        Orbital period `T` in seconds.
    e : float, optional
        Orbital eccentricity (`0 <= e < 1`).
    M_s, M_p : float, optional
        Optical-star and pulsar masses [g].  Their sum determines the
        relative Keplerian orbit and their ratio assigns the two barycentric
        orbits.  ``Orbit.M_s`` is an orbital-dynamics quantity only.
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
    M_s, M_p : float
        Optical-star and pulsar masses [g] used by the Keplerian two-body
        calculation.
    M : float
        Total mass ``M_s + M_p`` [g].
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
    xtab_p, ytab_p, ztab_p : ndarray or None
        Tabulated pulsar barycentric coordinates [cm].
    xtab_s, ytab_s, ztab_s : ndarray or None
        Tabulated optical-star barycentric coordinates [cm].
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
    decc_an(t)
        Time derivative of the eccentric anomaly at time(s) `t`.
    r(t)
        Separation at time(s) `t`.
    true_an(t)
        True anomaly at time(s) `t`.
    kepl_period(t)
        Keplerian period at time(s) `t`.
    t_from_true_an(nu)
        Time(s) since periastron for given true anomaly(ies) `nu`.
    x(t), y(t), z(t)
        Relative coordinates at time(s) `t`. For the pulsar coordinates, use
        x_p, y_p, z_p. For the optical star coordinates, use x_s, y_s, z_s.
    a_s, a_p, b_s, b_p
        Optical-star and pulsar barycentric semi-axes.
    r_s(t), r_p(t)
        Optical-star and pulsar distances from the barycenter.
    dx(t), dy(t), dz(t)
        Velocity components at time(s) `t`.
    vector_sp(t)
        Relative star→pulsar position vector.
    vector_v(t)
        Relative star→pulsar velocity vector.
    vector_s(t), vector_p(t)
        Optical-star and pulsar barycentric position vectors.
    vector_v_s(t), vector_v_p(t)
        Optical-star and pulsar barycentric velocity vectors.
    peek(ax=None, showtime=None, times_pos=(), color='k', xplot='time')
        Quick look at the orbit.
    
    
    Notes
    -----
    Times ``t`` are interpreted as offsets from periastron passage (``t=0``).
    All distances are in cm and velocities in cm s^-1.

"""%(known_names,)

class Orbit:
    __doc__ = orbit_docstring
    def __init__(self, sys_name=None, sys_params=None, T=None, e=None,
                 M_s=None, M_p=None,
                 nu_los=None, incl_los=None, n=1000, allow_missing=False):
        """Initialize the relative Kepler orbit and its barycentric mass split.

        Parameter precedence and coordinate conventions are documented on
        :class:`Orbit`.  Tabulation is performed immediately when ``n`` is not
        ``None``.
        """
        T_, e_, M_s_, M_p_, nu_los_, incl_los_ = unpack_params(
            ('T', 'e', 'M_s', 'M_p', 'nu_los', 'incl_los'),
            orb_type=sys_name, sys_params=sys_params,
            known_types=known_names,
            get_defaults_func=get_parameters,
            T=T, e=e, M_s=M_s, M_p=M_p, nu_los=nu_los,
            incl_los=incl_los, allow_missing=allow_missing)
        self.e = e_
        self.T = T_
        self.M_s = M_s_
        self.M_p = M_p_
        self.M = self.M_s + self.M_p
        self.nu_los = nu_los_
        self.incl_los = incl_los_
        self.unit_los = n_from_v(rotated_vector(alpha=nu_los_, incl=incl_los_))
        self.GM = G * self.M
        self.name = sys_name
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
        Calculate the periastron binary separation.

        Returns
        -------
        float
            Periastron distance.

        """
        return self.a * (1 - self.e)
    
    @property
    def r_apoastr(self):
        """
        Calculate the apoastron  binary separation.

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
        func_to_solve = lambda E: E - self.e * sin(E) - self.mean_motion(t)
        try:
            E = brentq(func_to_solve, -1e3, 1e3)
            return E
        except:
            print("Something's wrong with `Ecc_novec`, maybe you multiplied by 86400 more than once?")
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
            return float(self._ecc_an_novec(float(t_)))
        
        E_ = np.array([
            self._ecc_an_novec(t_now) for t_now in t_
            ])
        return E_
        
        
    def r(self, t):
        """
        Magnitude of the relative star-to-pulsar separation at time ``t``.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            r(t).

        """
        return self.a * (1 - self.e * cos(self.ecc_an(t)))
    
    def kepl_period(self, t):
        """
        Keplerian period at the time t, defined as 2pi/Omega_kepl

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            T_kepl

        """
        return 2. * pi * np.sqrt(self.r(t)**3 / self.GM)
    
    def d_ecc_an(self, t):
        """
        Time derivative of the eccentric anomaly,
        :math:`\\dot{E} = (a/r) 2\\pi/T`.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            dot E.

        """
        return self.a / self.r(t) * 2 * pi / self.T
       
        
    def true_an(self, t):
        """
        Relative true anomaly at time ``t``.

        This is the angle from +X (the star→pulsar periastron direction) to
        ``vector_sp(t)``; it is independent of the barycentric mass ratio.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            nu_true(t).

        """
        ecc_ = self.ecc_an(t)
        b_ = self.e / (1. + np.sqrt(1. - self.e**2))
        return ecc_ + 2. * np.arctan(b_ * sin(ecc_) / (1. - b_ * cos(ecc_))) 
    

    def t_from_true_an(self, nu):
        """
        Time since periastron for a pulsar for a given true anomaly nu.
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
    
        # Normalize nu to (-π, π] for a consistent branch
        nu_norm = (nu + pi) % twopi - pi
    
        # Convert true anomaly -> eccentric anomaly using a quadrant-safe formula
        # E = 2 * atan2( sqrt(1-e) * sin(ν/2), sqrt(1+e) * cos(ν/2) )
        s = sin(0.5 * nu_norm)
        c = cos(0.5 * nu_norm)
        E = 2.0 * np.arctan2(np.sqrt(1.0 - self.e) * s, np.sqrt(1.0 + self.e) * c)
    
        # Wrap E to (-π, π] to match the chosen branch
        E = (E + pi) % twopi - pi
        M = E - self.e * sin(E)
        # Add the correct number of full revolutions from the *unwrapped* nu
        # k is the integer number of 2pi turns implied by nu with t=0 at nu=0.
        # For nu >= 0: k = floor(nu / 2π); for ν < 0: k = ceil(nu / 2π)
        # This makes small negative nu map to small negative t (not near -T).
        turns = np.where(nu >= 0.0, np.floor(nu / twopi+0.5),
                         np.ceil(nu / twopi-0.5)).astype(float)
        M_ext = M + turns * twopi
        n = twopi / self.T
        t = M_ext / float(n)
        return np.asarray(t)

    
    @property
    def t_los(self):
        """
        Time of the pulsar line-of-sight passage (inferior conjunction).

        Returns
        -------
        float
            t_los.

        """
        if abs(self.true_an(self.T/2.) - self.nu_los) < 1e-6:
            return self.T/2.
        else:
            to_solve = lambda t_: self.true_an(t_) - self.nu_los
            t_to_obs = brentq(to_solve, -self.T/2., self.T/2.)
            return t_to_obs
    
    def x(self, t):
        """
        X component of the relative star→pulsar vector.

        The periastron direction is +X.  This is not the pulsar's
        barycentric X coordinate; use ``x_p`` for that.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            x(t).

        """
        return self.a * (cos(self.ecc_an( t)) - self.e)

    def y(self, t):
        """
        Y component of the relative star→pulsar vector.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            y(t).

        """
        return self.b * sin(self.ecc_an( t))

    def z(self, t):
        """
        Z component of the relative star→pulsar vector.

        The present Keplerian model is planar, so this returns zero.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        np.ndarray
            z(t).

        """
        return t * 0
    

    def vector_sp(self, t):
        """
        Relative position vector from optical star to pulsar.

        Parameters
        ----------
        t : np.ndarray
            Time relative to the periastron passage [s].

        Returns
        -------
        ndarray, shape (3,) or (3, N)
            ``vector_p(t) - vector_s(t)`` in cm.  A scalar ``t`` gives
            shape ``(3,)``; array input gives shape ``(3, N)``.

        """
        return np.array([self.x( t), self.y( t), self.z( t)])   
    
    def dx(self, t):
        """Relative star→pulsar X velocity [cm s^-1]."""
        return -self.a * self.d_ecc_an(t) * sin(self.ecc_an(t))
    
    def dy(self, t):
        """Relative star→pulsar Y velocity [cm s^-1]."""
        return self.b * self.d_ecc_an(t) * cos(self.ecc_an(t))
    
    def dz(self, t):
        """Relative star→pulsar Z velocity [cm s^-1], identically zero."""
        return t * 0.
    
    def vector_v(self, t):
        """Relative velocity ``d(vector_sp)/dt`` [cm s^-1]."""
        return np.array([self.dx(t), self.dy(t), self.dz(t)])
    
    @property
    def a_s(self):
        """Optical-star barycentric semi-major axis [cm]."""
        return self.a * self.M_p / self.M
    
    @property
    def a_p(self):
        """Pulsar barycentric semi-major axis [cm]."""
        return self.a * self.M_s / self.M
    
    @property
    def b_s(self):
        """Optical-star barycentric semi-minor axis [cm]."""
        return self.b * self.M_p / self.M
    
    @property
    def b_p(self):
        """Pulsar barycentric semi-minor axis [cm]."""
        return self.b * self.M_s / self.M
    
    def r_s(self, t):
        """Optical-star distance from the barycenter [cm]."""
        return self.r(t) * self.M_p / self.M
    
    def r_p(self, t):
        """Pulsar distance from the barycenter [cm]."""
        return self.r(t) * self.M_s / self.M
    
    def x_s(self, t):
        """Optical-star barycentric X coordinate [cm]."""
        return -self.M_p / self.M * self.x(t)
    
    def y_s(self, t):
        """Optical-star barycentric Y coordinate [cm]."""
        return -self.M_p / self.M * self.y(t)
    
    def z_s(self, t):
        """Optical-star barycentric Z coordinate [cm]."""
        return -self.M_p / self.M * self.z(t)
    
    def x_p(self, t):
        """Pulsar barycentric X coordinate [cm]."""
        return self.M_s / self.M * self.x(t)
    
    def y_p(self, t):
        """Pulsar barycentric Y coordinate [cm]."""
        return self.M_s / self.M * self.y(t)
    
    def z_p(self, t):
        """Pulsar barycentric Z coordinate [cm]."""
        return self.M_s / self.M * self.z(t)
    
    def vector_s(self, t):
        """Optical-star barycentric position vector [cm]."""
        return -self.M_p / self.M * self.vector_sp(t)
    
    def vector_p(self, t):
        """Pulsar barycentric position vector [cm]."""
        return self.M_s / self.M * self.vector_sp(t)
    
    def vector_v_s(self, t):
        """Optical-star barycentric velocity vector [cm s^-1]."""
        return -self.M_p / self.M * self.vector_v(t)
    
    def vector_v_p(self, t):
        """Pulsar barycentric velocity vector [cm s^-1]."""
        return self.M_s / self.M * self.vector_v(t)

    def _calculate(self):
        """
        Tabulate the relative and barycentric orbit coordinates.

        Sets ``xtab_s``, ``ytab_s``, ``ztab_s``, ``rtab_s`` and their
        ``_p`` counterparts, together with ``ttab``, ``rtab``, and
        ``nu_truetab``.

        """
        _E_tab = np.linspace(-2.5 * pi, 2.5 * pi, int(self.n))
        t_tab = self.T / (2. * pi) * (_E_tab - self.e * sin(_E_tab))
        self.xtab_s = self.x_s( t_tab)
        self.ytab_s = self.y_s( t_tab)
        self.ztab_s = self.z_s( t_tab)    
        self.ttab = t_tab
        self.xtab_p = self.x_p( t_tab)
        self.ytab_p = self.y_p( t_tab)
        self.ztab_p = self.z_p( t_tab)    
        
        self.rtab_s = self.r_s( t_tab)
        self.rtab_p = self.r_p( t_tab)
        self.rtab = self.r( t_tab)
        
        
        self.nu_truetab = self.true_an( t_tab)

    def peek(self, ax=None,
             showtime = None,
             times_pos = (),
             show_star=False,
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

        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=3,
                                   figsize=(8, 8))
            
        if showtime is None:
            showtime = [-self.T/2, self.T/2]
        show_cond  = np.logical_and(self.ttab > showtime[0], 
                                    self.ttab < showtime[1])
            
        # ax[0].set_aspect('equal')
        ax[0].plot(self.xtab_p[show_cond], self.ytab_p[show_cond], color=color) # plot the pulsar orbit
        if show_star:
            ax[0].plot(self.xtab_s[show_cond], self.ytab_s[show_cond], color='r') # plot the star orbit
        ax[0].scatter(x=0, y=0, color='k', marker='x') # show a barycenter
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
        ax[2].plot(self.ttab[show_cond]/x_norma, np.rad2deg(self.nu_truetab[show_cond]), color=color)
        ax[1].axvline(x=self.t_los/x_norma, color=color, alpha=0.3)
        ax[2].axvline(x=self.t_los/x_norma, color=color, alpha=0.3)

        ax[1].set_ylabel(r'$r_\mathrm{sp}$, cm')
        ax[2].set_ylabel(r'$\nu_\mathrm{true}$, deg')
        ax[1].set_xlabel(xlabel_)
        ax[2].set_xlabel(xlabel_)
        
        
        for t_pos in times_pos:
            # self.rtab_p = self.r_p( t_tab)
            ax[0].scatter(x=self.x_p( t_pos),
                              y=self.y_p( t_pos), color=color) # draw a point at time t_pos
            if show_star:
                ax[0].scatter(x=self.x_s( t_pos),
                                  y=self.y_s( t_pos), color=color) # draw a star at time t_pos
            ax[1].scatter(x=t_pos/x_norma, y=self.r( t_pos), color=color)
            ax[2].scatter(x=t_pos/x_norma,
                          y=self.true_an( t_pos) * 180. / pi, color=color)
