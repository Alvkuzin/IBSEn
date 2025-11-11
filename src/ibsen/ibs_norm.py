import numpy as np
from numpy import pi, sin, cos, tan
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pathlib import Path
import xarray as xr
from ibsen.utils import beta_from_g, absv, \
 vector_angle, rotate_z, rotate_z_xy, n_from_v, plot_with_gradient
 
from astropy import constants as const

C_LIGHT = 2.998e10
DAY = 86400
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)

### --------------------- For shock front shape --------------------------- ###
_here = Path(__file__).parent          # points to pulsar/
_ibs_data_file = _here / "tab_data" / "Shocks4.nc"
ds_sh = xr.load_dataset(_ibs_data_file)


class IBS_norm: #!!!
    """
    Dimensionless intrabinary shock (IBS) geometry and kinematics.

    This class builds a two-horn, planar IBS shape in **normalized units** where
    the instantaneous star–pulsar separation is 1. The shape can be obtained
    from a pre-tabulated CRW-like solution (interpolated from
    ``tab_data/Shocks4.nc``) and is then postprocessed to provide arclength,
    tangents, asymptotics, flow directions, Doppler factors, and scattering
    angles. The line of sight (LoS) direction is user-set and normalized on
    input. All distances here are *dimensionless*; use a separate class
    ibs:IBS to rescale to physical units.

    Parameters
    ----------
    beta : float
        Momentum-flux ratio of the winds.
    s_max : float
        Arclength cutoff, so that only  points with ``s < s_max`` are kept.
        Default is 1.0.
    gamma_max : float, optional
        Maximum bulk Lorentz factor reached at ``s = s_max_g``. Default is 3.0.
    s_max_g : float, optional
        Arclength at which ``gamma == gamma_max`` (dimensionless). Default 4.0.
    n : int, optional
        Number of sampling points on **one** horn before mirroring. Default 31.
    unit_los : array_like, shape (3,), optional
        Line-of-sight unit vector; will be normalized internally.
        Default is ``[1, 0, 0]`` (in the direction of +X).

    Attributes
    ----------
    x, y : ndarray, shape (2*n,)
        IBS coordinates in the orbital plane (dimensionless).
    theta : ndarray
        Pulsar-to-IBS angle at each point (rad).
    r : ndarray
        Pulsar-to-IBS distance (dimensionless).
    s : ndarray
        Signed arclength along the IBS (dimensionless), increasing from the
        lower horn through the apex to the upper horn. ``s_max`` is updated to
        ``max(s)`` after construction.
    theta1 : ndarray
        Star-to-IBS angle at each point (rad).
    r1 : ndarray
        Star-to-IBS distance (dimensionless).
    tangent : ndarray
        Tangent direction angle along the curve (rad).
    ds_dtheta : ndarray
        ``ds/dtheta`` along the curve (dimensionless).
    thetainf : float
        Asymptotic angle returned by the tabulated solution (rad).
    x_apex : float
        Apex radius from the pulsar in normalized units (dimensionless).
    unit_r, unit_r1 : list of ndarray(3,)
        Unit vectors of pulsar→IBS and star→IBS directions at each point.
    unit_r_mid, unit_r1_mid : list of ndarray(3,)
        Mid-segment averages of the above (length ``len(x)-1``).
    unit_beta, unit_beta_mid : list of ndarray(3,)
        Unit flow directions along the IBS (pointwise and mid-segment).
    vec_beta, vec_beta_mid : list of ndarray(3,)
        Flow 3-vectors ``beta * unit_beta`` .
    unit_rsp : ndarray(3,)
        Unit vector from the star to the pulsar (dimensionless).
    r_sp : float
        Star–pulsar separation in the same normalized units (equals 1 here).
    g, g_mid : ndarray
        Bulk Lorentz factor along points / midpoints, via ``gma(s)``.
    beta_vel, beta_vel_mid : ndarray
        Speed in units of ``c`` along points / midpoints.
    dopl, dopl_mid : ndarray
        Doppler factors at points / midpoints for the given LoS.
    scattering_angle, scattering_angle_mid : ndarray
        Star–IBS–observer scattering angle (lab frame) at points / midpoints (rad).
    scattering_angle_comoving, scattering_angle_comoving_mid : ndarray
        Same angle evaluated in the bulk-comoving frame (rad).

    Methods
    -------
    calculate()
        Build the IBS from the tabulated dataset and populate all arrays/lists.
    s_interp(s_, what)
        Interpolate an attribute (e.g. ``'x'``, ``'y'``) at arclength coordinate ``s_``.
    gma(s)
        Bulk Lorentz factor as a function of arclength.
    theta_inf : property
        Asymptotic angle from the CRW equation.
    theta1_CRW(theta)
        Star-side angle as a function of pulsar-side angle (CRW 1996).
    rotate(phi)
        Return a rotated copy of the IBS (counter-clockwise by ``phi`` radians).
    calculate_ibs_shape_crw(full_return=False)
        Compute an analytic CRW-like curve directly (no table), primarily for checks.
    approx_IBS(full_output=False)
        Interpolate the pre-tabulated IBS (``Shocks4.nc``) at the requested ``beta``.
    doppler_factor(g_vel, ang_) : staticmethod
        Doppler factor ``1 / (g * (1 - beta*cos ang))`` for given ``g`` and angle.
    peek(fig=None, ax=None, ibs_color='k', to_label=True)
        Quick-look plot; can color by Doppler factor or scattering angles.

    Notes
    -----
    * All lengths are **dimensionless**; without rotation, the star is at 
        approximately ``(+1, 0)``,
      the pulsar at the origin, and the nominal star–pulsar line is along the
      ±X-axis (internally ``vec_rsp = [-1, 0, 0]``). Use a separate scaled class
      IBS to convert to cgs units.
    * The default construction path uses the pre-tabulated dataset
      ``tab_data/Shocks4.nc`` (loaded at import) and resamples to a symmetric,
      evenly spaced grid in ``y`` before mirroring to two horns.

    References
    ----------
    Canto, Raga & Wilkin (1996), ApJ 469, 729 — analytic bow-shock solution.

    """
    def __init__(self, beta, s_max=1.0, gamma_max=3.0, s_max_g=4., n=31,
                unit_los=np.array([1, 0, 0])):
        self.beta = beta # winds momenta ratio
        self.gamma_max = gamma_max # max bulk motion Lorentz factor at s = s_max_g
        self.s_max = s_max # dimentionless arclength of the IBS at which it should be cut
        self.s_max_g = s_max_g # the arclength at which the flow reaches gamma_max
        self.n = n # number of points on one horn of IBS
        self.unit_los = unit_los # unit vector in the Line of Sight direction

        self.calculate()

    
    def calculate(self):
        """
        Calculates the IBS: tables of x, y, theta, r, s, theta1, r1, tang, 
        d s / d theta, and sets them to the corresponding attributes. 
        Sets also theta_inf, and x_apex, unit vectors r, r1, r_sp, 
        bulk motion velosities; and vectors of dimentionless (non-unit) velosities
        beta. For x, y, s, and theta, creates minus/plus/mid/diff arrays of 
        length len(x)-1.

        Returns
        -------
        None.

        """
        (xp, yp, tp, rp, sp, t1p, r1p, tanp, ds_dtp, theta_inf_,
         r_apex) = IBS_norm.approx_IBS(self, full_output=True)
        self.x = xp
        self.y = yp
        self.theta = tp
        self.r = rp
        self.s = sp
        self.theta1 = t1p
        self.r1 = r1p
        self.tangent = tanp
        self.ds_dtheta = ds_dtp
        self.thetainf = theta_inf_
        self.x_apex = r_apex 
        
        norms_r = []
        norms_r1 = []
        unit_beta_= []
        self.unit_los = n_from_v(np.array(self.unit_los) )

        vec_rsp = np.array([-1, 0, 0])
        unit_rsp = n_from_v(vec_rsp)
        for x_, y_, th_, tan_ in zip(xp, yp, tp, tanp):
            n_ = np.array([x_, y_, 0])
            n_r_ = n_from_v(n_)
            n_r1_ = n_from_v(vec_rsp + n_)
            norms_r.append(n_r_)
            norms_r1.append(n_r1_)
            
            if th_ < 0: # if lower horn
                unit_beta_vec = -np.array([cos(tan_), sin(tan_), 0])
            else: # if upper horn
                unit_beta_vec = np.array([cos(tan_), sin(tan_), 0])
            unit_beta_.append(n_from_v(unit_beta_vec))
        
        
        # norms_r, norms_r1, unit_beta_ = [np.array(ar_) for ar_ in (norms_r, norms_r1, unit_beta_)]
        self.unit_r = norms_r
        self.unit_r1 = norms_r1
        self.unit_rsp = unit_rsp
        self.r_sp = absv(vec_rsp)
        self.unit_beta = unit_beta_
        beta_vecs = [_x * _y for _x, _y in zip(unit_beta_, beta_from_g(self.g))]
        self.vec_beta = beta_vecs
        self.s_max=np.max(sp)
        
        self.s_m = self.s[:self.s.size-1]
        self.s_p = self.s[1:]
        self.s_mid = 0.5*(self.s_m + self.s_p)
        self.ds = self.s_p-self.s_m

        self.x_m = self.x[:self.s.size-1]
        self.x_p = self.x[1:]
        self.x_mid = 0.5*(self.x_m + self.x_p)
        self.dx = self.x_p-self.x_m

        self.y_m = self.y[:self.s.size-1]
        self.y_p = self.y[1:]    
        self.y_mid = 0.5*(self.y_m + self.y_p)
        self.dy = self.y_p-self.y_m

        self.theta_m = self.theta[:self.s.size-1]
        self.theta_p = self.theta[1:]
        self.theta_mid = 0.5*(self.theta_m + self.theta_p)
        self.dtheta = self.theta_p - self.theta_m
        
        unit_r_mid_ = []
        unit_r1_mid_ = []
        unit_beta_mid_ = []
        # vec_beta_mid_ = []
        for i_m in range(0, self.s.size-1):
            # print(norms_r[i_m+1])
            unit_r_mid_.append(0.5 * (norms_r[i_m+1] + norms_r[i_m]) )
            unit_r1_mid_.append(0.5 * (norms_r1[i_m+1] + norms_r1[i_m]))
            unit_beta_mid_.append(0.5 * (unit_beta_[i_m+1] + unit_beta_[i_m]))
            # vec_beta_mid_.append(0.5 * (beta_vecs[i_m+1] + beta_vecs[i_m]))
        
        self.unit_r_mid = unit_r_mid_
        self.unit_r1_mid = unit_r1_mid_
        self.unit_beta_mid = unit_beta_mid_
        self.vec_beta_mid = [_x * _y for _x, _y in zip(unit_beta_mid_, beta_from_g(self.g_mid))]

    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_. 
        MIND THE DIMENSIONLESS! 
 
        Parameters
        ----------
        s_ : np.ndarray
            The arclength along the upper horn of the IBS to find the value at.
            Dimensionless (for non-scaled IBS).

        Returns
        -------
        The desired value of ibs.what in the coordinate s_. 

        """
        try:
            data = getattr(self, what)
        except AttributeError:
            raise ValueError(f"No such attribute '{what}' in IBS.")
        ##### here I set fill_value='extrapolate' instead of raising an error
        ##### or like filling with NaNs, cause the values at the ends of an 
        ##### IBS sometimes behave weirdly, and we DO need these values. So
        ##### since this is the internal function that should not be used
        ##### by an external user, we put `extrapolate` and use it VERY
        ##### cautiously!!!
        interpolator = interp1d(self.s, data, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return interpolator(s_)
    
    @property
    def int_an(self):
        """
        The dimentionless time of bulk motion from apex to s_max_g. 
        \int_0^{s_max_g} ds / beta(s).
        Returns
        -------
        float
            t(s_max_g) - t(0) in units of r_sp / c.

        """
        return self.s_max_g / (self.gamma_max - 1.) * (
         (self.gamma_max**2 - 1.)**0.5 - 1. )

    
    def gma(self, s):
        """
        The bulk motion Lorentz factor at the arclength s.

        Parameters
        ----------
        s : np.ndarray
            The arclength along the IBS [dimensionless].

        Returns
        -------
        np.ndarray
            Gamma(s). 

        """
        return 1. + (self.gamma_max - 1.) * np.abs(s) / self.s_max_g        

        
    @property
    def g(self):
        """
        The bulk motion Lorentz factor at every point on the IBS.

        Returns
        -------
        np.ndarray
            Gamma(s_ibs).

        """
        return IBS_norm.gma(self, s = self.s)
    
    @property
    def g_mid(self):
        """
        The bulk motion Lorentz factor at mid points on the IBS.

        Returns
        -------
        np.ndarray
            Gamma(s_mid).

        """
        return IBS_norm.gma(self, s = self.s_mid)
    
    
    @property
    def beta_vel(self):
        """
        The bulk motion velocity in units of c at every point on the IBS.

        Returns
        -------
        np.ndarray 
            beta(s_ibs).

        """
        return beta_from_g(g_vel = self.g)
    
    @property
    def beta_vel_mid(self):
        """
        The bulk motion velocity in units of c at mid points on the IBS.

        Returns
        -------
        np.ndarray 
            beta(s_mid).

        """
        return beta_from_g(g_vel = self.g_mid)

    
    @property
    def theta_inf_analytical(self):
        """
        The asymptotic angle of the IBS in radians (> pi/2).

        Returns
        -------
        float
            theta_inf [rad].

        """
        to_solve1 = lambda tinf: tinf - tan(tinf) - pi / (1. - self.beta)
        try:
            th_inf = brentq(to_solve1, pi/2 + 1e-5, pi - 1e-5)
            return th_inf
        except:
            return np.nan
    
        
    def theta1_CRW(self, theta):
        """
        Theta1 (star-to-IBS angle) as a function
          of theta in the model of Canto, Raga, Wilkin (1996).

        Parameters
        ----------
        theta : float
            Pulsar-to-IBS angle [rad].

        Returns
        -------
        float
            theta1(theta) [rad].

        """
        if theta == 0:
            return 0
        else:
            th1_inf = pi - self.theta_inf 
            to_solve2 = lambda t1: t1 / tan(t1) - 1. - self.beta * (theta / tan(theta) - 1)
            th1 = brentq(to_solve2, 1e-10, th1_inf)
            return th1
        
    def rotate(self, phi):
        """
        Rotates the shock at the angle phi COUNTERCLOCKWISE. Note that the vector of 
        the line of sight unit_los is not rotated.
        
        Parameters
        ----------
        phi : float
            The angle to rotate at [rad].

        Returns
        -------
        rotated_ibs : class IBS_norm
            The class of dimentionleess IBS corresponding to the rotated IBS.

        """

        rotated_ibs = self.__class__(beta=self.beta, gamma_max=self.gamma_max, s_max=self.s_max,
                               s_max_g = self.s_max_g, n=self.n, unit_los=self.unit_los)
        rotated_ibs.x, rotated_ibs.y  = rotate_z_xy(self.x, self.y, phi)
        rotated_ibs.x_m, rotated_ibs.y_m  = rotate_z_xy(self.x_m, self.y_m, phi)
        rotated_ibs.x_p, rotated_ibs.y_p  = rotate_z_xy(self.x_p, self.y_p, phi)
        rotated_ibs.x_mid, rotated_ibs.y_mid  = rotate_z_xy(self.x_mid, self.y_mid, phi)
        
        rotated_ibs.tangent = self.tangent + phi
        for i in range(self.s.size):
            rotated_ibs.unit_r[i] = rotate_z(self.unit_r[i], phi)
            rotated_ibs.unit_beta[i] = rotate_z(self.unit_beta[i], phi)
            rotated_ibs.vec_beta[i] = rotate_z(self.vec_beta[i], phi)
            rotated_ibs.unit_r1[i] = rotate_z(self.unit_r1[i], phi)
            if i != self.s.size-1:
                rotated_ibs.unit_r_mid[i] = rotate_z(self.unit_r_mid[i], phi)
                rotated_ibs.unit_r1_mid[i] = rotate_z(self.unit_r1_mid[i], phi)
                rotated_ibs.vec_beta_mid[i] = rotate_z(self.vec_beta_mid[i], phi)
                rotated_ibs.unit_beta_mid[i] = rotate_z(self.unit_beta_mid[i], phi)
            
            
            
        rotated_ibs.unit_rsp = rotate_z(self.unit_rsp, phi)

        return rotated_ibs
    
    def calculate_ibs_shape_crw(self, full_return = False):
        """
        Calculates the IBS shape in the model of Canto, Raga, Wilkin (1996)
        https://ui.adsabs.harvard.edu/abs/1996ApJ...469..729C/abstract

        Parameters
        ----------
        full_return : bool, optional
            Whether to return less or more. The default is False.

        Returns
        -------
        Tuple
            If full_return=True, returnes 7 arrays of length n: x, y, theta,
            r, s, theta1, r1. If full_return=False, returnes 5 arrays of length n:
            x, y, theta, r, s.

        """
        th_inf = self.theta_inf
        beta_ = self.beta
        if beta_ > 1e-3:
            ## use the numerical solution for theta1(theta)
            thetas = np.linspace(1e-3, th_inf-1e-3, self.n)
            theta1s = np.zeros(thetas.size)
            theta1s = np.array([self.theta1_CRW(thetas[i]) for i in range(theta1s.size)])
        if beta_ <= 1e-3:
            ## use the approximate analytical solution for theta1(theta)
            thetas = np.linspace(1e-3, th_inf*(1 - beta_**2), self.n)
            theta1s = (7.5 * (-1. + (1. + 0.8 * beta_ * (1 - thetas / tan(thetas)) )**0.5) )**0.5
        rs = sin(theta1s) / sin(thetas + theta1s)
        ys = rs * sin(thetas)
        xs = rs * cos(thetas)
        r1s = ( (1-xs)**2 + ys**2)**0.5
        ds2 = np.zeros(rs.size)
        ss = np.zeros(rs.size)
        
        ds2[1:] = np.array([ (xs[i] - xs[i-1])**2 + 
                        (ys[i] - ys[i-1])**2 for i in range(1, rs.size)])
        ss[1:] = np.array([np.sum(ds2[0:i+1]**0.5) for i in range(1, rs.size)])
        inds = np.where(ss < 1.5 * self.s_max) #!!!
        thetas, theta1s, rs, xs, ys, ss, r1s = [arr[inds] for arr in (thetas, theta1s, rs, xs, ys, ss, r1s)]
        if not full_return:    
            return xs, ys, thetas, rs, ss
        if full_return:
            return xs, ys, thetas, rs, ss, theta1s, r1s
        
    def approx_IBS(self, full_output = False):
        """
        The tabulated shape of the IBS in the model of Canto, Raga, Wilkin (1996)
        https://ui.adsabs.harvard.edu/abs/1996ApJ...469..729C/abstract. It reads
        the pre-calculated file TabData/Shocks4.nc. 
        
        Parameters
        ----------
        full_output : bool, optional
            Whether to return less or more. The default is False.
    
        Returns
        -------
        tuple
            The shape of the IBS: the tuple of its characteristics. If full_output
            = False, then the tuple is (x, y). If full_output
            = True, then the tuple is (x, y, theta, r, s, theta1, r1, 
            theta_tangent, ds_dtheta, theta_inf (float), r_in_apex (float)). 
            All quantities are dimentionless, so that the distance between the 
            star and the pulsar = 1.
    
        """
        b = np.abs(np.log10(self.beta))
        # first, find the shape in given b by interpolation
        intpl = ds_sh.interp(abs_logbeta=b)
        # and get its x, y, theta, r, s, theta1, r1 as np.arrays
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = (intpl.x, intpl.y, intpl.theta,
                        intpl.r, intpl.s, intpl.theta1, intpl.r1, )
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [np.array(arr) for arr in (xs_, ys_,
                                        ts_, rs_, ss_, t1s_, r1s_)]
        
        tang = np.arctan(np.gradient(ys_, xs_, edge_order=2))
        ds_dt_ = np.gradient(ss_, ts_, edge_order=2)
        if isinstance(self.s_max, float) or isinstance(self.s_max, int):
            self.s_max = float(self.s_max)
            # leave only the points where arclength < s_max. This may not work good
            # when the b is super big, like > 8-9, as many points will be cut from
            # the already sparse arrays
            ok = np.where(ss_ < self.s_max)
        if isinstance(self.s_max, str):
            if self.s_max == 'bow':
                # leave only the part of the shock in forward half-sphere from the 
                # pulsar
                ok = np.where(xs_ >= 0)
            if self.s_max == 'incl':
                # leave only the parts where the angle between the flow and the line
                # from pulsar is < 90 + 10
                ok = np.where(ts_ + np.abs(tang) <= pi/2 + 10/180*pi)
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_, tang, ds_dt_ = [arr[ok] for arr in (xs_, ys_,
                                            ts_, rs_, ss_, t1s_, r1s_, tang, ds_dt_)]
        # now we interpolate the values onto the equally-spaced grid over y with
        # na nods, since this is the best way to interpolate (not over s)
        intx, ints, intth, intr, intth1, intr1, inttan, intdsdt = (interp1d(ys_, xs_),
                interp1d(ys_, ss_),
                interp1d(ys_, ts_), interp1d(ys_, rs_), interp1d(ys_, t1s_), 
                interp1d(ys_, r1s_), interp1d(ys_, tang), interp1d(ys_, ds_dt_))    
        yplot = np.linspace(np.min(ys_)*1.001, np.max(ys_)*0.999, int(self.n))
        xp, tp, rp, sp, t1p, r1p, tanp, ds_dtp = (intx(yplot), intth(yplot), intr(yplot),
                                          ints(yplot),
                intth1(yplot), intr1(yplot), inttan(yplot), intdsdt(yplot))
        yp = yplot
        
        xp = np.concatenate((xp[::-1], xp))
        yp = np.concatenate((-yp[::-1], yp))
        tp = np.concatenate((-tp[::-1], tp))
        rp = np.concatenate((rp[::-1], rp))
        sp = np.concatenate((-sp[::-1], sp))
        t1p = np.concatenate((-t1p[::-1], t1p))
        r1p = np.concatenate((r1p[::-1], r1p))
        tanp = np.concatenate((-tanp[::-1], pi+tanp))
        ds_dtp = np.concatenate((ds_dtp[::-1], ds_dtp))
        if full_output:
            return (xp, yp, tp, rp, sp, t1p, r1p, tanp, ds_dtp, intpl.theta_inf.item(),
        intpl.r_apex.item())
        if not full_output:
            return xp, yp
            
    @staticmethod
    def doppler_factor(g_vel, ang_):
        """
        The doppler factor for a blob moving with a Lorentz factor g_vel at an angle
        ang_ to the observer: 1 / g / (1 - beta * cos(ang_)).

        Parameters
        ----------
        g_vel : np.ndarray
            Lorentz factor of the blob.
        ang_ : np.ndarray
            The angle between the blob velocity and the LoS [rad].

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        beta_vels = beta_from_g(g_vel=g_vel)
        return 1 / g_vel / (1 - beta_vels * cos(ang_))
            
            
    @property
    def dopl_angle(self):
        """
        The angle between the velocity and the LoS direction for every point
        on the IBS. np.ndarray
        of length len(self.x).
        """
        angs = np.array(vector_angle(self.unit_los, self.unit_beta))
        return angs
    
    @property
    def dopl_angle_mid(self):
        """
        The angle between the velocity and the LoS direction for mid points of
        IBS. np.ndarray of length len(self.x)-1.
        """
        angs = np.array(vector_angle(self.unit_los, self.unit_beta_mid))
        return angs
 
    @property
    def dopl(self):
        """IBS doppler factors delta in every point of the IBS"""
        return IBS_norm.doppler_factor(g_vel = self.g, ang_ = self.dopl_angle)
    @property
    def dopl_mid(self):
        """IBS doppler factors delta in mid points of the IBS"""
        return IBS_norm.doppler_factor(g_vel = self.g_mid, ang_ = self.dopl_angle_mid)
    

    @property
    def scattering_angle(self):
        """
        The scattering angle for a photon from the star scattered at the IBS
        towards the observer in the lab frame. For every point of the IBS.
        """
        return np.array(vector_angle(self.unit_r1, self.unit_los))


    @property
    def scattering_angle_mid(self):
        """
        The scattering angle for a photon from the star scattered at the IBS
        towards the observer in the lab frame. For mid points of the IBS.
        """
        return np.array(vector_angle(self.unit_r1_mid, self.unit_los))

    @property
    def scattering_angle_comoving(self):
        """
        The scattering angle for a photon from the star scattered at the IBS
        towards the observer in the frame co-moving with the bulk motion 
        velocity. For every point of the IBS.
        """        
        return np.array(vector_angle(self.unit_r1, self.unit_los,
                                     self.vec_beta, True))
    
    @property
    def scattering_angle_comoving_mid(self):
        """
        The scattering angle for a photon from the star scattered at the IBS
        towards the observer in the frame co-moving with the bulk motion 
        velocity. For mid points of the IBS.
        """        
        return np.array(vector_angle(self.unit_r1_mid, self.unit_los,
                                     self.vec_beta_mid, True))    
        
    def peek(self, fig=None, ax=None,
             ibs_color='k', to_label=True):
        """
        Quick look at the IBS.

        Parameters
        ----------
        fig : fig object of pyplot, optional
             The default is None.
        ax : ax object of pyplot, optional
            DESCRIPTION. The default is None.
        ibs_color : str, optional
            Can be one of {'doppler', 'scattering', 'scattering_comoving'} 
            to colorcode the IBS by the doppler factor, or the scattering angle
            in the lab or comoving frame, respectively; or any matplotlib color. 
            The default is 'k'.
        to_label : bool, optional
            Whether to put a label `beta=...` on a plot.
              The default is True.

        Raises
        ------
        ValueError
            If the ibs_color is not one of the recognizable options.

        Returns
        -------
        None.

        """
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))    

        if to_label:
            label = rf'$\beta = {self.beta}$'
        else:
            label = None
        xstar_, ystar_, zstar_ = -self.unit_rsp
        xlos_, ylos_, zlos_ = self.unit_los

        if ibs_color in ( 'doppler', 'scattering', 'scattering_comoving'):
            # print(2)
            if ibs_color == 'doppler':
                color_param = self.dopl
                bar_label = r'doppler factor $\delta$'
            elif ibs_color == 'scattering':
                # print(3)
                color_param = self.scattering_angle / pi
                bar_label = r'scattering angle / $\pi$'
            elif ibs_color == 'scattering_comoving':
                color_param = self.scattering_angle_comoving / pi
                bar_label = r'comoving scattering angle / $\pi$'
            else:
                raise ValueError(f"Unknown ibs_color: {ibs_color}. "
                                 "Use 'doppler', 'scattering' or 'scattering_comoving'.")

            plot_with_gradient(fig=fig, ax=ax, xdata=self.x, ydata=self.y,
                            some_param=color_param, colorbar=True, lw=2, ls='-',
                            colorbar_label=bar_label)

        else:
            try:
                ax.plot(self.x, self.y, color=ibs_color, label = label)
                ax.plot([-5*xstar_, 5*xstar_], [-5*ystar_, 5*ystar_], 
                        color=ibs_color, ls='--', alpha = 0.3)
            except:
                raise ValueError('Probably wrong color keyword :(((')

        # if not rescaled:

        ax.scatter(xstar_, ystar_, color='b') # star is originally in (1, 0)
        ax.scatter(0, 0, color='r') # pulsar is in (0, 0)

        ax.plot([0, 5*xlos_], [0, 5*ylos_], color='g', ls='--', alpha = 0.3)

        x_scale = np.max(np.array([
            np.abs(np.min(self.x)), np.abs(np.max(self.x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(self.y)), np.abs(np.max(self.y))            
            ]))
            
        ax.set_xlim(-1.2*x_scale, 1.2*x_scale )
        ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        
            
        ax.legend()
