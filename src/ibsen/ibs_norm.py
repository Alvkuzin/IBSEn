import numpy as np
from numpy import sin, cos
from ibsen.utils import beta_from_g, absv, \
 vector_angle, rotate_z, rotate_z_xy, n_from_v, plot_with_gradient
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import xarray as xr

### --------------------- For shock front shape --------------------------- ###
_here = Path(__file__).parent          
_ibs_data_file = _here / "tab_data" / "Shocks4.nc"
ds_sh = xr.load_dataset(_ibs_data_file)
### ----------------------------------------------------------------------- ###
 
class BaseIBSNorm_2D:
    """
    Base class for the 2D IBS. It needs mixins that define kinematics
    (method self.gma(s)) and the geometry of one horn of the IBS
    (attributes x_1horn, y_1horn, ... <same for s, theta, theta1, r, r1, 
     tang, ds_dtheta>, and also unit_los).
    """
    def __init__(self, unit_los=np.array([1, 0, 0])):
        self.unit_los = unit_los
        self._set_reflected_horn()
        self._set_midparams_and_vecs()
        self._set_beta_vecs()
        
    
    def _set_reflected_horn(self):
        """
        Given the values at one (upper) horn of the IBS, define another
        horn.
        """
        xp, yp, sp, tp, t1p, rp, r1p, tanp, ds_dthetap = [getattr(self, f"{name}_1horn" ) 
            for  name in ("x", "y", "s", "theta", "theta1", "r", "r1", "tang", "ds_dtheta")]
        self.x = np.concatenate((xp[::-1], xp))
        self.y = np.concatenate((-yp[::-1], yp))
        self.theta = np.concatenate((-tp[::-1], tp))
        self.r = np.concatenate((rp[::-1], rp))
        self.s = np.concatenate((-sp[::-1], sp))
        self.theta1 = np.concatenate((-t1p[::-1], t1p))
        self.r1 = np.concatenate((r1p[::-1], r1p))
        self.tang = np.concatenate((-tanp[::-1], np.pi+tanp))
        self.ds_dtheta = np.concatenate((ds_dthetap[::-1], ds_dthetap))
        
        norms_r = []
        norms_r1 = []
        unit_beta_= []
        self.unit_los = n_from_v(np.array(self.unit_los) )
        vec_rsp = self.unit_rsp 
        for x_, y_, th_, tan_ in zip(self.x, self.y, self.theta, self.tang):
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
        self.unit_beta = unit_beta_
    
    def _set_mid_attrs(self, name):
        """
        For an attribute named X, sets attributes X_m and X_p as self.X[:-1]
        and self.X[1:], as well as X_mid as their average and dX as their difference.
        """
        arr = getattr(self, name)
        _m, _p = arr[:-1], arr[1:]
        _mid, _d = 0.5 * (_p + _m), (_p - _m)
        
        setattr(self, f"{name}_m", _m)
        setattr(self, f"{name}_p", _p)
        setattr(self, f"{name}_mid", _mid)
        setattr(self, f"d{name}", _d)
        
        
    def _set_midparams_and_vecs(self):
        for name in ("x", "y", "s", "theta", "r", "r1", "tang", "ds_dtheta"):
            self._set_mid_attrs(name)
        unit_r_mid_ = []
        unit_r1_mid_ = []
        unit_beta_mid_ = []
        # vec_beta_mid_ = []
        for i_m in range(0, self.s.size-1):
            unit_r_mid_.append(0.5 * (self.unit_r[i_m+1] + self.unit_r[i_m]) )
            unit_r1_mid_.append(0.5 * (self.unit_r1[i_m+1] + self.unit_r1[i_m]))
            unit_beta_mid_.append(0.5 * (self.unit_beta[i_m+1] + self.unit_beta[i_m]))
        
        self.unit_r_mid = unit_r_mid_
        self.unit_r1_mid = unit_r1_mid_
        self.unit_beta_mid = unit_beta_mid_
        
    def _set_beta_vecs(self):
        self.vec_beta = [_unit_beta * _v for _unit_beta, _v in
                         zip(self.unit_beta, beta_from_g(self.g))]
        self.vec_beta_mid = [_unit_beta * _v for _unit_beta, _v 
                             in zip(self.unit_beta_mid, beta_from_g(self.g_mid))]
        
        
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
        
    def gma(self, s):
        raise NotImplementedError
    
    @property
    def g(self):
        """
        The bulk motion Lorentz factor at every point on the IBS.

        Returns
        -------
        np.ndarray
            Gamma(s_ibs).

        """
        return self.gma(s = self.s)
    
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
            Relative to the initial IBS object, changed attributes are: x, y, x_m, y_m, 
            x_p, y_p, x_mid, y_mid, tangent, unit_r/unit_r_mid, 
            unit_beta/unit_beta_mid, unit_r1/unit_r1_mid, vec_beta/vec_beta_mid

        """

        rotated_ibs = self.__class__(beta=self.beta, gamma_max=self.gamma_max, s_max=self.s_max,
                               s_max_g = self.s_max_g, n=self.n, unit_los=self.unit_los)
        rotated_ibs.x, rotated_ibs.y  = rotate_z_xy(self.x, self.y, phi)
        rotated_ibs.x_m, rotated_ibs.y_m  = rotate_z_xy(self.x_m, self.y_m, phi)
        rotated_ibs.x_p, rotated_ibs.y_p  = rotate_z_xy(self.x_p, self.y_p, phi)
        rotated_ibs.x_mid, rotated_ibs.y_mid  = rotate_z_xy(self.x_mid, self.y_mid, phi)
        
        rotated_ibs.tang = self.tang + phi
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
    def scattering_angle_comov(self):
        """
        The scattering angle for a photon from the star scattered at the IBS
        towards the observer in the frame co-moving with the bulk motion 
        velocity. For every point of the IBS.
        """        
        return np.array(vector_angle(self.unit_r1, self.unit_los,
                                     self.vec_beta, True))
    
    @property
    def scattering_angle_mid_comov(self):
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

        if ibs_color in ( 'doppler', 'scattering', 'scattering_comov', 'scattering_comoving'):
            if ibs_color == 'doppler':
                color_param = self.dopl
                bar_label = r'doppler factor $\delta$'
            elif ibs_color == 'scattering':
                color_param = self.scattering_angle / np.pi
                bar_label = r'scattering angle / $\pi$'
            elif ibs_color in ('scattering_comoving', 'scattering_comov'):
                color_param = self.scattering_angle_comov / np.pi
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
    
    

class IBS_kinematics:
    """
    A mixin defining the kinematics.
    """
    def __init__(self, *args, **kwargs):
        self.gamma_max = kwargs.pop("gamma_max", 1.5)
        self.s_max_g = kwargs.pop("s_max_g", 1.5)
        super().__init__(*args, **kwargs)
        
    def gma(self, s):
        return 1. + (self.gamma_max - 1.) * np.abs(s) / self.s_max_g  
    



class IBSGeometryOneHorn:
    """
    The mixin passed to this class should contain as attributes: 
    x_1horn, y_1horn, s_1horn, tang_1horn, ds_dtheta_1horn, x_apex.
    It, itself, is an intermediate class that defines some auxillary parameters
    of the IBS, mid-values, and some vectors.
    """
    def __init__(self, *args, **kwargs):
        self.beta = kwargs.pop("beta")
        self.n = kwargs.pop("n", 15)
        self.s_max = kwargs.pop("s_max", 1.0)
        
                
        self._set_ibs_coords()
        self._set_auxil_params()
        self._set_vectors()
        
        super().__init__(*args, **kwargs)

        
    def _set_ibs_coords(self):
        """For one horn (at phi=const) of the IBS, mixins should set the
        attributes x_1horn, y_1horn, s_1horn, tang_1horn, ds_dtheta_1horn,
        which are the arrays of length n:
            - x_1horn, 
            - y_1horn, 
            - s_1horn (arclength along the IBS, starting from the apex where s=0),
            - tang_1horn (tangent angle: an angle between the tangent vector to a point
              directed away from the apex, and the x-axis),
            - ds_dtheta_1horn (ds/dtheta: literally the derivative ds/dtheta),
        and a scalar attribute:
            - x_apex (the distance from the pulsar to the apex).
            
        These are the parameters that cannot be properly calculated from just
        the (x, y) provided. Everything else, like r, r1, theta, ... we calculate
        in this base class,
            """
        raise NotImplementedError
        
    def _set_auxil_params(self):
        """
        After the ibs coords are set, calculate everything else of interest.
        """
        _x, _y = self.x_1horn, self.y_1horn
        self.theta_1horn = np.arctan2( _y,  _x)
        self.theta1_1horn = np.arctan2( _y, 1 - _x)
        self.r_1horn = np.sqrt(_x**2 + _y**2)
        self.r1_1horn = np.sqrt((1 - _x)**2 + _y**2)
        
    def _set_vectors(self):
        """Sets arrays of vectors: unit_r, unit_r1, unit_beta (unit vectors 
        in directions of r, r1, and in the direction of bulk motion), as
        well as vectors unit_rsp and vec_rsp (which is the same)."""
        norms_r_1horn = []
        norms_r1_1horn = []
        unit_beta_1horn= []
        # self.unit_los = n_from_v(np.array(self.unit_los) )

        vec_rsp = np.array([-1, 0, 0])
        unit_rsp = n_from_v(vec_rsp)
        for x_, y_, th_, tan_ in zip(self.x_1horn, self.y_1horn, 
                                    self.theta_1horn, self.tang_1horn):
            n_ = np.array([x_, y_, 0])
            n_r_ = n_from_v(n_)
            n_r1_ = n_from_v(vec_rsp + n_)
            norms_r_1horn.append(n_r_)
            norms_r1_1horn.append(n_r1_)            
            unit_beta_1horn.append(n_from_v(np.array([cos(tan_), sin(tan_), 0])))
        
        
        # norms_r, norms_r1, unit_beta_ = [np.array(ar_) for ar_ in (norms_r, norms_r1, unit_beta_)]
        self.unit_r_1horn = norms_r_1horn
        self.unit_r1_1horn = norms_r1_1horn
        self.unit_rsp = unit_rsp
        self.r_sp = absv(vec_rsp)
        self.unit_beta_1horn = unit_beta_1horn
        
        
        
class IBSToyModel: 
    """
    A mixin describing an IBS in a shape of a hemisphere. Not fully working,
    it's more for testing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def radius(self):
        return self.beta**0.5 / (1 + self.beta**0.5)
    
    def _y_circle_1horn(self, x):
        return np.sqrt(self.radius**2 - x**2)
    
    def _set_ibs_coords(self):
        all_thetas = np.linspace(0, np.pi/2, 1000)
        all_xs = self.radius * np.cos(all_thetas)
        all_ys = self._y_circle_1horn(all_xs)
        all_theta1s = np.arctan2(all_ys, 1 - all_xs)
        all_r1s = ( (1 - all_xs)**2 + all_ys**2)**0.5
        all_rs = np.sqrt(all_xs**2 + all_ys**2)
        all_ds2 = np.zeros(all_rs.size)
        all_ss = np.zeros(all_rs.size)
        
        all_ds2[1:] = np.array([ (all_xs[i] - all_xs[i-1])**2 + 
                        (all_ys[i] - all_ys[i-1])**2 for i in range(1, all_rs.size)])
        all_ss[1:] = np.array([np.sum(all_ds2[0:i+1]**0.5) for i in range(1, all_rs.size)])
        all_tangs = np.arctan(np.gradient(all_ys, all_xs, edge_order=2))
        all_ds_dthetas = np.gradient(all_ss, all_thetas, edge_order=2)
        #######################################################################
        ys_when_s_is_smax = all_ys[np.argmin(np.abs(all_ss-self.s_max)) - 1]
        ys_needed = np.linspace(np.min(all_ys)*1.001, ys_when_s_is_smax*0.999, int(self.n))
        thetas_needed, theta1s_needed, rs_needed, xs_needed, ss_needed, \
            r1s_needed, tangs_needed, dsdts_needed = [np.interp(x=ys_needed, xp=all_ys, fp=_ar)
            for _ar in (all_thetas, all_theta1s, all_rs, all_xs, all_ss, all_r1s, all_tangs, all_ds_dthetas)]

        self.x_1horn = xs_needed
        self.y_1horn = ys_needed
        self.s_1horn = ss_needed
        self.tang_1horn = tangs_needed
        self.ds_dtheta_1horn = dsdts_needed
        self.x_apex = self.radius
        

        
        
class IBS_CRW96: 
    """
    Defines one horn of the CRW model of the IBS. It reads the values from the
    tabulated data and interpolates onto the required amount of IBS points.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def thetainf(self):
        """
        The asymptotic angle of the IBS in radians (> pi/2)
        in the model of Canto, Raga, Wilkin (1996)

        Returns
        -------
        float
            theta_inf [rad].

        """
        to_solve1 = lambda tinf: tinf - np.tan(tinf) - np.pi / (1. - self.beta)
        try:
            th_inf = brentq(to_solve1, np.pi/2 + 1e-5, np.pi - 1e-5)
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
            th1_inf = np.pi - self.theta_inf_analytical 
            to_solve2 = lambda t1: t1 / np.tan(t1) - 1. - self.beta * (theta / np.tan(theta) - 1)
            th1 = brentq(to_solve2, 1e-10, th1_inf)
            return th1
        
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
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = (intpl.x, intpl.y, intpl.theta,
                        intpl.r, intpl.s, intpl.theta1, intpl.r1, )
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_ = [np.array(arr) for arr in (xs_, ys_,
                                        ts_, rs_, ss_, t1s_, r1s_)]
        
        tang = np.arctan(np.gradient(ys_, xs_, edge_order=2))
        ds_dt_ = np.gradient(ss_, ts_, edge_order=2)
        r_apex_analyt = self.beta**0.5 / (self.beta**0.5 + 1)
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
                ok = np.where(np.abs(ts_) <= np.pi/2)
            if self.s_max == 'incl':
                # leave only the parts where the angle between the flow and the line
                # from pulsar is < 90 + 10
                ok = np.where(ts_ + np.abs(tang) <= np.pi/2 + 10/180*np.pi)
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
        
        if full_output:
            # return (xp, yp, tp, rp, sp, t1p, r1p, tanp, ds_dtp, self.theta_inf_analytical,
            #         r_apex_analyt)
            return (xp, yp, sp, tanp, ds_dtp, r_apex_analyt)
            
        
        if not full_output:
            return xp, yp
    
    def _y_circle_1horn(self, x):
        return np.sqrt(self.radius**2 - x**2)
    
    def _set_ibs_coords(self):
        xs_needed, ys_needed, ss_needed, tangs_needed, dsdts_needed, _r_apex = self.approx_IBS(True)
        self.x_1horn = xs_needed
        self.y_1horn = ys_needed
        self.s_1horn = ss_needed
        self.tang_1horn = tangs_needed
        self.ds_dtheta_1horn = dsdts_needed
        self.x_apex = _r_apex
        
    
class IBS_norm_toy(IBSToyModel, IBSGeometryOneHorn, IBS_kinematics, BaseIBSNorm_2D):
    """Toy IBS_norm class."""
    pass

class IBS_norm(IBS_CRW96, IBSGeometryOneHorn, IBS_kinematics, BaseIBSNorm_2D):
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
    pass

    