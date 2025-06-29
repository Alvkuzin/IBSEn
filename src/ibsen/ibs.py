# pulsar/ibs.py
import numpy as np
from numpy import pi, sin, cos, tan
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pathlib import Path
import xarray as xr
from .winds import Winds
from .utils import lor_trans_b_iso, lor_trans_ug_iso, beta_from_g

from astropy import constants as const

C_LIGHT = 2.998e10
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

########## some helper function for drawing, whatev ############################ 
####################################################################################
####################################################################################
def plot_with_gradient(fig, ax, xdata, ydata, some_param, colorbar=False, lw=2,
                       ls='-', colorbar_label='grad', minimum=None, maximum=None):
    """
    to draw the plot (xdata, ydata) on the axis ax with color along the curve
    marking some_param. The color changes from blue to red as some_param increases.
    You may provide your own min and max values for some_param:
    minimum and maximum, then the color will be scaled according to them.
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

### --------------------- For shock front shape --------------------------- ###
_here = Path(__file__).parent          # points to pulsar/
_ibs_data_file = _here / "tab_data" / "Shocks4.nc"
ds_sh = xr.load_dataset(_ibs_data_file)

class IBS:
    """
    TODO: Maybe create 2 classes. 1st: IBS_dimentionless, without any winds, just 
    with beta. 2nd: IBS_real, it would take winds and optional time and 
    return rescaled, rotated ibs.
    """
    def __init__(self, beta, s_max, gamma_max=None, s_max_g=4., n=31, one_horn=True,  
                 winds = None, t_to_calculate_beta_eff = None):
        self.beta = beta
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.n = n
        self.one_horn = one_horn
        self.winds = winds
        self.x = None
        self.y = None
        self.theta = None
        self.r = None
        self.s = None
        self.theta1 = None
        self.r1 = None
        self.tangent = None
        self.thetainf = None
        self.x_apex = None
        self.t_forbeta = t_to_calculate_beta_eff

        self.calculate()
    
    
    # @staticmethod    
    # def beta_from_g(g_vel):
    #     if isinstance(g_vel, np.ndarray):
    #         res = np.zeros_like(g_vel)
    #         cond = (g_vel > 1.0) 
    #         res[cond] = np.sqrt((g_vel[cond]-1.0) * (g_vel[cond]+1.0)) / g_vel[cond]
    #     else:
    #         if g_vel > 1.0:
    #             res =  np.sqrt((g_vel-1.0) * (g_vel+1.0)) / g_vel
    #         else:
    #             res = 0.0
    #     return res 
    
    @staticmethod
    def vel_from_g(g_vel):
        return C_LIGHT * beta_from_g(g_vel) 
    
    
    
    def calculate(self):
        if isinstance(self.winds, Winds):
            self.beta = self.winds.beta_eff(self.t_forbeta)
        (xp, yp, tp, rp, sp, t1p, r1p, tanp, theta_inf_,
         r_apex) = IBS.approx_IBS(self, full_output=True)
        self.x = xp
        self.y = yp
        self.theta = tp
        self.r = rp
        self.s = sp
        self.theta1 = t1p
        self.r1 = r1p
        self.tangent = tanp
        self.thetainf = theta_inf_
        self.x_apex = r_apex 
        
        # self.u_g_apex = IBS.u_g_density(self, r_from_s)
        



    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_. Returned is the value for only one (upper) horn of the shock.
        MIND THE DIMENSIONLESS! 
 
        Parameters
        ----------
        s_ : np.ndarray
            The arclength along the upper horn of the IBS to find the value at.
            Dimensionless.

        Returns
        -------
        The desired value of ibs.what in the coordinate s_. 

        """
        try:
            data = getattr(self, what)
        except AttributeError:
            raise ValueError(f"No such attribute '{what}' in IBS.")
        s_to_interp = self.s[self.y >= 0]
        data_to_interp = data[self.y >= 0]
        interpolator = interp1d(s_to_interp, data_to_interp, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return interpolator(s_)
    
    @property
    def int_an(self):
        return self.s_max_g / (self.gamma_max - 1) * (self.gamma_max**2 - 1)**0.5 

    
    def gma(self, s):
        return 1. + (self.gamma_max - 1.) * s / self.s_max_g        
        # return self.gamma_max

        
    @property
    def g(self):
        return IBS.gma(self, s = self.s)
    
    
    @property
    def beta_vel(self):
        return beta_from_g(g_vel = self.g)
    
    def lor_trans_b_iso_on_ibs(self, B_iso):
        return lor_trans_b_iso(B_iso=B_iso, gamma=self.g)
    
    def lor_trans_ug_iso_on_ibs(self, ug_iso):
        return lor_trans_ug_iso(ug_iso=ug_iso, gamma=self.g)
    
    
    
    @property
    def theta_inf(self):
        to_solve1 = lambda tinf: tinf - tan(tinf) - pi / (1. - self.beta)
        th_inf = brentq(to_solve1, pi/2 + 1e-5, pi - 1e-5)
        return th_inf
    
        
    def theta1_CRW(self, theta):
        if theta == 0:
            return 0
        else:
            th1_inf = pi - self.theta_inf 
            to_solve2 = lambda t1: t1 / tan(t1) - 1. - self.beta * (theta / tan(theta) - 1)
            th1 = brentq(to_solve2, 1e-10, th1_inf)
            return th1
        
    def rotate(self, phi):
        """
        Rotates the shock at the angle phi CLOCKWISE 
        
        Parameters
        ----------
        phi : TYPE
            DESCRIPTION.

        Returns
        -------
        rotated_ibs : TYPE
            DESCRIPTION.

        """
        c_, s_ = cos(phi), sin(phi)
        x_rotated_ = c_ * self.x - s_ * self.y
        y_rotated_ = s_ * self.x + c_ * self.y
        rotated_ibs = self.__class__(beta=self.beta, gamma_max=self.gamma_max, s_max=self.s_max,
                               s_max_g = self.s_max_g, n=self.n, one_horn=self.one_horn,
                               winds=self.winds, t_to_calculate_beta_eff=self.t_forbeta)
        rotated_ibs.x = x_rotated_
        rotated_ibs.y = y_rotated_
        rotated_ibs.tangent = self.tangent + phi
        # modif_tangent = self.tangent-phi
        # rotated_ibs = IBS.modified_copy(self, x=x_rotated_, y=y_rotated_, tangent=modif_tangent)
        return rotated_ibs
        
        
        
        
    
    def calculate_ibs_shape_crw(self, full_return = False):
        """
        Calculates the IBS shape in the model of Canto, Raga, Wilkin (1996)
        https://ui.adsabs.harvard.edu/abs/1996ApJ...469..729C/abstract

        Parameters
        ----------
        beta : float
            The winds momenta relation [dimless].
        s_max : float
            The dimentionless arclength of the IBS at which it should be cut.
        n : int
            The number of points on the one horn of IBS.
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
            thetas = np.linspace(1e-3, th_inf-1e-3, self.n)
            theta1s = np.zeros(thetas.size)
            theta1s = np.array([self.theta1_CRW(thetas[i]) for i in range(theta1s.size)])
        if beta_ <= 1e-3:
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
        b : float
            b = | log10 (beta_eff) |. Should be > 0.1.
        n : int
            The number of nods in the grid.
        s_max : float or str
            Describes where the IBS should be cut. If float, then it is treated as
            the dimentionless arclength of the IBS at which it should be cut 
            (should be less than 5.0). If 'bow', then the part of the IBS with
            theta < 90 deg is left. If 'incl', these parts of the shock left for
            which the angle between the radius-vector from the pulsar and the 
            tanential is < 90 + 10 deg
            
        full_output : bool, optional
            Whether to return less or more. The default is False.
    
        Returns
        -------
        tuple
            The shape of the IBS: the tuple of its characteristics. If full_output
            = False, then the tuple is (x, y). If full_output
            = True, then the tuple is (x, y, theta, r, s, theta1, r1, 
            theta_tangent, theta_inf (float), r_in_apex (float)). 
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
        xs_, ys_, ts_, rs_, ss_, t1s_, r1s_, tang = [arr[ok] for arr in (xs_, ys_,
                                            ts_, rs_, ss_, t1s_, r1s_, tang)]
        # now we interpolate the values onto the equally-spaced grid over y with
        # na nods, since this is the best way to interpolate (not over s)
        intx, ints, intth, intr, intth1, intr1, inttan = (interp1d(ys_, xs_),
                interp1d(ys_, ss_),
                interp1d(ys_, ts_), interp1d(ys_, rs_), interp1d(ys_, t1s_), 
                interp1d(ys_, r1s_), interp1d(ys_, tang))    
        yplot = np.linspace(np.min(ys_)*1.001, np.max(ys_)*0.999, int(self.n))
        xp, tp, rp, sp, t1p, r1p, tanp = (intx(yplot), intth(yplot), intr(yplot),
                                          ints(yplot),
                intth1(yplot), intr1(yplot), inttan(yplot))
        yp = yplot
        if self.one_horn:
            if full_output:
                return (xp, yp, tp, rp, sp, t1p, r1p, tanp, intpl.theta_inf.item(),
            intpl.r_apex.item())
            if not full_output:
                return xp, yp
        if not self.one_horn:
            xp = np.concatenate((xp[::-1], xp))
            yp = np.concatenate((-yp[::-1], yp))
            tp = np.concatenate((-tp[::-1], tp))
            rp = np.concatenate((rp[::-1], rp))
            sp = np.concatenate((sp[::-1], sp))
            t1p = np.concatenate((-t1p[::-1], t1p))
            r1p = np.concatenate((r1p[::-1], r1p))
            tanp = np.concatenate((-tanp[::-1], pi+tanp))
            if full_output:
                return (xp, yp, tp, rp, sp, t1p, r1p, tanp, intpl.theta_inf.item(),
            intpl.r_apex.item())
            if not full_output:
                return xp, yp
            
    @staticmethod
    def doppler_factor(g_vel, ang_):
        beta_vels = beta_from_g(g_vel=g_vel)
        return 1 / g_vel / (1 - beta_vels * cos(ang_))
            
    def dopl(self, nu_true, nu_los=None):
        """
        The doppler factor from the bulk motion along the shock.

        Parameters
        ----------
        nu_true : np.ndarray
            The angle between the S-to-periastron direction and the IBS 
            symmetry line. In case the apex lies on the S-P line, 
            this is the true anomaly of the P.
            
        nu_los : np.ndarray
            The angle between the S-to-periastron direction and the projection
            of the line-of-sight onto the pulsar orbit. For PSRB, 2.3 rad.

        Returns
        -------
        np.ndarray of length n of bulk motion doppler-factors.

        """
        
        
        angs = np.zeros((self.s).size)
        
        try:
            if nu_los is None:
                _nu_los = self.winds.orbit.nu_los
            if nu_los is not None: _nu_los = nu_los
        except:
            if nu_los is None: 
                raise ValueError('you should provide nu_los')
            if nu_los is not None: _nu_los = nu_los
                
        
        angs[self.y < 0] = _nu_los - nu_true - (self.tangent)[self.y < 0]
        angs[self.y >= 0] = _nu_los - nu_true + (pi - self.tangent)[self.y >= 0]
        
        return IBS.doppler_factor(g_vel = self.g, ang_ = angs)
    
    def rescale_to_position(self):
        """
        If winds:Winds was provided, rescale the IBS to the real units at
        the time t_to_calculate_beta_eff [s] and rotate it so that its 
        line of symmetry is S-P line. Rescaled are: x, y, r, s, r1, x_apex.
        The tangent is added pi - nu_tr to.
        
        ---
        Returns new rescaled ibs_resc:IBS
        """
        if isinstance(self.winds, Winds):
            _r_sp = self.winds.orbit.r(self.t_forbeta)
            _nu_tr = self.winds.orbit.true_an(self.t_forbeta)
            ibs_resc = IBS.rotate(self, phi=pi + _nu_tr)
            x_sh, y_sh = ibs_resc.x, ibs_resc.y
            x_sh, y_sh = x_sh * _r_sp, y_sh * _r_sp
            x_sh += _r_sp * cos(_nu_tr)
            y_sh += _r_sp * sin(_nu_tr)
            ibs_resc.x = x_sh
            ibs_resc.y = y_sh
            ibs_resc.s = self.s * _r_sp
            ibs_resc.s_max = self.s_max * _r_sp
            ibs_resc.s_max_g = self.s_max_g * _r_sp
            ibs_resc.r = self.r * _r_sp
            ibs_resc.r1 = self.r1 * _r_sp
            ibs_resc.x_apex = self.x_apex * _r_sp
            return ibs_resc

        if not isinstance(self.winds, Winds):
            raise ValueError("You should provide winds:Winds to rescale the IBS to "
                             "the position of the pulsar. Use winds = Winds(...) "
                             "to create the winds object.")
    
    @property
    def real_dopl(self):
        """Only for the rotated ibs!"""
                
        angs = np.zeros((self.s).size)
        _nu_los = self.winds.orbit.nu_los
        
        angs[self.theta > 0] = (self.tangent)[self.theta > 0] - _nu_los
        angs[self.theta <= 0] = -_nu_los  + (self.tangent)[self.theta <= 0] + pi
        
        return IBS.doppler_factor(g_vel = self.g, ang_ = angs)
        # _nu_tr = self.winds.orbit.true_an(self.t_forbeta)
        # return IBS.dopl(self, _nu_tr, nu_los=0)
        
    
    def peek(self, ax=None, show_winds=False,
             ibs_color='k', to_label=True,
             showtime=None):
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))    

        if to_label:
            label = rf'$\beta = {self.beta}$'
        else:
            label = None

        if ibs_color not in ('doppler', ):
            ax.plot(self.x, self.y, color=ibs_color, label = label)
            ax.axvline(x = self.x_apex, color=ibs_color, alpha = 0.3)

        # find out if IBS is rescaled to the position of the pulsar
        rescaled = False
        if np.max(self.s) > 10: rescaled = True

        if ibs_color == 'doppler':

            if rescaled:
                dopls = self.real_dopl
            else:
                raise ValueError("You should rescale the IBS to the position of the pulsar "
                                 "before plotting the doppler factor. Use rescale_to_position() "
                                 "method of IBS class. Note that for this you should " 
                                 "provide winds:Winds to the IBS class.")
            plot_with_gradient(fig=None, ax=ax, xdata=self.x, ydata=self.y,
                               some_param=dopls, colorbar=False, lw=2, ls='-',
                               colorbar_label='Doppler factor')
        if show_winds:
            if not isinstance(self.winds, Winds):
                raise ValueError("You should provide winds:Winds to show the winds.")
            self.winds.peek(ax=ax, showtime=showtime,
                            plot_rs=False)

        if not rescaled:
            ax.scatter(1, 0, color='b') # star is in (1, 0)
            ax.scatter(0, 0, color='r') # pulsar is in (0, 0)
            ax.axhline(y=0, color='k', ls='--', alpha = 0.3)
            ax.axvline(x=0, color='k', ls='--', alpha = 0.3)

        if rescaled:
            puls_vector = self.winds.orbit.vector_sp(self.t_forbeta)
            _xp, _yp = puls_vector[0], puls_vector[1]
            ax.scatter(_xp, _yp, color='b') # pulsaR
            # ...and the star was alreay plotted in the winds.peek()
            ax.plot( [1.5*_xp, 1.5*_yp], [0, 0], color='k', alpha=0.3, ls='--',)

            ##################################################################################
            show_cond  = np.logical_and(self.winds.orbit.ttab > showtime[0], 
                                        self.winds.orbit.ttab < showtime[1])
            orb_x, orb_y = self.winds.orbit.xtab[show_cond], self.winds.orbit.ytab[show_cond]
            x_scale = np.max(np.array([
                np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
                ]))
            y_scale = np.max(np.array([
                    np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
                ]))
            
            ax.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.winds.orbit.r_periastr) )
            ax.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        
            
        ax.legend()
 
            
            
