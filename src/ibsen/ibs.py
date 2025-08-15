# pulsar/ibs.py
import numpy as np
from numpy import pi, sin, cos, tan
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pathlib import Path
import xarray as xr
from ibsen.winds import Winds
from ibsen.ibs_norm import IBS_norm
from ibsen.utils import beta_from_g, absv, \
 vector_angle, rotate_z, rotate_z_xy, n_from_v, plot_with_gradient
 
from astropy import constants as const

C_LIGHT = 2.998e10
DAY = 86400
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)

class IBS: #!!!
    def __init__(self, s_max, gamma_max=None, s_max_g=4., n=31, 
                 winds = None, t_to_calculate_beta_eff = None):
        self.gamma_max = gamma_max
        self.s_max = s_max
        self.s_max_g = s_max_g
        self.n = n
        self.winds = winds
        self.t_forbeta = t_to_calculate_beta_eff
        
        self.calculate()
        
    def calculate(self):
        """General calculation of everything"""
        self.calculate_normalized_ibs()
        self.rescale_to_position()
    
    def calculate_normalized_ibs(self):
        """calculate the effective beta from Winds at time t_forbeta and 
            initialize the normalized IBS with this beta.
        """
        self.beta = self.winds.beta_eff(self.t_forbeta)
        self.r_sp = self.winds.orbit.r(self.t_forbeta)
        unit_los_ = np.array([cos(self.winds.orbit.nu_los),
                             sin(self.winds.orbit.nu_los),
                             0])
        _nu_tr = self.winds.orbit.true_an(self.t_forbeta)

        self.ibs_n = IBS_norm(beta=self.beta, s_max=self.s_max,
                        gamma_max=self.gamma_max, s_max_g=self.s_max_g, n=self.n,
                    unit_los=unit_los_).rotate(pi + _nu_tr)
        
    def rescale_to_position(self):
        """
        Rescale the IBS to the real units at
        the time t_to_calculate_beta_eff [s] and rotate it so that its 
        line of sy_mmetry is S-P line. Rescaled are: x, y, r, s, r1, x_apex.
        The tangent is added pi + nu_tr to (at the stage of 'IBS.rotate').
        To the x and y - coordinates of the IBS there is also added the
        vector of the real s-p distance at the moment (in cm).
        Yes, read this terrible English sentence which I translated from 
        Russian in my head. Suffer.
        ---
        Returns new rescaled ibs_resc:IBS
        """
        _r_sp = self.winds.orbit.r(self.t_forbeta)
        _nu_tr = self.winds.orbit.true_an(self.t_forbeta)
        x_sh, y_sh = self.ibs_n.x, self.ibs_n.y
        x_sh, y_sh = x_sh * _r_sp, y_sh * _r_sp
        x_sh += _r_sp * cos(_nu_tr)
        y_sh += _r_sp * sin(_nu_tr)
        self.x = x_sh
        self.y = y_sh
        self.s, self.s_max, self.s_max_g, self.r, self.r1, self.x_apex, self.ds_dtheta = [
            _stuff * _r_sp for _stuff in (self.ibs_n.s, self.ibs_n.s_max, 
                            self.ibs_n.s_max_g, self.ibs_n.r, self.ibs_n.r1, 
                            self.ibs_n.x_apex, self.ibs_n.ds_dtheta)]
        
        self.s_m, self.s_p, self.s_mid, self.ds, self.x_m, self.x_p, self.x_mid, self.dx, \
        self.y_m, self.y_p, self.y_mid, self.dy = [_stuff * _r_sp for _stuff in (
            self.ibs_n.s_m, self.ibs_n.s_p, self.ibs_n.s_mid, self.ibs_n.ds, 
            self.ibs_n.x_m, self.ibs_n.x_p, self.ibs_n.x_mid, self.ibs_n.dx,
            self.ibs_n.y_m, self.ibs_n.y_p, self.ibs_n.y_mid, self.ibs_n.dy)]          
    
    def s_interp(self, s_, what):
        """
        Returns the interpolated value of 'what' (x, y, ...) at the coordinate 
        s_ [cm].
 
        Parameters
        ----------
        s_ : np.ndarray
            The arclength along the upper horn of the IBS to find the value at.
            [cm].

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
        return 1. + (self.gamma_max - 1.) * np.abs(s) / self.s_max_g        
        # return self.gamma_max

        
    @property
    def g(self):
        return IBS.gma(self, s = self.s)
        
    def peek(self, fig=None, ax=None, show_winds=False,
             ibs_color='k', to_label=True,
             showtime=None):
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))    

        if to_label:
            label = rf'$\beta = {self.beta}$'
        else:
            label = None



        if ibs_color in ( 'doppler', 'scattering', 'scattering_comoving'):

            # print(2)
            if ibs_color == 'doppler':
                color_param = self.dopl
            elif ibs_color == 'scattering':
                # print(3)
                color_param = self.scattering_angle
            elif ibs_color == 'scattering_comoving':
                color_param = self.scattering_angle_comoving
            else:
                raise ValueError(f"Unknown ibs_color: {ibs_color}. "
                                 "Use 'doppler', 'scattering' or 'scattering_comoving'.")

            plot_with_gradient(fig=fig, ax=ax, xdata=self.x, ydata=self.y,
                            some_param=color_param, colorbar=True, lw=2, ls='-',
                            colorbar_label=ibs_color)

        else:
            try:
                ax.plot(self.x, self.y, color=ibs_color, label = label)            
            except:
                raise ValueError('Probably wrong color keyword')

        if show_winds:
            if not isinstance(self.winds, Winds):
                raise ValueError("You should provide winds:Winds to show the winds.")
            self.winds.peek(ax=ax, showtime=showtime,
                            plot_rs=False)

        # if rescaled:
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

    # def __getattr__(self, name):
    #     # called if attribute not found on IBS; forward to normalized object
    #     return getattr(self.ibs_n, name)
    
    def __getattr__(self, name):
        ibs_n_ = self.__dict__.get("ibs_n", None)
        if ibs_n_ is None:
            raise AttributeError(name)
        return getattr(ibs_n_, name)
 
            
if __name__ == "__main__":
    from ibsen.orbit import Orbit
    import matplotlib.pyplot as plt   
    # example of how to use the IBS class
    orbit_ = Orbit(period=100*DAY, e=0, tot_mass=30*2e33, nu_los=np.pi*0.71577)
    winds_ = Winds(orbit=orbit_, sys_name='psrb', f_d=0)
    
    fig, ax = plt.subplots(2, 1)
    import time
    start = time.time()
    ibs = IBS(gamma_max=5, s_max=2, s_max_g=2,
              t_to_calculate_beta_eff=25*DAY, winds=winds_, n=21) 
        # print(ibs.theta_inf/np.pi)
    ibs.peek(show_winds=True, to_label=False, showtime=(-60*DAY, 60*DAY),
             ibs_color='scattering', ax=ax[0], fig=fig)
    # # # ax[1].plot(ibs1.s, ibs1.y, label='y(s)')

    # ax[1].scatter(ibs.s, ibs.scattering_angle_comoving/pi, label='scattering / pi', c='k', s=4)
    # ax[1].scatter(ibs.s_mid, ibs.scattering_angle_comoving_mid/pi, label='scattering / pi', c='r', s=4)
    
    # ax[1].scatter(ibs.s, ibs.dopl, label='dopl', c='k', s=4)
    # ax[1].scatter(ibs.ibs_n.s*ibs.r_sp, ibs.ibs_n.dopl, label='dopl', c='b', s=4)
    # ax[1].scatter(ibs.s_mid, ibs.dopl_mid, label='dopl', c='r', s=4)
    ax[1].scatter(ibs.s, ibs.ds_dtheta, label='dopl', c='k', s=4)
    ax[1].scatter(ibs.s, ibs.ibs_n.ds_dtheta*ibs.r_sp, label='dopl', c='r', s=4)
    
    s = np.linspace(-2, 3, 51) * 1e13
    ax[1].scatter(s, ibs.s_interp(s, 'ds_dtheta'), label='dopl', c='m', s=4)
    
    