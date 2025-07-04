import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import brentq

from astropy import constants as const
# from astropy import units as u


from ibsen.get_obs_data import get_parameters
from .utils import rotated_vector, mydot, mycross, n_from_v, absv
from ibsen.orbit import Orbit
import matplotlib.pyplot as plt


G = float(const.G.cgs.value)
C_LIGHT = float(const.c.cgs.value)
SIGMA_BOLTZ = float(const.sigma_sb.cgs.value)


R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

def unpack_star(sys_name=None, Topt=None, Ropt=None, Mopt=None):
    """
    Unpack star parameters with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        Topt, Ropt, Mopt: float or None
            Explicit values that override defaults.

    Returns:
        Tuple of Topt, Ropt, Mopt
    """
    # Step 1: Determine the source of defaults
    if isinstance(sys_name, str):
        known_types = ['psrb', 'rb', 'bw']
        if sys_name not in known_types:
            raise ValueError(f"Unknown orbit type: {sys_name}")
        defaults = get_parameters(sys_name)
    elif isinstance(sys_name, dict):
        defaults = sys_name
    else:
        defaults = {}

    # Step 2: Build final values, giving priority to explicit arguments
    Topt_final = Topt if Topt is not None else defaults.get('Topt')
    Ropt_final = Ropt if Ropt is not None else defaults.get('Ropt')
    Mopt_final = Mopt if Mopt is not None else defaults.get('Mopt')

    result = [Topt_final, Ropt_final, Mopt_final]
    return tuple(result)


class Winds:
    def __init__(self, orbit: Orbit, 
                 sys_name=None, 
                 Ropt=None, Topt=None, Mopt=None, 
                 M_ns = 1.4*M_SOLAR, f_p = 0.1, 
                 alpha=0, incl=30./180.*pi,   
                 f_d=10., np_disk = 3., delta=0.01, 
                 height_exp = 0.5,
                 rad_prof = 'pl', r_trunk = None,
                 
                 ns_field_model = 'linear', ns_field_surf = None, ns_r_scale = None,
                 ns_L_spindown = None, ns_sigma_magn = None,
                 opt_field_model = 'linear', opt_field_surf = None, opt_r_scale = None,
                 ):
        Topt_, Ropt_, Mopt_ = unpack_star(sys_name=sys_name, Topt=Topt, Ropt=Ropt, Mopt=Mopt)
        self.orbit = orbit
        self.Topt = Topt_
        self.Ropt = Ropt_
        self.Mopt = Mopt_
        self.m_ns = M_ns
        self.alpha = alpha
        self.incl = incl
        self.f_d = f_d
        self.f_p = f_p
        
        self.np_disk = np_disk
        self.delta = delta
        self.height_exp = height_exp
        self.rad_prof = rad_prof
        if r_trunk is None:
            self.r_trunk = 5 * Ropt_
        else:
            self.r_trunk = r_trunk
            
        self.ns_field_model = ns_field_model
        self.ns_field_surf = ns_field_surf
        self.ns_r_scale = ns_r_scale
        self.ns_L_spindown = ns_L_spindown
        self.ns_sigma_magn = ns_sigma_magn
        
        
        self.opt_field_model = opt_field_model
        self.opt_field_surf = opt_field_surf
        self.opt_r_scale = opt_r_scale
        
    
    def ns_field(r_to_p, model='linear', B_surf = None, r_scale = None,
                 L_spindown = None, sigma_magn = None):
        
            
        """
        The magnetic field of the NS [G] at the distance r_to_p.
        
        Parameters
        ----------
        r_to_p : TYPE
            DESCRIPTION.
        model : str, optional
            How to calculate the magnetic field from the NS. Options:
            -  'linear': B = B_surf * (r_scale / r_to_p). You should provide 
            B_surf and r_scale.
            - 'dipole': B = B_surf * (r_scale / r_to_p)^3. You should provide 
            B_surf and r_scale.
            - 'from_L_sigma': according to Kennel & Coroniti 1984a,b:
                B = sqrt(L_spindown * sigma_magn / c / r_to_p^2). 
                You should provide L_spindown and sigma_magn.
            
            The default is 'linear'.
            
            
        B_surf : float, optional
            The field at the NS surface [G]. The default is None.
        r_scale : float, optional
            The scale radius for model = 'liear' or model='dipole'.
            The default is None.
        L_spindown : float, optional
            The spin-down luminosity of the NS [erg / s]. The default is None.
        sigma_magn : float, optional
            Magnetization of the piulsar wind. The default is None.

        Returns
        -------
        The magnetic field of the NS [G] at the distance r_to_p.

        """    
        model_opts = ['linear', 'dipole', 'from_L_sigma']

        if model not in (model_opts):
            raise ValueError('the NS field model should be one of:',
                             model_opts)
        if model == 'linear':
            B_puls = B_surf * (r_scale / r_to_p)
        if model == 'dipole':
            B_puls = B_surf * (r_scale / r_to_p)**3
        
        if model == 'from_L_sigma':        
            B_puls = (L_spindown * sigma_magn / C_LIGHT / r_to_p**2)**0.5
        
        return B_puls
    
    def opt_field(r_to_s, model = 'linear', r_scale=None, B_surf=None):
               
            
        """
        The magnetic field of the opt. star [G] at the distance r_to_s.
        
        Parameters
        ----------
        r_to_a : float
            The distance from the star [cm] to the point.
        model : str, optional
            How to calculate the magnetic field from the star. Options:
            -  'linear': B = B_surf * (r_scale / r_to_p). You should provide 
            B_surf and r_scale.
            - 'dipole': B = B_surf * (r_scale / r_to_p)^3. You should provide 
            B_surf and r_scale.

            The default is 'linear'.
            
            
        B_surf : float, optional
            The field at the star surface [G]. The default is None.
        r_scale : float, optional
            The scale radius for model = 'liear' or model='dipole'.
            The default is None.

        Returns
        -------
        The magnetic field of the NS [G] at the distance r_to_p.

        """    
        model_opts = ['linear', 'dipole']

        if model not in (model_opts):
            raise ValueError('the opt star field model should be one of:',
                             model_opts)
        if model == 'linear':
            B_puls = B_surf * (r_scale / r_to_s)
        if model == 'dipole':
            B_puls = B_surf * (r_scale / r_to_s)**3
        
        return B_puls
    
    def u_g_density(r_from_s, r_star, T_star):      
        factor = 2. * (1. - (1. - (r_star / r_from_s)**2)**0.5 )
        u_dens = SIGMA_BOLTZ * T_star**4 / C_LIGHT * factor
        return u_dens
    
    
        
    @property
    def n_disk(self):
        return rotated_vector(alpha = self.alpha, incl = self.incl)


    def Dist_to_disk(self, rvec):
        return mydot(rvec, self.n_disk)
    
    @property
    def times_of_disk_passage(self):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        Dist_to_disk_time = lambda t: mydot(self.orbit.vector_sp(t), self.n_disk)
        t1 = brentq(Dist_to_disk_time, -self.orbit.T/2., 0)
        t2 = brentq(Dist_to_disk_time, 0, self.orbit.T/2.)
        return t1, t2

    @property
    def vectors_of_disk_passage(self):
        t1, t2 = self.times_of_disk_passage
        vec1 = self.orbit.vector_sp(t1)
        vec2 = self.orbit.vector_sp(t2)
        return vec1, vec2


    def vec_r_to_dp(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        radius = self.orbit.vector_sp(t)
        ndisk = self.n_disk
        return mydot(radius, ndisk) * ndisk
    
    def vec_r_in_dp(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        radius = self.orbit.vector_sp(t)
        # ndisk = N_disk(alpha, incl)
        # d_to_disk = mydot(radius, ndisk)
        return radius - Winds.r_to_dp(self, t)
    

    def n_DiskMatter(self, t):
        # Torb_, e_, Mtot_ = unpack_orbit(orb_p, Torb, e, Mtot, to_return='T e M')   
        n_indisk = n_from_v(Winds.vec_r_in_sp(self, t))
        ndisk = self.n_disk
        return mycross(ndisk, n_indisk)
    
    def pulsar_wind_pressure(self, r_from_p):
        return self.f_p * (self.Ropt / r_from_p)**2
    
    def polar_wind_pressure(self, r_from_s):
        return (self.Ropt / r_from_s)**2
    
    def disk_height(self, r):
        return self.delta * r * (r / self.Ropt)**self.height_exp
    
    def decr_disk_pressure(self, vec_r_from_s):
        ndisk = self.n_disk
        r_fromdisk = mydot(vec_r_from_s, ndisk) * ndisk
        r_indisk = vec_r_from_s - r_fromdisk
        r_to_d = absv(r_fromdisk) 
        r_indisk = vec_r_from_s - r_fromdisk
        r_in_d = absv(r_indisk)
        z0 = Winds.disk_height(self, r_in_d)
        vert = np.exp(-r_to_d**2 / 2 / z0**2)
        if self.rad_prof == 'pl':
            rad = (self.Ropt / r_in_d)**self.np_disk
        if self.rad_prof == 'broken_pl':
            if r_in_d < self.r_trunk:
                rad = (self.Ropt / r_in_d)**self.np_disk
            if r_in_d >= self.r_trunk:
                rad = (self.Ropt / self.r_trunk)**self.np_disk * (self.r_trunk / r_in_d)**2
        return self.f_d * rad * vert 
    
    def _dist_se_1d_notvectorized(self, t):   
        r_sp_vec = self.orbit.vector_sp(t)
        nwind = n_from_v(r_sp_vec) # unit vector from S to P
        r_sp = absv(r_sp_vec)
        pres_w = lambda r_se: Winds.polar_wind_pressure(self, r_from_s = r_se)
        pres_d = lambda r_se: Winds.decr_disk_pressure(self, vec_r_from_s = nwind * r_se)
        pres_p = lambda r_se: Winds.pulsar_wind_pressure(self, r_from_p = np.abs(r_sp - r_se))
        to_solve = lambda r_se: pres_d(r_se) + pres_w(r_se) - pres_p(r_se)
        rse = brentq(to_solve, self.Ropt, r_sp*(1-1e-8))
        ### ---------------- test if the solution is good -------------------------
        p_ref = pres_p(rse)
        max_rel_err = np.max(to_solve(rse) / p_ref)
        if max_rel_err > 1e-3:
            raise ValueError(f'Huge relative error of {max_rel_err} in the solution of the '
                          'disk-wind balance equation at {t/DAY}    days. ')
        return rse
    
    def dist_se_1d(self, t):   
        if isinstance(t, np.ndarray):
            res = np.array( [Winds._dist_se_1d_notvectorized(self, t_) for t_ in t] )
        else:
            res = Winds._dist_se_1d_notvectorized(self, t)
        return res
    
    def beta_eff(self, t):
        r_sp = self.orbit.r(t)
        r_se = Winds.dist_se_1d(self, t)
        r_pe = r_sp - r_se
        return (r_pe / r_se)**2
    
    def magn_fields_apex(self, t):
        r_se = Winds.dist_se_1d(self, t)
        r_pe = self.orbit.r(t) - r_se
        B_ns_apex = Winds.ns_field(r_to_p =r_pe, model=self.ns_field_model,
                              B_surf = self.ns_field_surf,
                              r_scale = self.ns_r_scale,
                              L_spindown = self.ns_L_spindown,
                              sigma_magn =self.ns_sigma_magn)
        B_opt_apex = Winds.opt_field(r_to_s = r_se,
                                     model = self.opt_field_model,
                                     r_scale=self.opt_r_scale,
                                     B_surf=self.opt_field_surf)
        
        return B_ns_apex, B_opt_apex
    
    def u_g_density_apex(self, t):  
        r_se = Winds.dist_se_1d(self, t)
        return Winds.u_g_density(r_from_s = r_se,
                                 r_star = self.Ropt,
                                 T_star = self.Topt)
    
    def peek(self, ax=None,
             showtime = None,
             plot_rs = True,):
        if ax is None:
            if plot_rs:
                fig, ax = plt.subplots(nrows=1, ncols=2)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1)

        if plot_rs:  ax0 = ax[0]
        else: ax0 = ax

        if showtime is None:
            showtime = [-self.orbit.T/2, self.orbit.T/2]
            
        _t = np.linspace(showtime[0], showtime[1], 307)
 
            
        show_cond  = np.logical_and(self.orbit.ttab > showtime[0], 
                                    self.orbit.ttab < showtime[1])
        
        ############# ------ disk passage related stuff ------- ###############

        t1, t2 = self.times_of_disk_passage
        # print('disk equator passage times [days]:')
        # print(t1/DAY, t2/DAY)
        vec_disk1, vec_disk2 = self.vectors_of_disk_passage

        
        orb_x, orb_y = self.orbit.xtab[show_cond], self.orbit.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x))
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y))            
                ]))
        
        coord_scale = np.max(np.array([
            np.min(orb_x), np.max(orb_x), np.min(orb_y), np.max(orb_y),
            np.max( (orb_x**2 + orb_y**2)**0.5 )
            ]))
        ################### ------ drawing the orbit again ------- ################
        ax0.plot(orb_x, orb_y)                                                    
        ax0.scatter(0, 0, c='r')                                                  
        ax0.plot([np.min(orb_x),
                    np.max(orb_x)], [0, 0],
                    color='k', ls='--')      
        ax0.plot([0, coord_scale*cos(self.orbit.nu_los)],                            
                [0, coord_scale*sin(self.orbit.nu_los)], color='g', ls='--')        
        xx1, yy1, zz1 = vec_disk1                                                 
        xx2, yy2, zz2 = vec_disk2                                                 
        ax0.plot([xx1, xx2], [yy1, yy2], color='orange', ls='--', lw=2)    
        # if display=='whole':                                                      
        #     ax[0].set_xlim(-self.orbit.a*2, self.orbit.a*2)                                           
        #     ax[0].set_ylim(-self.orbit.b*2, self.orbit.b*2)                                           
                                                                                
        # if display == 'near_per':                                                 

        ############################################################################

        # if display == 'whole':
        #     x_forp = np.linspace(-self.orbit.a*2, self.orbit.a*2, 301)
        #     y_forp = np.linspace(-self.orbit.b*2, self.orbit.b*2, 305)
        # if display == 'near_per':
        #     x_forp = np.linspace(-self.orbit.r_periastr*2, self.orbit.r_periastr*2, 301)
        #     y_forp = np.linspace(-self.orbit.b*1.6, self.orbit.b*1.6, 305)
        x_forp = np.linspace(np.min(orb_x)*3, np.max(orb_x)*4, 301)
        y_forp = np.linspace(np.min(orb_y)*2, np.max(orb_y)*2, 201)
        

        XX, YY = np.meshgrid(x_forp, y_forp, indexing='ij')
        disk_ps = np.zeros((x_forp.size, y_forp.size))
        for ix in range(x_forp.size):
            for iy in range(y_forp.size):
                vec_from_s_ = np.array([x_forp[ix], y_forp[iy], 0])
                r_ = (x_forp[ix]**2 + y_forp[iy]**2)**0.5
                disk_ps[ix, iy] = (Winds.decr_disk_pressure(self, vec_from_s_) 
                                +
                                Winds.polar_wind_pressure(self, r_)
                                )
        disk_ps = np.log10(disk_ps)

        # P_norm = (disk_ps - np.min(disk_ps)) / (np.max(disk_ps) - np.min(disk_ps))

        from matplotlib.colors import ListedColormap, Normalize

        ########### some magic for displaying the winds, never mind ################
        orange_rgba = np.array([1.0, 0.5, 0.0, 1.0]) 
        n_levels = 20
        colors = np.tile(orange_rgba[:3], (n_levels, 1))  
        alphas = np.linspace(0, 1, n_levels)           
        colors = np.column_stack((colors, alphas))     
        custom_cmap = ListedColormap(colors)
        disk_ps[(XX**2 + YY**2)**0.5 < self.Ropt] = np.nan
        disk_ps[disk_ps < np.nanmax(disk_ps)-4.5] = np.nan
        # norm = Normalize(vmin=np.min(disk_ps), vmax=np.max(disk_ps))
        ax0.contourf(XX, YY, disk_ps, levels=n_levels, cmap=custom_cmap)          

        ax0.set_xlim(-1.2*x_scale, 1.2*min(x_scale, self.orbit.r_periastr) )
        ax0.set_ylim(-1.2*y_scale, 1.2*y_scale) 
        #######################################################################    
        
        if plot_rs:
            dists_se = Winds.dist_se_1d(self, _t)
            rs = self.orbit.r(_t)
            ax[1].plot(_t/DAY, dists_se, label='se', ls='--')
            ax[1].plot(_t/DAY, rs, label = 'sp', ls='-')
            ax[1].plot(_t/DAY, rs-dists_se, label='pe', ls=':')
            ax[1].axvline(x=t1/DAY, color='k', alpha=0.3)
            ax[1].axvline(x=t2/DAY, color='k', alpha=0.3)
            ax[1].axvline(x=self.orbit.t_los/DAY, color='g', ls='--', alpha=0.3)
            
            ax[0].set_title('overview')
            ax[1].set_title(r'$r_\mathrm{SP}, r_\mathrm{SE}, r_\mathrm{PE}$')
            ax[1].legend()
        

        