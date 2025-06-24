# pulsar/spectrum.py
import numpy as np
from numpy import pi, sin, cos, exp

from .utils import lor_trans_e_spec_iso, lor_trans_b_iso, lor_trans_ug_iso, loggrid
from scipy.integrate import trapezoid
import astropy.units as u
from ibsen.get_obs_data import get_parameters
import ibsen.absorbtion.absorbtion as absb
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d, RegularGridInterpolator

# import time
import naima
from naima.models import Synchrotron, InverseCompton

sed_unit = u.erg / u.s / u.cm**2

def pl(x, p, norm):
    return norm * x**(-p)

def unpack_dist(sys_name=None, dist=None):
    """
    Unpack distance with priority to explicit arguments.

    Parameters:
        orb_type: dict, str, or None
            - If None: return the explicitly passed values.
            - If dict: use it as a source of defaults.
            - If str: use get_parameters(orb_type) to get a dict of defaults.
        dist: float or None
            Explicit values that override defaults.

    Returns:
        float dist
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
    dist_final = dist if dist is not None else defaults.get('D')

    return dist_final


class SpectrumIBS:
    def __init__(self, els,
                 delta_power=4, lorentz_boost=True, simple=False,
                 abs_photoel=True, abs_gg=False, nh_tbabs=0.8, syn_only=False,
                 distance = None):
        self.els = els
        self._orb = self.els.ibs.winds.orbit
        self._ibs = self.els.ibs
        
        self.delta_power = delta_power
        self.lorentz_boost = lorentz_boost
        self.simple = simple
        self.abs_photoel = abs_photoel
        self.abs_gg = abs_gg
        self.nh_tbabs = nh_tbabs
        self.syn_only = syn_only
        
        _dist = unpack_dist(self._orb.name, distance)
        self.distance = _dist
                

    def calculate_sed_on_ibs(self, E = np.logspace(2, 14, 1000),
                             to_set_onto_ibs=True, to_return=False,):
        
        # phi is an angle between LoS and shock front X axis
        _b, _u = self.els._b, self.els._u
        

        try:
            dNe_de_IBS, e_vals = self.els.dNe_de_IBS, self.els.e_vals
        except:
            print('no dNe_de_IBS in els, calculating...')
            dNe_de_IBS, e_vals = self.els.calculate(to_return=True)
            
        ug_a = self.els.u_g_apex
        rsp = self.els.r_sp
        _Topt = self._ibs.winds.Topt
            
            
        # -------------------------------------------------------------------------
        # (1) for each segment of IBS, we calulate a spectrum and put it into
        # the 2-dimentional array SED_s_E. The E_ph is always the same and it is 
        # E but extended: min(E) / max(delta) < E_ph < max(E) * max(delta)
        # if simple=True, then the simple apex-spectrum rescaling and integration
        # is performed, without calculating a spec for each s-segment
        # -------------------------------------------------------------------------
        nu_tr = self._orb.true_an(self._ibs.t_forbeta)
        dopls = self._ibs.dopl(nu_true = nu_tr) # for both horns
        
        # d_max =  d_boost(Gamma, 0)
        d_max = np.max(dopls)
        # print('d max = ', d_max)
        # print('start 3')
        Nphot = int(2 * E.size * 
                    np.log10(d_max**2 * 1.21 * np.max(E) / np.min(E)) /
                    np.log10(np.max(E) / np.min(E)) )
        E_ext = np.logspace(np.log10(np.min(E)/d_max/1.1),
                            np.log10(np.max(E)*d_max*1.1), Nphot)
        # this SED_s_E we calculate only for 1 horn
        if self._ibs.one_horn:
            s_1d_dim = self._ibs.s * rsp
        if not self._ibs.one_horn:
            # print(self._ibs.s[self._ibs.n : 2*self._ibs.n-1].size)
            # print(self._ibs.s[: self._ibs.n-1].size)
            # print(self._ibs.s.size)
            s_1d_dim = self._ibs.s[self._ibs.n : 2*self._ibs.n] * rsp
            dNe_de_IBS_1horn = dNe_de_IBS[self._ibs.n : 2*self._ibs.n, :]
            
        _n = 0.5 * (1 - cos( self._ibs.s_interp(s_=s_1d_dim/rsp, what = 'theta') ))
        # print(rsp.shape)
        # print(s_1d_dim.shape)
        # print(dNe_de_IBS.shape)
        
        SED_s_E = np.zeros((s_1d_dim.size, E_ext.size))
        if self.simple:
            # average e-spectrum along the shock. It will be used as the e-spectrum
            # in case s_adv = False
            dNe_de_1d = trapezoid(dNe_de_IBS_1horn, s_1d_dim, axis=0) / np.max(s_1d_dim)
            ok = np.where(dNe_de_1d > 0)
            e_spec_for_naima = naima.models.TableModel(e_vals[ok]*u.eV, (dNe_de_1d[ok])/u.eV )
            
            # calculating a spectrum in the apex once and then insert it in every
            # s-segment
            E_dim = E_ext * u.eV

            Sync = Synchrotron(e_spec_for_naima, B = self.els.B_apex * u.G, nEed=173)
            sed_synchr = Sync.sed(E_dim, distance = self.distance * u.cm)
            SED_sy_apex = sed_synchr / sed_unit
            
            if not self.syn_only:
                seed_ph = ['star', _Topt * u.K, ug_a * u.erg / u.cm**3]
                IC = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=71)
                sed_IC = IC.sed(E_dim, distance = self.distance * u.cm)
                # and putting a total dimentionLESS spec into SED_s_E
                SED_ic_apex = sed_IC / sed_unit
            
        for i_ibs in range(0, s_1d_dim.size):
            # rescaling B and u_g to the point on an IBS. I do it even in case
            # simple = True, so that I can retrieve the values later in addition 
            # to the SED 
            B_here = self.els.B_apex * _b[i_ibs]
            u_g_here = self.els.u_g_apex * _u[i_ibs]
            # Calculating B, u_g, and electron spectrum in the frame comoving
            # along the shock with a bulk Lorentz factor of Gammas[i_ibs]
            if self.lorentz_boost:
                # Gamma_here = Gammas[i_ibs]
                gamma_here = self.els.ibs.gma(s = s_1d_dim[i_ibs]/rsp)
                # B_comov = LorTrans_B_iso(B_here, Gamma_here)
                # u_g_comov = LorTrans_ug_iso(u_g_here, Gamma_here)
                B_comov = lor_trans_b_iso(B_iso=B_here, gamma=gamma_here)
                u_g_comov = lor_trans_ug_iso(ug_iso=u_g_here, gamma=gamma_here)
                if not self.simple:
                    e_vals_comov, dN_de_comov = lor_trans_e_spec_iso(E_lab=e_vals,
                        dN_dE_lab=dNe_de_IBS_1horn[i_ibs, :], gamma=gamma_here)
            else:
                B_comov = B_here
                u_g_comov = u_g_here
                e_vals_comov, dN_de_comov = e_vals, dNe_de_IBS_1horn[i_ibs, :]
            
            if not self.simple:
                # Preparing e_spec so in can be fed to Naima
                if np.max(dN_de_comov) == 0:
                    continue
                ok = np.where(dN_de_comov > 0)
                e_spec_for_naima = naima.models.TableModel(e_vals_comov[ok]*u.eV, (dN_de_comov[ok])/u.eV )
                E_dim = E_ext * u.eV

                # calculating an actual spectrum
                Sync = Synchrotron(e_spec_for_naima, B = B_comov * u.G)
                sed_syncr = Sync.sed(E_dim, distance = self.distance * u.cm)
                sed_tot = sed_syncr 

                if not self.syn_only:
                    seed_ph = ['star', _Topt * u.K, u_g_comov * u.erg / u.cm**3]
                    IC = InverseCompton(e_spec_for_naima, seed_photon_fields = [seed_ph], nEed=101)
                    sed_IC = IC.sed(E_dim, distance = self.distance * u.cm)
                    sed_tot += sed_IC
                # and putting a total dimentionLESS spec into SED_s_E
                SED_s_E[i_ibs, :] = sed_tot / sed_unit
                
            if self.simple:
                # and putting a total dimentionLESS apex spec into SED_s_E
                sy_here = SED_sy_apex * (B_comov / self.els.B_apex)**2 * _n[i_ibs] 
                sed_tot = sy_here
                SED_s_E[i_ibs, :] = sy_here  

                if not self.syn_only:
                    ic_here = SED_ic_apex * (u_g_comov / ug_a) * _n[i_ibs]
                    sed_tot += ic_here
                    
                SED_s_E[i_ibs, :] = sed_tot
                
        # if there are 2 horns in ibs, fill the values on the lower horn with
        # same SEDs as in the upper one        
        # if not self._ibs.one_horn:
        #     SED_s_E_2horns = np.zeros(((self._ibs.s).size, E_dim.size ))
        #     SED_s_E_2horns[:self._ibs.n-1, :] = SED_s_E[::-1, :] # in reverse order from s=smax to ~0
        #     SED_s_E_2horns[self._ibs.n : 2*self._ibs.n-1, :] = SED_s_E
        #     SED_s_E = SED_s_E_2horns
                
        
        # -------------------------------------------------------------------------
        # (4) finally, we integrate over IBS the value
        # delta_doppl^delta_power * SED(E_ph / delta_doppl)
        # -------------------------------------------------------------------------

        # It's maybe not the best idea to use RGI here, seems like it sometimes
        # interpolates too rough. But I haven't figured out a way to use interp1d 
        
        
        
        RGI = RegularGridInterpolator((s_1d_dim, E_ext), SED_s_E, bounds_error=False,
        fill_value=0., method = 'linear') # for 1 horn
        
        # ang_up = pi - phi + th_tan_up # shape (ss.size, )
        # ang_low = pi - phi + th_tan_low # shape (ss.size, )
        
        deltas_up = dopls[self._ibs.n : 2*self._ibs.n]
        deltas_down = dopls[:self._ibs.n]
        
        
        E_new_up = (E[None, :] / deltas_up[:, None])
        E_new_down = (E[None, :] / deltas_down[:, None])
        
        # E_new = (E[None, :] / dopls[:, None])
        
        pts_up = np.column_stack([
            np.repeat(s_1d_dim, E.size),  # shape (M*P_sel,)
            E_new_up.ravel()        # shape (M*P_sel,)
        ])
        pts_down = np.column_stack([
            np.repeat(s_1d_dim, E.size), # shape (M*P_sel,)
            E_new_down.ravel()      # shape (M*P_sel,)
        ])
        
        # pts = np.column_stack([
        #     np.repeat(s_1d_dim, E.size), E_new.ravel()     
        # ])
        
        
        vals_up = RGI(pts_up) # shape (M*P_sel,)
        vals_down = RGI(pts_down)               
        
        # vals = RGI(pts)
        
        I_interp_up = vals_up.reshape(s_1d_dim.size, E.size)    # → (M, P_sel)
        I_interp_down = vals_down.reshape(s_1d_dim.size, E.size)    # → (M, P_sel)
        # I_interp = vals.reshape(self._ibs.s.size, E.size) # (e/delta)

        div = np.max(s_1d_dim)
        sed_s_e_up = I_interp_up * deltas_up[:, None]**self.delta_power
        sed_s_e_down = I_interp_down * deltas_down[:, None]**self.delta_power
        
        sed_e_up = trapezoid(sed_s_e_up, s_1d_dim, axis=0) /  div
        sed_e_down = trapezoid(sed_s_e_down, s_1d_dim, axis=0) /  div
        
        sed_tot = sed_e_up + sed_e_down
        # print(sed_tot.shape)
        # print(sed_s_e_down.shape)
        
        if self.abs_photoel:
            sed_tot = sed_tot * absb.abs_photoel(E=E, Nh = self.nh_tbabs)
        if self.abs_gg:
            if self._orb.name != 'psrb': 
                print('Using gg-abs tabulated for psrb')
            sed_tot = sed_tot * absb.abs_gg_tab(E=E,
                nu_los = self._orb.nu_los, t = self._ibs.t_forbeta, Teff=_Topt)
            
        sed_s_ = np.zeros((2 * s_1d_dim.size, E.size))
        sed_s_[:s_1d_dim.size, :] = sed_s_e_down[::-1, :]
        sed_s_[s_1d_dim.size : 2*s_1d_dim.size, :] = sed_s_e_up

        if to_set_onto_ibs:
            self.sed_s = sed_s_
            self.sed = sed_tot
            self.e_ph = E
            self.sed_spl = interp1d(E, sed_tot)
            # self.sed_logspl = interp1d(np.log10(E), np.log10(sed_tot))
            
        if to_return:    
            return E, sed_tot, sed_s_
    
    
    def flux(self, e1, e2):
        try:
            sed_spl_ = self.sed_spl
        except:
            raise ValueError('The specrum has not been set yet.')
        _E = loggrid(e1, e2, n_dec = 50)
        sed_here = sed_spl_(_E)
        return trapezoid(sed_here / _E, _E)
            
    
    def fluxes(self,  bands):
        fluxes_ = []
        for band in bands:
            e1, e2 = band
            fluxes_.append(SpectrumIBS.flux(self, e1, e2))
        return np.array(fluxes_)
    
    def index(self, e1, e2):
        try:
            sed_spl_ = self.sed_spl
        except:
            raise ValueError('The specrum has not been set yet.')
        _E = loggrid(e1, e2, n_dec = 50)
        sed_here = sed_spl_(_E)
        popt, pcov = curve_fit(f = pl, xdata = _E,
                               ydata = sed_here,
                               p0=(0.5, 
                                   sed_here[0] * _E[0]**0.5
                                   ))
        return popt[0] + 2 
    
    def indexes(self, bands):
        indexes_ = []
        for band in bands:
            e1, e2 = band
            indexes_.append(SpectrumIBS.index(self, e1, e2))
        return np.array(indexes_)
        
            
            
