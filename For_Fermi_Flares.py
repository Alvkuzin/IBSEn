#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:03:21 2025

@author: alvkuzin
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sin, cos, exp
import astropy.units as u
from astropy import constants as const
from scipy.integrate import trapezoid, cumulative_trapezoid, solve_ivp, quad
from scipy.interpolate import interp1d
import time
# from joblib import Parallel, delayed
from ibsen.utils import beta_from_g
from ibsen.ibs import IBS
# import xarray as xr
# from pathlib import Path

# from .orbit import Orbit

def solve_BlandKoenig(s_tilde, a, n, s0_blob, gamma_jet): 
    # gamma_jet should be a func of s (dimless in units of d_orb)
    # a --- without `r_0`, because i multiply `a` by `s0_blob` explicitly
    def rhs(x_til, beta):
        beta_j_here = beta_from_g( gamma_jet( x_til * s0_blob  ) )
        gamma_= 1 / np.sqrt(1. - beta**2)
        parenthesis = (beta_j_here - beta)
        if parenthesis <= 0: anz = 0
        else: anz = parenthesis**1.2
        
        return s0_blob * a * anz * beta_j_here**0.8 / beta / gamma_**1.8 * x_til**(-0.6 * n)

    sol = solve_ivp(fun = rhs, t_span = [1, np.max(s_tilde)*1.1],
                    y0 = (1e-6,), dense_output=True, rtol=1e-4)
    
    beta = sol.sol(s_tilde)[0]
    return beta

def emission_bor_blob(ibs, nu_los, r_sp, a, n, s0_blob):
    s_arr = np.linspace(s0_blob, 4, 1000)
    gamma_jet = lambda _s: ibs.gma(_s)
    beta = solve_BlandKoenig(s_tilde = s_arr / s0_blob, a = a, n=n,
                                     s0_blob=s0_blob, gamma_jet = gamma_jet)
    gamma_blob = (1 - beta**2)**(-0.5)
    spl = interp1d(s_arr, gamma_blob)

    def time_from_s_pos(_s): # in [sec]
        # dumb_array_s = np.linspace(s0_blob, _s, 100000)
        # dumb_array_s = np.logspace(np.log10(s0_blob), np.log10(_s), 10000000)
        # return trapezoid(r_sp / real_velocity(dumb_array_s), dumb_array_s)
        _g_here = lambda _ss: spl(_ss)
        real_velocity = lambda _ss: 3e10 * (beta_from_g(_g_here(_ss)) + 1e-3)
        return quad(func = lambda _ss: r_sp / real_velocity(_ss),
                    a = s0_blob,
                    b = _s, epsrel = 1e-2)[0]
    s_provided = ibs.s
    s_provoded_1horn = s_provided[ibs.theta >= 0]
    tangents_1horn = ibs.tangent[ibs.theta >= 0]
    beta_j_1horn = ibs.beta_vel[ibs.theta >= 0]
    s_for_blob =  s_provoded_1horn[s_provoded_1horn >= s0_blob]
    tangents_for_blob = tangents_1horn[s_provoded_1horn >= s0_blob]
    betas_j_blob = beta_j_1horn[s_provoded_1horn >= s0_blob]
    times_for_blob = np.zeros(s_for_blob.size)
    for i in range(s_for_blob.size):
        times_for_blob[i] = time_from_s_pos(_s = s_for_blob[i])
        
    gamma_blob_on_ibs = spl(s_for_blob)
    beta_blob_ob_ibs = beta_from_g(gamma_blob_on_ibs)
    deltas = ibs.doppler_factor(g_vel = gamma_blob_on_ibs,
                        ang_ = tangents_for_blob - nu_los)
    blob_size = (gamma_blob_on_ibs**(-0.4) * 
                 (betas_j_blob - beta_blob_ob_ibs)**(-0.4) * 
                 (s_for_blob/s0_blob)**(0.2*n) *
                 betas_j_blob**0.4
                 )
    fl = deltas**3.5 # like 3 from doppl. boost and 0.5 from ``synchr``
    fl = fl / blob_size**3 # propto rho for bremstr
    fl = fl * (betas_j_blob - beta_blob_ob_ibs) # propto e_vel_relative for brems
    return times_for_blob, fl 
    
    # def s_at_ibs_t(t_):
        


"""
    Draw several curves gamma_bulb(s) 
"""

# a = 1
# n = 0.5
# s0 = 1.5
# # for a in (1e-2, 1e-1, 1, 1e1, 1e2):
# # for n in (0, 0.3, 0.5, 1, 1.5):  
# for a in (1e-1, 1, 1e1):
#     if a < 1:
#         ls = '--'
#     elif a==1: ls = ':'
#     else:
#         ls = '-'
#     for s0 in (1.1, 1.5, 2, 3):
#         c_ = (3 - s0) / (3 - 1.1)
#         color = [1-c_, 0, c_]
#         s_max_g = 4
#         s_ = np.linspace(s0, s_max_g, 1000)
#         # n = 0.5
#         gamma0 = 3.
#         gamma_jet = lambda _s: 1. + (gamma0 - 1.) * _s / s_max_g
#         beta = solve_BlandKoenig(s_tilde = s_ / s0, a = a, n=n,
#                                  s0_blob=s0, gamma_jet = gamma_jet)
#         gamma = (1 - beta**2)**(-0.5)
#         plt.plot(s_, gamma-1, label = rf"$a = {a}, n = {n}, s_0 = {s0}$",
#                  ls=ls, color=color)
#         plt.axhline(y = gamma0 - 1)
#         plt.legend()
#         plt.ylim(1e-3, gamma0*1.03)
#         plt.xscale('log')
#         # plt.yscale('log')
#         plt.xlabel('s')
#         plt.ylabel(r'$\Gamma_\mathrm{blob}-1$')

start = time.time()

beta_eff = 0.01
s_max = 1
gamma0 = 3
s_max_g = 4
nu_los = 120 * pi / 180
a = 1e1
n = 2
s0 = 0.1
# for a in (1e-1, 1, 1e1, 1e2, 1e3, 1e4):
# for n in (0, 0.5, 1, 2):
# for s0 in (0.1, 0.5, 1, 1.5, 3):
# for beta_eff in (1e-3, 1e-2, 1e-1):
for nu_los_deg in (0, 60, 90, 120, 150, 180):
    nu_los = nu_los_deg / 180 * pi
    ibs = IBS(beta=beta_eff, n=201, s_max = s_max, one_horn=False,
              s_max_g = s_max_g, gamma_max=gamma0)
    
    
    # plt.scatter(ibs.x, ibs.y)
    # plt.plot([0, 3 * cos(nu_los)], [0, 3 * sin(nu_los)], c='g', ls='--')
    
    t, f = emission_bor_blob(ibs = ibs, nu_los = nu_los,
                             r_sp = 2 * 1.5e13, a=a, n=n, s0_blob = s0)
    
    plt.plot(t/60, f)
    plt.xlabel('t, min')
    # plt.yscale('log')
print(time.time() - start)