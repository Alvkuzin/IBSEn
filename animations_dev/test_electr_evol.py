# main.py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from scipy.integrate import trapezoid
# from matplotlib.cm import get_cmap
import importlib
import ibsen
importlib.reload(ibsen)
from ibsen.orbit import Orbit
from ibsen.ibs import IBS
# from pulsar.spec import Spectrum
# from pulsar.lc import LightCurve
import ibsen.absorbtion.absorbtion as Absb
from ibsen.el_ev import ElectronsOnIBS
from ibsen.winds import Winds

import numpy as np
from numpy import sin, cos, pi

"""
This is a little script that animates the movement of the pulsar around the 
star with the intrabinary shock, which depends on the winds.
"""


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

DAY = 86400.

# nu_los = 2.3
# nu_los = 0


# nu_true=nu_los

### ----------------------------------------------------------------------- ###
### ----------------------------------------------------------------------- ###
gamma = 2.15 # max bulk motion gamma at ibs
s_max = 1. # where to cut the ibs
# s_max = 'bow'

alpha_deg = -8.
incl_deg = 25.

to_show = 'near_per'   ### for now, no other options
# to_show = 'whole'

sys_name = 'psrb'
# sys_name = 'rb'

f_d = 50 # disk
# f_d = 0 # no disk
# f_d = 1e3 # super strong disk

f_p = 0.01

delta = 0.02 # relative half-thickness of the disk (at r=Ropt)


### ----------------------------------------------------------------------- ###
### ----------------------------------------------------------------------- ###

# beta = 0.01

orb = Orbit(sys_name = sys_name, n=1000)

nu_los = orb.nu_los

winds = Winds(orbit=orb, sys_name = sys_name, alpha=alpha_deg/180*pi, incl=incl_deg*pi/180,
              f_d=f_d, f_p=f_p, delta=delta, np_disk=3, rad_prof='broken_pl',
              r_trunk=None)

### ----------------------------------------------------------------------- ###
####  ---- calc the map of the disk log(pressure) for visualising -------- ####
### ----------------------------------------------------------------------- ###

if to_show=='whole':
    x_forp = np.linspace(-orb.a*2, orb.a*2, 301)
    y_forp = np.linspace(-orb.b*2, orb.b*2, 305)
if to_show=='near_per':
    x_forp = np.linspace(-5.13e13, 6.1e13, 301)
    y_forp = np.linspace(-8.5e13, 9.4e13, 305)
    
XX, YY = np.meshgrid(x_forp, y_forp, indexing='ij')
disk_ps = np.zeros((x_forp.size, y_forp.size))
for ix in range(x_forp.size):
    for iy in range(y_forp.size):
        vec_from_s_ = np.array([x_forp[ix], y_forp[iy], 0])
        r_ = (x_forp[ix]**2 + y_forp[iy]**2)**0.5
        disk_ps[ix, iy] = (winds.decr_disk_pressure(vec_from_s_) 
                           +
                           winds.polar_wind_pressure(r_)
                           )
disk_ps = np.log10(disk_ps)

P_norm = (disk_ps - np.min(disk_ps)) / (np.max(disk_ps) - np.min(disk_ps))

# Create a custom colormap with fixed orange and varying alpha
orange_rgba = np.array([1.0, 0.5, 0.0, 1.0])  # RGB for orange + dummy alpha
n_levels = 20
colors = np.tile(orange_rgba[:3], (n_levels, 1))  # repeat RGB
alphas = np.linspace(0, 1, n_levels)              # vary alpha
colors = np.column_stack((colors, alphas))       # RGBA
custom_cmap = ListedColormap(colors)
disk_ps[disk_ps < np.max(disk_ps)-4.5] = np.nan
norm = Normalize(vmin=np.min(disk_ps), vmax=np.max(disk_ps))


t1, t2 = winds.times_of_disk_passage
print(t1/DAY, t2/DAY)
vec_disk1, vec_disk2 = winds.vectors_of_disk_passage

orb_x, orb_y = orb.xtab, orb.ytab

fps = 30
duration = 10  # seconds
num_frames = int(duration * fps)
interval = 1000 / fps  # in milliseconds
fig, ax = plt.subplots(nrows=2)

if to_show=='whole':
    tanim = np.linspace(-orb.T/2, orb.T/2, num_frames) 
if to_show=='near_per':
    tanim = np.linspace(-40, 40, num_frames) * DAY    

# tanim = np.linspace(-40, 110, num_frames) * DAY
# tanim = np.linspace(-200, 110, num_frames) * DAY



betas_eff = winds.beta_eff(tanim)
# all_dopls = np.zeros(())

def init():
    ax[0].plot(orb_x, orb_y)
    # ax.imshow(disk_ps, extent=[x_forp.min(), x_forp.max(), y_forp.min(), y_forp.max()],
    #        origin='lower', cmap='plasma', alpha=alpha)
    ax[0].contourf(XX, YY, disk_ps, levels=n_levels, cmap=custom_cmap)
    ax[0].scatter(0, 0, c='k')
    ax[0].plot([np.min(orb_x), np.max(orb_x)], [0, 0], color='k', ls='--')
    ax[0].plot([0, 3*orb.r_apoastr*cos(nu_los)], [0, 3*orb.r_apoastr*sin(nu_los)], color='g', ls='--')
    xx1, yy1, zz1 = vec_disk1
    xx2, yy2, zz2 = vec_disk2
    ax[0].plot([xx1, xx2], [yy1, yy2], color='orange', ls='--')
    
    
    
    # ax1.plot(tanim/DAY, f_sim)
    
    
### ---------------------- precalculate ibs and e ------------------------- ###

Nibs = 47
x_sh2d, y_sh2d, dopl2d, ntotinj2d, ntot2d, maxsed2d = [ np.zeros((tanim.size, 2*Nibs))*i for i in range(6) ]
evals2d = []
dNe_de_IBS_avg = []

for i in range(tanim.size):
    t = tanim[i]
    ibs = IBS(beta=None, winds=winds, gamma_max=gamma, s_max=s_max, s_max_g=2., n=Nibs, one_horn=False,
               t_to_calculate_beta_eff=t
               )
    r = orb.r(t=t)
    rsp = r - winds.dist_se_1d(t)
    els = ElectronsOnIBS(Bp_apex=3. * 1e12/rsp, ibs=ibs,
                         cooling='stat_mimic',
                         # cooling='adv',
                         
                         p_e=1.7) # toy values for B and u_g, scaled for r
    
    es = np.logspace(9, 14, 103)
    S_, E_ = np.meshgrid(ibs.s * orb.r(t), es, indexing='ij')
    inj = els.f_inject(S_, E_)
    nu_tr = orb.true_an(t=t)
    ibs_rot = ibs.rotate(phi=pi+nu_tr)
    x_sh, y_sh = ibs_rot.x, ibs_rot.y
    x_sh, y_sh = x_sh * r, y_sh * r
    x_sh += r * cos(nu_tr)
    y_sh += r * sin(nu_tr)
    dopls = ibs.dopl(nu_true = nu_tr)
    x_sh2d[i, :] = x_sh
    y_sh2d[i, :] = y_sh
    dopl2d[i, :] = dopls
    ntotinj2d[i, :] = trapezoid(inj, es, axis=1)
    
    dNe_de_IBS, e_vals = els.calculate(to_return=True)
    ntot2d[i, :] = trapezoid(dNe_de_IBS, e_vals, axis=1)
    # evals2d[i, :] = e_vals
    evals2d.append(e_vals)
    # dNe_de_IBS_3d.append(dNe_de_IBS)
    dNe_de_IBS_avg.append( trapezoid(dNe_de_IBS[Nibs:2*Nibs-1, :], ibs.s[Nibs:2*Nibs-1], axis=0) / np.max(ibs.s) )
    # dNe_de_IBS_avg.append( -trapezoid(dNe_de_IBS[:Nibs-1, :], ibs.s[:Nibs-1], axis=0) / np.max(ibs.s) )
    
    # dNe_de_IBS_avg.append( dNe_de_IBS[Nibs-1, :])
    
    maxsed2d[i, :] = np.array([
        e_vals[np.argmax(dNe_de_IBS[i_s, :] *e_vals**2)]
        for i_s in range(x_sh.size)
        ])
    
    
    
    
evals2d = np.array(evals2d) 
dNe_de_IBS_avg = np.array(dNe_de_IBS_avg)

def update(i):
    """Draw the i-th frame."""

    ax[0].clear()
    ax[1].clear()
    # ax.clear()
    # fig.clear()
    init()

    t = tanim[i]
    ax[0].set_title(t/orb.T)
    x, y, z = orb.vector_sp(t=t)
    # nu_tr = orb.true_an(t=t)
    ax[0].scatter(x, y, c='r')
    ax[0].plot([0, 4*x], [0, 4*y], c='r', ls='--')

    x_sh, y_sh, dopls, ntot, maxsed = x_sh2d[i], y_sh2d[i], dopl2d[i], ntot2d[i], maxsed2d[i]
    
    colored_param = np.log10(dopls)

    plot_with_gradient(fig, ax[0], xdata=x_sh, ydata=y_sh, 
                       some_param=colored_param,
                       minimum=np.min(np.log10(dopl2d)),
                       maximum=np.max(np.log10(dopl2d)))
    
    ### for showing the number of particles VS s with color = e_max(sed)
    # plot_with_gradient(fig, ax[1], xdata=ibs_rot.s, ydata=ntot, some_param=np.log10(maxsed))
    # ax[1].set_ylim(0, np.max(ntot2d))
    
    # evals, dNe_de_avg = evals2d[i], 
    
    sed_e_here = evals2d[i]**2 * dNe_de_IBS_avg[i]
    ax[1].plot(evals2d[i], sed_e_here)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(np.max(dNe_de_IBS_avg*evals2d[i]**2)/1e3, 
                   np.max(dNe_de_IBS_avg*evals2d[i]**2))
    ax[1].set_xlim(1e9, 5.1e14)
    
    
    

    
    # ax.set_xlim(-orb.a*1.2, orb.a/2)
    # ax.set_ylim(-orb.b*1.3, orb.b*1.3)
    
    
    # ax.set_xlim(-orb.r_apoastr*1.2, orb.a*1.2)
    # ax.set_ylim(-orb.b*1.2, orb.b*1.2
    
    if to_show=='whole':
        ax[0].set_xlim(-orb.a*2, orb.a*2)
        ax[0].set_ylim(-orb.b*2, orb.b*2)
    
    if to_show == 'near_per':
        ax[0].set_xlim(-0.5e14, 0.3e14)
        ax[0].set_ylim(-orb.b*1.2, orb.b*1.2)
    

# ani = FuncAnimation(
#     fig, update,
#     frames=num_frames,
#     interval=interval,
# )

### To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=30,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('anim_psrb.gif', writer=writer)

# 

#### ----------------------- draw one frame ----------------------------- #####
# i = int(num_frames/2)
# init()
# t = tanim[i]
# x, y, z = orb.vector_sp(t=t)
# nu_tr = orb.true_an(t=t)
# fig, ax = plt.subplots(nrows=2)
# ax[0].scatter(x, y, c='r')
# ax[0].plot([0, 4*x], [0, 4*y], c='r', ls='--')
# ibs = IBS(beta=betas_eff[i], gamma_max=gamma, s_max=s_max, s_max_g=4, n=25, one_horn=False,
#            t_to_calculate_beta_eff=t
#           )
# ibs_rot = ibs.rotate(phi=pi+nu_tr)
# x_sh, y_sh = ibs_rot.x, ibs_rot.y
# r = orb.r(t=t)
# x_sh, y_sh = x_sh * r, y_sh * r
# x_sh += r * cos(nu_tr)
# y_sh += r * sin(nu_tr)
# dopls = ibs_rot.dopl(nu_true = nu_tr, nu_los=nu_los)
# plot_with_gradient(fig, ax[0], xdata=x_sh, ydata=y_sh, some_param=dopls, colorbar=True)
# ax[0].set_xlim(-orb.a*2, orb.a*2)
# ax[0].set_ylim(-orb.b*2, orb.b*2)
# ax[1].plot()
# plt.show()







