# main.py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from scipy.integrate import trapezoid
from matplotlib.cm import get_cmap
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

gamma = 4.15
s_max = 1.
# s_max = 'bow'

nu_los = 2.3
# nu_los = 0

alpha_deg = -8.
incl_deg = 25.
# nu_true=nu_los
sys_name = 'psrb'
# sys_name = 'circ'

# beta = 0.01

orb = Orbit(sys_name = sys_name, n=1000)
winds = Winds(orbit=orb, sys_name = sys_name, alpha=alpha_deg/180*pi, incl=incl_deg*pi/180,
              f_d=50, f_p=0.01, delta=0.02, np_disk=3, rad_prof='broken_pl',
              r_trunk=None)

# calc the map of the disk log(pressure) for visualising
# rs = np.linspace(0.01, 3, 105)
# phis = np.linspace(0, 2*pi-1e-3, 103)
# rr, phiphi = np.meshgrid(rs, phis, indexing='ij')
# XX = rr * cos(phiphi)
# YY = rr * sin(phiphi)
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

# ibs = IBS(beta=beta, gamma_max=gamma, s_max=s_max, s_max_g=4, n=25, one_horn=False,)
orb_x, orb_y = orb.xtab, orb.ytab

fps = 30
duration = 10  # seconds
num_frames = int(duration * fps)
interval = 1000 / fps  # in milliseconds
fig, ax = plt.subplots(nrows=2)


tanim = np.linspace(-40, 40, num_frames) * DAY
# tanim = np.linspace(-40, 110, num_frames) * DAY
# tanim = np.linspace(-200, 110, num_frames) * DAY
# tanim = np.linspace(-orb.T/2, orb.T/2, num_frames) 



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
    ax[0].plot([xx1, xx2], [yy1, yy2], color='orange', ls=':')
    
    
    
    # ax1.plot(tanim/DAY, f_sim)

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
    nu_tr = orb.true_an(t=t)
    ax[0].scatter(x, y, c='r')
    ax[0].plot([0, 4*x], [0, 4*y], c='r', ls='--')
    # ibs = IBS(beta=betas_eff[i], gamma_max=gamma, s_max=s_max, s_max_g=2., n=250, one_horn=False,
    #            t_to_calculate_beta_eff=t
    #           )
    ibs = IBS(beta=None, winds=winds, gamma_max=gamma, s_max=s_max, s_max_g=2., n=51, one_horn=False,
               t_to_calculate_beta_eff=t
               )
    
    els = ElectronsOnIBS(Bp_apex=1., u_g_apex=1., ibs=ibs, to_cut_theta=False, to_inject_theta='3d')
    es = np.logspace(9, 14, 103)
    S_, E_ = np.meshgrid(ibs.s * orb.r(t), es, indexing='ij')
    inj = els.f_inject(S_, E_)

    ibs_rot = ibs.rotate(phi=pi+nu_tr)
    x_sh, y_sh = ibs_rot.x, ibs_rot.y
    r = orb.r(t=t)
    x_sh, y_sh = x_sh * r, y_sh * r
    x_sh += r * cos(nu_tr)
    y_sh += r * sin(nu_tr)
    dopls = ibs.dopl(nu_true = nu_tr, nu_los=nu_los)
    
    # colored_param = dopls
    colored_param = inj[:, 50] 
    # colored_param = inj
    
    
    plot_with_gradient(fig, ax[0], xdata=x_sh, ydata=y_sh, some_param=colored_param)
    
    Ntot = trapezoid(inj, es, axis=1)
    plot_with_gradient(fig, ax[1], xdata=ibs_rot.s, ydata=Ntot,
                       some_param=np.array([np.argmax(inj[is_, :]) for is_ in range(ibs.s.size) ]))
    
    
    # ax.plot(ibs_rot.theta, dopls)
    
    
    # ibs = IBS(beta=1, gamma_max=gamma, s_max=s_max, s_max_g=4, n=25, one_horn=False,
    #           winds = winds, t_to_calculate_beta_eff=t
    #           )
    # ibs_real = ibs.rescale_to_position()
    # plot_with_gradient(fig, ax, xdata=ibs_real.x, ydata=ibs_real.y, 
    #                    some_param=ibs_real.real_dopl(nu_los))

    # ax.plot(x_sh, y_sh)
    
    # ax.set_xlim(-orb.a*1.2, orb.a/2)
    # ax.set_ylim(-orb.b*1.3, orb.b*1.3)
    
    
    # ax.set_xlim(-orb.r_apoastr*1.2, orb.a*1.2)
    # ax.set_ylim(-orb.b*1.2, orb.b*1.2
    
    # ax.set_xlim(-orb.a*2, orb.a*2)
    # ax.set_ylim(-orb.b*2, orb.b*2)
    
    ax[0].set_xlim(-0.5e14, 0.3e14)
    ax[0].set_ylim(-orb.b*1.2, orb.b*1.2)
    

ani = FuncAnimation(
    fig, update,
    frames=num_frames,
    interval=interval,
)
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







