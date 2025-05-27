import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.optimize import brentq
from pathlib import Path
from numpy import pi, sin, cos, tan
from scipy.interpolate import interp1d
import Orbit as Orb
from ShapeIBS import approx_IBS
from joblib import Parallel, delayed

DAY = 86400.



from SpecIBS import dummy_LC
import time
start = time.time()

pe = 1.7
Topt = 3.3e4
Bx = 5e11
Bopt = 0
Gamma = 2.5
nu_los = 2.3


# orb_p = 'circ'
orb_p = 'psrb'

beta_ef = 0.01
# s_max = 'bow'
s_max = 3
orb_p_psrb = Orb.Get_PSRB_params(orb_p)
P, exc = [orb_p_psrb[key] for key in ('P', 'e')]

### ----------------------------------------------------------------- #####
###  ---------- Draw a realistic IBS with the orbital shape ------------ ##
### ----------------------------------------------------------------- #####   
tpl = np.linspace(-P/2, P/2, 1000) * DAY
r_per = Orb.r_peri(orb_p)


fps = 30
duration = 4  # seconds
num_frames = int(duration * fps)
interval = 1000 / fps  # in milliseconds

# === Create the figure ===
fig, ax = plt.subplots(nrows=2)
ax0, ax1 = ax
# ax.grid(True)
xs, ys, zs = Orb.Vector_S_P(tpl, orb_p) / r_per


x_sh, y_sh, th_sh, s_sh, r_sh = approx_IBS(b = np.abs(np.log10(beta_ef)),
                            Na = 100, s_max = s_max, full_output=False)
x_sh = np.concatenate(( x_sh[::-1], x_sh))    
y_sh = np.concatenate((-y_sh[::-1], y_sh))

tanim = np.linspace(-40, 110, num_frames) * DAY
# tanim = np.linspace(-200, 110, num_frames) * DAY
# tanim = np.linspace(-P/2, P/2, num_frames) * DAY


def func_simple(i):
    f_sim_ = dummy_LC(tanim[i], Bx, Bopt, Topt, E0_e=1, Ecut_e=1, Ampl_e=1, p_e=pe,
                    beta_IBS=beta_ef, Gamma=Gamma, s_max_em=s_max,
                    s_max_g=4., simple=True, orb=orb_p,
                    eta_a = 1, lorentz_boost=True, cooling='no')   
    return f_sim_

f_sim = Parallel(n_jobs=10)(delayed(func_simple)(i) for i in range(0, len(tanim)))
f_sim=np.array(f_sim)
# label = 'simple calculation'
# label = f'no adv, eta = {eta}'
# plt.plot(tplot/DAY, f_sim/(f_sim[np.argmin(np.abs(tplot))]), label = label, ls='--')

def init():
    ax0.plot(xs, ys)
    ax0.scatter(0, 0, c='k')
    ax0.plot([np.min(xs), np.max(xs)], [0, 0], color='k', ls='--')
    ax0.plot([0, 10*cos(nu_los)], [0, 10*sin(nu_los)], color='g', ls='--')
    ax1.plot(tanim/DAY, f_sim)

def update(i):
    """Draw the i-th frame."""

    ax0.clear()
    ax1.clear()
    
    init()

    t = tanim[i]
    x, y, z = Orb.Vector_S_P(t, orb_p) / r_per
    nu_tr = Orb.True_an(t, orb_p)
    ax0.scatter(x, y, c='r')
    ax0.plot([0, 4*x], [0, 4*y], c='r', ls='--')
    r = Orb.Radius(t, orb_p) / r_per
    
    # r_sh = np.column_stack((x_sh, y_sh))
    # first, rescale the shock
    x_sh_i, y_sh_i = [ar * r for ar in (x_sh, y_sh)]
    # then, transport it at the distance r_SP(t)
    x_sh_i += -r 
    # finally, rotate each vector r_sh at the angle 180-nu_tr(t) clockwise
    th = pi - nu_tr
    rotation_matrix = np.array([
    [np.cos(th), np.sin(th)],
    [-np.sin(th), np.cos(th)]
        ])
    points = np.vstack((x_sh_i, y_sh_i))
    
    # Apply rotation
    rotated_points = rotation_matrix @ points
    x_rotated, y_rotated = rotated_points
    ax0.plot(x_rotated, y_rotated)
    ax0.set_aspect('equal')
    lim = 1.3
    ax0.set_xlim(np.min(xs)*lim, np.max(xs)*lim*(1+exc)/(1-exc))
    ax0.set_ylim(np.min(ys)*lim, np.max(ys)*lim)
    # ax.set_xlim(-23.5, 5.5)
    # ax.set_ylim(-9.5, 9.5)
    ax1.scatter(tanim[i]/DAY, f_sim[i], color='r')
    ax1.set_xlim(np.min(tanim)/DAY-3, np.max(tanim)/DAY+3)
    ax1.set_ylim(0, 1.2*np.max(f_sim))
    


# === Create animation ===
ani = FuncAnimation(
    fig, update,
    frames=num_frames,
    interval=interval,
)

plt.show()

