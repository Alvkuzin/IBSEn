"""
It's written mainly by ChatGPT. Sorry I'm not a coder jeez!!!!!!!!!! 
"""
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QSlider,
    QComboBox, QPushButton, QCheckBox,
)

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from ibsen import Orbit, Winds, IBS
from ibsen.gui.base import  ToolWindowBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ibsen.get_obs_data import known_names

def _ibs_color_param(ibs, ibs_color: str) -> np.ndarray:
    key = ibs_color.strip().lower()

    if key in ("doppler", "dopl", "doppl", "dopler", "doppler factor"):
        return np.asarray(ibs.dopl)
    if key in ("scattering", "scattering angle"):
        return np.asarray(ibs.scattering_angle)
    if key in (
        "scattering_comov", "scattering_angle_comov",
        "scattering comov", "scattering angle comov",
        "scattering comoving", "scattering angle comoving",
    ):
        if hasattr(ibs, "scattering_angle_comov"):
            return np.asarray(getattr(ibs, "scattering_angle_comov"))
        if hasattr(ibs, "scattering_angle_comoving"):
            return np.asarray(getattr(ibs, "scattering_angle_comoving"))
        raise AttributeError("IBS has no scattering_angle_comov/scattering_angle_comoving attribute")
        
    if key in ("gamma-gamma (1 tev)"):
        return np.asarray(ibs.gg_abs(e_phot=1e12, what_return="tau"))
    if key in ("gamma-gamma (100 gev)"):
        return np.asarray(ibs.gg_abs(e_phot=1e11, what_return="tau"))
    if key in ("gamma-gamma (10 tev)"):
        return np.asarray(ibs.gg_abs(e_phot=1e13, what_return="tau"))
    
    if not hasattr(ibs, key):
        raise AttributeError(f"No attribute '{ibs_color}' in {type(ibs).__name__}")
    return np.asarray(getattr(ibs, key))


def winds_and_ibs(
    t: float,
    sys_name: str,
    f_d: float,
    f_p: float,
    delta: float,
    alpha_deg: float,
    incl_deg: float,
    b_ns_13: float,
    b_opt_13: float,
    nu_los_deg: float,
    s_max: float,
    gamma_max: float,
    ibs_color: str,
):
    orbit = Orbit(sys_name, nu_los=np.deg2rad(nu_los_deg), n=1002)

    winds = Winds(
        orbit=orbit,
        sys_name=sys_name,
        allow_missing=False,
        M_ns=1.4 * 2e33,
        f_p=f_p,
        alpha=np.deg2rad(alpha_deg),
        incl=np.deg2rad(incl_deg),
        f_d=f_d,
        np_disk=3.0,
        delta=delta,
        height_exp=0.5,
        rad_prof="pl",
        r_trunk=None,
        t_forwinds=t,
        ns_b_model="linear",
        ns_b_ref=b_ns_13,
        ns_r_ref=1e13,
        opt_b_model="linear",
        opt_b_ref=b_opt_13,
        opt_r_ref=1e13,
    )

    ibs = IBS(
        t_to_calculate_beta_eff=t,
        winds=winds,
        s_max=s_max,
        gamma_max=gamma_max,
        s_max_g=4.0,
        n=31,
    )

    x = np.asarray(ibs.x)
    y = np.asarray(ibs.y)
    c = _ibs_color_param(ibs, ibs_color)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]

    return orbit, winds, (x, y), c


def add_gradient_line(ax, x, y, c, *, cmap="coolwarm", lw=2.5, alpha=1.0, vmin=None, vmax=None):
    """
    Add a LineCollection gradient line to ax. Returns the LineCollection (mappable).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    c = np.asarray(c).ravel()

    if x.size < 2:
        return None

    if vmin is None:
        vmin = float(np.nanmin(c))
    if vmax is None:
        vmax = float(np.nanmax(c))
    norm = Normalize(vmin=vmin, vmax=vmax)

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    ax.add_collection(lc)
    return lc


class IBSWindow(ToolWindowBase): #!!!
    TOOL_NAME = "Winds + IBS in real scale."
    IS_HEAVY = True  # Winds.peek() is expensive

    def __init__(self):
        super().__init__()

        # Configure the single axis created by ToolWindowBase
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        self._divider = make_axes_locatable(self.ax)
        self._cax = self._divider.append_axes("right", size="4%", pad=0.05)  # fixed space for cbar
        self._cbar = None
        self._ibs_lc = None
        # self._cbar = None

        # ---- Controls (right panel)
        # Insert after the title label at index 1
        self.controls_layout.insertWidget(1, QLabel("sys_name"))
        self.sys_name = QComboBox()
        self.sys_name.setEditable(True)
        self.sys_name.addItems(known_names)
        self.sys_name.setCurrentText("psrb")
        self.controls_layout.insertWidget(2, self.sys_name)
        
        lim_layout, self.limits = self.make_linear_slider("Axes limits (in orbital half-phases)",
                                        0.0, 1.0, 0.01, 0.7)
        self.controls_layout.insertLayout(3, lim_layout)

        t_layout, self.t_days = self.make_linear_slider("t [days]", -100.0, 100.0, 1.0, 0.0)
        self.controls_layout.insertLayout(3, t_layout)

        fd_layout, self.f_d = self.make_linear_slider("f_d", 0.0, 1000.0, 5.0, 100.0)
        self.controls_layout.insertLayout(4, fd_layout)

        fp_layout, self.f_p = self.make_linear_slider("f_p", 0.0, 1.0, 0.01, 0.1)
        self.controls_layout.insertLayout(5, fp_layout)

        delta_layout, self.delta = self.make_log10_slider("delta", 0.001, 0.3, 0.01, 0.01)
        self.controls_layout.insertLayout(6, delta_layout)

        alpha_layout, self.alpha_deg = self.make_linear_slider("alpha [deg]", 0.0, 180.0, 1.0, 18.0)
        self.controls_layout.insertLayout(7, alpha_layout)

        incl_layout, self.incl_deg = self.make_linear_slider("incl [deg]", 0.0, 180.0, 1.0, 30.0)
        self.controls_layout.insertLayout(8, incl_layout)

        bns_layout, self.b_ns_13 = self.make_log10_slider("b_ns_13", 0.01, 100.0, 0.01, 1.0)
        self.controls_layout.insertLayout(9, bns_layout)

        bopt_layout, self.b_opt_13 = self.make_log10_slider("b_opt_13", 0.01, 100.0, 0.01, 1.0)
        self.controls_layout.insertLayout(10, bopt_layout)

        nu_layout, self.nu_los_deg = self.make_linear_slider("nu_los [deg]", 0.0, 360.0, 1.0, 90.0)
        self.controls_layout.insertLayout(11, nu_layout)

        s_layout, self.s_max = self.make_linear_slider("s_max", 0.5, 4.0, 0.01, 1.0)
        self.controls_layout.insertLayout(12, s_layout)

        g_layout, self.gamma_max = self.make_linear_slider("gamma_max", 1.0, 4.0, 0.01, 2.0)
        self.controls_layout.insertLayout(13, g_layout)

        self.controls_layout.insertWidget(14, QLabel("IBS color"))
        self.ibs_color = QComboBox()
        self.ibs_color.addItems(["doppler", "scattering", "scattering_comoving",
                                 "b", "b_comov", "ug", "ug_comov",
                                 "gamma-gamma (100 GeV)", "gamma-gamma (1 TeV)",
                                 "gamma-gamma (10 TeV)"])
        self.ibs_color.setCurrentText("doppler")
        self.controls_layout.insertWidget(15, self.ibs_color)

        # ---- Signals
        assert self.apply_btn is not None

        def hook_slider(s):
            s.valueChanged.connect(lambda _: (self.update_slider_label(s), self.schedule_update()))

        for s in (
            self.limits, self.t_days, self.f_d, self.f_p, self.delta, self.alpha_deg, self.incl_deg,
            self.b_ns_13, self.b_opt_13, self.nu_los_deg, self.s_max, self.gamma_max
        ):
            hook_slider(s)

        self.sys_name.currentIndexChanged.connect(lambda _: self.schedule_update())
        self.sys_name.editTextChanged.connect(lambda _: self.schedule_update())
        self.ibs_color.currentIndexChanged.connect(lambda _: self.schedule_update())

        self.apply_btn.clicked.connect(self.update_plot)

        self.update_plot()
        
    def _set_lims(self):
        axis_limits = float(self.slider_value(self.limits))
        t_limit = self._orb.t_from_true_an(axis_limits * np.pi)
        show_cond  = np.logical_and(self._orb.ttab > -t_limit, 
                                    self._orb.ttab < t_limit)
        
        ############# ------ disk passage related stuff ------- ###############

        t1, t2 = self._winds.times_of_disk_passage
        # print('disk equator passage times [days]:')
        # print(t1/DAY, t2/DAY)
        _r_scale = max(self._orb.r(t1), self._orb.r(t2))
        
        orb_x, orb_y = self._orb.xtab[show_cond], self._orb.ytab[show_cond]
        x_scale = np.max(np.array([
            np.abs(np.min(orb_x)), np.abs(np.max(orb_x)), 1.5*_r_scale,
            np.abs(np.max(self._ibs_x)), np.abs(np.max(self._ibs_x)),
            self._orb.r_periastr
            ]))
        y_scale = np.max(np.array([
                np.abs(np.min(orb_y)), np.abs(np.max(orb_y)), 1.5*_r_scale, 
                np.abs(np.max(self._ibs_y)), np.abs(np.max(self._ibs_y)),
                self._orb.r_periastr
                ]))
        self.ax.set_xlim(-1.2 * x_scale, 1.2 * x_scale)
        self.ax.set_ylim(-1.2 * y_scale, 1.2 * y_scale)
        

    def update_plot(self):
        if self.status_lbl is not None:
            self.status_lbl.setText("Computing…")

        try:
            sys_name = self.sys_name.currentText().strip()

            t = float(self.slider_value(self.t_days)) * 86400.0
            f_d = float(self.slider_value(self.f_d))
            f_p = float(self.slider_value(self.f_p))
            delta = float(self.slider_value(self.delta))
            alpha_deg = float(self.slider_value(self.alpha_deg))
            incl_deg = float(self.slider_value(self.incl_deg))
            b_ns_13 = float(self.slider_value(self.b_ns_13))
            b_opt_13 = float(self.slider_value(self.b_opt_13))
            nu_los_deg = float(self.slider_value(self.nu_los_deg))
            s_max = float(self.slider_value(self.s_max))
            gamma_max = float(self.slider_value(self.gamma_max))
            ibs_color = self.ibs_color.currentText()

            _orb, winds, (x, y), c = winds_and_ibs(
                t=t,
                sys_name=sys_name,
                f_d=f_d,
                f_p=f_p,
                delta=delta,
                alpha_deg=alpha_deg,
                incl_deg=incl_deg,
                b_ns_13=b_ns_13,
                b_opt_13=b_opt_13,
                nu_los_deg=nu_los_deg,
                s_max=s_max,
                gamma_max=gamma_max,
                ibs_color=ibs_color,
            )
            self._orb = _orb
            self._winds = winds
            self._ibs_x = x            
            self._ibs_y = y

            # Clear & redraw background
            self.ax.clear()
    

            # Draw winds/orbit overview only
            winds.peek(ax=self.ax, plot_rs=False)

            if self._ibs_lc is not None:
                try:
                    self._ibs_lc.remove()
                except Exception:
                    pass
                self._ibs_lc = None
            
            # draw new IBS gradient line
            self._ibs_lc = add_gradient_line(self.ax, x, y, c, lw=2.5)
            
            # create colorbar once, then update it
            if self._ibs_lc is not None:
                if self._cbar is None:
                    self._cbar = self.fig.colorbar(self._ibs_lc, cax=self._cax)
                else:
                    self._cbar.update_normal(self._ibs_lc)
            
                self._cbar.set_label(ibs_color)

            self.ax.set_title("Winds + IBS")
            self.ax.set_aspect("equal", adjustable="box")
            self._set_lims()

            self.canvas.draw_idle()

            if self.status_lbl is not None:
                self.status_lbl.setText("Ready")

        except Exception as e:
            if self.status_lbl is not None:
                self.status_lbl.setText(f"Error: {type(e).__name__}: {e}")
                
def main():

    # app = QApplication(sys.argv)
    w = IBSWindow()
    w.resize(1200, 700)
    w.show()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()