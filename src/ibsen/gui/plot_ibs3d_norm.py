"""
It's written mainly by ChatGPT. Sorry I'm not a coder jeez!!!!!!!!!! 
"""
import numpy as np
from PySide6.QtWidgets import (
    # QCheckBox, 
    QComboBox, 
    # QHBoxLayout, 
    QLabel, 
    # QMainWindow,
    # QPushButton, QSlider, QVBoxLayout, QWidget
)

from ibsen.gui.base import plot_surface_quads_local, ToolWindowBase



from ibsen.utils import rotated_vector
from ibsen.ibs_norm import IBS_norm3D 

def ibs_plotter(beta, n, n_phi, s_max, gamma_max, 
                theta_los, phi_los,
                theta_rot, phi_rot, angle_rot,
                ibs_color):
    ibs3d = IBS_norm3D(beta=beta, n=n, n_phi=n_phi, s_max=s_max, gamma_max=gamma_max, s_max_g=s_max,
                unit_los=rotated_vector(alpha=phi_los, incl=theta_los)
                )
    rot_ax = rotated_vector(phi_rot, theta_rot)
    ibs_rotated = ibs3d.rotate(phi=angle_rot, vec_ax=rot_ax)
    ibs_color = ibs_color.lower()
    if ibs_color in ('doppler', 'dopl', 'doppl', 'dopler', 'doppler factor'):
        color_param = ibs_rotated.dopl
    elif ibs_color in ('scattering', 'scattering angle'):
        color_param = ibs_rotated.scattering_angle
    elif ibs_color in ('scattering_comov', 'scattering_angle_comov'
                       'scattering comov', 'scattering angle comov',
                       'scattering comoving', 'scattering angle comoving'
                       ):
        color_param = ibs_rotated.scattering_angle_comov
    else:
        ### finds the attribute via getarray
        if not hasattr(ibs_rotated, ibs_color):
            raise AttributeError(f"No attribute '{ibs_color}' in {type(ibs3d).__name__}")
        color_param = getattr(ibs_rotated, ibs_color)
        
    return ibs_rotated.r_vec, color_param

class IBS3DWindow(ToolWindowBase):
    TOOL_NAME = "IBS 3D"
    IS_HEAVY = False  # IBS is “fast”: update immediately
    THREE_DIMENSIONAL_PLOT= True
    def __init__(self) -> None:
        super().__init__()

        # --- replace base 2D axis with 3D axis ---
        # self.fig.clear()
        # self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas.draw_idle()

        # internal artists
        self._surf_poly = None
        self._cbar = None
        self._cbar_label = None
        self._arrow_los = None
        self._arrow_rot = None

        # --- controls (use base right panel layout) ---
        self._remove_last_stretch(self.controls_layout)

        self.controls_layout.addWidget(QLabel("<b>IBS parameters</b>"))

        lay, self.s_beta = self.make_log10_slider("beta", vmin=1e-4, vmax=1.0, step_log10=1e-2, initial=0.1)
        self.controls_layout.addLayout(lay)

        lay, self.s_smax = self.make_linear_slider("s_max", vmin=0.5, vmax=4.0, step=0.05, initial=1.5)
        self.controls_layout.addLayout(lay)

        lay, self.s_gmax = self.make_linear_slider("gamma_max", vmin=1.0, vmax=5.0, step=0.05, initial=2.0)
        self.controls_layout.addLayout(lay)

        self.controls_layout.addWidget(QLabel("<b>Resolution</b>"))
        lay, self.s_n = self.make_linear_slider("n", vmin=10, vmax=80, step=1, initial=24)
        self.controls_layout.addLayout(lay)
        lay, self.s_nphi = self.make_linear_slider("n_phi", vmin=12, vmax=120, step=1, initial=32)
        self.controls_layout.addLayout(lay)

        self.controls_layout.addWidget(QLabel("<b>LOS angles</b>"))
        lay, self.s_theta = self.make_linear_slider("theta [deg]", vmin=0.0, vmax=180.0, step=1.0, initial=54.0)
        self.controls_layout.addLayout(lay)
        lay, self.s_phi = self.make_linear_slider("phi [deg]", vmin=0.0, vmax=360.0, step=1.0, initial=116.0)
        self.controls_layout.addLayout(lay)

        self.controls_layout.addWidget(QLabel("<b>Rotation angles</b>"))
        lay, self.s_theta_rot = self.make_linear_slider("theta rot [deg]", vmin=0.0, vmax=180.0, step=1.0, initial=90.0)
        self.controls_layout.addLayout(lay)
        lay, self.s_phi_rot = self.make_linear_slider("phi rot [deg]", vmin=0.0, vmax=360.0, step=1.0, initial=0.0)
        self.controls_layout.addLayout(lay)
        lay, self.s_rot = self.make_linear_slider("rot angle around axis [deg]", vmin=0.0, vmax=360.0, step=1.0, initial=0.0)
        self.controls_layout.addLayout(lay)

        self.controls_layout.addWidget(QLabel("<b>Color</b>"))
        self.color = QComboBox()
        self.color.addItems(["scattering_comov", "scattering", "doppler", "none"])
        self.color.setCurrentText("scattering_comov")
        self.controls_layout.addWidget(self.color)

        self.controls_layout.addStretch(1)

        # --- wiring ---
        for sl in (
            self.s_beta, self.s_smax, self.s_gmax,
            self.s_n, self.s_nphi,
            self.s_theta, self.s_phi,
            self.s_theta_rot, self.s_phi_rot, self.s_rot,
        ):
            sl.valueChanged.connect(lambda _=None, s=sl: self._on_slider(s))

        self.color.currentIndexChanged.connect(lambda _=None: self.schedule_update())

        # initial draw
        self.update_plot()

    def _on_slider(self, slider) -> None:
        # Update label (override formatting for integer sliders)
        if slider in (self.s_n, self.s_nphi):
            v = int(round(self.slider_value(slider)))
            slider._label.setText(f"{self._slider_meta[slider].name} = {v}")  # type: ignore[attr-defined]
        else:
            self.update_slider_label(slider)

        self.schedule_update()

    def _set_arrow(self, which: str, origin, vec, length=1.7, color="g", lw=2.0):
        x0, y0, z0 = origin
        vx, vy, vz = vec

        attr = "_arrow_" + which
        old = getattr(self, attr, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass

        q = self.ax.quiver(
            x0, y0, z0,
            length * vx, length * vy, length * vz,
            arrow_length_ratio=0.12,
            linewidth=lw,
            color=color,
            normalize=False,
        )
        setattr(self, attr, q)

    def update_plot(self) -> None:
        try:
            beta = float(self.slider_value(self.s_beta))
            s_max = float(self.slider_value(self.s_smax))
            gamma_max = float(self.slider_value(self.s_gmax))

            n = int(round(self.slider_value(self.s_n)))
            n_phi = int(round(self.slider_value(self.s_nphi)))

            theta = np.deg2rad(float(self.slider_value(self.s_theta)))
            phi = np.deg2rad(float(self.slider_value(self.s_phi)))

            theta_rot = np.deg2rad(float(self.slider_value(self.s_theta_rot)))
            phi_rot = np.deg2rad(float(self.slider_value(self.s_phi_rot)))
            rot_rot = np.deg2rad(float(self.slider_value(self.s_rot)))

            ibs_color = self.color.currentText()
            if ibs_color == "none":
                ibs_color = None

            # remove old surface
            if self._surf_poly is not None:
                try:
                    self._surf_poly.remove()
                except Exception:
                    pass
                self._surf_poly = None

            rvec, param = ibs_plotter(
                beta=beta, n=n, n_phi=n_phi,
                s_max=s_max, gamma_max=gamma_max,
                theta_los=theta, phi_los=phi,
                theta_rot=theta_rot, phi_rot=phi_rot, angle_rot=rot_rot,
                ibs_color=ibs_color,
            )

            self._surf_poly, mappable, _ = plot_surface_quads_local(
                ax=self.ax,
                coords=rvec,
                param=param,
                linewidth=0.05,
                edgecolor="k",
                close_phi=True,
            )

            # colorbar handling
            want_cbar = (mappable is not None)
            if want_cbar:
                if self._cbar is None:
                    self._cbar = self.fig.colorbar(mappable, ax=self.ax, pad=0.05)
                else:
                    self._cbar.update_normal(mappable)
            else:
                if self._cbar is not None:
                    self._cbar.remove()
                    self._cbar = None
                    self._cbar_label = None

            # arrows
            self._set_arrow("los", origin=(0, 0, 0), vec=rotated_vector(phi, theta), length=1.7, color="g", lw=2.0)
            self._set_arrow("rot", origin=(0, 0, 0), vec=rotated_vector(phi_rot, theta_rot), length=1.7, color="r", lw=2.0)

            # aspect + limits
            try:
                self.ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass

            _scale = max(2.0, float(s_max))
            self.ax.set_xlim(-_scale, _scale)
            self.ax.set_ylim(-_scale, _scale)
            self.ax.set_zlim(-_scale, _scale)

            self.canvas.draw_idle()

        except Exception as e:
            # IBS tool isn't "heavy", so no status label by default.
            # Print or add a QLabel if you want UI feedback.
            print(f"[IBS3DWindow] Error: {type(e).__name__}: {e}")

def main():
    # app = QApplication(sys.argv)
    w = IBS3DWindow()
    w.resize(1200, 700)
    w.show()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()