import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QSlider, QComboBox, QPushButton
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from ibsen.ibs_norm import IBS_norm
from ibsen.gui.base import GradientPlot

                
def points_and_color(beta, nu_los, s_max, gamma_max, phi_rot, ibs_color):
    unit_los = np.array([np.cos(nu_los), np.sin(nu_los), 0])
    _ibs0 = IBS_norm(beta, s_max=s_max, gamma_max=gamma_max, s_max_g=4., n=13,
                unit_los = unit_los)
    _ibs = _ibs0.rotate(phi_rot)
    ibs_color = ibs_color.lower()
    if ibs_color in ('doppler', 'dopl', 'doppl', 'dopler', 'doppler factor'):
        color_param = _ibs.dopl
    elif ibs_color in ('scattering', 'scattering angle'):
        color_param = _ibs.scattering_angle
    elif ibs_color in ('scattering_comov', 'scattering_angle_comov'
                       'scattering comov', 'scattering angle comov',
                       'scattering comoving', 'scattering angle comoving'
                       ):
        color_param = _ibs.scattering_angle_comov
    else:
        ### finds the attribute via getarray
        if not hasattr(_ibs, ibs_color):
            raise AttributeError(f"No attribute '{ibs_color}' in {type(_ibs).__name__}")
        color_param = getattr(_ibs, ibs_color)
    
    return _ibs.x, _ibs.y, color_param
                
class IBSNormWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive model explorer")

        # ---- Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # ---- Left/Right layout with splitter
        main = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter)

        # =======================
        # Left: plot
        # =======================
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        (self.line_los,) = self.ax.plot([], [], lw=2, ls="--", color='C2')
        (self.line_symm,) = self.ax.plot([], [], lw=2, ls=":", color='C1')
        self.ax.scatter(0, 0, c='r')
        self.opt_star_scatter = self.ax.scatter([], [], color='b')
        self.grad_plot = GradientPlot(fig=self.fig, ax=self.ax, colorbar=True, cbar_label=None)

        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        # =======================
        # Right: controls (vertical column)
        # =======================
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)

        controls.addWidget(QLabel("<b>Parameters</b>"))

        # --- sliders stacked vertically ---
        # log(beta)
        self.beta_layout, self.logbeta = self._make_slider(
            name="log(beta)", min_val=-4.0, max_val=-0.1, step=0.01, initial=-1.0
        )
        controls.addLayout(self.beta_layout)

        # nu_los in degrees
        self.nu_layout, self.nu_deg = self._make_slider(
            name="nu_los (deg)", min_val=0.0, max_val=360.0, step=1.0, initial=90.0
        )
        controls.addLayout(self.nu_layout)

        # s_max  (choose a range that makes sense for your model)
        self.smax_layout, self.s_max = self._make_slider(
            name="s_max", min_val=0.5, max_val=4.0, step=0.05, initial=1.0
        )
        controls.addLayout(self.smax_layout)

        # gamma_max (range is model-dependent; pick what’s sensible)
        self.gmax_layout, self.gamma_max = self._make_slider(
            name="gamma_max", min_val=1.0, max_val=5.0, step=0.01, initial=1.5
        )
        controls.addLayout(self.gmax_layout)

        # phi_rot in degrees
        self.phi_layout, self.phi_deg = self._make_slider(
            name="phi_rot (deg)", min_val=0.0, max_val=360.0, step=1.0, initial=0.0
        )
        controls.addLayout(self.phi_layout)

        controls.addSpacing(8)
        controls.addWidget(QLabel("<b>Color coding</b>"))

        # --- dropdown + buttons ---
        self.colorcode = QComboBox()
        self.colorcode.addItems(["Doppler factor", "scattering", "scattering comoving"])
        controls.addWidget(self.colorcode)

        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        btn_row.addStretch(1)
        btn_row.addWidget(self.reset_btn)
        controls.addLayout(btn_row)

        # push everything to top (nice “panel” look)
        controls.addStretch(1)

        splitter.addWidget(controls_widget)

        # initial splitter sizes: plot bigger than panel
        splitter.setSizes([900, 320])

        # ---- Wire up events
        self._connect_signals()

        # First draw
        self.update_plot()

    def _make_slider(self, name: str, min_val: float, max_val: float, step: float, initial: float):
        """
        Qt sliders are integer-based. We scale float -> int.
        Returns (layout, slider).
        """
        scale = int(round(1.0 / step))
        imin = int(round(min_val * scale))
        imax = int(round(max_val * scale))
        iinit = int(round(initial * scale))

        layout = QVBoxLayout()

        title = QLabel(f"{name} = {initial:.2f}")
        layout.addWidget(title)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(imin)
        slider.setMaximum(imax)
        slider.setValue(iinit)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (imax - imin) // 20))
        layout.addWidget(slider)

        slider._scale = scale
        slider._label = title
        slider._name = name
        slider._initial = initial

        return layout, slider

    def _slider_value(self, slider: QSlider) -> float:
        return slider.value() / slider._scale

    def _connect_signals(self):
        def on_slider_changed(slider: QSlider):
            val = self._slider_value(slider)
            slider._label.setText(f"{slider._name} = {val:.2f}")
            self.update_plot()

        # sliders
        for s in (self.logbeta, self.nu_deg, self.s_max, self.gamma_max, self.phi_deg):
            s.valueChanged.connect(lambda _, ss=s: on_slider_changed(ss))

        # dropdown
        self.colorcode.currentIndexChanged.connect(lambda _: self.update_plot())

        # reset
        self.reset_btn.clicked.connect(self.reset_controls)

    def reset_controls(self):
        # set slider values back to initial
        for s in (self.logbeta, self.nu_deg, self.s_max, self.gamma_max, self.phi_deg):
            s.setValue(int(round(s._initial * s._scale)))

        self.colorcode.setCurrentText("Doppler factor")
        self.update_plot()

    def update_plot(self):
        beta = 10 ** self._slider_value(self.logbeta)
        nu_los = np.deg2rad(self._slider_value(self.nu_deg))
        s_max = self._slider_value(self.s_max)
        gamma_max = self._slider_value(self.gamma_max)
        phi_rot = np.deg2rad(self._slider_value(self.phi_deg))
        colorcode = self.colorcode.currentText().lower()

        x, y, param = points_and_color(
            beta=beta,
            nu_los=nu_los,
            s_max=s_max,
            gamma_max=gamma_max,
            phi_rot=phi_rot,
            ibs_color=colorcode,
        )

        self.grad_plot.update(x, y, param, autoscale=False)

        self.line_los.set_data([0, 3 * np.cos(nu_los)], [0, 3 * np.sin(nu_los)])
        self.line_symm.set_data([-3 * np.cos(phi_rot), 3 * np.cos(phi_rot)],
                                [-3 * np.sin(phi_rot), 3 * np.sin(phi_rot)])
        self.opt_star_scatter.set_offsets([[np.cos(phi_rot), np.sin(phi_rot)]])
        
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)

        self.canvas.draw_idle()

def main():
    # app = QApplication(sys.argv)
    w = IBSNormWindow()
    w.resize(900, 650)
    w.show()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()
