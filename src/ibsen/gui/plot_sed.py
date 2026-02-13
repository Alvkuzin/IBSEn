import numpy as np
from numpy import pi

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QSlider, QComboBox, QPushButton,
    QCheckBox
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# from ibsen.utils import fit_norm, interplg
from ibsen.gui.base import fit_norm_here
from ibsen import Orbit, Winds, IBS, ElectronsOnIBS, SpectrumIBS
from gammapy.estimators import FluxPoints

NORMALIZATION_INITIAL = 1E37


def read_sed_txt_dat(path):
    """
    Reads the SED called sed_{key}{suffix}.txt

    returns : dict['e', 'sed', 'dsed', 'dsed_minus', 'dsed_plus',
                     'de', 'de_minus', 'de_plus']
    e in TeV
    seds in erg/cm2/s
    """

    da = np.genfromtxt(path,
                       delimiter=' ', usecols=[0, 1, 2, 3, 4, 5], 
                       names=['e', 'sed', 'de_p', 'de_m', 'dsed_p', 'dsed_m'])

   
    e, sed, de_p, de_m, dsed_p, dsed_m = [da[key] for key in ('e', 'sed',
                                                              'de_p', 'de_m',
                                                'dsed_p', 'dsed_m')]
    de = 0.5 * (de_m + de_p)
    dsed = 0.5 * (dsed_m + dsed_p)
    
    res = {'e': e*1e12, 'sed': sed, 'dsed': dsed,
              'dsed_minus': dsed_m, 'dsed_plus': dsed_p,
        'de': de*1e12, 'de_minus': de_m*1e12, 'de_plus': de_p*1e12,}
    return res    

def get_sed_values(sed_gammapy, return_de=False):
    """
    From a FluxPoints gammapy object extracts e [TeV], sed [erg s-1 cm-2],
    dsed [erg s-1 cm-2].
        """
    tab  = sed_gammapy.to_table(sed_type='e2dnde')
    e, sed, dsed = tab['e_ref'].value, \
        (tab['e2dnde'].to("erg s-1 cm-2")).value, \
            (tab['e2dnde_err'].to("erg s-1 cm-2")).value
    de = (tab['e_max'].value - tab['e_min'].value) / 2.0
    if return_de:
        return e, de, sed, dsed
    return e, sed, dsed

def read_sed_gammapy(path):
    sed = FluxPoints.read(path)
    e, sed, dsed = get_sed_values(sed)
    return {'e': e*1e12, 'sed':sed, 'dsed': dsed}


    
                
def sed(t=0.0, f_d=100.0, b_13=1.0, gamma_max=2.0, s_max=1.0, cooling='stat_ibs',
        p_e=2, e_cut=5e12, eta_a=1, ic_ani=False, 
                        abs_gg=False, method='simple', lorentz_boost=True,
                        abs_photoel=False, nh_tbabs=0.8):
    orb = Orbit(sys_name = 'psrb')
    winds = Winds(orbit=orb, sys_name = 'psrb', alpha=18/180*pi, incl=30*pi/180,
              f_d=f_d, f_p=0.1, delta=0.01, np_disk=3, rad_prof='pl', r_trunk=None,
             height_exp=0.25,  t_forwinds=t,
             ns_b_ref=b_13, ns_r_ref=1e13)
    ibs = IBS(winds = winds,    
      t_to_calculate_beta_eff=t, 
      gamma_max=gamma_max,  
      s_max=s_max,       
      s_max_g=4,      
      n=21,           
      abs_gg_filename = None, # with tabulated gg-opacities; optional
      ) 
    els = ElectronsOnIBS(ibs=ibs, 
                        norm_e = NORMALIZATION_INITIAL,
                 cooling=cooling, 
                 p_e=p_e,
                 ecut = e_cut,  
                 emax = 5e13, 
                 emin_grid=3e8,
                 emax_grid=1e14,
                 eta_a=eta_a,
                 ) 
    spec = SpectrumIBS(els=els, mechanisms=['s', 'i'], method=method,
                       lorentz_boost=lorentz_boost, abs_gg=abs_gg, sys_name='psrb',
                       ic_ani=ic_ani, abs_photoel=abs_photoel, nh_tbabs=nh_tbabs)
    E = np.geomspace(3e2, 1e14, 100)
    spec.calculate(e_ph=E)
    return spec.e_ph, spec.sed_sy, spec.sed_ic


# =========================
# GUI
# =========================

class SEDWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SED explorer")

        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter)

        # -------- Left: plot --------
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        # self.obs_artists = []   # list of "artist groups" so we can remove later
        self.obs_data = []  # list of dicts: {"name": str, "x":..., "y":..., "dy":..., "artists":[...]}

        self._setup_menu()
        
                

        
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("E [eV]")
        self.ax.set_ylabel("SED")
        self.ax.grid(True, which="both", alpha=0.3)

        (self.line_sy,) = self.ax.plot([], [], label="Synchrotron", color='b', ls='--')
        (self.line_ic,) = self.ax.plot([], [], label="IC", color='r', ls='--')
        (self.line_tot,) = self.ax.plot([], [], label="Total", color='k', ls='-')
        self._conf_interv_poly = self.ax.fill_between([], [], [], alpha=0.2, color='k') 

        self.ax.legend(loc="best")

        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        # -------- Right: controls --------
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)

        controls.addWidget(QLabel("<b>Parameters</b>"))

        # t in days: -100..100, later multiply by 86400
        self.t_layout, self.t_days = self._make_slider_linear(
            name="t [days]", min_val=-100.0, max_val=100.0, step=1.0, initial=0.0
        )
        controls.addLayout(self.t_layout)

        # f_d linear 0..1000
        self.fd_layout, self.f_d = self._make_slider_linear(
            name="f_d", min_val=0.0, max_val=1000.0, step=5.0, initial=100.0
        )
        controls.addLayout(self.fd_layout)

        # b_13 log 0.01..100
        self.b_layout, self.b_13 = self._make_slider_log10(
            name="b_13", min_val=0.01, max_val=100.0, step_log10=0.01, initial=1.0
        )
        controls.addLayout(self.b_layout)

        # gamma_max linear 1..4
        self.gm_layout, self.gamma_max = self._make_slider_linear(
            name="gamma_max", min_val=1.0, max_val=4.0, step=0.01, initial=2.0
        )
        controls.addLayout(self.gm_layout)

        # s_max linear 0.5..4
        self.sm_layout, self.s_max = self._make_slider_linear(
            name="s_max", min_val=0.5, max_val=4.0, step=0.01, initial=1.0
        )
        controls.addLayout(self.sm_layout)

        # p_e linear 1..3
        self.pe_layout, self.p_e = self._make_slider_linear(
            name="p_e", min_val=1.0, max_val=3.0, step=0.01, initial=2.0
        )
        controls.addLayout(self.pe_layout)

        # e_cut log 1e11..1e14
        self.ec_layout, self.e_cut = self._make_slider_log10(
            name="e_cut [eV]", min_val=1e11, max_val=1e14, step_log10=0.01, initial=5e12
        )
        controls.addLayout(self.ec_layout)

        # eta_a log 0.1..10
        self.eta_layout, self.eta_a = self._make_slider_log10(
            name="eta_a", min_val=0.1, max_val=10.0, step_log10=0.01, initial=1.0
        )
        controls.addLayout(self.eta_layout)
        
        # nH linear 0.2...3.0
        self.nh_layout, self.nh = self._make_slider_linear(
            name="nH", min_val=0.2, max_val=3.0, step=0.01, initial=0.8
        )
        controls.addLayout(self.nh_layout)
        
        controls.addSpacing(8)
        controls.addWidget(QLabel("<b>Options</b>"))

        # cooling dropdown
        controls.addWidget(QLabel("cooling"))
        self.cooling = QComboBox()
        self.cooling.addItems(['no',
            "stat_apex", "stat_ibs", "stat_mimic",
            "leak_apex", "leak_ibs", "leak_mimic",
            "adv"
        ])
        self.cooling.setCurrentText("stat_ibs")
        controls.addWidget(self.cooling)

        # method dropdown (typo “emthod” in your text; using "method")
        controls.addWidget(QLabel("method"))
        self.method = QComboBox()
        self.method.addItems(["full", "simple", "apex"])
        self.method.setCurrentText("simple")
        controls.addWidget(self.method)

        # boolean checkboxes (True/False)
        checkbox_1_row = QHBoxLayout()
        checkbox_2_row = QHBoxLayout()
        
        self.ic_ani_cb = QCheckBox("ic_ani")
        self.ic_ani_cb.setChecked(False)
        
        self.abs_gg_cb = QCheckBox("abs_gg")
        self.abs_gg_cb.setChecked(False)
        
        self.lorentz_boost_cb = QCheckBox("lorentz_boost")
        self.lorentz_boost_cb.setChecked(True)
        
        self.photoel_abs_cb = QCheckBox("Photoel. absorbtion")
        self.photoel_abs_cb.setChecked(False)
        
        checkbox_1_row.addWidget(self.ic_ani_cb)
        checkbox_1_row.addWidget(self.abs_gg_cb)
        checkbox_2_row.addWidget(self.lorentz_boost_cb)        
        checkbox_2_row.addWidget(self.photoel_abs_cb)
        
        controls.addLayout(checkbox_1_row)
        controls.addLayout(checkbox_2_row)

        # --- Fit controls ---
        controls.addSpacing(10)
        controls.addWidget(QLabel("<b>Fitting</b>"))
        
        self.fit_norm_cb = QCheckBox("Fit norm")
        self.fit_norm_cb.setChecked(False)
        controls.addWidget(self.fit_norm_cb)
        
        controls.addWidget(QLabel("Fit to dataset:"))
        self.fit_dataset = QComboBox()
        self.fit_dataset.setEnabled(False)  # enabled only when Fit norm is checked
        controls.addWidget(self.fit_dataset)

        # Apply / Reset / Auto-update
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.reset_btn)
        controls.addLayout(btn_row)

        self.auto = QCheckBox("Auto update (debounced)")
        self.auto.setChecked(False)
        controls.addWidget(self.auto)

        self.status = QLabel("")
        controls.addWidget(self.status)

        controls.addStretch(1)
        splitter.addWidget(controls_widget)
        splitter.setSizes([900, 330])

        # Debounce timer (avoid recompute on every tiny slider move)
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)  # ms
        self._debounce.timeout.connect(self.update_plot)

        self._connect_signals()
        self.update_plot()

    # ---------- Widget helpers ----------
    def _make_slider_linear(self, name: str, min_val: float, max_val: float, step: float, initial: float):
        if step <= 0:
            raise ValueError("step must be > 0")
        n_steps = int(round((max_val - min_val) / step))
        if n_steps <= 0:
            raise ValueError("Bad slider range/step")
    
        # clamp initial
        initial = max(min_val, min(max_val, initial))
        iinit = int(round((initial - min_val) / step))
    
        layout = QVBoxLayout()
        title = QLabel(f"{name} = {initial:.4g}")
        layout.addWidget(title)
    
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(n_steps)
        slider.setValue(iinit)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, n_steps // 20))
        layout.addWidget(slider)
    
        slider._mode = "linear_step"
        slider._min = float(min_val)
        slider._step = float(step)
        slider._label = title
        slider._name = name
        slider._initial = float(initial)
    
        return layout, slider

    def _make_slider_log10(self, name: str, min_val: float, max_val: float, step_log10: float, initial: float):
        # slider is linear in log10(value)
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_init = np.log10(initial)

        scale = int(round(1.0 / step_log10))
        imin = int(round(log_min * scale))
        imax = int(round(log_max * scale))
        iinit = int(round(log_init * scale))

        layout = QVBoxLayout()
        title = QLabel(f"{name} = {initial:.4g}")
        layout.addWidget(title)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(imin)
        slider.setMaximum(imax)
        slider.setValue(iinit)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (imax - imin) // 20))
        layout.addWidget(slider)

        slider._mode = "log10"
        slider._scale = scale
        slider._label = title
        slider._name = name
        slider._initial = initial
        slider._min = min_val
        slider._max = max_val

        return layout, slider

    def _slider_value(self, slider: QSlider) -> float:
        if slider._mode == "linear_step":
            return slider._min + slider.value() * slider._step
        if slider._mode == "log10":
            logv = slider.value() / slider._scale
            return float(10 ** logv)
        raise ValueError("Unknown slider mode")

    def _update_slider_label(self, slider: QSlider):
        v = self._slider_value(slider)
        slider._label.setText(f"{slider._name} = {v:.4g}")

    def _make_bool_dropdown(self, name: str, default: bool):
        lab = QLabel(name)
        box = QComboBox()
        box.addItems(["False", "True"])
        box.setCurrentText("True" if default else "False")
        return {"label": lab, "box": box, "default": default, "name": name}

    def _bool_value(self, bool_box: QComboBox) -> bool:
        return bool_box.currentText() == "True"
    
    def _setup_menu(self):
        menubar = self.menuBar()
    
        upload_menu = menubar.addMenu("Upload obs")
    
        act_txt = QAction("Upload txt/dat…", self)
        act_txt.triggered.connect(self.upload_txt_dat)
        upload_menu.addAction(act_txt)
    
        act_gpy = QAction("Upload Gammapy SED fits…", self)
        act_gpy.triggered.connect(self.upload_gammapy_sed)
        upload_menu.addAction(act_gpy)
    
        upload_menu.addSeparator()
    
        act_clear = QAction("Clear observational data", self)
        act_clear.triggered.connect(self.clear_observations)
        upload_menu.addAction(act_clear)
    
    
    def upload_txt_dat(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select txt/dat observation file",
            "",
            "Text data (*.txt *.dat *.csv);;All files (*)"
        )
        if not path:
            return
        res = read_sed_txt_dat(path)
        x, y, dy = [res[key] for key in ('e', 'sed', 'dsed')]
        self._add_observation(x, y, dy, label=self._nice_label(path))
    
    
    def upload_gammapy_sed(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Gammapy SED FITS file",
            "",
            "FITS files (*.fits *.fit *.fz);;All files (*)"
        )
        if not path:
            return
        res = read_sed_gammapy(path)
        x, y, dy = [res[key] for key in ('e', 'sed', 'dsed')]
        self._add_observation(x, y, dy, label=self._nice_label(path))
    
    
    def _add_observation(self, x, y, dy, label="Obs"):
        x = np.asarray(x)
        y = np.asarray(y)
        dy_arr = None if dy is None else np.asarray(dy)
    
        # For log-log plots, keep positive finite points
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if dy_arr is not None:
            m = m & np.isfinite(dy_arr) & (dy_arr >= 0)
            dy_arr = dy_arr[m]
        x, y = x[m], y[m]
    
        if x.size == 0:
            return
    
        # Plot
        if dy_arr is None:
            sc = self.ax.scatter(x, y, marker="o", s=25, label=label)
            artists = [sc]
        else:
            cont = self.ax.errorbar(x, y, yerr=dy_arr, fmt="o", capsize=2, label=label)
            line, caplines, barlinecols = cont
            artists = [line] + list(caplines) + list(barlinecols)
    
        # Store dataset
        self.obs_data.append({"name": label, "x": x, "y": y, "dy": dy_arr,
                              "artists": artists, 'yscale': np.max(y)})
    
        # Update the "fit to dataset" dropdown
        self.fit_dataset.addItem(label)
        if self.fit_dataset.count() == 1:
            self.fit_dataset.setCurrentIndex(0)
    
        self.ax.legend(loc="best")
        self.canvas.draw_idle()
    
    
    def clear_observations(self):
        for ds in self.obs_data:
            for artist in ds["artists"]:
                try:
                    artist.remove()
                except Exception:
                    pass
        self.obs_data.clear()
    
        # Clear fit dropdown too
        self.fit_dataset.clear()
        self.ax.legend(loc="best")
        self.canvas.draw_idle()
    
    
    def _nice_label(self, path: str) -> str:
        """
        Short label from filename; customize if you want.
        """
        import os
        return os.path.basename(path)


    # ---------- Signals ----------
    def _connect_signals(self):
        # all sliders
        sliders = [self.t_days, self.f_d, self.b_13, self.gamma_max,
                   self.s_max, self.p_e, self.e_cut, self.eta_a, self.nh]

        def on_any_change(slider=None):
            if slider is not None:
                self._update_slider_label(slider)
            if self.auto.isChecked():
                self._debounce.start()

        for s in sliders:
            s.valueChanged.connect(lambda _, ss=s: on_any_change(ss))

        # dropdowns
        self.cooling.currentIndexChanged.connect(lambda _: on_any_change())
        self.method.currentIndexChanged.connect(lambda _: on_any_change())
        # self.ic_ani["box"].currentIndexChanged.connect(lambda _: on_any_change())
        # self.abs_gg["box"].currentIndexChanged.connect(lambda _: on_any_change())
        # self.lorentz_boost["box"].currentIndexChanged.connect(lambda _: on_any_change())

        self.ic_ani_cb.stateChanged.connect(lambda _: on_any_change())
        self.abs_gg_cb.stateChanged.connect(lambda _: on_any_change())
        self.lorentz_boost_cb.stateChanged.connect(lambda _: on_any_change())
        self.photoel_abs_cb.stateChanged.connect(lambda _: on_any_change())
        
        self.fit_norm_cb.toggled.connect(self.fit_dataset.setEnabled)
        self.fit_norm_cb.toggled.connect(lambda _: on_any_change())
        self.fit_dataset.currentIndexChanged.connect(lambda _: on_any_change())

        
        # buttons
        self.apply_btn.clicked.connect(self.update_plot)
        self.reset_btn.clicked.connect(self.reset_controls)

    def reset_controls(self):
        for s in [self.t_days, self.f_d, self.gamma_max, self.s_max,
                  self.p_e, self.nh]:
            # linear_step sliders
            if s._mode == "linear_step":
                s.setValue(int(round((s._initial - s._min) / s._step)))
                self._update_slider_label(s)
    
        for s in [self.b_13, self.e_cut, self.eta_a]:
            # log10 sliders
            s.setValue(int(round(np.log10(s._initial) * s._scale)))
            self._update_slider_label(s)
    
        self.cooling.setCurrentText("stat_ibs")
        self.method.setCurrentText("simple")
        # self.ic_ani["box"].setCurrentText("False")
        # self.abs_gg["box"].setCurrentText("False")
        # self.lorentz_boost["box"].setCurrentText("True")

        self.ic_ani_cb.setChecked(False)
        self.abs_gg_cb.setChecked(False)
        self.lorentz_boost_cb.setChecked(True)
        self.photoel_abs_cb.setChecked(True)
        
        self.status.setText("")
        self.update_plot()

    # ---------- Plot update ----------
    def update_plot(self):
        try:
            self.status.setText("Computing…")
            QApplication.processEvents()

            t_days = self._slider_value(self.t_days)
            t_sec = t_days * 86400.0
            

            
            params = dict(
                t=t_sec,
                f_d=self._slider_value(self.f_d),
                b_13=self._slider_value(self.b_13),
                gamma_max=self._slider_value(self.gamma_max),
                s_max=self._slider_value(self.s_max),
                cooling=self.cooling.currentText(),
                p_e=self._slider_value(self.p_e),
                e_cut=self._slider_value(self.e_cut),
                eta_a=self._slider_value(self.eta_a),
                ic_ani=self.ic_ani_cb.isChecked(),
                abs_gg=self.abs_gg_cb.isChecked(),
                method=self.method.currentText(),
                lorentz_boost=self.lorentz_boost_cb.isChecked(),
                abs_photoel=self.photoel_abs_cb.isChecked(),
                nh_tbabs=self._slider_value(self.nh)
            )
            # print(params)

            E, sed_sy, sed_ic = sed(**params)

            # Avoid non-positive values on log scale:
            sed_sy = np.asarray(sed_sy)
            sed_ic = np.asarray(sed_ic)
            sed_tot = sed_sy + sed_ic
            
            # if fit enabled and a dataset exists:
            if self.fit_norm_cb.isChecked() and self.fit_dataset.count() > 0:
                name = self.fit_dataset.currentText()
            
                # find the dataset by name
                ds = next((d for d in self.obs_data if d["name"] == name), None)
                if ds is not None:
                    optimal_norm_e, sed_tot, dn, sed_tot_low, sed_tot_high = fit_norm_here(
                        x_obs=ds["x"], 
                        y_obs=ds["y"], dy_obs=ds["dy"], x_model=E,
                        y_model=sed_tot, norm_init=NORMALIZATION_INITIAL,
                        grid_scale='log', return_err=True)
                    
 
                    sed_sy = optimal_norm_e / NORMALIZATION_INITIAL * sed_sy
                    sed_ic = optimal_norm_e / NORMALIZATION_INITIAL * sed_ic
            
            # Mask invalid/nonpositive for log scale
            m = np.isfinite(E) & np.isfinite(sed_tot) & (E > 0) & (sed_tot > 0)
            E = E[m]
            sed_sy = sed_sy[m]
            sed_ic = sed_ic[m]
            sed_tot = sed_tot[m]
            
            # Update band: remove old poly and make a new one
            if self.fit_norm_cb.isChecked() and self.fit_dataset.count() > 0:    
                if self._conf_interv_poly is not None:
                    self._conf_interv_poly.remove()
                self._conf_interv_poly = self.ax.fill_between(E, sed_tot_low,
                                sed_tot_high, alpha=0.2, color='k')
                
            self.line_sy.set_data(E, np.clip(sed_sy, 1e-300, np.inf))
            self.line_ic.set_data(E, np.clip(sed_ic, 1e-300, np.inf))
            self.line_tot.set_data(E, np.clip(sed_tot, 1e-300, np.inf))
            
            # autoscale nicely
            self.ax.relim()
            self.ax.autoscale_view()
            y_scale = np.max(sed_tot)
            if self.obs_data is not None:
                for data_ in self.obs_data:
                    y_scales_data = np.array([d['yscale'] for d in self.obs_data])
                    y_scale = max(y_scale, y_scales_data.max())
            self.ax.set_ylim(y_scale/1e3, y_scale*1.5)

            self.canvas.draw_idle()
            self.status.setText("Ready")
        except Exception as e:
            self.status.setText(f"Error: {type(e).__name__}: {e}")


def main():

    # app = QApplication(sys.argv)
    w = SEDWindow()
    w.resize(1200, 700)
    w.show()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()