"""
It's written mainly by ChatGPT. Sorry I'm not a coder jeez!!!!!!!!!! 
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table
from astropy.io import ascii

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QToolBox
)

from ibsen.gui.base import fit_norm_here, ToolWindowBase
from ibsen import LightCurve
from ibsen.get_obs_data import known_names

NORMALIZATION_INITIAL = 1E37
DAY = 86400.


def read_fits_write_txt(fits_name, txt_name, to_save=True,
                        x_axis_offset=0, to_return=False,
                        flux_type='flux', **plot_kwargs):
    """
    Reads the fits file obtaned by 
    lightcurve.write(fits_name, sed_type=<'flux' OR 'eflux'>)
    and exctracts the data: t+-dt, flux+- df.
    You can provide x_axis_offset to subtract from time.
    
        - if to_save: saves the data into text file txt_name.txt
        - if to_plot: plots the data. Use **plot_kwargs
        - `flux_type` species which format of the flux is in the .fits file:
            -- 'flux' [1 / s / cm2], default, OR
            -- 'eflux' [TeV / s / cm2] (the physical flux) which will be transformed 
                into [erg / s / cm2].
        - if to_return: returns a dict of np.arrays 
        {"t", "flux", "t_mjd", "dt", "df_p", "df_m"}

    """


    # read FLUXPOINTS table
    tbl = Table.read(fits_name, hdu="FLUXPOINTS")

    # columns: time vs flux (works whether or not they have units)
    time_min = getattr(tbl["time_min"], "value", tbl["time_min"])
    time_max = getattr(tbl["time_max"], "value", tbl["time_max"])
    time_mjd = 0.5 * (time_min + time_max)
    time = time_mjd - x_axis_offset
    dt = 0.5 * (time_max-time_min)
    
    if flux_type == 'flux':
        flux = np.squeeze(np.asarray(getattr(tbl["flux"], "value", tbl["flux"])))
        try:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_errp"], "value", tbl["flux_errp"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_errn"], "value", tbl["flux_errn"])))
        except:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            
    elif flux_type == 'eflux':
        flux = np.squeeze(np.asarray(getattr(tbl["eflux"], "value", tbl["eflux"])))
        try:
            dfp = np.squeeze(np.asarray(getattr(tbl["eflux_errp"], "value", tbl["eflux_errp"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["eflux_errn"], "value", tbl["eflux_errn"])))
        except:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
        ### manually transform TeV/s/cm2 --> erg/s/cm2
        flux, dfp, dfm = [ar_ * 1.6 for ar_ in (flux, dfp, dfm)]
    else: raise ValueError('flux_type can be only `flux` or `eflux`')

    if to_save:
        data_ = np.array([time, flux, time_mjd, dt, dfp, dfm]).T
        header_ = "t flux t_mjd dt df_p df_m"
        np.savetxt(txt_name+'.txt', data_, header=header_, delimiter=' ',
                   fmt='%s')
        np.savetxt(txt_name+'.dat', data_, header=header_, delimiter=' ',
                   fmt='%s')
        

        
    if to_return:
        return {"t": time, "flux":flux, "t_mjd":time_mjd, "dt":dt, 
                "df_p":dfp, "df_m": dfm}
    
def read_lightcurve_columns(
    path: str,
    *,
    # candidate names are checked case-insensitively; underscores/spaces ignored
    t_names=("t", "time", "mjd", "jd", "bjd", "tbjd"),
    dt_names=("dt", "deltat", "time_err", "timeerr", "sigma_t", "sigmat", "err_t"),
    f_names=("f", "flux", "flx", "rate", "counts", "count_rate"),
    df_names=("df", "dflux", "flux_err", "fluxerr", "sigma_f", "sigmaflux", 
              "err_f", "eflux", "df_p", "flux_errp", "eflux_errp"),
        ):
    """
    Read an ASCII table (with optional '#' commented header) and extract columns for:
      - t (time)
      - dt (time uncertainty) [optional -> NaNs if missing]
      - f (flux)
      - df (flux uncertainty) [optional -> NaNs if missing]

    Returns:
        t, dt, f, df, meta
    where meta includes:
        - 'table': the astropy Table
        - 'colmap': mapping of {'t': name|None, 'dt':..., 'f':..., 'df':...}
    """
    # Astropy can usually guess the format/delimiter; comment="#" handles typical headers.
    tbl: Table = ascii.read(
        path,
        guess=True,
        comment="#",
        fast_reader=True,
    )

    if len(tbl.colnames) == 0:
        raise ValueError("No columns detected. Is the file empty or not a table?")

    # Normalization helper for robust matching
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        return s

    colnames = list(tbl.colnames)
    norm_map = {name: norm(name) for name in colnames}

    def find_col(candidates) -> str | None:
        cand_norm = {norm(c) for c in candidates}
        # exact normalized match first
        for name, nn in norm_map.items():
            if nn in cand_norm:
                return name
        # looser: contains candidate token (e.g. "flux" in "raw_flux")
        for name, nn in norm_map.items():
            for c in cand_norm:
                if c and (c in nn):
                    return name
        return None

    t_col = find_col(t_names)
    dt_col = find_col(dt_names)
    f_col = find_col(f_names)
    df_col = find_col(df_names)

    # helpers to convert to float numpy arrays (handle masked columns)
    def to_float_array(colname: str) -> np.ndarray:
        col = tbl[colname]
        arr = np.array(col, dtype=float)
        # If masked, fill masked entries with NaN
        if hasattr(col, "mask"):
            arr = np.array(col.filled(np.nan), dtype=float)
        return arr

    if t_col is None:
        raise ValueError(f"Could not find time column. Tried names: {t_names}. Found: {colnames}")
    if f_col is None:
        raise ValueError(f"Could not find flux column. Tried names: {f_names}. Found: {colnames}")

    # Required (or possibly missing if require_* = False)
    t = to_float_array(t_col) if t_col is not None else np.full(len(tbl), np.nan)
    f = to_float_array(f_col) if f_col is not None else np.full(len(tbl), np.nan)

    # Optional -> NaNs
    dt = to_float_array(dt_col) if dt_col is not None else np.full(len(tbl), np.nan)
    df = to_float_array(df_col) if df_col is not None else np.full(len(tbl), np.nan)

    meta = {
        "table": tbl,
        "colmap": {"t": t_col, "dt": dt_col, "f": f_col, "df": df_col},
        "all_columns": colnames,
    }
    return t, dt, f, df, meta
        
def read_lightcurve_special_case(path):
    da = np.genfromtxt(path,
                   delimiter=' ', usecols=[0, 1, 4, 5, 3], 
                   names=['t', 'f', 'df_p', 'df_m', 'dt'])
    t, f, df_p, df_m, dt = [da[key] for key in ('t', 'f', 'df_p', 'df_m', 'dt')]
    df_p = np.nan_to_num(df_p)
    df_m = np.nan_to_num(df_m)
    df = 0.5 * (df_m + df_p)
    return t, f, df

def read_lightcurve(path):
    ext = Path(path).suffix.lower()
    if ext in ('.txt', '.dat', '.csv'):
        t, dt, f, df, _ = read_lightcurve_columns(path=path)
    elif ext == '.fits':
        res = read_fits_write_txt(fits_name = path, txt_name=None, plot_kwargs=None,
                                  to_save=False, x_axis_offset=0, to_return=True,
                                  flux_type='eflux')
        t, dt, f, df = [res[keyw] for keyw in ("t", "dt", "flux", "df_p")]
    else:
        raise ValueError("The format of the file not recognized.")
    _sort = np.argsort(t)
    t, dt, f, df = [ar[_sort] for ar in (t, dt, f, df)]
    return t, dt, f, df

def compute_lc(ts, bands, sys_name='psrb',
               ### stars params:
               b_ns_13=1,
               b_opt_13=1, 
               ### disk params:
               f_d=100, 
               delta=0.01,
               np_disk=3,
               height_exp=0.5,
               alpha_deg=18,
               incl_deg=30,
               ### ibs params:
               gamma_max=2, 
               n_ibs=13,
               s_max=1,
               ### electron spec parameters:
               cooling='stat_mimic',
               p_e=2,
               e_cut=1e13,
               eta_a=1,
               emin=1e10,
               emax=1e14,
               ### spectrum params
               method='simple',
               delta_pow=3,
               lorentz_boost=True,
               abs_gg=False,
               abs_photoel=True,
               ic_ani=False,
               nh_tbabs=0.8,
               to_parall=True,
               ):
    lc_ = LightCurve(times=ts, bands=bands, 
                     to_parall=to_parall, n_cores=10,
                        sys_name=sys_name, 
                        ### -----------------------------------------------
                        ns_b_ref=b_ns_13, ns_r_ref=1e13,
                        opt_b_ref=b_opt_13, opt_r_ref=1e13,
                        ### -----------------------------------------------
                        f_d=f_d, delta=delta, np_disk=np_disk,
                        height_exp=height_exp, alpha_deg=alpha_deg,
                        incl_deg=incl_deg,
                        ### -----------------------------------------------
                        gamma_max=gamma_max, s_max=s_max, n_ibs=n_ibs,
                        ### -----------------------------------------------
                        norm_e=NORMALIZATION_INITIAL,
                        cooling=cooling, p_e=p_e, ecut=e_cut,
                        eta_a=eta_a, emin=emin, emax=emax,
                        ### -----------------------------------------------
                        method=method, mechanisms=["syn", "ic"],
                        delta_power=delta_pow, lorentz_boost=lorentz_boost,
                        abs_gg=abs_gg, abs_photoel=abs_photoel,
                        ic_ani=ic_ani, nh_tbabs=nh_tbabs
                      )
    lc_.calculate()
    fluxes = lc_.fluxes
    return fluxes


class LightCurveWindow(ToolWindowBase):
    TOOL_NAME = "Light curve viewer"
    IS_HEAVY = True
    DEBOUNCE_MS = 250

    def __init__(self):
        super().__init__()

        # ---------- Internal state ----------
        self.bands = []         # list of dicts, one per row
        self.axes = []          # matplotlib Axes list, one per band
        self.model_lines = []   # one line per band
        self._bands_dirty = False

        # ---------- Plot area ----------
        # Base gives self.fig and a default self.ax; we will not use self.ax
        # We'll manage our own axes stack in rebuild_axes()
        self.fig.clf()
        self.canvas.draw_idle()

        # ---------- Right panel UI ----------
        self._build_controls()

        # Menu
        self._setup_menu()

        # Wire heavy controls from base
        assert self.apply_btn is not None
        self.apply_btn.clicked.connect(self.update_plot)  # base expects update_plot
        assert self.auto_cb is not None
        self.auto_cb.setChecked(False)
        self.fit_row = None  # or -1


        # Start with 1 band
        self.add_band_row(emin=1e3, emax=1e4)

        # Build axes + first model
        self.rebuild_axes()
        self.update_plot()

    # ---------------- Controls (right panel) ----------------
    def _build_controls(self):
        """
        Build the right-panel controls:
          - Top row: sys_name dropdown + to_parall checkbox
          - Time grid sliders (always visible)
          - Bands table (compact)
          - Accordion (QToolBox) with 5 categories:
              Stars / Disk / IBS / e-spec / Spec
        Defaults match compute_lc() signature.
        """
        insert_at = 1  # right after the Tool title label created by ToolWindowBase
    
        # ---------- Top row: sys_name + to_parall ----------
        top_row_w = QWidget()
        top_row = QHBoxLayout(top_row_w)
        top_row.setContentsMargins(0, 0, 0, 0)
    
        top_row.addWidget(QLabel("sys_name:"))
        self.sys_name = QComboBox()
        self.sys_name.addItems(known_names)
        self.sys_name.setCurrentText("psrb")
        top_row.addWidget(self.sys_name, 1)
    
        self.to_parall = QCheckBox("parallel")
        self.to_parall.setChecked(True)  # default: to_parall=True
        top_row.addWidget(self.to_parall)
    
        self.controls_layout.insertWidget(insert_at, top_row_w)
        insert_at += 1
    
        self.sys_name.currentIndexChanged.connect(lambda _: self.schedule_update())
        self.to_parall.stateChanged.connect(lambda _: self.schedule_update())
    
        # ---------- Time grid (always visible) ----------
        self.controls_layout.insertWidget(insert_at, QLabel("<b>Time grid</b>"))
        insert_at += 1
    
        lay, self.tmin_days = self.make_linear_slider("t_min [days]", -200.0, 200.0, 1.0, -170.0)
        self.controls_layout.insertLayout(insert_at, lay); insert_at += 1
    
        lay, self.tmax_days = self.make_linear_slider("t_max [days]", -200.0, 200.0, 1.0, 110.0)
        self.controls_layout.insertLayout(insert_at, lay); insert_at += 1
    
        lay, self.nt = self.make_linear_slider("N(t)", 10.0, 400.0, 1.0, 150.0)
        self.controls_layout.insertLayout(insert_at, lay); insert_at += 1
    
        # ---------- Bands table ----------
        self.controls_layout.insertWidget(insert_at, QLabel("<b>Bands</b>"))
        insert_at += 1
    
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Emin [eV]", "Emax [eV]",
            "Upload", "Clear",
            "Fit norm", "Fit to dataset",
            "N data"
        ])
        self.table.verticalHeader().setVisible(False)
    
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
    
        # compact: ~3 rows visible
        rows_visible = 3
        h = self.table.horizontalHeader().height() + int(rows_visible * self.table.verticalHeader().defaultSectionSize()) + 8
        self.table.setMaximumHeight(h)
    
        self.controls_layout.insertWidget(insert_at, self.table)
        insert_at += 1
    
        # Buttons under table
        btn_row_widget = QWidget()
        btn_row = QHBoxLayout(btn_row_widget)
        btn_row.setContentsMargins(0, 0, 0, 0)
    
        self.add_band_btn = QPushButton("Add band")
        self.del_band_btn = QPushButton("Delete selected")
        self.clear_all_btn = QPushButton("Clear all obs")
    
        btn_row.addWidget(self.add_band_btn)
        btn_row.addWidget(self.del_band_btn)
        btn_row.addWidget(self.clear_all_btn)
    
        self.controls_layout.insertWidget(insert_at, btn_row_widget)
        insert_at += 1
    
        # ---------- Accordion: Parameters ----------
        self.controls_layout.insertWidget(insert_at, QLabel("<b>Parameters</b>"))
        insert_at += 1
    
        self.param_box = QToolBox()
        self.controls_layout.insertWidget(insert_at, self.param_box)
        insert_at += 1
    
        def _page_widget(title: str) -> QVBoxLayout:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(6)
            self.param_box.addItem(w, title)
            return lay
    
        def _wire_slider(sl):
            sl.valueChanged.connect(lambda _, ss=sl: (self.update_slider_label(ss), self.schedule_update()))
    
        # ----- Stars parameters -----
        lay_stars = _page_widget("Stars parameters")
    
        lay, self.b_ns_13 = self.make_log10_slider("b_ns_13", 0.01, 100.0, 0.003, 0.53)
        lay_stars.addLayout(lay); _wire_slider(self.b_ns_13)
    
        lay, self.b_opt_13 = self.make_log10_slider("b_opt_13", 0.01, 100.0, 0.1, 0.01)
        lay_stars.addLayout(lay); _wire_slider(self.b_opt_13)
    
        # ----- Disk parameters -----
        lay_disk = _page_widget("Disk parameters")
    
        lay, self.f_d = self.make_log10_slider("f_d", 1.0, 10000.0, 0.001, 470.0)
        lay_disk.addLayout(lay); _wire_slider(self.f_d)
    
        lay, self.delta = self.make_log10_slider("delta", 1e-4, 3e-1, 0.001, 0.023)
        lay_disk.addLayout(lay); _wire_slider(self.delta)
    
        lay, self.np_disk = self.make_linear_slider("np_disk", 1.0, 6.0, 0.1, 3.0)
        lay_disk.addLayout(lay); _wire_slider(self.np_disk)
    
        lay, self.height_exp = self.make_linear_slider("height_exp", 0.0, 2.0, 0.05, 0.5)
        lay_disk.addLayout(lay); _wire_slider(self.height_exp)
    
        lay, self.alpha_deg = self.make_linear_slider("alpha_deg", -90.0, 90.0, 
                                                      1.0, 18.0)
        lay_disk.addLayout(lay); _wire_slider(self.alpha_deg)
    
        lay, self.incl_deg = self.make_linear_slider("incl_deg", 0.0, 90.0, 1.0, 30.0)
        lay_disk.addLayout(lay); _wire_slider(self.incl_deg)
    
        # ----- IBS parameters -----
        lay_ibs = _page_widget("IBS parameters")
    
        lay, self.gamma_max = self.make_linear_slider("gamma_max", 1.0001, 5.0, 0.01, 2.3)
        lay_ibs.addLayout(lay); _wire_slider(self.gamma_max)
    
        lay, self.s_max = self.make_linear_slider("s_max", 0.5, 4.0, 0.01, 1.0)
        lay_ibs.addLayout(lay); _wire_slider(self.s_max)
        
        lay, self.n_ibs = self.make_linear_slider("n_ibs", 5, 40, 1, 15)
        lay_ibs.addLayout(lay); _wire_slider(self.n_ibs)
        
        # ----- e-spec parameters -----
        lay_espec = _page_widget("e-spec parameters")
    
        lay_espec.addWidget(QLabel("cooling"))
        self.cooling = QComboBox()
        self.cooling.addItems([
            "no", "stat_apex", "stat_ibs", "stat_mimic",
            "leak_apex", "leak_ibs", "leak_mimic",
            "adv"
        ])
        self.cooling.setCurrentText("stat_ibs")
        lay_espec.addWidget(self.cooling)
        self.cooling.currentIndexChanged.connect(lambda _: self.schedule_update())
    
        lay, self.p_e = self.make_linear_slider("p_e", 1.0, 3.0, 0.01, 1.7)
        lay_espec.addLayout(lay); _wire_slider(self.p_e)
    
        lay, self.e_cut = self.make_log10_slider("e_cut [eV]", 1e11, 1e14, 0.05, 7e12)
        lay_espec.addLayout(lay); _wire_slider(self.e_cut)
    
        lay, self.eta_a = self.make_log10_slider("eta_a", 0.01, 100.0, 0.02, 0.03)
        lay_espec.addLayout(lay); _wire_slider(self.eta_a)
    
        lay, self.emin = self.make_log10_slider("emin [eV]", 1e8, 1e11, 0.05, 1e9)
        lay_espec.addLayout(lay); _wire_slider(self.emin)
    
        lay, self.emax = self.make_log10_slider("emax [eV]", 1e11, 5e14, 0.05, 1e13)
        lay_espec.addLayout(lay); _wire_slider(self.emax)
    
        # ----- Spec parameters -----
        lay_spec = _page_widget("Spec parameters")
    
        lay_spec.addWidget(QLabel("method"))
        self.method = QComboBox()
        self.method.addItems(["simple", "full", "apex"])
        self.method.setCurrentText("simple")
        lay_spec.addWidget(self.method)
        self.method.currentIndexChanged.connect(lambda _: self.schedule_update())
    
        lay, self.delta_pow = self.make_linear_slider("delta_pow", 0.0, 6.0, 0.1, 3.0)
        lay_spec.addLayout(lay); _wire_slider(self.delta_pow)
    
        self.lorentz_boost = QCheckBox("lorentz_boost")
        self.lorentz_boost.setChecked(True)
        lay_spec.addWidget(self.lorentz_boost)
        self.lorentz_boost.stateChanged.connect(lambda _: self.schedule_update())
    
        self.abs_gg = QCheckBox("abs_gg")
        self.abs_gg.setChecked(False)
        lay_spec.addWidget(self.abs_gg)
        self.abs_gg.stateChanged.connect(lambda _: self.schedule_update())
    
        self.abs_photoel = QCheckBox("abs_photoel")
        self.abs_photoel.setChecked(True)
        lay_spec.addWidget(self.abs_photoel)
        self.abs_photoel.stateChanged.connect(lambda _: self.schedule_update())
    
        self.ic_ani = QCheckBox("ic_ani")
        self.ic_ani.setChecked(False)
        lay_spec.addWidget(self.ic_ani)
        self.ic_ani.stateChanged.connect(lambda _: self.schedule_update())
    
        # nh_tbabs: linear-ish; keep reasonable range
        lay, self.nh_tbabs = self.make_linear_slider("nh_tbabs", 0.0, 5.0, 0.05, 0.8)
        lay_spec.addLayout(lay); _wire_slider(self.nh_tbabs)
    
        # Start with the first page open
        self.param_box.setCurrentIndex(0)
    
        # Wire time sliders
        for s in (self.tmin_days, self.tmax_days, self.nt):
            _wire_slider(s)
    
        # Connect band buttons and table signals
        self.add_band_btn.clicked.connect(lambda: self.add_band_row(emin=1e3, emax=1e4))
        self.del_band_btn.clicked.connect(self.delete_selected_band)
        self.clear_all_btn.clicked.connect(self.clear_all_observations)
        self.table.itemChanged.connect(self._on_table_item_changed)

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # ---------------- Band table management ----------------
    def add_band_row(self, emin: float, emax: float):
        self.table.blockSignals(True)
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(f"{emin:g}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{emax:g}"))

            up = QPushButton("Upload…")
            up.setProperty("row", row)
            up.clicked.connect(self._upload_clicked)
            self.table.setCellWidget(row, 2, up)

            cl = QPushButton("Clear")
            cl.setProperty("row", row)
            cl.clicked.connect(self._clear_band_clicked)
            self.table.setCellWidget(row, 3, cl)

            fit_cb = QCheckBox()
            fit_cb.setChecked(False)
            fit_cb.setProperty("row", row)
            fit_cb.stateChanged.connect(self._fit_cb_changed)
            self.table.setCellWidget(row, 4, fit_cb)

            fit_dd = QComboBox()
            fit_dd.setProperty("row", row)
            fit_dd.setEnabled(False)
            fit_dd.currentIndexChanged.connect(self._fit_dd_changed)
            self.table.setCellWidget(row, 5, fit_dd)

            itN = QTableWidgetItem("0")
            itN.setFlags(itN.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, itN)

            self.bands.append({
                "emin": float(emin),
                "emax": float(emax),
                "datasets": [],
                "fit_enabled": False,
                "fit_dataset_name": None,
                "ax": None,
                "model_line": None,
            })

            self._reindex_row_widgets()
        finally:
            self.table.blockSignals(False)

        # structural change -> rebuild axes
        self.rebuild_axes()
        self.schedule_update()

    def delete_selected_band(self):
        row = self.table.currentRow()
        if row < 0:
            return

        # delete artists
        self._delete_band(row)

        self.table.removeRow(row)
        self.bands.pop(row)

        self._reindex_row_widgets()

        # structural change -> rebuild axes
        self.rebuild_axes()
        self.schedule_update()

    def _delete_band(self, row: int):
        for ds in self.bands[row]["datasets"]:
            for art in ds["artists"]:
                try:
                    art.remove()
                except Exception:
                    pass

    def _reindex_row_widgets(self):
        for r in range(self.table.rowCount()):
            for col in (2, 3, 4, 5):
                w = self.table.cellWidget(r, col)
                if w is not None:
                    w.setProperty("row", r)

    def _on_table_item_changed(self, _item):
        # no validation here
        self._bands_dirty = True
        self.schedule_update()

    # ---------------- Axes rebuild ----------------
    def rebuild_axes(self):
        self.fig.clf()
        self.axes = []
        self.model_lines = []

        nb = len(self.bands)
        if nb == 0:
            self.canvas.draw_idle()
            return

        gs = self.fig.add_gridspec(nb, 1, hspace=0.15)

        for i in range(nb):
            if i == 0:
                ax = self.fig.add_subplot(gs[i, 0])
            else:
                ax = self.fig.add_subplot(gs[i, 0], sharex=self.axes[0])

            ax.grid(True, alpha=0.3)
            if i < nb - 1:
                ax.tick_params(labelbottom=False)

            (line,) = ax.plot([], [], label="Model")

            self.axes.append(ax)
            self.model_lines.append(line)

            self.bands[i]["ax"] = ax
            self.bands[i]["model_line"] = line
            self._update_axis_label(i)

        self.axes[-1].set_xlabel("t [days]")
        self.canvas.draw_idle()

        self._redraw_all_observations()

    def _read_and_validate_bands(self):
        bands = []
        for row in range(self.table.rowCount()):
            it0 = self.table.item(row, 0)
            it1 = self.table.item(row, 1)
            if it0 is None or it1 is None:
                raise ValueError(f"Band {row+1}: missing Emin/Emax")

            s0 = it0.text().strip()
            s1 = it1.text().strip()

            try:
                emin = float(s0)
                emax = float(s1)
            except ValueError:
                raise ValueError(
                    f"Band {row+1}: cannot parse Emin/Emax as numbers (got '{s0}', '{s1}')"
                )

            if emin <= 0 or emax <= 0:
                raise ValueError(f"Band {row+1}: energies must be positive")
            if emin >= emax:
                raise ValueError(f"Band {row+1}: Emin must be smaller than Emax (got {emin:g} ≥ {emax:g})")

            bands.append((emin, emax))
        return bands

    def _update_axis_label(self, row: int):
        ax = self.bands[row].get("ax")
        if ax is None:
            return
        emin = self.bands[row]["emin"]
        emax = self.bands[row]["emax"]
        ax.set_ylabel(f"Flux\n{emin:.2g}-{emax:.2g} eV")

    # ---------------- Observations ----------------
    def _upload_clicked(self):
        btn = self.sender()
        row = int(btn.property("row"))
        self.upload_obs_for_band(row)

    def _clear_band_clicked(self):
        btn = self.sender()
        row = int(btn.property("row"))
        self.clear_observations_for_band(row)

    def upload_obs_for_band(self, row: int):
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload light curve data", "",
            "Data (*.txt *.dat *.csv *.fits *.fit);;All files (*)"
        )
        if not path:
            return

        t, dt, y, dy = read_lightcurve(path)  # your function
        name = self._nice_label(path)

        artists = self._plot_obs(row, t, dt, y, dy, label=name)

        self.bands[row]["datasets"].append({
            "name": name,
            "t": np.asarray(t),
            "y": np.asarray(y),
            "dy": None if dy is None else np.asarray(dy),
            "dt": None if dt is None else np.asarray(dt),
            "artists": artists
        })

        self.table.item(row, 6).setText(str(len(self.bands[row]["datasets"])))
        self._refresh_fit_dropdown(row)

        self.canvas.draw_idle()

    def _plot_obs(self, row: int, t, dt, y, dy, label: str):
        ax = self.bands[row]["ax"]
        if ax is None:
            return []

        t = np.asarray(t)
        y = np.asarray(y)
        m = np.isfinite(t) & np.isfinite(y)
        if dy is not None:
            dy = np.asarray(dy)
            m = m & np.isfinite(dy) & (dy >= 0)
            dy = dy[m]
        t, y = t[m], y[m]

        if dy is None:
            sc = ax.scatter(t / DAY, y, s=18, label=label)
            ax.legend(loc="best")
            return [sc]
        transpar = max(0.2, min(0.8, 50/t.size))
        cont = ax.errorbar(t, y, yerr=dy, xerr=dt, fmt="o", capsize=2, label=label,
                           alpha=transpar)
        line, caplines, barlinecols = cont
        ax.legend(loc="best")
        return [line] + list(caplines) + list(barlinecols)

    def clear_observations_for_band(self, row: int):
        for ds in self.bands[row]["datasets"]:
            for art in ds["artists"]:
                try:
                    art.remove()
                except Exception:
                    pass

        self.bands[row]["datasets"].clear()
        self.table.item(row, 6).setText("0")
        self._refresh_fit_dropdown(row, cleared=True)
        self.canvas.draw_idle()

    def clear_all_observations(self):
        for r in range(len(self.bands)):
            self.clear_observations_for_band(r)

    def _redraw_all_observations(self):
        for row, band in enumerate(self.bands):
            for ds in band["datasets"]:
                ds["artists"] = self._plot_obs(row, t=ds["t"], y=ds["y"],
                                               dy=ds["dy"],dt=ds["dt"],
                                               label=ds["name"])
            self._refresh_fit_dropdown(row)

    # ---------------- Per-band fit norm UI ----------------
    def _fit_cb_changed(self, state):
        cb = self.sender()
        row = int(cb.property("row"))
        checked = (state == Qt.Checked)
    
        if checked:
            self.fit_row = row
            self._uncheck_all_fit_checkboxes_except(row)
            self._update_fit_controls_enabled()
    
            dd: QComboBox = self.table.cellWidget(row, 5)
            band = self.bands[row]
            # default select first dataset if available
            if dd is not None and dd.count() > 0 and not band.get("fit_dataset_name"):
                band["fit_dataset_name"] = dd.itemText(0)
                dd.blockSignals(True)
                dd.setCurrentIndex(0)
                dd.blockSignals(False)
        else:
            if self.fit_row == row:
                self.fit_row = None
            self._update_fit_controls_enabled()
    
        # IMPORTANT: don't depend on schedule_update behavior
        if self.auto_cb.isChecked():
            self.schedule_update()
       
    def _uncheck_all_fit_checkboxes_except(self, keep_row: int):
        for r in range(self.table.rowCount()):
            cb: QCheckBox = self.table.cellWidget(r, 4)
            if cb is None:
                continue
            if r != keep_row and cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                
    def _update_fit_controls_enabled(self):
        # Enable only the dropdown for fit_row (if it exists and has datasets)
        for r in range(self.table.rowCount()):
            dd: QComboBox = self.table.cellWidget(r, 5)
            cb: QCheckBox = self.table.cellWidget(r, 4)
            if dd is None or cb is None:
                continue
    
            if self.fit_row == r and cb.isChecked() and len(self.bands[r]["datasets"]) > 0:
                dd.setEnabled(True)
            else:
                dd.setEnabled(False)
            
    def _fit_dd_changed(self, _):
        dd = self.sender()
        row = int(dd.property("row"))
        if self.fit_row != row:
            return
        name = dd.currentText().strip() if dd.currentText() else None
        self.bands[row]["fit_dataset_name"] = name if name else None
        if self.auto_cb.isChecked():
            self.schedule_update()
        else:
            self.status_lbl.setText("Fit dataset changed — press Apply (or enable Auto update).")
            
    def _get_fit_selection_from_table(self):
        """Return (fit_row, dataset_name) or (None, None)."""
        for r in range(self.table.rowCount()):
            cb = self.table.cellWidget(r, 4)  # Fit norm checkbox column
            if cb is not None and cb.isChecked():
                dd = self.table.cellWidget(r, 5)  # dropdown column
                name = dd.currentText().strip() if (dd is not None and dd.currentText()) else None
                return r, name
        return None, None

    def _refresh_fit_dropdown(self, row: int, cleared: bool = False):
        dd: QComboBox = self.table.cellWidget(row, 5)
        cb: QCheckBox = self.table.cellWidget(row, 4)

        dd.blockSignals(True)
        dd.clear()
        for ds in self.bands[row]["datasets"]:
            dd.addItem(ds["name"])
        dd.blockSignals(False)

        if cleared or len(self.bands[row]["datasets"]) == 0:
            self.bands[row]["fit_dataset_name"] = None
            dd.setEnabled(False)
            cb.setChecked(False)
            self.bands[row]["fit_enabled"] = False
        else:
            if self.bands[row]["fit_dataset_name"] is None:
                self.bands[row]["fit_dataset_name"] = self.bands[row]["datasets"][0]["name"]
                dd.setCurrentIndex(0)
            dd.setEnabled(cb.isChecked())

    # ---------------- Base hook: schedule_update & update_plot ----------------
    def schedule_update(self):
        # Use ToolWindowBase's schedule_update (debounced when auto is on)
        super().schedule_update()

    def update_plot(self):
        # (ToolWindowBase expects update_plot name)
        if self.status_lbl is not None:
            self.status_lbl.setText("Computing…")

        # Validate + commit bands on Apply/auto
        try:
            bands_valid = self._read_and_validate_bands()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid band", str(e))
            if self.status_lbl is not None:
                self.status_lbl.setText("Error")
            return

        for i, (emin, emax) in enumerate(bands_valid):
            self.bands[i]["emin"] = emin
            self.bands[i]["emax"] = emax
            self._update_axis_label(i)

        # Build time grid from sliders
        tmin = float(self.slider_value(self.tmin_days)) * DAY
        tmax = float(self.slider_value(self.tmax_days)) * DAY
        n = int(round(self.slider_value(self.nt)))
        n = max(2, n)

        if tmax <= tmin:
            QMessageBox.warning(self, "Invalid time grid", "t_max must be > t_min.")
            if self.status_lbl is not None:
                self.status_lbl.setText("Error")
            return

        ts = np.linspace(tmin, tmax, n)
        bands = [(b["emin"], b["emax"]) for b in self.bands]

        try:

            print("Computing LC...")
            fluxes = compute_lc(ts, bands, 
                        sys_name=self.sys_name.currentText(),
                        to_parall=self.to_parall.isChecked(),
                        b_ns_13=float(self.slider_value(self.b_ns_13)),
                        b_opt_13=float(self.slider_value(self.b_opt_13)),
                        f_d=float(self.slider_value(self.f_d)),
                        delta=float(self.slider_value(self.delta)),
                        np_disk=float(self.slider_value(self.np_disk)),
                        height_exp=float(self.slider_value(self.height_exp)),
                        alpha_deg=float(self.slider_value(self.alpha_deg)),
                        incl_deg=float(self.slider_value(self.incl_deg)),
                        gamma_max=float(self.slider_value(self.gamma_max)),
                        s_max=float(self.slider_value(self.s_max)),
                        n_ibs=int(self.slider_value(self.n_ibs)),
                        cooling=self.cooling.currentText(),
                        p_e=float(self.slider_value(self.p_e)),
                        e_cut=float(self.slider_value(self.e_cut)),
                        eta_a=float(self.slider_value(self.eta_a)),
                        emin=float(self.slider_value(self.emin)),
                        emax=float(self.slider_value(self.emax)),
                        method=self.method.currentText(),
                        delta_pow=float(self.slider_value(self.delta_pow)),
                        nh_tbabs=float(self.slider_value(self.nh_tbabs)),
                        abs_gg=self.abs_gg.isChecked(),
                        ic_ani=self.ic_ani.isChecked(),
                        abs_photoel=self.abs_photoel.isChecked(),
                        lorentz_boost=self.lorentz_boost.isChecked(),
                           )

            opt_norm = NORMALIZATION_INITIAL
            
            fit_row, fit_name = self._get_fit_selection_from_table()
            
            if fit_row is not None and fit_name:
                band = self.bands[fit_row]
                ds = next((d for d in band["datasets"] if d["name"] == fit_name), None)
                if ds is not None:
                    y_fit_band = fluxes[:, fit_row]
            
                    opt_norm, _ = fit_norm_here(
                        x_obs=ds["t"], y_obs=ds["y"], dy_obs=ds["dy"],
                        x_model=ts/DAY, y_model=y_fit_band,
                        norm_init=NORMALIZATION_INITIAL,
                        grid_scale="linear",
                        return_err=False
                    )

            for i in range(len(self.bands)):
                print('replotting...')
                y_model = opt_norm / NORMALIZATION_INITIAL * fluxes[:, i]
                self.model_lines[i].set_data(ts / DAY, y_model)
                self.axes[i].relim()
                self.axes[i].autoscale_view()
            print("Ready!")
            self.canvas.draw_idle()
            if self.status_lbl is not None:
                self.status_lbl.setText("Ready")

        except Exception as e:
            if self.status_lbl is not None:
                self.status_lbl.setText(f"Error: {type(e).__name__}: {e}")

    # ---------------- small helpers ----------------
    def _nice_label(self, path: str) -> str:
        import os
        return os.path.basename(path)