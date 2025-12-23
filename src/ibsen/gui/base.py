from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QSlider, QPushButton, QCheckBox
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from ibsen.utils import interplg, fit_norm
from scipy.interpolate import interp1d

def fit_norm_here(x_obs, y_obs, dy_obs, x_model, y_model, norm_init,
                  grid_scale='log', return_err=False):
    """
    Fits the model normalizatoin: (x_model, y_model) to the observations:
        (x_obs, y_obs, dy_obs).
    
    y_model should have been calculated with norm_init. 
    
    The fit is performed over only these observational points x_obs_i which are inside
    the model: min(x_model) < x_obs_i < max(x_model).
    
    grid_scale : str, one of: {'log', 'linear'} --- shows how to interpolate
    model onto x_obs. Default 'log'.
    
    
    Returns
    -------
    norm_opt : float
        Optimal normalization.
    y_model_renormalized : np.ndarray of length y_model.size
        Renormalized model y.
        
    and, if return_err==True,
    dn_rel (float), relative error for N,
    y_low, y_high: np.arrays of size y_model.size, confidence interval boundaries.
    """
    
    obs_ok = (np.isfinite(x_obs) & np.isfinite(y_obs) & np.isfinite(dy_obs) 
              & (x_obs > np.min(x_model)) & (x_obs < np.max(x_model)) )
    x_obs, y_obs, dy_obs = [ar[obs_ok] for ar in (x_obs, y_obs, dy_obs)]
    model_ok = np.isfinite(x_model) & np.isfinite(y_model)
    x_model, y_model = [ar[model_ok] for ar in (x_model, y_model)]
    y_model_normalized = y_model  / norm_init

    if grid_scale.lower() in ('linear', 'lin'):
        interp_ = interp1d(x=x_model, y=y_model_normalized, bounds_error="extrapolate",
                fill_value=(y_model_normalized[0], y_model_normalized[-1]))
        y_model_normalized_in_xobs = interp_(x_obs)
    elif grid_scale.lower() in ('log', 'log10'):
        y_model_normalized_in_xobs = interplg(x_obs, x_model, 
                            y_model_normalized, bounds_error="extrapolate",
                fill_value=(np.log10(y_model_normalized[0]), 
                            np.log10(y_model_normalized[-1])))
    else:
        raise ValueError("grid_scale should be one of: 'lin', 'log'")
    if not return_err:
        norm_opt, y_model_opt = fit_norm(ydata = y_obs, dy_data = dy_obs, 
                                         y0_normalized=y_model_normalized_in_xobs)
        return norm_opt, y_model * norm_opt / norm_init
    if return_err:
        norm_opt, y_model_opt, dnorm_opt = fit_norm(ydata = y_obs, dy_data = dy_obs, 
                                         y0_normalized=y_model_normalized_in_xobs,
                                         return_err=True)
        dn_rel = dnorm_opt / norm_opt
        y_model_renorm = y_model * norm_opt / norm_init
        y_low, y_high = y_model_renorm * (1 - dn_rel), y_model_renorm * (1 + dn_rel)
        
        return norm_opt, y_model_renorm, dn_rel, y_low, y_high
    

@dataclass
class SliderMeta:
    name: str
    mode: str              # "linear_step" or "log10"
    vmin: float
    step: float            # for linear_step
    scale: int             # for log10
    initial: float
    
class GradientPlot:
    def __init__(self, fig, ax, *, cmap="coolwarm", lw=2, ls="-", alpha=1.0,
                 scatter=False, marker="o", s=20, colorbar=False, cbar_label="grad"):
        self.fig = fig
        self.ax = ax
        self.cmap = cmap
        self.lw = lw
        self.ls = ls
        self.alpha = alpha
        self.scatter = scatter
        self.marker = marker
        self.s = s
        self.want_colorbar = colorbar
        self.cbar_label = cbar_label

        self.artist = None   # LineCollection or PathCollection
        self.cbar = None

    def _clean_data(self, x, y, p):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        p = np.asarray(p).ravel()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
        return x[m], y[m], p[m]

    def update(self, x, y, p, *, vmin=None, vmax=None, autoscale=True):
        x, y, p = self._clean_data(x, y, p)
        if x.size == 0:
            return

        vmin_here = np.min(p) if vmin is None else vmin
        vmax_here = np.max(p) if vmax is None else vmax
        norm = Normalize(vmin=vmin_here, vmax=vmax_here)

        if self.scatter or x.size < 2:
            # --- scatter mode ---
            if self.artist is None or self.artist.__class__.__name__ != "PathCollection":
                # first time: create
                if self.artist is not None:
                    self.artist.remove()
                self.artist = self.ax.scatter(
                    x, y, c=p, cmap=self.cmap, norm=norm,
                    s=self.s, marker=self.marker, linewidths=0, alpha=self.alpha
                )
            else:
                # update existing scatter
                self.artist.set_offsets(np.c_[x, y])
                self.artist.set_array(p)
                self.artist.set_norm(norm)

        else:
            # --- gradient line via LineCollection ---
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if self.artist is None or not isinstance(self.artist, LineCollection):
                if self.artist is not None:
                    self.artist.remove()
                lc = LineCollection(segments, cmap=self.cmap, norm=norm)
                lc.set_array(p[:-1])
                lc.set_linewidth(self.lw)
                lc.set_linestyle(self.ls)
                lc.set_alpha(self.alpha)
                self.artist = self.ax.add_collection(lc)
            else:
                self.artist.set_segments(segments)
                self.artist.set_array(p[:-1])
                self.artist.set_norm(norm)
                # linewidth/linestyle/alpha can also be updated if you want:
                # self.artist.set_linewidth(self.lw)

        # Axis limits
        if autoscale:
            # For LineCollection, autoscale needs the collection to be considered:
            self.ax.relim()
            self.ax.autoscale_view()

        # Colorbar: create once, then update
        if self.want_colorbar:
            if self.cbar is None:
                self.cbar = self.fig.colorbar(self.artist, ax=self.ax, label=self.cbar_label)
            else:
                self.cbar.update_normal(self.artist)


class ToolWindowBase(QMainWindow):
    """
    Base window with:
      - splitter: plot on the left, controls on the right
      - matplotlib figure/canvas/axes
      - slider helper methods (linear and log10)
      - optional heavy-tool update controls (Apply + Auto + debounce)
    """

    TOOL_NAME: str = "Tool"
    IS_HEAVY: bool = False            # set True for SED/LC; False for IBS
    DEBOUNCE_MS: int = 250

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.TOOL_NAME)

        # Core UI
        self._build_split_ui()

        # heavy controls (optional)
        self.apply_btn: Optional[QPushButton] = None
        self.auto_cb: Optional[QCheckBox] = None
        self.status_lbl: Optional[QLabel] = None
        self._debounce: Optional[QTimer] = None

        if self.IS_HEAVY:
            self._build_heavy_controls()

        # storage for slider metas
        self._slider_meta: dict[QSlider, SliderMeta] = {}

    # ---------- Layout scaffold ----------
    def _build_split_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        self.splitter = QSplitter(Qt.Horizontal)
        main.addWidget(self.splitter)

        # left: plot container
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        plot_layout.addWidget(self.canvas)
        self.splitter.addWidget(self.plot_widget)

        # right: controls container
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.addWidget(QLabel(f"<b>{self.TOOL_NAME}</b>"))
        self.controls_layout.addStretch(1)
        self.splitter.addWidget(self.controls_widget)

        self.splitter.setSizes([900, 320])

    def _build_heavy_controls(self) -> None:
        # Put heavy controls at the bottom of the right panel.
        # We temporarily remove the stretch and add it back.
        self._remove_last_stretch(self.controls_layout)

        self.apply_btn = QPushButton("Apply")
        self.auto_cb = QCheckBox("Auto update (debounced)")
        self.auto_cb.setChecked(False)
        self.status_lbl = QLabel("")

        self.controls_layout.addWidget(self.apply_btn)
        self.controls_layout.addWidget(self.auto_cb)
        self.controls_layout.addWidget(self.status_lbl)
        self.controls_layout.addStretch(1)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self.update_plot)

    @staticmethod
    def _remove_last_stretch(layout: QVBoxLayout) -> None:
        # crude but effective: remove the last item if it's a stretch
        count = layout.count()
        if count <= 0:
            return
        item = layout.itemAt(count - 1)
        if item is not None and item.spacerItem() is not None:
            layout.takeAt(count - 1)

    # ---------- Slider helpers ----------
    def make_linear_slider(self, name: str, vmin: float, vmax: float, step: float, initial: float) -> tuple[QVBoxLayout, QSlider]:
        """
        Linear slider with arbitrary step (works for step>1 too).
        Uses integer index i: value = vmin + i*step.
        """
        if step <= 0:
            raise ValueError("step must be > 0")
        n_steps = int(round((vmax - vmin) / step))
        if n_steps <= 0:
            raise ValueError("Bad slider range/step")

        initial = float(np.clip(initial, vmin, vmax))
        iinit = int(round((initial - vmin) / step))

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

        self._slider_meta[slider] = SliderMeta(name=name, mode="linear_step", vmin=vmin, step=step, scale=0, initial=initial)
        slider._label = title  # type: ignore[attr-defined]

        return layout, slider

    def make_log10_slider(self, name: str, vmin: float, vmax: float, step_log10: float, initial: float) -> tuple[QVBoxLayout, QSlider]:
        """
        Slider uniform in log10(value).
        """
        log_min = float(np.log10(vmin))
        log_max = float(np.log10(vmax))
        log_init = float(np.log10(initial))

        scale = int(round(1.0 / step_log10))
        if scale <= 0:
            raise ValueError("step_log10 too large")

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

        self._slider_meta[slider] = SliderMeta(name=name, mode="log10", vmin=0.0, step=0.0, scale=scale, initial=float(initial))
        slider._label = title  # type: ignore[attr-defined]

        return layout, slider

    def slider_value(self, slider: QSlider) -> float:
        meta = self._slider_meta[slider]
        if meta.mode == "linear_step":
            return meta.vmin + slider.value() * meta.step
        if meta.mode == "log10":
            return float(10 ** (slider.value() / meta.scale))
        raise ValueError("Unknown slider mode")

    def update_slider_label(self, slider: QSlider) -> None:
        meta = self._slider_meta[slider]
        v = self.slider_value(slider)
        slider._label.setText(f"{meta.name} = {v:.4g}")  # type: ignore[attr-defined]

    # ---------- Update scheduling ----------
    def connect_control(self, widget, callback: Callable[[], None]) -> None:
        """
        Convenience: connect control changes to update scheduling.
        For heavy tools: schedules/debounces if auto is on.
        For fast tools: updates immediately.
        """
        def _trigger(*_):
            callback()

        # You’ll call widget.signal.connect(...) in your tool.
        # This helper is just a pattern — use schedule_update() below.
        raise NotImplementedError("Use schedule_update() directly from your signal handlers.")

    def schedule_update(self) -> None:
        if not self.IS_HEAVY:
            self.update_plot()
            return

        assert self.auto_cb is not None and self._debounce is not None
        if self.auto_cb.isChecked():
            self._debounce.start()

    # ---------- To be implemented by subclasses ----------
    def update_plot(self) -> None:
        raise NotImplementedError

