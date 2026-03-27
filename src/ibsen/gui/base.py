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

    

@dataclass
class SliderMeta:
    name: str
    mode: str              # "linear_step" or "log10"
    vmin: float
    step: float            # for linear_step
    scale: int             # for log10
    initial: float


class ToolWindowBase(QMainWindow):
    """
    Base window with:
      - splitter: plot on the left, controls on the right
      - matplotlib figure/canvas/axes
      - slider helper methods (linear and log10)
      - optional heavy-tool update controls (Apply + Auto + debounce)
    """

    TOOL_NAME: str = "Tool"
    IS_HEAVY: bool = False            # set True for SED/LC; False for IBS/IBS3D
    DEBOUNCE_MS: int = 250
    THREE_DIMENSIONAL_PLOT: bool = False

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
        if not self.THREE_DIMENSIONAL_PLOT:
            self.ax = self.fig.add_subplot(111)
        if self.THREE_DIMENSIONAL_PLOT:
            self.ax = self.fig.add_subplot(111, projection="3d")

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

