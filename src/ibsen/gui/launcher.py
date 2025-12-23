from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel

from ibsen.gui.plot_ibs_norm import IBSNormWindow
from ibsen.gui.plot_sed import SEDWindow
from ibsen.gui.plot_ibs import IBSWindow as WindsIBSWindow
from ibsen.gui.plot_lc import LightCurveWindow


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IBSEn  tools")

        self._open_windows: list[QMainWindow] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("<h2>IBSEn</h2>Yeah, that's not Ibsen...</h2><p>Select a tool:</p>")
        layout.addWidget(title)

        col = QVBoxLayout()
        layout.addLayout(col)

        btn_ibs_norm = QPushButton("New normalized IBS window")
        btn_ibs = QPushButton("New Winds + IBS window")
        btn_sed = QPushButton("New SED window")
        btn_lc = QPushButton("New Light Curve window")
        
        
        col.addWidget(btn_ibs_norm)
        col.addWidget(btn_ibs)     
        col.addWidget(btn_sed)
        col.addWidget(btn_lc)
        

        # Later you can add: btn_lc = QPushButton("New Light Curve window")

        layout.addStretch(1)
        btn_ibs_norm.clicked.connect(self.open_ibs_norm)
        btn_ibs.clicked.connect(self.open_ibs)
        btn_sed.clicked.connect(self.open_sed)
        btn_lc.clicked.connect(self.open_lc)
        

    def _keep(self, w: QMainWindow) -> None:
        """Keep a reference; remove it when the window closes."""
        self._open_windows.append(w)

        def _drop_ref() -> None:
            try:
                self._open_windows.remove(w)
            except ValueError:
                pass

        # destroyed fires when Qt deletes the object
        w.destroyed.connect(_drop_ref)

    def open_ibs_norm(self) -> None:
        w = IBSNormWindow()
        w.show()
        self._keep(w)
        
    def open_ibs(self) -> None:
        w = WindsIBSWindow()
        w.show()
        self._keep(w)

    def open_sed(self) -> None:
        w = SEDWindow()
        w.show()
        self._keep(w)
        
    def open_lc(self) -> None:
        w = LightCurveWindow()
        w.show()
        self._keep(w)
        

