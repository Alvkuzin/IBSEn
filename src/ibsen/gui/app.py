import sys
from PySide6.QtWidgets import QApplication

from ibsen.gui.launcher import LauncherWindow


def main() -> None:
    app = QApplication(sys.argv)

    w = LauncherWindow()
    w.resize(500, 300)
    w.show()

    sys.exit(app.exec())
