"""GUI entry point."""
from __future__ import annotations

import io
import os
import sys


def _fix_std_streams() -> None:
    # A windowed exe has no console: sys.stdout/stderr are None, which crashes libraries that print.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, io.StringIO())


def main() -> int:
    _fix_std_streams()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from ..paths import INSTALL_DIR
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Clipper")
    app.setStyle("Fusion")
    icon = INSTALL_DIR / "assets" / "clipper.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
