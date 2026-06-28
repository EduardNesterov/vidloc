"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow

# Light, neutral styling that reads well on macOS.
STYLESHEET = """
QWidget { font-size: 13px; }
QGroupBox {
    font-weight: 600;
    border: 1px solid rgba(127,127,127,0.35);
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton#startBtn {
    background: #2d7d46; color: white; font-weight: 600;
    border: none; border-radius: 8px; padding: 10px 18px;
}
QPushButton#startBtn:disabled { background: #6b6b6b; }
QPushButton#cancelBtn {
    background: #a23b3b; color: white; border: none;
    border-radius: 8px; padding: 10px 18px;
}
QProgressBar {
    border: 1px solid rgba(127,127,127,0.4); border-radius: 6px;
    text-align: center; height: 20px;
}
QProgressBar::chunk { background: #2d7d46; border-radius: 5px; }
QPlainTextEdit { font-family: 'Menlo',monospace; font-size: 11px; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Vidloc")
    app.setApplicationDisplayName("Vidloc — локалізатор відео (UA)")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
