"""Dialog to draw rectangular regions (watermark / subtitle band) on a preview
frame. Returns regions as normalized (x, y, w, h) in 0..1.
"""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_ACCENT = QColor("#2d7d46")
_FILL = QColor(45, 125, 70, 70)


class _Canvas(QWidget):
    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self._src = pixmap
        self._scaled = pixmap
        self.rects: List[QRect] = []
        self._start = None
        self._cur = None
        self.setMinimumSize(640, 360)

    def resizeEvent(self, _e):
        self._scaled = self._src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def _img_rect(self) -> QRect:
        sw, sh = self._scaled.width(), self._scaled.height()
        return QRect((self.width() - sw) // 2, (self.height() - sh) // 2, sw, sh)

    def mousePressEvent(self, e):
        self._start = e.position().toPoint()
        self._cur = self._start
        self.update()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self._cur = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._start is not None:
            r = QRect(self._start, e.position().toPoint()).normalized()
            if r.width() > 5 and r.height() > 5:
                self.rects.append(r)
            self._start = self._cur = None
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        ir = self._img_rect()
        p.drawPixmap(ir.topLeft(), self._scaled)
        p.setPen(QPen(_ACCENT, 2))
        for r in self.rects:
            p.fillRect(r, _FILL)
            p.drawRect(r)
        if self._start is not None and self._cur is not None:
            r = QRect(self._start, self._cur).normalized()
            p.fillRect(r, _FILL)
            p.drawRect(r)

    def clear(self):
        self.rects = []
        self.update()

    def normalized(self) -> List[Tuple[float, float, float, float]]:
        ir = self._img_rect()
        out = []
        for r in self.rects:
            rr = r.intersected(ir)
            if rr.width() < 2 or rr.height() < 2:
                continue
            out.append((
                round((rr.x() - ir.x()) / ir.width(), 4),
                round((rr.y() - ir.y()) / ir.height(), 4),
                round(rr.width() / ir.width(), 4),
                round(rr.height() / ir.height(), 4),
            ))
        return out


class RegionSelector(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None, title: str = "Виділіть область"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        self.canvas = _Canvas(pixmap)
        lay.addWidget(self.canvas, 1)
        hint = QLabel("Перетягни мишею, щоб виділити область. Можна кілька рамок.")
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        clr = QPushButton("Очистити")
        clr.clicked.connect(self.canvas.clear)
        ok = QPushButton("Готово")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Скасувати")
        cancel.clicked.connect(self.reject)
        row.addWidget(clr)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)

    def regions(self) -> List[Tuple[float, float, float, float]]:
        return self.canvas.normalized()
