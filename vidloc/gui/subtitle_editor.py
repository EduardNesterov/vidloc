"""Visual subtitle editor (frames + timeline + per-caption style).

The user positions each subtitle by dragging / resizing a box on the video frame,
retimes it on a caption timeline, edits its text, and styles it in the right panel.
The result — a list of :class:`vidloc.models.Caption` — is the single source of truth:
the pipeline burns it into the picture **and** drives the TTS dubbing from the same
list, so editing a subtitle also changes the voiceover.

The video preview (smooth scrub + audio via an H.264 proxy, jog "shuttle" audio) is
the same machinery as :mod:`vidloc.gui.region_editor`.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import replace
from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioOutput,
    QAudioSink,
    QMediaDevices,
    QMediaPlayer,
    QVideoSink,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..media import proxy
from ..models import Caption, SubtitleStyle
from ..subtitles.ass import BOX_FONT_FACTOR
from .worker import ProxyWorker, TranslateWorker

MAX_CAPTIONS = 200
_SPLIT_WORDS = 4        # target words per piece when splitting into short captions
_ACCENT = QColor("#2d7d46")
_SEL = QColor("#37b6ff")
_FILL = QColor(45, 125, 70, 50)
_DEL = QColor("#c8442e")
_GRAB = 9
_DEL_SZ = 18

_ALIGNS = [("По центру рамки", 5), ("Знизу рамки", 2), ("Зверху рамки", 8)]


# --------------------------------------------------------------------------- #
# Canvas: caption boxes over the frame, with a WYSIWYG text preview
# --------------------------------------------------------------------------- #
class _CaptionCanvas(QWidget):
    def __init__(self, pixmap: QPixmap, captions: List[Caption],
                 default_style: SubtitleStyle, on_change: Callable,
                 video_w: int = 0, video_h: int = 0):
        super().__init__()
        self._src = pixmap
        self._scaled = pixmap
        self.captions = captions
        self.default_style = default_style
        self.video_w = video_w
        self.video_h = video_h
        self.selected = len(captions) - 1
        self.playhead = 0.0
        self.on_change = on_change
        self._mode: Optional[str] = None
        self._idx = -1
        self._handle: Optional[str] = None
        self._start: Optional[QPoint] = None
        self._orig: Optional[QRect] = None
        self._newrect: Optional[QRect] = None
        self.setMinimumSize(560, 315)
        self.setMouseTracking(True)

    def set_pixmap(self, pm: QPixmap) -> None:
        self._src = pm
        self.resizeEvent(None)
        self.update()

    def resizeEvent(self, _e):
        self._scaled = self._src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    # -- coordinate conversion (same math as region_editor._Canvas) -------- #
    def _img_rect(self) -> QRect:
        sw, sh = self._scaled.width(), self._scaled.height()
        return QRect((self.width() - sw) // 2, (self.height() - sh) // 2, sw, sh)

    def _to_widget(self, rect) -> QRect:
        ir = self._img_rect()
        return QRect(int(ir.x() + rect[0] * ir.width()), int(ir.y() + rect[1] * ir.height()),
                     max(1, int(rect[2] * ir.width())), max(1, int(rect[3] * ir.height())))

    def _to_norm(self, r: QRect):
        ir = self._img_rect()
        if ir.width() < 2 or ir.height() < 2:
            return (0.0, 0.0, 0.0, 0.0)
        rr = r.intersected(ir)
        return (round((rr.x() - ir.x()) / ir.width(), 4),
                round((rr.y() - ir.y()) / ir.height(), 4),
                round(rr.width() / ir.width(), 4),
                round(rr.height() / ir.height(), 4))

    def _handles(self, r: QRect) -> dict:
        return {"tl": QRect(r.left() - _GRAB, r.top() - _GRAB, 2 * _GRAB, 2 * _GRAB),
                "tr": QRect(r.right() - _GRAB, r.top() - _GRAB, 2 * _GRAB, 2 * _GRAB),
                "bl": QRect(r.left() - _GRAB, r.bottom() - _GRAB, 2 * _GRAB, 2 * _GRAB),
                "br": QRect(r.right() - _GRAB, r.bottom() - _GRAB, 2 * _GRAB, 2 * _GRAB)}

    def _del_btn(self, r: QRect) -> QRect:
        return QRect(r.right() - _DEL_SZ // 2, r.top() - _DEL_SZ - 2, _DEL_SZ, _DEL_SZ)

    # -- caption ops ------------------------------------------------------- #
    def add_caption(self, rect=None) -> bool:
        if len(self.captions) >= MAX_CAPTIONS:
            return False
        t0 = round(self.playhead, 2)
        self.captions.append(Caption(
            text="Новий субтитр", start=t0, end=t0 + 2.0,
            rect=tuple(rect) if rect else (0.1, 0.78, 0.8, 0.14)))
        self.selected = len(self.captions) - 1
        self.on_change()
        self.update()
        return True

    def delete_caption(self, i: int) -> None:
        if 0 <= i < len(self.captions):
            del self.captions[i]
            self.selected = min(self.selected, len(self.captions) - 1)
            self.on_change()
            self.update()

    def select(self, i: int) -> None:
        self.selected = i if 0 <= i < len(self.captions) else -1
        self.on_change()
        self.update()

    def _hit_order(self) -> List[int]:
        """Indices to hit-test, most-grabbable first: the selected caption, then the
        captions actually on screen (active) at the playhead. Boxes that aren't drawn
        (invisible, overlapping the same band) are NOT grabbable — otherwise reverse
        iteration would always grab the last caption. You position a caption after
        selecting / scrubbing to it (which makes it the visible one)."""
        order: List[int] = []
        if 0 <= self.selected < len(self.captions):
            order.append(self.selected)
        for i in range(len(self.captions)):
            if i not in order and self.captions[i].active_at(self.playhead):
                order.append(i)
        return order

    # -- mouse (create / move / resize / delete / select) ------------------ #
    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        order = self._hit_order()
        for i in order:
            if self._del_btn(self._to_widget(self.captions[i].rect)).contains(pos):
                self.delete_caption(i)
                return
        for i in order:
            r = self._to_widget(self.captions[i].rect)
            for name, hr in self._handles(r).items():
                if hr.contains(pos):
                    self._mode, self._idx, self._handle = "resize", i, name
                    self._orig, self._start = QRect(r), pos
                    self.select(i)
                    return
        for i in order:
            r = self._to_widget(self.captions[i].rect)
            if r.contains(pos):
                self._mode, self._idx = "move", i
                self._orig, self._start = QRect(r), pos
                self.select(i)
                return
        if len(self.captions) < MAX_CAPTIONS and self._img_rect().contains(pos):
            self._mode, self._start = "create", pos
            self._newrect = QRect(pos, pos)

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if self._mode == "create":
            self._newrect = QRect(self._start, pos).normalized()
            self.update()
        elif self._mode == "move":
            self.captions[self._idx].rect = self._to_norm(
                self._clamp(self._orig.translated(pos - self._start)))
            self.on_change()
            self.update()
        elif self._mode == "resize":
            self.captions[self._idx].rect = self._to_norm(
                self._resize(self._orig, self._handle, pos))
            self.on_change()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._mode == "create" and self._start is not None:
            r = QRect(self._start, e.position().toPoint()).normalized()
            if r.width() > 10 and r.height() > 8:
                self.add_caption(self._to_norm(r))
        self._mode = self._handle = self._newrect = None
        self._idx = -1
        self.update()

    def _clamp(self, r: QRect) -> QRect:
        ir = self._img_rect()
        x = min(max(r.left(), ir.left()), ir.right() - r.width() + 1)
        y = min(max(r.top(), ir.top()), ir.bottom() - r.height() + 1)
        return QRect(x, y, r.width(), r.height())

    def _resize(self, orig: QRect, handle: str, pos: QPoint) -> QRect:
        ir = self._img_rect()
        px = min(max(pos.x(), ir.left()), ir.right())
        py = min(max(pos.y(), ir.top()), ir.bottom())
        left, top, right, bottom = orig.left(), orig.top(), orig.right(), orig.bottom()
        if "l" in handle:
            left = px
        if "r" in handle:
            right = px
        if "t" in handle:
            top = py
        if "b" in handle:
            bottom = py
        r = QRect(QPoint(min(left, right), min(top, bottom)),
                  QPoint(max(left, right), max(top, bottom)))
        if r.width() < 10:
            r.setWidth(10)
        if r.height() < 8:
            r.setHeight(8)
        return r

    # -- style resolution (caption override → global default) -------------- #
    def _font_px(self, cap: Caption, box: QRect) -> int:
        ir = self._img_rect()
        if cap.font_size_pct is not None:
            return max(8, int(ir.height() * cap.font_size_pct / 100.0))
        return max(8, int(box.height() * BOX_FONT_FACTOR))

    def _outline_px(self, cap: Caption, font_px: int) -> int:
        """Outline width in preview (widget) pixels. An explicit ``outline_w`` is in
        video (ASS PlayRes) px, so scale it by the displayed image / video ratio to
        match what will be burned; auto = a fixed fraction of the font (same 0.11 as
        the ASS writer)."""
        if cap.outline_w is not None and self.video_h:
            return max(0, round(cap.outline_w * self._img_rect().height() / self.video_h))
        return max(1, round(font_px * 0.11))

    def _draw_caption_text(self, p: QPainter, box: QRect, cap: Caption) -> None:
        d = self.default_style
        family = cap.font or d.font
        px = self._font_px(cap, box)
        f = QFont(family)
        f.setPixelSize(px)
        f.setBold(True if cap.bold is None else cap.bold)
        p.setFont(f)
        txt = cap.text or ""
        if (d.uppercase if cap.uppercase is None else cap.uppercase):
            txt = txt.upper()
        an = cap.align or 5
        flag = Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap
        flag |= (Qt.AlignmentFlag.AlignVCenter if an == 5 else
                 Qt.AlignmentFlag.AlignBottom if an == 2 else Qt.AlignmentFlag.AlignTop)
        outline = QColor(cap.outline_color or d.outline_color)
        primary = QColor(cap.primary_color or d.primary_color)
        ow = self._outline_px(cap, px)
        if ow > 0:
            # Fill a disk of radius ``ow`` with offset copies → a solid outline at any
            # width. Step keeps the draw count bounded for thick outlines (a preview
            # approximation; the ASS \bord is exact).
            p.setPen(outline)
            step = max(1, (2 * ow + 1) // 12)
            for dx in range(-ow, ow + 1, step):
                for dy in range(-ow, ow + 1, step):
                    if (dx or dy) and dx * dx + dy * dy <= ow * ow:
                        p.drawText(box.translated(dx, dy), flag, txt)
        p.setPen(primary)
        p.drawText(box, flag, txt)

    def paintEvent(self, _e):
        p = QPainter(self)
        ir = self._img_rect()
        p.drawPixmap(ir.topLeft(), self._scaled)
        for i, cap in enumerate(self.captions):
            sel = i == self.selected
            active = cap.active_at(self.playhead)
            # Declutter: draw only the caption(s) on screen now (active) + the selected
            # one. With many captions sharing the same band, drawing them all would be
            # an unreadable pile — and only the visible ones are grabbable (see
            # _hit_order), so what you see is what you can move.
            if not (active or sel):
                continue
            r = self._to_widget(cap.rect)
            if active:
                self._draw_caption_text(p, r, cap)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_SEL if sel else _ACCENT, 2))
            p.drawRect(r)
            p.setPen(QColor("white"))
            p.drawText(r.adjusted(4, 2, 0, 0),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, str(i + 1))
            if sel:
                p.setPen(Qt.PenStyle.NoPen)
                for cx, cy in ((r.left(), r.top()), (r.right(), r.top()),
                               (r.left(), r.bottom()), (r.right(), r.bottom())):
                    p.fillRect(QRect(cx - 4, cy - 4, 8, 8), _SEL)
                db = self._del_btn(r)
                p.setBrush(_DEL)
                p.setPen(QPen(QColor("white"), 2))
                p.drawEllipse(db)
                p.drawLine(db.left() + 5, db.top() + 5, db.right() - 5, db.bottom() - 5)
                p.drawLine(db.right() - 5, db.top() + 5, db.left() + 5, db.bottom() - 5)
        if self._newrect is not None:
            p.fillRect(self._newrect, _FILL)
            p.setPen(QPen(_ACCENT, 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self._newrect)
        p.end()


# --------------------------------------------------------------------------- #
# Timeline: one block per caption on a single track
# --------------------------------------------------------------------------- #
class _CaptionTimeline(QWidget):
    M = 12
    BR = 6

    def __init__(self, captions: List[Caption]):
        super().__init__()
        self.captions = captions
        self.duration = 1.0
        self.selected = -1
        self.playhead = 0.0
        self.on_change: Callable = lambda: None
        self.on_scrub: Callable = lambda t: None
        self.on_select: Callable = lambda i: None
        self.on_split: Callable = lambda: None
        self._mode: Optional[str] = None
        self._i = -1
        self._edge = 0
        self._start_x = 0
        self._orig = None
        self.setMinimumHeight(64)
        self.setMouseTracking(True)

    def set_data(self, duration: float, selected: int) -> None:
        self.duration = max(0.01, duration)
        self.selected = selected
        self.update()

    def _usable(self) -> int:
        return max(1, self.width() - 2 * self.M)

    def _x(self, t: float) -> float:
        return self.M + (t / self.duration) * self._usable()

    def _t(self, x: float) -> float:
        return min(max((x - self.M) / self._usable(), 0.0), 1.0) * self.duration

    def _block_at(self, x: float, y: float):
        """Return (index, edge) where edge is 0=start, 1=end, -1=body, or None."""
        for i in reversed(range(len(self.captions))):
            c = self.captions[i]
            x0, x1 = self._x(c.start), self._x(c.end)
            if abs(x - x0) <= self.BR:
                return i, 0
            if abs(x - x1) <= self.BR:
                return i, 1
            if x0 < x < x1:
                return i, -1
        return None

    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        if e.button() == Qt.MouseButton.RightButton:
            self._menu(pos)
            return
        hit = self._block_at(pos.x(), pos.y())
        if hit is None:
            self.playhead = self._t(pos.x())          # click empty → scrub
            self.on_scrub(self.playhead)
            self.update()
            return
        i, edge = hit
        self.selected = i
        self.on_select(i)
        if edge == -1:
            self._mode, self._i = "move", i
            self._start_x = pos.x()
            self._orig = (self.captions[i].start, self.captions[i].end)
        else:
            self._mode, self._i, self._edge = "resize", i, edge
        self.update()

    def mouseMoveEvent(self, e):
        if self._mode is None:
            return
        x = e.position().toPoint().x()
        c = self.captions[self._i]
        if self._mode == "resize":
            if self._edge == 0:
                c.start = round(min(self._t(x), c.end - 0.1), 3)
                c.start = max(0.0, c.start)
            else:
                c.end = round(max(self._t(x), c.start + 0.1), 3)
                c.end = min(self.duration, c.end)
        elif self._mode == "move":
            dt = self._t(x) - self._t(self._start_x)
            t0, t1 = self._orig
            w = t1 - t0
            nt0 = min(max(0.0, t0 + dt), self.duration - w)
            c.start, c.end = round(nt0, 3), round(nt0 + w, 3)
        self.update()
        self.on_change()

    def mouseReleaseEvent(self, _e):
        self._mode = None
        self._i = -1

    def _menu(self, pos):
        m = QMenu(self)
        a_add = m.addAction("Додати субтитр тут")
        hit = self._block_at(pos.x(), pos.y())
        a_split = m.addAction("Розбити на короткі (3–4 слова)") if hit is not None else None
        a_del = m.addAction("Видалити субтитр") if hit is not None else None
        act = m.exec(self.mapToGlobal(pos))
        if act is a_add:
            t0 = round(self._t(pos.x()), 2)
            self.captions.append(Caption(text="Новий субтитр", start=t0, end=t0 + 2.0,
                                         rect=(0.1, 0.78, 0.8, 0.14)))
            self.selected = len(self.captions) - 1
            self.on_select(self.selected)
            self.on_change()
            self.update()
        elif a_split is not None and hit is not None and act is a_split:
            self.selected = hit[0]
            self.on_select(hit[0])
            self.on_split()
        elif a_del is not None and hit is not None and act is a_del:
            del self.captions[hit[0]]
            self.selected = min(self.selected, len(self.captions) - 1)
            self.on_select(self.selected)
            self.on_change()
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        y, bh = 10, self.height() - 24
        p.fillRect(QRect(self.M, y, self._usable(), bh), QColor("#1f1f1f"))
        if not self.captions:
            p.setPen(QColor("gray"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Додайте субтитр (＋), намалюйте рамку на кадрі "
                       "або «Підтягнути переклад»")
        for i, c in enumerate(self.captions):
            x0, x1 = int(self._x(c.start)), int(self._x(c.end))
            sel = i == self.selected
            p.fillRect(QRect(x0, y, max(3, x1 - x0), bh),
                       QColor(55, 182, 255, 90) if sel else _FILL)
            p.setPen(QPen(_SEL if sel else _ACCENT, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRect(x0, y, max(3, x1 - x0), bh))
            snippet = (c.text or "").split("\n")[0][:18]
            p.setPen(QColor("white") if sel else QColor(200, 200, 200))
            p.drawText(QRect(x0 + 4, y, max(3, x1 - x0) - 6, bh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"{i + 1}. {snippet}")
        px = int(self._x(self.playhead))
        p.setPen(QPen(QColor("#ffcc00"), 2))
        p.drawLine(px, 2, px, self.height() - 6)
        p.end()


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
class SubtitleEditor(QDialog):
    def __init__(self, pixmap: QPixmap, captions: Optional[List[Caption]] = None,
                 default_style: Optional[SubtitleStyle] = None, duration: float = 0.0,
                 frame_provider: Optional[Callable] = None, src: Optional[str] = None,
                 proxy_ready: Optional[str] = None, proxy_worker: Optional[ProxyWorker] = None,
                 job=None, workspace_root: str = "", video_w: int = 0, video_h: int = 0,
                 parent=None, title: str = "Редактор субтитрів"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 820)
        self.captions: List[Caption] = list(captions or [])
        self.default_style = default_style or SubtitleStyle()
        self.duration = duration
        self._frame_provider = frame_provider
        self._src = src
        self._proxy_ready = proxy_ready
        self._prewarm_worker = proxy_worker
        self._job = job
        self._workspace_root = workspace_root
        self.video_w = video_w
        self.video_h = video_h

        # media/preview state (same as region_editor) ------------------- #
        self._player_ready = False
        self._priming = False
        self._proxy_loaded = False
        self._player: Optional[QMediaPlayer] = None
        self._proxy_worker: Optional[ProxyWorker] = None
        self._tr_worker: Optional[TranslateWorker] = None
        self._pcm = b""
        self._audio_sink: Optional[QAudioSink] = None
        self._jog_io = None

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 3)

        self.canvas = _CaptionCanvas(pixmap, self.captions, self.default_style,
                                     self._on_canvas_change, video_w, video_h)
        left.addWidget(self.canvas, 1)
        hint = QLabel("Кадр: тягни на вільному місці — нова рамка субтитра, кут — розмір "
                      "(розтягування масштабує текст), всередині — рух, × — видалити. "
                      "Знизу — час кожного субтитра (тягни блок / краї, правий клік — меню).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        left.addWidget(hint)

        self.slider = None
        self.btn_play = None
        if duration > 0 and (src or frame_provider is not None):
            self.btn_play = QPushButton("▶")
            self.btn_play.setFixedWidth(40)
            self.btn_play.clicked.connect(self._toggle_play)
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, max(1, int(duration * 1000)))
            self.slider.sliderPressed.connect(self._on_slider_pressed)
            self.slider.sliderMoved.connect(self._on_slider_moved)
            self.slider.sliderReleased.connect(self._on_slider_released)
            play_row = QHBoxLayout()
            play_row.addWidget(self.btn_play)
            play_row.addWidget(self.slider, 1)
            left.addLayout(play_row)
            self.lbl_prep = QLabel("")
            self.lbl_prep.setStyleSheet("color:#37b6ff;")
            self.lbl_prep.setVisible(False)
            left.addWidget(self.lbl_prep)
            self._setup_player()

        self.timeline = _CaptionTimeline(self.captions)
        self.timeline.on_change = self._on_timeline_change
        self.timeline.on_scrub = self._on_timeline_scrub
        self.timeline.on_select = self._on_timeline_select
        self.timeline.on_split = self._split_caption
        left.addWidget(self.timeline)

        row = QHBoxLayout()
        add = QPushButton("＋ Додати субтитр")
        add.clicked.connect(lambda: self.canvas.add_caption())
        self.btn_split = QPushButton("✂ Розбити на короткі")
        self.btn_split.setToolTip(
            "Розбити обрану репліку на короткі субтитри по ~3–4 слова. Озвучка при цьому "
            "лишається цілісною (говорить речення повністю) — темп не збивається.")
        self.btn_split.clicked.connect(self._split_caption)
        self.btn_pull = QPushButton("Підтягнути переклад")
        self.btn_pull.clicked.connect(self._pull_translation)
        self.btn_pull.setEnabled(bool(self._workspace_root and self._job))
        self.btn_pull.setToolTip(
            "Заповнити субтитри перекладеними репліками з розпізнаного тексту."
            if self.btn_pull.isEnabled() else
            "Спершу виконайте крок «Розпізнати» для цього відео.")
        clr = QPushButton("Очистити")
        clr.clicked.connect(self._clear)
        row.addWidget(add)
        row.addWidget(self.btn_split)
        row.addWidget(self.btn_pull)
        row.addWidget(clr)
        row.addStretch(1)
        cancel = QPushButton("Скасувати")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Готово")
        ok.clicked.connect(self.accept)
        # Don't let Space/Enter fire a button (Space is our play/pause) — the whole
        # row is mouse-driven; auto-default would hijack the spacebar.
        for b in (add, self.btn_split, self.btn_pull, clr, cancel, ok):
            b.setAutoDefault(False)
            b.setDefault(False)
        row.addWidget(cancel)
        row.addWidget(ok)
        left.addLayout(row)

        root.addWidget(self._style_panel(), 0)
        self._shut = False
        # Spacebar = play/pause anywhere in the editor (unless typing caption text).
        QApplication.instance().installEventFilter(self)
        self._refresh()

    # -- spacebar → play/pause -------------------------------------------- #
    def eventFilter(self, obj, event) -> bool:
        if (event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and QApplication.focusWidget() is not self.ed_text):
            self._toggle_play()
            return True
        return super().eventFilter(obj, event)

    # -- right-side style panel ------------------------------------------- #
    def _style_panel(self) -> QWidget:
        self.grp = QGroupBox("Субтитр")
        form = QFormLayout(self.grp)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setPlaceholderText("Текст субтитра…")
        self.ed_text.setFixedHeight(80)
        self.ed_text.textChanged.connect(self._on_text_changed)
        form.addRow("Текст:", self.ed_text)

        self.sp_start = QSpinBox()
        self.sp_start.setRange(0, max(1, int(self.duration * 1000)))
        self.sp_start.setSuffix(" мс")
        self.sp_start.setSingleStep(100)
        self.sp_start.valueChanged.connect(self._on_time_changed)
        self.sp_end = QSpinBox()
        self.sp_end.setRange(0, max(1, int(self.duration * 1000)))
        self.sp_end.setSuffix(" мс")
        self.sp_end.setSingleStep(100)
        self.sp_end.valueChanged.connect(self._on_time_changed)
        form.addRow("Початок:", self.sp_start)
        form.addRow("Кінець:", self.sp_end)

        self.cmb_font = QComboBox()
        self.cmb_font.addItem("— за умовчанням —", None)
        from . import fonts
        for fam in fonts.load():
            self.cmb_font.addItem(fam, fam)
        self.cmb_font.currentIndexChanged.connect(self._on_style_changed)
        form.addRow("Шрифт:", self.cmb_font)

        self.chk_bold = QCheckBox("Жирний")
        self.chk_bold.toggled.connect(self._on_style_changed)
        self.chk_caps = QCheckBox("ВЕЛИКІ ЛІТЕРИ")
        self.chk_caps.toggled.connect(self._on_style_changed)
        form.addRow(self.chk_bold)
        form.addRow(self.chk_caps)

        self.sp_size = QSpinBox()
        self.sp_size.setRange(0, 30)
        self.sp_size.setSpecialValueText("Авто (від рамки)")
        self.sp_size.setSuffix(" %")
        self.sp_size.valueChanged.connect(self._on_style_changed)
        form.addRow("Розмір:", self.sp_size)

        self.sp_outline = QSpinBox()
        self.sp_outline.setRange(0, 40)
        self.sp_outline.setSpecialValueText("Авто")
        self.sp_outline.setSuffix(" px")
        self.sp_outline.valueChanged.connect(self._on_style_changed)
        form.addRow("Обведення:", self.sp_outline)

        self.cmb_align = QComboBox()
        for label, data in _ALIGNS:
            self.cmb_align.addItem(label, data)
        self.cmb_align.currentIndexChanged.connect(self._on_style_changed)
        form.addRow("Вирівнювання:", self.cmb_align)

        form.addRow("Колір тексту:", self._color_row("primary_color"))
        form.addRow("Колір обведення:", self._color_row("outline_color"))

        apply_all = QPushButton("Застосувати до всіх (стиль + позиція)")
        apply_all.clicked.connect(self._apply_style_to_all)
        form.addRow(apply_all)
        note = QLabel("«Шрифт/розмір/колір» діють на обрану репліку. Кнопка копіює її "
                      "стиль І положення рамки на всі субтитри. Порожні значення = "
                      "глобальний стиль.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return self.grp

    def _color_row(self, field: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        swatch = QLabel()
        swatch.setFixedSize(30, 18)
        btn = QPushButton("Обрати…")
        btn.clicked.connect(lambda: self._pick_color(field))
        lay.addWidget(swatch)
        lay.addWidget(btn)
        lay.addStretch(1)
        setattr(self, f"_swatch_{field}", swatch)
        return w

    def _pick_color(self, field: str) -> None:
        cap = self._selected()
        if cap is None:
            return
        cur = getattr(cap, field) or getattr(self.default_style, field)
        c = QColorDialog.getColor(QColor(cur), self, "Колір")
        if c.isValid():
            setattr(cap, field, c.name().upper())
            self._refresh()

    # -- selection helpers ------------------------------------------------- #
    def _selected(self) -> Optional[Caption]:
        i = self.canvas.selected
        return self.captions[i] if 0 <= i < len(self.captions) else None

    def _refresh(self) -> None:
        """Repaint canvas + timeline and reload the panel from the selection."""
        self.timeline.set_data(self.duration or 1.0, self.canvas.selected)
        self.canvas.update()
        self._load_panel()

    def _load_panel(self) -> None:
        cap = self._selected()
        self.grp.setEnabled(cap is not None)
        widgets = [self.ed_text, self.sp_start, self.sp_end, self.cmb_font, self.chk_bold,
                   self.chk_caps, self.sp_size, self.sp_outline, self.cmb_align]
        for w in widgets:
            w.blockSignals(True)
        if cap is not None:
            self.grp.setTitle(f"Субтитр №{self.canvas.selected + 1}")
            if self.ed_text.toPlainText() != cap.text:
                self.ed_text.setPlainText(cap.text)
            self.sp_start.setValue(int(cap.start * 1000))
            self.sp_end.setValue(int(cap.end * 1000))
            self.cmb_font.setCurrentIndex(max(0, self.cmb_font.findData(cap.font)))
            self.chk_bold.setChecked(True if cap.bold is None else cap.bold)
            self.chk_caps.setChecked(
                self.default_style.uppercase if cap.uppercase is None else cap.uppercase)
            self.sp_size.setValue(0 if cap.font_size_pct is None else int(cap.font_size_pct))
            self.sp_outline.setValue(0 if cap.outline_w is None else int(cap.outline_w))
            self.cmb_align.setCurrentIndex(max(0, self.cmb_align.findData(cap.align or 5)))
            self._paint_swatch("primary_color", cap.primary_color or self.default_style.primary_color)
            self._paint_swatch("outline_color", cap.outline_color or self.default_style.outline_color)
        for w in widgets:
            w.blockSignals(False)

    def _paint_swatch(self, field: str, hexcol: str) -> None:
        sw = getattr(self, f"_swatch_{field}", None)
        if sw is not None:
            sw.setStyleSheet(f"background:{hexcol}; border:1px solid #555;")

    # -- panel → caption --------------------------------------------------- #
    def _on_text_changed(self) -> None:
        cap = self._selected()
        if cap is not None:
            cap.text = self.ed_text.toPlainText()
            self.canvas.update()
            self.timeline.update()

    def _on_time_changed(self) -> None:
        cap = self._selected()
        if cap is None:
            return
        s = self.sp_start.value() / 1000.0
        e = self.sp_end.value() / 1000.0
        if e <= s:
            e = s + 0.1
        cap.start, cap.end = round(s, 3), round(e, 3)
        self.timeline.update()
        self.canvas.update()

    def _on_style_changed(self, *_a) -> None:
        cap = self._selected()
        if cap is None:
            return
        cap.font = self.cmb_font.currentData()
        cap.bold = self.chk_bold.isChecked()
        cap.uppercase = self.chk_caps.isChecked()
        cap.font_size_pct = None if self.sp_size.value() == 0 else float(self.sp_size.value())
        cap.outline_w = None if self.sp_outline.value() == 0 else float(self.sp_outline.value())
        cap.align = self.cmb_align.currentData()
        self.canvas.update()

    def _apply_style_to_all(self) -> None:
        cap = self._selected()
        if cap is None:
            return
        for c in self.captions:
            if c is cap:
                continue
            # Position (rect) is copied too, so "apply to all" also places every
            # caption where this one sits — the common case for consistent subtitles.
            for f in ("rect", "font", "bold", "uppercase", "font_size_pct", "outline_w",
                      "align", "primary_color", "outline_color"):
                setattr(c, f, getattr(cap, f))
        self.canvas.update()

    def _split_caption(self) -> None:
        """Split the selected caption into short ~3–4-word captions that share a group
        id. They burn as separate short lines; the dub re-joins the group into one
        utterance (see runner), so the voiceover tempo is not chopped."""
        i = self.canvas.selected
        if not (0 <= i < len(self.captions)):
            return
        cap = self.captions[i]
        words = (cap.text or "").split()
        if len(words) <= _SPLIT_WORDS:
            return
        n = max(2, math.ceil(len(words) / _SPLIT_WORDS))
        base, extra = divmod(len(words), n)
        pieces, idx = [], 0
        for k in range(n):
            take = base + (1 if k < extra else 0)
            pieces.append(" ".join(words[idx:idx + take]))
            idx += take
        gid = 1 + max((c.group or 0 for c in self.captions), default=0)
        total = sum(len(pn) for pn in pieces) or 1
        span = max(0.0, cap.end - cap.start)
        new_caps, t = [], cap.start
        for pn in pieces:
            d = span * (len(pn) / total)
            new_caps.append(replace(cap, text=pn, start=round(t, 3),
                                    end=round(t + d, 3), group=gid))
            t += d
        new_caps[-1].end = round(cap.end, 3)   # snap last edge (no float drift)
        self.captions[i:i + 1] = new_caps
        self.canvas.selected = i
        self.timeline.selected = i
        self._refresh()

    # -- canvas / timeline callbacks -------------------------------------- #
    def _on_canvas_change(self) -> None:
        self.timeline.selected = self.canvas.selected
        self._refresh()

    def _on_timeline_change(self) -> None:
        self.canvas.update()
        self._load_panel()

    def _on_timeline_select(self, i: int) -> None:
        self.canvas.selected = i
        cap = self._selected()
        # Jump the preview to this caption so you can see it (and its style) on frame.
        if cap is not None and not cap.active_at(self.timeline.playhead):
            self._seek(min(cap.start + 0.05, cap.end), move_slider=True)
        self.canvas.update()
        self._load_panel()

    def _on_timeline_scrub(self, t: float) -> None:
        self._seek(t, move_slider=True)

    # -- pull translation -------------------------------------------------- #
    def _pull_translation(self) -> None:
        if not (self._workspace_root and self._job):
            return
        if self.captions and QMessageBox.question(
                self, "Підтягнути переклад",
                "Замінити наявні субтитри перекладеними репліками?") != \
                QMessageBox.StandardButton.Yes:
            return
        self.btn_pull.setEnabled(False)
        self.btn_pull.setText("Завантаження моделі…")
        self._tr_worker = TranslateWorker(self._job, self._workspace_root, self)
        self._tr_worker.sig_progress.connect(self._on_translation_progress)
        self._tr_worker.sig_done.connect(self._on_translation_done)
        self._tr_worker.sig_fail.connect(self._on_translation_fail)
        self._tr_worker.start()

    def _on_translation_progress(self, f: float, msg: str) -> None:
        self.btn_pull.setText(f"Переклад… {int((f or 0.0) * 100)}%")

    def _on_translation_done(self, segments: list) -> None:
        self.btn_pull.setEnabled(True)
        self.btn_pull.setText("Підтягнути переклад")
        caps: List[Caption] = []
        for seg in segments:
            txt = (seg.get("text_tr") or seg.get("text") or "").strip()
            if not txt:
                continue
            caps.append(Caption(text=txt, start=float(seg.get("start", 0.0)),
                                end=float(seg.get("end", 0.0)) or float(seg.get("start", 0.0)) + 1.5,
                                rect=(0.1, 0.78, 0.8, 0.14)))
        if not caps:
            QMessageBox.information(self, "Переклад", "Порожній переклад.")
            return
        self.captions[:] = caps
        self.canvas.selected = 0
        self.timeline.selected = 0
        self._refresh()
        # Land the preview on the first caption so the pulled text is visible at once.
        self._seek(min(caps[0].start + 0.05, caps[0].end), move_slider=True)

    def _on_translation_fail(self, message: str) -> None:
        self.btn_pull.setEnabled(True)
        self.btn_pull.setText("Підтягнути переклад")
        QMessageBox.critical(self, "Переклад", message)

    def _clear(self) -> None:
        self.captions.clear()
        self.canvas.selected = -1
        self.timeline.selected = -1
        self._refresh()

    def result_captions(self) -> List[Caption]:
        return [c for c in self.captions if (c.text or "").strip()]

    # ------------------------------------------------------------------ #
    # Preview player (H.264 proxy + jog audio) — mirrors region_editor
    # ------------------------------------------------------------------ #
    def _set_playhead(self, t: float) -> None:
        self.timeline.playhead = t
        self.canvas.playhead = t
        self.timeline.update()
        self.canvas.update()
        self._sync_selection_to_playhead()

    def _sync_selection_to_playhead(self) -> None:
        """Make the selection (and the right-panel editable text) follow the playhead:
        scrubbing to a moment selects the caption shown there, so you edit whatever is
        on screen. Skipped while editing any field in the panel (don't yank the edit
        away) and over gaps with no active caption (keep the current selection)."""
        fw = QApplication.focusWidget()
        if fw is not None and (fw is self.ed_text or self.grp.isAncestorOf(fw)):
            return
        t = self.canvas.playhead
        cur = self.canvas.selected
        # Keep the current selection while it is still on screen (handles overlaps and
        # avoids fighting an explicit pick).
        if 0 <= cur < len(self.captions) and self.captions[cur].active_at(t):
            return
        idx = next((i for i, c in enumerate(self.captions) if c.active_at(t)), -1)
        if idx == -1 or idx == cur:
            return
        self.canvas.selected = idx
        self.timeline.selected = idx
        self.timeline.update()
        self.canvas.update()
        self._load_panel()

    def _seek(self, t: float, *, move_slider: bool) -> None:
        t = min(max(0.0, t), self.duration)
        self._set_playhead(t)
        if move_slider and self.slider is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 1000))
            self.slider.blockSignals(False)
        if self._player_ready and self._player is not None:
            self._player.setPosition(int(t * 1000))
        elif self._frame_provider is not None:
            pm = self._frame_provider(t)
            if pm is not None and not pm.isNull():
                self.canvas.set_pixmap(pm)

    def _setup_player(self) -> None:
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._sink = QVideoSink(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.setVideoSink(self._sink)
        self._sink.videoFrameChanged.connect(self._on_video_frame)
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_playstate)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        if self.btn_play is not None:
            self.btn_play.setEnabled(False)
        if self._proxy_ready and os.path.exists(self._proxy_ready):
            self._on_proxy_ready(self._proxy_ready)
        elif self._prewarm_worker is not None and self._prewarm_worker.isRunning():
            self.lbl_prep.setText("Готую плавний перегляд (перекодування у H.264)…")
            self.lbl_prep.setVisible(True)
            self._prewarm_worker.sig_progress.connect(self._on_proxy_progress)
            self._prewarm_worker.sig_ready.connect(self._on_proxy_ready)
        elif self._src:
            self.lbl_prep.setText("Готую плавний перегляд (перекодування у H.264)…")
            self.lbl_prep.setVisible(True)
            self._proxy_worker = ProxyWorker(self._src, self)
            self._proxy_worker.sig_progress.connect(self._on_proxy_progress)
            self._proxy_worker.sig_ready.connect(self._on_proxy_ready)
            self._proxy_worker.start()

    def _on_proxy_progress(self, f: float) -> None:
        if not self._player_ready:
            self.lbl_prep.setText(f"Готую плавний перегляд… {int(f * 100)}% "
                                  "(поки що перемотка по кадру на відпускання)")

    def _on_proxy_ready(self, path: str) -> None:
        if not self._player or self._proxy_loaded:
            return
        self._proxy_loaded = True
        self._player.setSource(QUrl.fromLocalFile(path))
        if self._src:
            threading.Thread(
                target=lambda: setattr(self, "_pcm", proxy.load_pcm(self._src)),
                daemon=True).start()

    def _on_media_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and not self._priming \
                and not self._player_ready:
            self._priming = True
            self._audio_out.setMuted(True)
            self._player.play()
            QTimer.singleShot(150, self._finish_prime)

    def _finish_prime(self) -> None:
        if self._player is None:
            return
        self._player.pause()
        self._audio_out.setMuted(False)
        self._priming = False
        self._player_ready = True
        self.lbl_prep.setVisible(False)
        if self.btn_play is not None:
            self.btn_play.setEnabled(True)
        self._player.setPosition(int(self.timeline.playhead * 1000))

    def _on_video_frame(self, frame) -> None:
        if not frame.isValid():
            return
        img: QImage = frame.toImage()
        if not img.isNull():
            self.canvas.set_pixmap(QPixmap.fromImage(img))

    def _on_position(self, ms: int) -> None:
        t = ms / 1000.0
        self._set_playhead(t)
        if self.slider is not None and not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(ms)
            self.slider.blockSignals(False)

    def _on_playstate(self, state) -> None:
        if self.btn_play is not None:
            playing = state == QMediaPlayer.PlaybackState.PlayingState
            self.btn_play.setText("⏸" if playing else "▶")

    def _toggle_play(self) -> None:
        if not self._player_ready or self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_slider_pressed(self) -> None:
        if self._player is not None:
            self._player.pause()
        self._start_jog()

    def _on_slider_moved(self, v: int) -> None:
        t = v / 1000.0
        self._set_playhead(t)
        if self._player_ready and self._player is not None:
            self._player.setPosition(v)
            self._jog_grain(v)

    def _on_slider_released(self) -> None:
        self._stop_jog()
        if self.slider is None:
            return
        t = self.slider.value() / 1000.0
        self._set_playhead(t)
        if self._player_ready and self._player is not None:
            self._player.setPosition(int(t * 1000))
        elif self._frame_provider is not None:
            pm = self._frame_provider(t)
            if pm is not None and not pm.isNull():
                self.canvas.set_pixmap(pm)

    def _ensure_audio_sink(self) -> Optional[QAudioSink]:
        if self._audio_sink is not None:
            return self._audio_sink
        fmt = QAudioFormat()
        fmt.setSampleRate(proxy.JOG_SR)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            return None
        self._audio_sink = QAudioSink(dev, fmt, self)
        return self._audio_sink

    def _start_jog(self) -> None:
        if not self._pcm:
            return
        sink = self._ensure_audio_sink()
        if sink is not None:
            self._jog_io = sink.start()

    def _jog_grain(self, ms: int) -> None:
        if self._jog_io is None or not self._pcm or self._audio_sink is None:
            return
        win = int(0.05 * proxy.JOG_SR)
        off = int((ms / 1000.0) * proxy.JOG_SR)
        start = max(0, off * 2)
        grain = self._pcm[start:start + win * 2]
        if grain and self._audio_sink.bytesFree() >= len(grain):
            self._jog_io.write(grain)

    def _stop_jog(self) -> None:
        if self._audio_sink is not None:
            self._audio_sink.stop()
        self._jog_io = None

    def _shutdown(self) -> None:
        """Stop playback / audio and detach. Idempotent — runs on accept, reject or
        window close (accept()/reject() go through done(), which does NOT fire
        closeEvent, so stopping only in closeEvent left audio playing)."""
        if self._shut:
            return
        self._shut = True
        QApplication.instance().removeEventFilter(self)
        self._stop_jog()
        if self._player is not None:
            self._player.stop()
        if self._prewarm_worker is not None:
            for sig, slot in ((self._prewarm_worker.sig_ready, self._on_proxy_ready),
                              (self._prewarm_worker.sig_progress, self._on_proxy_progress)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        if self._proxy_worker is not None and self._proxy_worker.isRunning():
            self._proxy_worker.wait(2000)
        if self._tr_worker is not None and self._tr_worker.isRunning():
            self._tr_worker.wait(2000)

    def done(self, r) -> None:
        self._shutdown()
        super().done(r)

    def closeEvent(self, e) -> None:
        self._shutdown()
        super().closeEvent(e)
