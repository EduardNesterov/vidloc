"""Region + time editor (Phase 2 + 3 of timed removal).

Top: a preview frame where you add / move / resize / delete up to 15 rectangular
regions and select one. Middle: a scrub slider (frame preview via a provided
``frame_provider``). Bottom: a per-region timeline where each region's time
intervals are edited as draggable bracket pairs (right-click → add a segment).

Output: a list of :class:`vidloc.models.TimedRegion` (rect + intervals); an empty
interval list means the region erases for the whole video.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from PySide6.QtCore import QTimer, QUrl, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioOutput,
    QAudioSink,
    QMediaDevices,
    QMediaPlayer,
    QVideoSink,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..media import proxy
from ..models import TimedRegion
from .worker import ProxyWorker

MAX_REGIONS = 15
_ACCENT = QColor("#2d7d46")
_SEL = QColor("#37b6ff")
_FILL = QColor(45, 125, 70, 60)
_DEL = QColor("#c8442e")
_GRAB = 9
_DEL_SZ = 18


# --------------------------------------------------------------------------- #
# Spatial canvas: rectangles on the preview frame
# --------------------------------------------------------------------------- #
class _Canvas(QWidget):
    def __init__(self, pixmap: QPixmap, regions: List[TimedRegion], on_change: Callable):
        super().__init__()
        self._src = pixmap
        self._scaled = pixmap
        self.regions = regions          # shared list[TimedRegion]
        self.selected = len(regions) - 1
        self.on_change = on_change
        self._mode: Optional[str] = None
        self._idx = -1
        self._handle: Optional[str] = None
        self._start: Optional[QPoint] = None
        self._orig: Optional[QRect] = None
        self._newrect: Optional[QRect] = None
        self.setMinimumSize(620, 330)
        self.setMouseTracking(True)

    def set_pixmap(self, pm: QPixmap) -> None:
        self._src = pm
        self.resizeEvent(None)
        self.update()

    def resizeEvent(self, _e):
        self._scaled = self._src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    # -- coordinate conversion --
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

    # -- region ops (testable) --
    def add_region(self, rect=None) -> bool:
        if len(self.regions) >= MAX_REGIONS:
            return False
        self.regions.append(TimedRegion(rect=tuple(rect) if rect else (0.4, 0.4, 0.2, 0.2)))
        self.selected = len(self.regions) - 1
        self.on_change()
        self.update()
        return True

    def delete_region(self, i: int) -> None:
        if 0 <= i < len(self.regions):
            del self.regions[i]
            self.selected = min(self.selected, len(self.regions) - 1)
            self.on_change()
            self.update()

    def select(self, i: int) -> None:
        self.selected = i if 0 <= i < len(self.regions) else -1
        self.on_change()
        self.update()

    # -- mouse --
    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        for i in reversed(range(len(self.regions))):
            if self._del_btn(self._to_widget(self.regions[i].rect)).contains(pos):
                self.delete_region(i)
                return
        for i in reversed(range(len(self.regions))):
            r = self._to_widget(self.regions[i].rect)
            for name, hr in self._handles(r).items():
                if hr.contains(pos):
                    self._mode, self._idx, self._handle = "resize", i, name
                    self._orig, self._start = QRect(r), pos
                    self.select(i)
                    return
        for i in reversed(range(len(self.regions))):
            r = self._to_widget(self.regions[i].rect)
            if r.contains(pos):
                self._mode, self._idx = "move", i
                self._orig, self._start = QRect(r), pos
                self.select(i)
                return
        if len(self.regions) < MAX_REGIONS and self._img_rect().contains(pos):
            self._mode, self._start = "create", pos
            self._newrect = QRect(pos, pos)

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if self._mode == "create":
            self._newrect = QRect(self._start, pos).normalized()
            self.update()
        elif self._mode == "move":
            self.regions[self._idx].rect = self._to_norm(self._clamp(self._orig.translated(pos - self._start)))
            self.update()
        elif self._mode == "resize":
            self.regions[self._idx].rect = self._to_norm(self._resize(self._orig, self._handle, pos))
            self.update()

    def mouseReleaseEvent(self, e):
        if self._mode == "create" and self._start is not None:
            r = QRect(self._start, e.position().toPoint()).normalized()
            if r.width() > 8 and r.height() > 8:
                self.add_region(self._to_norm(r))
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
        if r.width() < 8:
            r.setWidth(8)
        if r.height() < 8:
            r.setHeight(8)
        return r

    def paintEvent(self, _e):
        p = QPainter(self)
        ir = self._img_rect()
        p.drawPixmap(ir.topLeft(), self._scaled)
        for i, reg in enumerate(self.regions):
            r = self._to_widget(reg.rect)
            sel = i == self.selected
            p.fillRect(r, _FILL)
            p.setPen(QPen(_SEL if sel else _ACCENT, 3 if sel else 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            p.setPen(QColor("white"))
            tag = str(i + 1) + ("" if not reg.intervals else f"  ⏱{len(reg.intervals)}")
            p.drawText(r.adjusted(5, 3, 0, 0),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, tag)
            p.setPen(Qt.PenStyle.NoPen)
            for cx, cy in ((r.left(), r.top()), (r.right(), r.top()),
                           (r.left(), r.bottom()), (r.right(), r.bottom())):
                p.fillRect(QRect(cx - 4, cy - 4, 8, 8), _SEL if sel else _ACCENT)
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
# Timeline: time intervals (bracket pairs) for the selected region
# --------------------------------------------------------------------------- #
class _Timeline(QWidget):
    M = 12          # left/right margin
    BR = 6          # bracket grab half-width

    def __init__(self):
        super().__init__()
        self.duration = 1.0
        self.region: Optional[TimedRegion] = None
        self.playhead = 0.0
        self.on_change: Callable = lambda: None
        self.on_scrub: Callable = lambda t: None
        self._mode: Optional[str] = None
        self._i = -1
        self._edge = 0
        self._start_x = 0
        self._orig = None
        self.setMinimumHeight(56)
        self.setMouseTracking(True)

    def set_region(self, region: Optional[TimedRegion], duration: float) -> None:
        self.region = region
        self.duration = max(0.01, duration)
        self.update()

    def _usable(self) -> int:
        return max(1, self.width() - 2 * self.M)

    def _x(self, t: float) -> float:
        return self.M + (t / self.duration) * self._usable()

    def _t(self, x: float) -> float:
        return min(max((x - self.M) / self._usable(), 0.0), 1.0) * self.duration

    def _interval_at(self, x: float):
        if self.region is None:
            return None
        for i, (t0, t1) in enumerate(self.region.intervals):
            if self._x(t0) <= x <= self._x(t1):
                return i
        return None

    # -- ops (testable) --
    def add_interval_at(self, t: float) -> None:
        if self.region is None:
            return
        span = max(1.0, self.duration * 0.08)
        t0 = min(max(0.0, t), self.duration - 0.2)
        t1 = min(self.duration, t0 + span)
        self.region.intervals.append([round(t0, 3), round(t1, 3)])
        self.update()
        self.on_change()

    def delete_interval(self, i: int) -> None:
        if self.region and 0 <= i < len(self.region.intervals):
            del self.region.intervals[i]
            self.update()
            self.on_change()

    def mousePressEvent(self, e):
        if self.region is None:
            return
        pos = e.position().toPoint()
        if e.button() == Qt.MouseButton.RightButton:
            self._menu(pos)
            return
        for i, (t0, t1) in enumerate(self.region.intervals):
            if abs(pos.x() - self._x(t0)) <= self.BR:
                self._mode, self._i, self._edge = "resize", i, 0
                return
            if abs(pos.x() - self._x(t1)) <= self.BR:
                self._mode, self._i, self._edge = "resize", i, 1
                return
        for i, (t0, t1) in enumerate(self.region.intervals):
            if self._x(t0) < pos.x() < self._x(t1):
                self._mode, self._i = "move", i
                self._start_x, self._orig = pos.x(), list(self.region.intervals[i])
                return
        self.playhead = self._t(pos.x())            # click empty area → scrub
        self.on_scrub(self.playhead)
        self.update()

    def mouseMoveEvent(self, e):
        if self.region is None or self._mode is None:
            return
        x = e.position().toPoint().x()
        iv = self.region.intervals
        if self._mode == "resize":
            t0, t1 = iv[self._i]
            if self._edge == 0:
                t0 = min(self._t(x), t1 - 0.05)
            else:
                t1 = max(self._t(x), t0 + 0.05)
            iv[self._i] = [round(max(0.0, t0), 3), round(min(self.duration, t1), 3)]
        elif self._mode == "move":
            dt = self._t(x) - self._t(self._start_x)
            t0, t1 = self._orig
            w = t1 - t0
            nt0 = min(max(0.0, t0 + dt), self.duration - w)
            iv[self._i] = [round(nt0, 3), round(nt0 + w, 3)]
        self.update()
        self.on_change()

    def mouseReleaseEvent(self, _e):
        self._mode = None
        self._i = -1

    def _menu(self, pos):
        m = QMenu(self)
        a_add = m.addAction("Додати відрізок")
        hit = self._interval_at(pos.x())
        a_del = m.addAction("Видалити відрізок") if hit is not None else None
        act = m.exec(self.mapToGlobal(pos))
        if act is a_add:
            self.add_interval_at(self._t(pos.x()))
        elif a_del is not None and act is a_del:
            self.delete_interval(hit)

    def paintEvent(self, _e):
        p = QPainter(self)
        y, bh = 10, self.height() - 22
        p.fillRect(QRect(self.M, y, self._usable(), bh), QColor("#262626"))
        if self.region is None:
            p.setPen(QColor("gray"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Оберіть рамку, щоб задати відрізки часу")
            p.end()
            return
        for (t0, t1) in self.region.intervals:
            x0, x1 = int(self._x(t0)), int(self._x(t1))
            p.fillRect(QRect(x0, y, max(2, x1 - x0), bh), _FILL)
            p.setPen(QPen(_ACCENT, 2))
            p.drawLine(x0, y, x0, y + bh)
            p.drawLine(x1, y, x1, y + bh)
            p.drawLine(x0, y, x0 + 5, y)
            p.drawLine(x0, y + bh, x0 + 5, y + bh)
            p.drawLine(x1 - 5, y, x1, y)
            p.drawLine(x1 - 5, y + bh, x1, y + bh)
        px = int(self._x(self.playhead))
        p.setPen(QPen(QColor("#ffcc00"), 2))
        p.drawLine(px, 2, px, self.height() - 6)
        p.end()


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
class RegionEditor(QDialog):
    def __init__(self, pixmap: QPixmap, regions: Optional[List[TimedRegion]] = None,
                 duration: float = 0.0, frame_provider: Optional[Callable] = None,
                 src: Optional[str] = None, proxy_ready: Optional[str] = None,
                 proxy_worker: Optional[ProxyWorker] = None,
                 parent=None, title: str = "Області та час для видалення"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(940, 780)
        self.regions: List[TimedRegion] = [r for r in (regions or [])][:MAX_REGIONS]
        self.duration = duration
        self._frame_provider = frame_provider
        self._src = src
        self._proxy_ready = proxy_ready     # prewarmed proxy path, if already done
        self._prewarm_worker = proxy_worker  # main window's in-flight prewarm, if any

        # -- media/preview state ------------------------------------------ #
        self._player_ready = False          # proxy loaded + primed → smooth scrub
        self._priming = False
        self._proxy_loaded = False          # source set on the player (once)
        self._player: Optional[QMediaPlayer] = None
        self._proxy_worker: Optional[ProxyWorker] = None
        self._pcm = b""                     # mono s16le buffer for jog audio
        self._audio_sink: Optional[QAudioSink] = None
        self._jog_io = None                 # QIODevice from the running sink

        lay = QVBoxLayout(self)
        self.canvas = _Canvas(pixmap, self.regions, self._refresh)
        lay.addWidget(self.canvas, 1)
        hint = QLabel("Рамка: тягни на вільному (нова), кут (розмір), всередині (рух), "
                      "× (видалити), клік — обрати. Внизу — відрізки часу обраної рамки "
                      "(правий клік по таймлайну — «Додати відрізок»).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)

        self.slider = None
        self.btn_play = None
        if duration > 0 and (src or frame_provider is not None):
            self.btn_play = QPushButton("▶")
            self.btn_play.setFixedWidth(40)
            self.btn_play.setToolTip("Відтворити / пауза (зі звуком)")
            self.btn_play.clicked.connect(self._toggle_play)
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, max(1, int(duration * 1000)))
            self.slider.sliderPressed.connect(self._on_slider_pressed)
            self.slider.sliderMoved.connect(self._on_slider_moved)
            self.slider.sliderReleased.connect(self._on_slider_released)
            play_row = QHBoxLayout()
            play_row.addWidget(self.btn_play)
            play_row.addWidget(self.slider, 1)
            lay.addLayout(play_row)
            self.lbl_prep = QLabel("")
            self.lbl_prep.setStyleSheet("color:#37b6ff;")
            self.lbl_prep.setVisible(False)
            lay.addWidget(self.lbl_prep)
            self._setup_player()

        sel_row = QHBoxLayout()
        self.lbl_sel = QLabel("")
        self.cmb_kind = QComboBox()
        self.cmb_kind.addItem("Об'єкт / лого (reuse фону)", "object")
        self.cmb_kind.addItem("Текст / субтитри (нейродетектор CRAFT)", "text")
        self.cmb_kind.addItem("Статичний фон (затерти раз, заморозити)", "static")
        self.cmb_kind.setToolTip(
            "Тип обраної рамки. «Об'єкт»: затирає весь прямокутник, повторно використовує "
            "фон і пере-затирає при русі фону. «Текст»: нейродетектор CRAFT (офлайн, "
            "апаратно через CoreML) знаходить текст і покриває його — надійно на "
            "стилізованих / низькоконтрастних / кириличних субтитрах. «Статичний фон»: "
            "затирає один раз і вставляє на всі кадри без перевірки — найшвидше і без "
            "мерехтіння, але дасть «наклейку», якщо фон за рамкою рухається.")
        self.cmb_kind.currentIndexChanged.connect(self._on_kind_changed)
        sel_row.addWidget(self.lbl_sel, 1)
        sel_row.addWidget(QLabel("Тип рамки:"))
        sel_row.addWidget(self.cmb_kind)
        sel_wrap = QWidget()
        sel_wrap.setLayout(sel_row)
        lay.addWidget(sel_wrap)
        self.timeline = _Timeline()
        self.timeline.on_change = self._refresh
        self.timeline.on_scrub = self._on_timeline_scrub
        lay.addWidget(self.timeline)

        row = QHBoxLayout()
        add = QPushButton("＋ Додати рамку")
        add.clicked.connect(lambda: self.canvas.add_region())
        clr = QPushButton("Очистити")
        clr.clicked.connect(self._clear)
        ok = QPushButton("Готово")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Скасувати")
        cancel.clicked.connect(self.reject)
        row.addWidget(add)
        row.addWidget(clr)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)
        self._refresh()

    def _refresh(self) -> None:
        i = self.canvas.selected
        if 0 <= i < len(self.regions):
            self.timeline.set_region(self.regions[i], self.duration or 1.0)
            n = len(self.regions[i].intervals)
            self.lbl_sel.setText(f"Відрізки рамки №{i + 1}: {n}  (порожньо = весь ролик)")
            self.cmb_kind.blockSignals(True)
            idx = self.cmb_kind.findData(self.regions[i].kind)
            self.cmb_kind.setCurrentIndex(idx if idx >= 0 else 0)
            self.cmb_kind.setEnabled(True)
            self.cmb_kind.blockSignals(False)
        else:
            self.timeline.set_region(None, self.duration or 1.0)
            self.lbl_sel.setText("Відрізки: оберіть рамку (клік по ній)")
            self.cmb_kind.setEnabled(False)
        self.canvas.update()

    def _on_kind_changed(self, _i: int) -> None:
        i = self.canvas.selected
        if 0 <= i < len(self.regions):
            self.regions[i].kind = self.cmb_kind.currentData()

    # -- scrubbing (timeline click / bracket drag) --------------------- #
    def _on_timeline_scrub(self, t: float) -> None:
        self._seek(t, move_slider=True)

    def _seek(self, t: float, *, move_slider: bool) -> None:
        """Jump preview + playhead to ``t`` seconds (player if ready, else frame)."""
        t = min(max(0.0, t), self.duration)
        self.timeline.playhead = t
        self.timeline.update()
        if move_slider and self.slider is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 1000))
            self.slider.blockSignals(False)
        if self._player_ready and self._player is not None:
            self._player.setPosition(int(t * 1000))       # frame arrives via sink
        elif self._frame_provider is not None:
            pm = self._frame_provider(t)                  # fallback: extract 1 frame
            if pm is not None and not pm.isNull():
                self.canvas.set_pixmap(pm)

    # -- media player (smooth scrub + audio via H.264 proxy) ----------- #
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
            self.btn_play.setEnabled(False)              # until the proxy is ready
        # Coordinate with the main window's prewarm so we never build the same
        # proxy twice (concurrent full-res decodes compete for the CPU).
        import os
        if self._proxy_ready and os.path.exists(self._proxy_ready):
            self._on_proxy_ready(self._proxy_ready)      # already built by prewarm
        elif self._prewarm_worker is not None and self._prewarm_worker.isRunning():
            self.lbl_prep.setText("Готую плавний перегляд (перекодування у H.264)…")
            self.lbl_prep.setVisible(True)
            self._prewarm_worker.sig_progress.connect(self._on_proxy_progress)
            self._prewarm_worker.sig_ready.connect(self._on_proxy_ready)  # ride along
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
        # jog-audio PCM: decode in a plain thread (only sets a bytes attribute).
        if self._src:
            threading.Thread(
                target=lambda: setattr(self, "_pcm", proxy.load_pcm(self._src)),
                daemon=True).start()

    def _on_media_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and not self._priming \
                and not self._player_ready:
            # Qt's ffmpeg backend only pushes frames once the pipeline has been
            # primed by playing; after a brief muted prime, a paused setPosition()
            # emits exactly one frame — that is our instant scrub.
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
        self.timeline.playhead = t
        self.timeline.update()
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

    # -- slider drag: instant scrub + jog ("shuttle") audio ------------ #
    def _on_slider_pressed(self) -> None:
        if self._player is not None:
            self._player.pause()             # continuous audio off; jog takes over
        self._start_jog()

    def _on_slider_moved(self, v: int) -> None:
        t = v / 1000.0
        self.timeline.playhead = t
        self.timeline.update()
        if self._player_ready and self._player is not None:
            self._player.setPosition(v)          # cheap: 1 frame from primed player
            self._jog_grain(v)
        # Fallback (proxy not ready): do NOT extract per move — full-res decode on
        # every drag event is what made it lag. The frame updates on release only.

    def _on_slider_released(self) -> None:
        self._stop_jog()
        if self.slider is None:
            return
        t = self.slider.value() / 1000.0
        self.timeline.playhead = t
        self.timeline.update()
        if self._player_ready and self._player is not None:
            self._player.setPosition(int(t * 1000))
        elif self._frame_provider is not None:
            pm = self._frame_provider(t)          # one extraction, like before
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
            self._jog_io = sink.start()      # push mode: write PCM grains to this

    def _jog_grain(self, ms: int) -> None:
        """Play a short PCM window around ``ms`` so dragging is audible (shuttle)."""
        if self._jog_io is None or not self._pcm or self._audio_sink is None:
            return
        win = int(0.05 * proxy.JOG_SR)                   # 50 ms grain
        off = int((ms / 1000.0) * proxy.JOG_SR)
        start = max(0, off * 2)
        grain = self._pcm[start:start + win * 2]         # s16 = 2 bytes/sample
        if grain and self._audio_sink.bytesFree() >= len(grain):
            self._jog_io.write(grain)

    def _stop_jog(self) -> None:
        if self._audio_sink is not None:
            self._audio_sink.stop()
        self._jog_io = None

    def closeEvent(self, e) -> None:
        self._stop_jog()
        if self._player is not None:
            self._player.stop()
        # We may be riding along on the main window's prewarm worker (which we do
        # not own) — drop our connections so it can't call back into a closed dialog.
        if self._prewarm_worker is not None:
            for sig, slot in ((self._prewarm_worker.sig_ready, self._on_proxy_ready),
                              (self._prewarm_worker.sig_progress, self._on_proxy_progress)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        if self._proxy_worker is not None and self._proxy_worker.isRunning():
            self._proxy_worker.wait(2000)
        super().closeEvent(e)

    def _clear(self) -> None:
        self.regions.clear()
        self.canvas.selected = -1
        self._refresh()

    def result_regions(self) -> List[TimedRegion]:
        return [r for r in self.regions if r.rect[2] > 0.005 and r.rect[3] > 0.005]
