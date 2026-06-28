"""Main application window: every job setting exposed as its own control."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import settings
from ..media import ffmpeg
from ..models import (
    ASRQuality,
    Container,
    InpaintQuality,
    JobConfig,
    OutputConfig,
    RemovalConfig,
    SourceLang,
    SubtitlePlacement,
    SubtitlePosition,
    SubtitleStyle,
    TranslateConfig,
    TranslateMode,
    TranslateQuality,
    TTS_VOICES,
    TTSEngine,
    VideoCodec,
)
from .worker import DownloadWorker, JobWorker, SaveWorker

STAGE_LABELS = {
    "inpaint": "Видалення об'єктів",
    "audio": "Аудіо",
    "asr": "Розпізнавання мовлення",
    "translate": "Переклад",
    "tts": "Озвучення",
    "render": "Збирання відео",
}

_COLOR_PRESETS = [
    ("Білий", "#FFFFFF"), ("Чорний", "#000000"), ("Жовтий", "#FFE600"),
    ("Червоний", "#FF2D2D"), ("Зелений", "#3DDB5A"), ("Блакитний", "#37B6FF"),
    ("Рожевий", "#FF5FA2"), ("Помаранчевий", "#FF8A1E"),
]


def _combo(items: List[Tuple[str, object]]) -> QComboBox:
    box = QComboBox()
    for label, data in items:
        box.addItem(label, data)
    return box


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vidloc — локалізатор відео (UA)")
        self.setMinimumSize(780, 860)
        self.input_path = ""
        self.source_context = ""
        self.worker: JobWorker | None = None
        self.dl_worker: DownloadWorker | None = None
        self.save_worker: SaveWorker | None = None
        self._wm_regions: list = []
        self._hs_region = None
        self._build_ui()
        self._sync_enabled()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        outer.addWidget(self._input_section())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setSpacing(10)
        col.addWidget(self._downloader_section())
        col.addWidget(self._removal_section())
        col.addWidget(self._translate_section())
        col.addWidget(self._substyle_section())
        col.addWidget(self._quality_section())
        col.addWidget(self._tts_section())
        col.addWidget(self._output_section())
        col.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        outer.addWidget(self._run_section())

    def _input_section(self) -> QGroupBox:
        box = QGroupBox("Вхідне відео")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.ed_input = QLineEdit()
        self.ed_input.setReadOnly(True)
        self.ed_input.setPlaceholderText("Оберіть відеофайл…")
        btn = QPushButton("Обрати файл…")
        btn.clicked.connect(self._pick_input)
        row.addWidget(self.ed_input, 1)
        row.addWidget(btn)
        lay.addLayout(row)

        url_row = QHBoxLayout()
        self.ed_url = QLineEdit()
        self.ed_url.setPlaceholderText("…або встав посилання: YouTube / Instagram / TikTok")
        self.btn_dl = QPushButton("Завантажити")
        self.btn_dl.clicked.connect(self._download_url)
        url_row.addWidget(self.ed_url, 1)
        url_row.addWidget(self.btn_dl)
        lay.addLayout(url_row)

        self.lbl_info = QLabel("—")
        self.lbl_info.setStyleSheet("color: gray;")
        lay.addWidget(self.lbl_info)
        return box

    def _downloader_section(self) -> QGroupBox:
        box = QGroupBox("Завантажувач — зберегти на диск (без перекладу)")
        form = QFormLayout(box)
        self.cmb_dlq = _combo([
            ("Найкраще", "best"), ("1080p", "1080"), ("720p", "720"), ("480p", "480"),
        ])
        self.chk_audio = QCheckBox("Тільки звук (mp3)")
        self.chk_audio.toggled.connect(lambda on: self.cmb_dlq.setEnabled(not on))
        self.btn_save = QPushButton("⬇  Зберегти у файл")
        self.btn_save.clicked.connect(self._save_url)
        form.addRow("Якість:", self.cmb_dlq)
        form.addRow(self.chk_audio)
        form.addRow(self.btn_save)
        note = QLabel("Бере посилання з поля «Вхідне відео» вище. "
                      "Збереже у папку «Завантаження» (Downloads).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return box

    def _removal_section(self) -> QGroupBox:
        box = QGroupBox("Видалення об'єктів (вотермарки / вшиті субтитри)")
        form = QFormLayout(box)
        self.chk_wm = QCheckBox("Прибрати водяні знаки")
        btn_wm = QPushButton("Виділити область…")
        btn_wm.clicked.connect(self._select_wm_region)
        self.lbl_wm = QLabel("не виділено")
        self.lbl_wm.setStyleSheet("color: gray;")
        wm_row = QHBoxLayout()
        wm_row.addWidget(btn_wm)
        wm_row.addWidget(self.lbl_wm, 1)
        wm_wrap = QWidget()
        wm_wrap.setLayout(wm_row)

        self.chk_hs = QCheckBox("Прибрати вшиті субтитри")
        btn_hs = QPushButton("Виділити смугу субтитрів…")
        btn_hs.clicked.connect(self._select_hs_region)
        self.lbl_hs = QLabel("не виділено")
        self.lbl_hs.setStyleSheet("color: gray;")
        hs_row = QHBoxLayout()
        hs_row.addWidget(btn_hs)
        hs_row.addWidget(self.lbl_hs, 1)
        hs_wrap = QWidget()
        hs_wrap.setLayout(hs_row)

        self.cmb_inpaint = _combo([
            ("Збалансовано (нейро ÷2) — швидше", "balanced"),
            ("Найкраще (нейро, повний розмір)", "best"),
            ("Швидко (delogo) — грубо", "fast"),
        ])

        form.addRow(self.chk_wm)
        form.addRow("Область вотермарки:", wm_wrap)
        form.addRow(self.chk_hs)
        form.addRow("Смуга субтитрів:", hs_wrap)
        form.addRow("Якість затирання:", self.cmb_inpaint)
        note = QLabel("Виділення доступне після вибору відео. Нейро-режим на M2 "
                      "повільний (хвилини на ролик).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return box

    def _preview_pixmap(self):
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Немає відео", "Спочатку оберіть відеофайл.")
            return None
        frame = os.path.join(tempfile.gettempdir(), "vidloc_preview.png")
        try:
            ts = max(0.0, ffmpeg.probe(self.input_path).duration * 0.5)
        except Exception:  # noqa: BLE001
            ts = 1.0
        subprocess.run([settings.FFMPEG, "-y", "-loglevel", "error", "-ss", str(ts),
                        "-i", self.input_path, "-frames:v", "1", frame], check=False)
        if not os.path.exists(frame):
            QMessageBox.warning(self, "Помилка", "Не вдалося отримати кадр із відео.")
            return None
        from PySide6.QtGui import QPixmap
        return QPixmap(frame)

    def _select_wm_region(self) -> None:
        pix = self._preview_pixmap()
        if pix is None:
            return
        from .region_selector import RegionSelector
        dlg = RegionSelector(pix, self, "Виділіть водяний знак")
        if dlg.exec():
            self._wm_regions = dlg.regions()
            self.lbl_wm.setText(f"виділено рамок: {len(self._wm_regions)}"
                                if self._wm_regions else "не виділено")
            if self._wm_regions:
                self.chk_wm.setChecked(True)

    def _select_hs_region(self) -> None:
        pix = self._preview_pixmap()
        if pix is None:
            return
        from .region_selector import RegionSelector
        dlg = RegionSelector(pix, self, "Виділіть смугу субтитрів")
        if dlg.exec():
            regs = dlg.regions()
            self._hs_region = regs[0] if regs else None
            self.lbl_hs.setText("виділено" if self._hs_region else "не виділено")
            if self._hs_region:
                self.chk_hs.setChecked(True)

    def _translate_section(self) -> QGroupBox:
        box = QGroupBox("Переклад → українська")
        form = QFormLayout(box)
        self.cmb_src = _combo([
            ("Авто-визначення", "auto"), ("English", "en"),
            ("Русский", "ru"), ("Italiano", "it"),
        ])
        self.cmb_mode = _combo([
            ("Тільки субтитри", "subtitles"),
            ("Тільки озвучка (дубляж)", "voice"),
            ("Субтитри + озвучка", "both"),
            ("Без перекладу (лише обробка відео)", "none"),
        ])
        self.cmb_mode.currentIndexChanged.connect(self._sync_enabled)
        self.cmb_subpos = _combo([
            ("М'які — окрема доріжка (вмикаються в плеєрі)", "soft"),
            ("Вшити у відео (hardsub)", "burn"),
        ])
        self.chk_keepsrc = QCheckBox("Додатково зберегти субтитри мовою оригіналу")
        form.addRow("Мова оригіналу:", self.cmb_src)
        form.addRow("Що робимо:", self.cmb_mode)
        form.addRow("Субтитри:", self.cmb_subpos)
        form.addRow(self.chk_keepsrc)
        return box

    def _substyle_section(self) -> QGroupBox:
        box = QGroupBox("Стиль субтитрів")
        form = QFormLayout(box)
        self.cmb_subfont = _combo([
            ("Montserrat Black (жирний)", "Montserrat Black"),
            ("Montserrat ExtraBold", "Montserrat ExtraBold"),
            ("Montserrat Bold", "Montserrat"),
        ])
        self.cmb_position = _combo([
            ("По центру екрана", "center"), ("Знизу", "bottom"),
        ])
        self.cmb_chunk = _combo([
            ("Короткі (~20)", 20), ("Середні (~28)", 28),
            ("Довші (~42)", 42), ("Цілі речення", 0),
        ])
        self.cmb_chunk.setCurrentIndex(1)
        self.sld_subsize = QSlider(Qt.Orientation.Horizontal)
        self.sld_subsize.setRange(4, 12)
        self.sld_subsize.setValue(7)
        self.lbl_subsize = QLabel("7%")
        self.sld_subsize.valueChanged.connect(lambda v: self.lbl_subsize.setText(f"{v}%"))
        sz_row = QHBoxLayout()
        sz_row.addWidget(self.sld_subsize, 1)
        sz_row.addWidget(self.lbl_subsize)
        sz_wrap = QWidget()
        sz_wrap.setLayout(sz_row)
        self.chk_caps = QCheckBox("ВЕЛИКІ ЛІТЕРИ")
        form.addRow("Шрифт:", self.cmb_subfont)
        form.addRow("Положення:", self.cmb_position)
        form.addRow("Довжина реплік:", self.cmb_chunk)
        form.addRow("Розмір:", sz_wrap)
        form.addRow("Колір тексту:", self._color_control("_text_color", "#FFFFFF"))
        form.addRow("Колір обведення:", self._color_control("_outline_color", "#000000"))
        form.addRow(self.chk_caps)
        note = QLabel("Шрифт / розмір / положення / колір / КАПС діють лише для «Вшити у відео» "
                      "(м'які субтитри стилізує плеєр). Довжина реплік — для обох.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return box

    def _color_control(self, attr_name: str, default: str) -> QWidget:
        """Preset colours + a «Свій колір…» picker. Stores the hex in self.<attr_name>."""
        setattr(self, attr_name, default)
        combo = _combo(_COLOR_PRESETS + [("Свій колір…", "custom")])
        for i in range(combo.count()):
            if combo.itemData(i) == default:
                combo.setCurrentIndex(i)
                break
        swatch = QLabel()
        swatch.setFixedSize(34, 18)

        def update_swatch() -> None:
            swatch.setStyleSheet(
                f"background:{getattr(self, attr_name)}; border:1px solid gray;")

        def on_change() -> None:
            data = combo.currentData()
            if data == "custom":
                from PySide6.QtGui import QColor
                from PySide6.QtWidgets import QColorDialog
                c = QColorDialog.getColor(QColor(getattr(self, attr_name)), self, "Колір")
                if c.isValid():
                    setattr(self, attr_name, c.name())
            else:
                setattr(self, attr_name, data)
            update_swatch()

        combo.currentIndexChanged.connect(on_change)
        update_swatch()
        row = QHBoxLayout()
        row.addWidget(combo, 1)
        row.addWidget(swatch)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    def _quality_section(self) -> QGroupBox:
        box = QGroupBox("Швидкість і якість розпізнавання / перекладу")
        form = QFormLayout(box)
        self.cmb_asr = _combo([
            ("Швидко (medium)", "fast"),
            ("Збалансовано (large-v3-turbo)", "balanced"),
            ("Найкраще (large-v3)", "best"),
        ])
        self.cmb_asr.setCurrentIndex(2)
        self.cmb_mt = _combo([
            ("Натуральний LLM — жива мова, сленг (реком.)", "natural"),
            ("Швидко (NLLB 600M) — дослівно", "fast"),
            ("Найкраще (NLLB 1.3B) — дослівно", "best"),
        ])
        self.sld_beam = QSlider(Qt.Orientation.Horizontal)
        self.sld_beam.setRange(1, 8)
        self.sld_beam.setValue(5)
        self.lbl_beam = QLabel("5")
        self.sld_beam.valueChanged.connect(lambda v: self.lbl_beam.setText(str(v)))
        beam_row = QHBoxLayout()
        beam_row.addWidget(self.sld_beam, 1)
        beam_row.addWidget(self.lbl_beam)
        beam_wrap = QWidget()
        beam_wrap.setLayout(beam_row)
        form.addRow("Якість розпізнавання:", self.cmb_asr)
        form.addRow("Якість перекладу:", self.cmb_mt)
        form.addRow("Глибина пошуку (beam):", beam_wrap)
        hint = QLabel("Більше beam = якісніше, але повільніше.")
        hint.setStyleSheet("color: gray;")
        form.addRow("", hint)
        return box

    def _tts_section(self) -> QGroupBox:
        box = QGroupBox("Озвучення (дубляж)")
        form = QFormLayout(box)
        self.cmb_tts = _combo([
            ("Ukrainian-TTS — 5 голосів (рекомендовано)", "uktts"),
            ("Piper — 3 голоси, інший тембр", "piper"),
            ("Coqui VITS — 1 голос (запасний)", "coqui_uk"),
        ])
        self.cmb_tts.currentIndexChanged.connect(self._refresh_voices)
        self.cmb_voice = QComboBox()
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(80, 120)       # 0.80x .. 1.20x
        self.sld_speed.setValue(100)
        self.lbl_speed = QLabel("1.00×")
        self.sld_speed.valueChanged.connect(
            lambda v: self.lbl_speed.setText(f"{v / 100:.2f}×"))
        sp_row = QHBoxLayout()
        sp_row.addWidget(self.sld_speed, 1)
        sp_row.addWidget(self.lbl_speed)
        sp_wrap = QWidget()
        sp_wrap.setLayout(sp_row)
        self.chk_bg = QCheckBox("Залишити оригінальний звук (музику) тихо під озвучкою")
        self.chk_bg.setChecked(False)
        form.addRow("Рушій:", self.cmb_tts)
        form.addRow("Голос:", self.cmb_voice)
        form.addRow("Темп озвучки:", sp_wrap)
        hint = QLabel("Темп підлаштовується під оригінал. Менше = спокійніше, більше = жвавіше.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow("", hint)
        form.addRow(self.chk_bg)
        self._refresh_voices()
        return box

    def _output_section(self) -> QGroupBox:
        box = QGroupBox("Вихідне відео")
        form = QFormLayout(box)
        self.cmb_container = _combo([
            ("MP4", "mp4"), ("MKV", "mkv"), ("MOV", "mov"), ("WebM", "webm"),
        ])
        self.cmb_container.currentIndexChanged.connect(self._suggest_output)
        self.cmb_codec = _combo([
            ("H.264 (сумісність)", "h264"),
            ("H.265 / HEVC (менший розмір)", "h265"),
            ("Копіювати без перекодування", "keep"),
        ])
        self.cmb_codec.currentIndexChanged.connect(self._sync_enabled)
        self.cmb_res = _combo([
            ("Як в оригіналі", 0), ("2160p (4K)", 2160), ("1440p", 1440),
            ("1080p", 1080), ("720p", 720), ("480p", 480),
        ])
        self.cmb_fps = _combo([
            ("Як в оригіналі", 0), ("24", 24), ("25", 25), ("30", 30), ("60", 60),
        ])
        self.cmb_abr = _combo([
            ("128 kbps", "128k"), ("192 kbps", "192k"),
            ("256 kbps", "256k"), ("320 kbps", "320k"),
        ])
        self.cmb_abr.setCurrentIndex(1)
        self.sld_crf = QSlider(Qt.Orientation.Horizontal)
        self.sld_crf.setRange(14, 30)
        self.sld_crf.setValue(20)
        self.lbl_crf = QLabel("20")
        self.sld_crf.valueChanged.connect(lambda v: self.lbl_crf.setText(str(v)))
        crf_row = QHBoxLayout()
        crf_row.addWidget(self.sld_crf, 1)
        crf_row.addWidget(self.lbl_crf)
        crf_wrap = QWidget()
        crf_wrap.setLayout(crf_row)

        form.addRow("Контейнер:", self.cmb_container)
        form.addRow("Відеокодек:", self.cmb_codec)
        form.addRow("Роздільність:", self.cmb_res)
        form.addRow("Якість (CRF, менше=краще):", crf_wrap)
        form.addRow("Частота кадрів:", self.cmb_fps)
        form.addRow("Бітрейт аудіо:", self.cmb_abr)
        self.chk_hw = QCheckBox("Швидке апаратне кодування (VideoToolbox)")
        form.addRow(self.chk_hw)

        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Шлях для збереження результату…")
        btn_out = QPushButton("Змінити…")
        btn_out.clicked.connect(self._pick_output)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_out)
        out_wrap = QWidget()
        out_wrap.setLayout(out_row)
        form.addRow("Зберегти як:", out_wrap)
        return box

    def _run_section(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Старт")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Скасувати")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_start, 1)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        lay.addWidget(self.progress)
        self.lbl_stage = QLabel("Готовий до роботи.")
        lay.addWidget(self.lbl_stage)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        lay.addWidget(self.log)
        return wrap

    # ------------------------------------------------------------------ #
    # Behaviour
    # ------------------------------------------------------------------ #
    def _refresh_voices(self) -> None:
        engine = TTSEngine(self.cmb_tts.currentData())
        self.cmb_voice.clear()
        for label, vid in TTS_VOICES.get(engine, []):
            self.cmb_voice.addItem(label, vid)

    def _sync_enabled(self) -> None:
        mode = self.cmb_mode.currentData()
        wants_text = mode in ("subtitles", "both")
        wants_voice = mode in ("voice", "both")
        wants_any = mode != "none"

        self.cmb_src.setEnabled(wants_any)
        self.cmb_asr.setEnabled(wants_any)
        self.cmb_mt.setEnabled(wants_any)
        self.sld_beam.setEnabled(wants_any)
        self.cmb_subpos.setEnabled(wants_text)
        self.chk_keepsrc.setEnabled(wants_text)
        for w in (self.cmb_tts, self.cmb_voice, self.chk_bg, self.sld_speed):
            w.setEnabled(wants_voice)
        # CRF only matters when we actually (re-)encode
        self.sld_crf.setEnabled(self.cmb_codec.currentData() != "keep")

    def _pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть відео", str(Path.home()),
            "Відео (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.flv);;Усі файли (*.*)",
        )
        if not path:
            return
        self.input_path = path
        self.source_context = ""
        self.ed_input.setText(path)
        try:
            info = ffmpeg.probe(path)
            self.lbl_info.setText(
                f"{info.resolution} · {info.duration:.1f} с · {info.fps:.0f} fps · "
                f"відео: {info.v_codec or '—'} · аудіо: {info.a_codec or 'немає'}")
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"Не вдалося прочитати файл: {exc}")
        self._suggest_output()

    def _pick_output(self) -> None:
        container = self.cmb_container.currentData()
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти як", self.ed_out.text() or str(settings.OUTPUT_DIR),
            f"Відео (*.{container})")
        if path:
            self.ed_out.setText(path)

    def _suggest_output(self) -> None:
        if not self.input_path:
            return
        container = self.cmb_container.currentData()
        stem = Path(self.input_path).stem
        self.ed_out.setText(str(settings.OUTPUT_DIR / f"{stem}_uk.{container}"))

    def _download_url(self) -> None:
        url = self.ed_url.text().strip()
        if not url:
            QMessageBox.information(self, "Посилання", "Встав посилання на відео.")
            return
        self.btn_dl.setEnabled(False)
        self.lbl_info.setText("Завантаження…")
        self.dl_worker = DownloadWorker(url)
        self.dl_worker.sig_progress.connect(
            lambda f, m: self.lbl_info.setText(f"{m} {int(f * 100)}%"))
        self.dl_worker.sig_log.connect(self._on_log)
        self.dl_worker.sig_done.connect(self._on_download_done)
        self.dl_worker.sig_fail.connect(self._on_download_fail)
        self.dl_worker.start()

    def _on_download_done(self, path: str, context: str) -> None:
        self.btn_dl.setEnabled(True)
        self.input_path = path
        self.source_context = context
        self.ed_input.setText(path)
        title = ""
        if context.startswith("Title:"):
            title = " · " + context.split("|")[0].replace("Title:", "").strip()
        try:
            info = ffmpeg.probe(path)
            self.lbl_info.setText(
                f"{info.resolution} · {info.duration:.1f} с · {info.fps:.0f} fps{title}")
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"Завантажено, але читання не вдалося: {exc}")
        self._suggest_output()

    def _on_download_fail(self, message: str) -> None:
        self.btn_dl.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_info.setText("Помилка завантаження ⚠")
        QMessageBox.critical(
            self, "Не вдалося завантажити",
            message + "\n\nПримітка: приватні або деякі Instagram-відео можуть "
            "потребувати входу (cookies).")

    def _save_url(self) -> None:
        url = self.ed_url.text().strip()
        if not url:
            QMessageBox.information(self, "Посилання", "Встав посилання у поле «Вхідне відео».")
            return
        dest = str(Path.home() / "Downloads")
        self.btn_save.setEnabled(False)
        self.lbl_info.setText("Завантаження…")
        self.save_worker = SaveWorker(
            url, dest, self.cmb_dlq.currentData(), self.chk_audio.isChecked())
        self.save_worker.sig_progress.connect(
            lambda f, m: self.lbl_info.setText(f"{m} {int(f * 100)}%"))
        self.save_worker.sig_log.connect(self._on_log)
        self.save_worker.sig_done.connect(self._on_save_done)
        self.save_worker.sig_fail.connect(self._on_download_fail)
        self.save_worker.start()

    def _on_save_done(self, path: str) -> None:
        self.btn_save.setEnabled(True)
        self.lbl_info.setText("Збережено ✓")
        box = QMessageBox(self)
        box.setWindowTitle("Збережено")
        box.setText(f"Файл збережено:\n{path}")
        reveal = box.addButton("Показати у Finder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Закрити", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is reveal:
            subprocess.run(["open", "-R", path], check=False)

    def _build_job(self) -> JobConfig:
        out = OutputConfig(
            container=Container(self.cmb_container.currentData()),
            codec=VideoCodec(self.cmb_codec.currentData()),
            height=(self.cmb_res.currentData() or None),
            crf=self.sld_crf.value(),
            fps=(self.cmb_fps.currentData() or None),
            audio_bitrate=self.cmb_abr.currentData(),
            hw_encode=self.chk_hw.isChecked(),
        )
        style = SubtitleStyle(
            font=self.cmb_subfont.currentData(),
            font_size_pct=float(self.sld_subsize.value()),
            position=SubtitlePosition(self.cmb_position.currentData()),
            max_chars=int(self.cmb_chunk.currentData()),
            uppercase=self.chk_caps.isChecked(),
            primary_color=self._text_color,
            outline_color=self._outline_color,
        )
        tr = TranslateConfig(
            mode=TranslateMode(self.cmb_mode.currentData()),
            source=SourceLang(self.cmb_src.currentData()),
            asr_quality=ASRQuality(self.cmb_asr.currentData()),
            translate_quality=TranslateQuality(self.cmb_mt.currentData()),
            beam_size=self.sld_beam.value(),
            subtitle_placement=SubtitlePlacement(self.cmb_subpos.currentData()),
            subtitle_style=style,
            keep_source_subtitles=self.chk_keepsrc.isChecked(),
            tts_engine=TTSEngine(self.cmb_tts.currentData()),
            tts_voice=self.cmb_voice.currentData(),
            tts_clone_reference=None,
            keep_background=self.chk_bg.isChecked(),
            tts_speed=self.sld_speed.value() / 100.0,
        )
        rm = RemovalConfig(
            remove_watermark=self.chk_wm.isChecked() and bool(self._wm_regions),
            watermark_regions=list(self._wm_regions),
            remove_hardsubs=self.chk_hs.isChecked() and self._hs_region is not None,
            hardsub_region=self._hs_region,
            quality=InpaintQuality(self.cmb_inpaint.currentData()),
        )
        return JobConfig(input_path=self.input_path,
                         output_path=self.ed_out.text(),
                         source_context=self.source_context,
                         output=out, removal=rm, translate=tr)

    def _start(self) -> None:
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Немає файлу", "Спочатку оберіть вхідне відео.")
            return
        job = self._build_job()
        self.log.clear()
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = JobWorker(job)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_stage.connect(self._on_stage)
        self.worker.sig_log.connect(self._on_log)
        self.worker.sig_done.connect(self._on_done)
        self.worker.sig_fail.connect(self._on_fail)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_stage.setText("Скасування…")

    # -- worker callbacks ------------------------------------------------ #
    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(frac * 1000))
        if msg:
            self.lbl_stage.setText(msg)

    def _on_stage(self, stage: str) -> None:
        self.lbl_stage.setText(STAGE_LABELS.get(stage, stage))

    def _on_log(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip())

    def _on_done(self, out_path: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_stage.setText("Готово ✅")
        box = QMessageBox(self)
        box.setWindowTitle("Готово")
        box.setText(f"Файл збережено:\n{out_path}")
        reveal = box.addButton("Показати у Finder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Закрити", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is reveal:
            subprocess.run(["open", "-R", out_path], check=False)

    def _on_fail(self, message: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_stage.setText("Помилка ⚠")
        QMessageBox.critical(self, "Помилка", message)
