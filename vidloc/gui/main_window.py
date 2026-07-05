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
    QDialog,
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

from .. import config, settings
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
    SubtitleStyle,
    TranslateConfig,
    TranslateMode,
    TranslateQuality,
    TTS_VOICES,
    TTSEngine,
    VideoCodec,
)
from .region_editor import RegionEditor
from .subtitle_editor import SubtitleEditor
from .worker import (
    CookieCheckWorker,
    DownloadWorker,
    JobWorker,
    ProxyWorker,
    RecognizeWorker,
    RemixWorker,
    SaveWorker,
)

STAGE_LABELS = {
    "inpaint": "Видалення об'єктів",
    "audio": "Аудіо",
    "asr": "Розпізнавання мовлення",
    "translate": "Переклад",
    "tts": "Озвучення",
    "render": "Збирання відео",
}

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
        self.workspace_path = ""          # set once recognition has run for this video
        self._recognize_after_dl = False  # run recognition once a download finishes
        self.worker: JobWorker | None = None
        self.remix_worker: RemixWorker | None = None
        self._last_job = None                 # the JobConfig of the last render (for re-mix)
        self._remix_ctx: dict | None = None   # {output, source, dub, ...} of the last dubbed render
        self.dl_worker: DownloadWorker | None = None
        self.save_worker: SaveWorker | None = None
        self.recog_worker: RecognizeWorker | None = None
        self._cookie_worker: CookieCheckWorker | None = None
        self._proxy_worker: ProxyWorker | None = None
        self._proxy_ready_path = ""        # editor preview proxy, once prewarmed
        self._can_start = False           # main «Старт» unlocked only after removal step
        self._removal_run = False         # a removal-only job is currently in flight
        self._clean_output = ""           # last produced watermark-free video
        self._hs_region = None
        self._regions: list = []          # multi-region editor: normalized (x,y,w,h) rects
        self._captions: list = []         # subtitle editor: list[Caption] (subs + dub source)
        self._sub_default_style = SubtitleStyle()  # global fallback for caption styling
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
        self.sec_downloader = self._downloader_section()
        self.sec_recog = self._recog_section()
        self.sec_removal = self._removal_section()
        self.sec_translate = self._translate_section()
        self.sec_subedit = self._subtitles_section()
        self.sec_tts = self._tts_section()
        self.sec_output = self._output_section()
        for sec in (self.sec_downloader, self.sec_recog, self.sec_removal,
                    self.sec_translate, self.sec_subedit, self.sec_tts,
                    self.sec_output):
            col.addWidget(sec)
        col.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        outer.addWidget(self._run_section())

        # Progressive disclosure: the first screen shows only «Вхідне відео» +
        # «Завантажувач». A video reveals «Розпізнавання»; once recognition has
        # run, that step disappears and the remaining sections appear.
        for sec in (self.sec_recog, self.sec_removal, self.sec_translate,
                    self.sec_subedit, self.sec_tts, self.sec_output):
            sec.setVisible(False)
        self.ed_url.textChanged.connect(self._on_url_changed)
        self.btn_start.setEnabled(False)   # unlocked only after the removal step

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

        lay.addLayout(self._cookies_row())

        self.lbl_info = QLabel("—")
        self.lbl_info.setStyleSheet("color: gray;")
        lay.addWidget(self.lbl_info)
        return box

    def _cookies_row(self) -> QHBoxLayout:
        """Instagram/private downloads need a logged-in browser session. Let the user
        pick which browser (or a cookies.txt) and test it — no env vars needed."""
        self.cmb_cookies = _combo([
            ("Авто (визначити сам)", "auto"),
            ("Google Chrome", "chrome"),
            ("Firefox", "firefox"),
            ("Safari", "safari"),
            ("Brave", "brave"),
            ("Microsoft Edge", "edge"),
            ("Chromium", "chromium"),
            ("Файл cookies.txt…", "file"),
        ])
        self.cmb_cookies.setToolTip(
            "Для приватних / залогінених відео (Instagram) застосунок бере куки з "
            "браузера, у якому ти увійшов у акаунт. Обери свій браузер і натисни "
            "«Перевірити». Safari потребує macOS-дозволу «Повний доступ до диска».")
        self.btn_check_cookies = QPushButton("Перевірити")
        self.btn_check_cookies.clicked.connect(self._check_cookies)
        self.lbl_cookies = QLabel("")
        self.lbl_cookies.setStyleSheet("color: gray;")
        self._load_cookies_ui()
        self.cmb_cookies.currentIndexChanged.connect(self._on_cookies_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel("Instagram-вхід:"))
        row.addWidget(self.cmb_cookies)
        row.addWidget(self.btn_check_cookies)
        row.addWidget(self.lbl_cookies, 1)
        return row

    def _load_cookies_ui(self) -> None:
        """Reflect the saved cookie choice in the combo + label."""
        self.cmb_cookies.blockSignals(True)
        cfile = config.get("cookies_file")
        if cfile:
            self.cmb_cookies.setCurrentIndex(self.cmb_cookies.findData("file"))
            self.lbl_cookies.setText(f"файл: {Path(cfile).name}")
        else:
            browser = config.get("cookies_browser") or "auto"
            idx = self.cmb_cookies.findData(browser)
            self.cmb_cookies.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_cookies.blockSignals(False)

    def _on_cookies_changed(self) -> None:
        val = self.cmb_cookies.currentData()
        if val == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Оберіть cookies.txt", str(Path.home()),
                "Cookies (*.txt);;Усі файли (*.*)")
            if path:
                config.set("cookies_file", path)
                config.set("cookies_browser", "")
                self.lbl_cookies.setText(f"файл: {Path(path).name}")
            else:
                self._load_cookies_ui()       # cancelled → restore previous choice
            return
        config.set("cookies_browser", val)
        config.set("cookies_file", "")
        self.lbl_cookies.setText("")

    def _check_cookies(self) -> None:
        val = self.cmb_cookies.currentData()
        browser = "" if val in ("auto", "file") else val
        cfile = config.get("cookies_file") if val == "file" else ""
        self.btn_check_cookies.setEnabled(False)
        self.lbl_cookies.setText("Перевіряю…")
        self.lbl_cookies.setStyleSheet("color:#37b6ff;")
        self._cookie_worker = CookieCheckWorker(browser, cfile, self)
        self._cookie_worker.sig_done.connect(self._on_cookies_checked)
        self._cookie_worker.start()

    def _on_cookies_checked(self, ok: bool, msg: str) -> None:
        self.btn_check_cookies.setEnabled(True)
        self.lbl_cookies.setText(msg)
        self.lbl_cookies.setStyleSheet("color:#2d7d46;" if ok else "color:#c8442e;")

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
        self.chk_do_removal = QCheckBox("Видаляти вотермарки / вшиті субтитри")
        self.chk_do_removal.setChecked(True)
        self.chk_do_removal.toggled.connect(self._on_removal_toggled)
        form.addRow(self.chk_do_removal)

        self.chk_hs = QCheckBox("Прибрати вшиті субтитри/вотермарки")
        self.btn_hs = QPushButton("Виділити область…")
        self.btn_hs.clicked.connect(self._select_hs_region)
        self.lbl_hs = QLabel("не виділено")
        self.lbl_hs.setStyleSheet("color: gray;")
        hs_row = QHBoxLayout()
        hs_row.addWidget(self.btn_hs)
        hs_row.addWidget(self.lbl_hs, 1)
        hs_wrap = QWidget()
        hs_wrap.setLayout(hs_row)

        self.cmb_inpaint = _combo([
            ("Збалансовано (нейро ÷2) — швидше", "balanced"),
            ("Найкраще (нейро, повний розмір)", "best"),
            ("Швидко (delogo) — грубо", "fast"),
        ])

        # encoder speed for the removal output (data = (libx264 preset, hardware VT?))
        self.cmb_enc = _combo([
            ("Швидко (x264 veryfast) — реком.", ("veryfast", False)),
            ("Якісно (x264 medium) — повільно", ("medium", False)),
            ("Апаратно (VideoToolbox) — швидко, більший файл", ("veryfast", True)),
        ])
        self.cmb_enc.setToolTip(
            "Кодування результату затирання. «Швидко (veryfast)» — приблизно вдвічі швидше "
            "за medium і навіть краще стискає, ніж апаратний; рекомендовано. «Апаратно "
            "(VideoToolbox)» — медіа-рушій M2, теж швидко, але файл більший.")

        self.chk_reuse = QCheckBox("Прискорення reuse-first (для рамок типу «Об'єкт/лого»)")
        self.chk_reuse.setChecked(True)
        self.chk_reuse.setToolTip(
            "Для рамок типу «Об'єкт/лого»: затирає LaMa лише там, де фон змінився, а на "
            "стабільних кадрах повторно використовує вже очищений фон. Кратно швидше на "
            "статичному лого та субтитрах із паузами. Тип рамки задається у редакторі "
            "областей; для рамок «Текст» reuse не застосовується.")

        self.chk_master = QCheckBox("Читати з H.264-майстра (швидкий декод AV1)")
        self.chk_master.setChecked(True)
        self.chk_master.setToolTip(
            "Затирання декодує підготовлений повнорозмірний H.264-майстер (той самий, "
            "що й для перегляду) замість повільного AV1 — декод ≈10× швидший, а воркер "
            "читає джерело двічі за прохід. Найбільший виграш на статиці / субтитрах із "
            "паузами. Вимкни, якщо потрібен піксель-в-піксель із оригінального AV1.")

        self.btn_regions = QPushButton("Області (декілька)…")
        self.btn_regions.clicked.connect(self._edit_regions)
        self.lbl_regions = QLabel("рамок: 0")
        self.lbl_regions.setStyleSheet("color: gray;")
        reg_row = QHBoxLayout()
        reg_row.addWidget(self.btn_regions)
        reg_row.addWidget(self.lbl_regions, 1)
        reg_wrap = QWidget()
        reg_wrap.setLayout(reg_row)

        form.addRow(self.chk_hs)
        form.addRow("Область:", hs_wrap)
        form.addRow("Кілька областей:", reg_wrap)
        form.addRow("Якість затирання:", self.cmb_inpaint)
        form.addRow("Кодування:", self.cmb_enc)
        form.addRow(self.chk_reuse)
        form.addRow(self.chk_master)
        note = QLabel("Виділення доступне після вибору відео. Тип кожної рамки (об'єкт "
                      "або текст) задається в редакторі областей. Нейро-режим на M2 "
                      "повільний (хвилини на ролик).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        self.btn_inpaint = QPushButton("Прибрати →")
        self.btn_inpaint.setObjectName("inpaintBtn")
        self.btn_inpaint.clicked.connect(self._on_run_removal)
        self.btn_skip_removal = QPushButton("Пропустити")
        self.btn_skip_removal.clicked.connect(self._on_skip_removal)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_inpaint, 1)
        btn_row.addWidget(self.btn_skip_removal)
        btn_wrap = QWidget()
        btn_wrap.setLayout(btn_row)
        form.addRow(btn_wrap)
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

    def _select_hs_region(self) -> None:
        pix = self._preview_pixmap()
        if pix is None:
            return
        from .region_selector import RegionSelector
        dlg = RegionSelector(pix, self, "Виділіть смугу субтитрів / вотермарку")
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
        self.cmb_mt = _combo([
            ("Натуральний LLM — жива мова, сленг (реком.)", "natural"),
            ("Dragoman — EN→UK спеціалізований", "dragoman"),
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
        self.chk_keepsrc = QCheckBox("Додатково зберегти субтитри мовою оригіналу")
        form.addRow("Мова оригіналу:", self.cmb_src)
        form.addRow("Що робимо:", self.cmb_mode)
        form.addRow("Якість перекладу:", self.cmb_mt)
        form.addRow("Глибина пошуку (beam):", beam_wrap)
        form.addRow(self.chk_keepsrc)
        note = QLabel("Субтитри вшиваються (burn) через «Редактор субтитрів» нижче. "
                      "Більше beam = якісніше, але повільніше.")
        note.setStyleSheet("color: gray;")
        form.addRow("", note)
        return box

    def _subtitles_section(self) -> QGroupBox:
        box = QGroupBox("Субтитри (візуальний редактор)")
        form = QFormLayout(box)
        self.btn_subedit = QPushButton("Відкрити редактор субтитрів…")
        self.btn_subedit.clicked.connect(self._edit_subtitles)
        self.lbl_subs = QLabel("субтитрів: 0")
        self.lbl_subs.setStyleSheet("color: gray;")
        row = QHBoxLayout()
        row.addWidget(self.btn_subedit)
        row.addWidget(self.lbl_subs, 1)
        wrap = QWidget()
        wrap.setLayout(row)
        form.addRow(wrap)
        note = QLabel("У редакторі розставляєш субтитри рамками на кадрі, задаєш час на "
                      "таймлайні та стиль (шрифт/колір/обведення) для кожної репліки. "
                      "«Підтягнути переклад» заповнить їх перекладеним текстом. Ці ж репліки "
                      "керують озвучкою — дубляж говорить те, що на екрані.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return box

    def _edit_subtitles(self) -> None:
        """Open the visual subtitle editor; captions persist on the window state."""
        pix = self._preview_pixmap()
        if pix is None:
            return
        dur, vw, vh = 0.0, 0, 0
        try:
            info = ffmpeg.probe(self.input_path)
            dur, vw, vh = info.duration, info.width, info.height
        except Exception:  # noqa: BLE001
            pass
        dlg = SubtitleEditor(
            pix, self._captions, default_style=self._sub_default_style, duration=dur,
            frame_provider=self._frame_at, src=self.input_path,
            proxy_ready=self._proxy_ready_path or None, proxy_worker=self._proxy_worker,
            job=self._build_job(), workspace_root=self.workspace_path,
            video_w=vw, video_h=vh, parent=self)
        if dlg.exec():
            self._captions = dlg.result_captions()
            self.lbl_subs.setText(f"субтитрів: {len(self._captions)}")

    def _recog_section(self) -> QGroupBox:
        box = QGroupBox("Розпізнавання тексту")
        form = QFormLayout(box)
        self.chk_do_recog = QCheckBox("Розпізнавати текст")
        self.chk_do_recog.setChecked(True)
        self.chk_do_recog.toggled.connect(self._on_recog_toggled)
        form.addRow(self.chk_do_recog)
        self.cmb_asr = _combo([
            ("Швидко (medium)", "fast"),
            ("Збалансовано (large-v3-turbo)", "balanced"),
            ("Найкраще (large-v3)", "best"),
        ])
        self.cmb_asr.setCurrentIndex(2)
        form.addRow("Модель розпізнавання:", self.cmb_asr)
        note = QLabel("Оберіть модель і натисніть «Розпізнати»: текст розпізнається "
                      "один раз і збережеться в кеш. Після цього крок зникне з налаштувань.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        self.btn_recog = QPushButton("Розпізнати →")
        self.btn_recog.setObjectName("recogBtn")
        self.btn_recog.clicked.connect(self._on_recognize)
        self.btn_skip_recog = QPushButton("Пропустити")
        self.btn_skip_recog.clicked.connect(self._on_skip_recog)
        row = QHBoxLayout()
        row.addWidget(self.btn_recog, 1)
        row.addWidget(self.btn_skip_recog)
        wrap = QWidget()
        wrap.setLayout(row)
        form.addRow(wrap)
        return box

    def _tts_section(self) -> QGroupBox:
        box = QGroupBox("Озвучення (дубляж)")
        form = QFormLayout(box)
        self.cmb_tts = _combo([
            ("StyleTTS2 Ukrainian — натурально (рекомендовано)", "styletts2_uk"),
            ("Ukrainian-TTS — 5 голосів", "uktts"),
            ("Piper — 3 голоси, інший тембр", "piper"),
            ("Coqui VITS — 1 голос (запасний)", "coqui_uk"),
            ("F5-TTS — клон голосу з оригіналу (експеримент)", "f5_uk"),
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
        # Original-track level under the dub: 0% = clean voiceover (replace original),
        # up to 100% = original at full. It is the *relative* level vs. the dub, so it
        # can be retuned onto a finished render without re-encoding (see «Пересобрати звук»).
        self.sld_bg = QSlider(Qt.Orientation.Horizontal)
        self.sld_bg.setRange(0, 100)
        self.sld_bg.setValue(0)
        self.lbl_bg = QLabel("вимк.")
        self.sld_bg.valueChanged.connect(
            lambda v: self.lbl_bg.setText("вимк." if v == 0 else f"{v}%"))
        bg_row = QHBoxLayout()
        bg_row.addWidget(self.sld_bg, 1)
        bg_row.addWidget(self.lbl_bg)
        bg_wrap = QWidget()
        bg_wrap.setLayout(bg_row)
        form.addRow("Рушій:", self.cmb_tts)
        form.addRow("Голос:", self.cmb_voice)
        form.addRow("Темп озвучки:", sp_wrap)
        hint = QLabel("Темп підлаштовується під оригінал. Менше = спокійніше, більше = жвавіше.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow("", hint)
        form.addRow("Оригінал під озвучкою:", bg_wrap)
        bg_hint = QLabel("0% = лише озвучка (чистий голос). Більше = гучніше музика/оригінал під нею.")
        bg_hint.setWordWrap(True)
        bg_hint.setStyleSheet("color: gray;")
        form.addRow("", bg_hint)
        # After a dubbed render, retune this level on the finished file in seconds —
        # rebuilds only the audio (stream-copies the picture), no re-encode / no re-TTS.
        self.btn_remix = QPushButton("Пересобрати звук останнього відео (без перекодування)")
        self.btn_remix.setEnabled(False)
        self.btn_remix.clicked.connect(self._remix)
        form.addRow("", self.btn_remix)
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
        self.cmb_mt.setEnabled(wants_any)
        self.sld_beam.setEnabled(wants_any)
        self.chk_keepsrc.setEnabled(wants_text)
        # The caption editor drives both burned subtitles AND dubbing, so it is
        # relevant whenever we produce text or voice. In voice-only mode the captions
        # are not burned — they just steer what the dub says (and when).
        self.sec_subedit.setEnabled(wants_text or wants_voice)
        for w in (self.cmb_tts, self.cmb_voice, self.sld_bg, self.sld_speed):
            w.setEnabled(wants_voice)
        # CRF only matters when we actually (re-)encode
        self.sld_crf.setEnabled(self.cmb_codec.currentData() != "keep")

    def _on_url_changed(self, text: str) -> None:
        """Pasting/typing a link into the URL field reveals «Розпізнавання»."""
        self.sec_recog.setVisible(bool(text.strip()))

    def _on_recognize(self) -> None:
        """«Розпізнати»: make sure the video is in a workspace, then run ASR."""
        if self.input_path and os.path.exists(self.input_path):
            self._do_recognize()
        elif self.ed_url.text().strip():
            # No local file yet — download the link first, then recognize.
            self._recognize_after_dl = True
            self._download_url()
        else:
            QMessageBox.information(
                self, "Немає відео",
                "Спочатку оберіть файл або встав посилання на відео.")

    def _do_recognize(self) -> None:
        """Cache the source into the workspace, then run (or reuse) ASR."""
        import shutil
        from .. import workspace as wsmod

        ws = wsmod.resolve(Path(self.input_path).stem)
        src = ws.source_video()
        if src is None:
            dst = ws.source / Path(self.input_path).name
            try:
                shutil.copy2(self.input_path, dst)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(
                    self, "Workspace", f"Не вдалося скопіювати відео у Source: {exc}")
                return
            src = dst
        ws.write_meta({
            "name": Path(self.input_path).stem,
            "source_context": self.source_context,
            "original_path": self.input_path,
        })
        self.workspace_path = str(ws.root)
        self.input_path = str(src)
        self.ed_input.setText(self.input_path)
        self._prewarm_proxy()                # overlap proxy build with ASR (lead time)

        model = self.cmb_asr.currentData()   # 'fast' | 'balanced' | 'best'
        if ws.load_transcript(model):
            QMessageBox.information(
                self, "Розпізнавання",
                f"Ця настройка розпізнавання ({model}) для цього відео вже виконана — "
                "використаю збережений транскрипт із кешу.")
            self._finish_recognition()
            return

        # Run ASR off the UI thread; the step disappears once it succeeds.
        self.btn_recog.setEnabled(False)
        self.btn_skip_recog.setEnabled(False)
        self.progress.setRange(0, 0)            # busy/indeterminate while recognising
        self.lbl_stage.setText("Розпізнавання тексту…")
        self.lbl_info.setText("Розпізнавання…")
        self.recog_worker = RecognizeWorker(
            self.input_path, self.workspace_path, model, self.cmb_src.currentData())
        self.recog_worker.sig_progress.connect(lambda f, m: self.lbl_stage.setText(m))
        self.recog_worker.sig_log.connect(self._on_log)
        self.recog_worker.sig_done.connect(self._on_recognize_done)
        self.recog_worker.sig_fail.connect(self._on_recognize_fail)
        self.recog_worker.start()

    def _on_recognize_done(self, n_segments: int) -> None:
        self._reset_progress()
        self.btn_recog.setEnabled(True)
        self.btn_skip_recog.setEnabled(True)
        self.lbl_info.setText(f"Розпізнано сегментів: {n_segments} ✓")
        self.lbl_stage.setText("Готовий до роботи.")
        self._finish_recognition()

    def _on_recognize_fail(self, message: str) -> None:
        self._reset_progress()
        self.btn_recog.setEnabled(True)
        self.btn_skip_recog.setEnabled(True)
        self.lbl_info.setText("Помилка розпізнавання ⚠")
        QMessageBox.critical(self, "Розпізнавання", message)

    def _reset_progress(self) -> None:
        """Restore the determinate 0–100 % bar after an indeterminate phase."""
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)

    def _on_recog_toggled(self, on: bool) -> None:
        """Master toggle: grey out / re-enable the recognition controls."""
        for w in (self.cmb_asr, self.btn_recog, self.btn_skip_recog):
            w.setEnabled(on)

    def _finish_recognition(self) -> None:
        """Recognition done → the step disappears; open the removal step."""
        self.sec_recog.setVisible(False)
        self._proceed_to_removal()

    def _on_skip_recog(self) -> None:
        """«Пропустити»: skip recognition — grey the block (kept visible so it can
        be re-enabled via the checkbox) and move on to the removal step."""
        self.chk_do_recog.setChecked(False)
        self._proceed_to_removal()

    def _proceed_to_removal(self) -> None:
        """Hide the downloader and open ONLY the removal step next."""
        self.sec_downloader.setVisible(False)   # download-to-disk no longer relevant
        self.chk_do_removal.setChecked(True)
        self.sec_removal.setVisible(True)
        self._suggest_output()
        self._prewarm_proxy()

    def _prewarm_proxy(self) -> None:
        """Build the region-editor preview proxy in the background so scrubbing is
        instant by the time the user opens «Області». Idempotent — safe to call at
        several points (workspace ready, removal step) without double-building; the
        AV1 decode is heavy, so we never want two builds racing for the CPU."""
        if not self.input_path or not os.path.exists(self.input_path):
            return
        from ..media import proxy
        cached = proxy.proxy_path(self.input_path)
        if cached.exists() and proxy.is_valid(str(cached)):
            self._proxy_ready_path = str(cached)          # already built earlier
            return
        if (self._proxy_worker is not None and self._proxy_worker.isRunning()
                and getattr(self._proxy_worker, "src", "") == self.input_path):
            return                                        # a build is already running
        self._proxy_ready_path = ""
        self._proxy_worker = ProxyWorker(self.input_path, self)
        self._proxy_worker.sig_ready.connect(self._on_proxy_ready)
        self._proxy_worker.start()

    def _on_proxy_ready(self, path: str) -> None:
        # Only accept the proxy that belongs to the CURRENT input. A prewarm started
        # for the original video can finish after we've adopted the cleaned file — its
        # (stale) path must not overwrite the cleaned file's proxy.
        from ..media import proxy
        if self.input_path and path == str(proxy.proxy_path(self.input_path)):
            self._proxy_ready_path = path

    # -- removal step ---------------------------------------------------- #
    def _on_removal_toggled(self, on: bool) -> None:
        """Master toggle: grey out / re-enable the whole removal block."""
        for w in (self.chk_hs, self.btn_hs, self.lbl_hs, self.cmb_inpaint, self.cmb_enc,
                  self.chk_reuse, self.chk_master,
                  self.btn_inpaint, self.btn_skip_removal, self.btn_regions):
            w.setEnabled(on)

    def _edit_regions(self) -> None:
        """Open the region + time editor; regions persist on the window state."""
        pix = self._preview_pixmap()
        if pix is None:
            return
        dur = 0.0
        try:
            dur = ffmpeg.probe(self.input_path).duration
        except Exception:  # noqa: BLE001
            pass
        dlg = RegionEditor(pix, self._regions, duration=dur,
                           frame_provider=self._frame_at, src=self.input_path,
                           proxy_ready=self._proxy_ready_path or None,
                           proxy_worker=self._proxy_worker, parent=self)
        if dlg.exec():
            self._regions = dlg.result_regions()
            self.lbl_regions.setText(f"рамок: {len(self._regions)}")

    def _frame_at(self, t: float):
        """Extract one frame at ``t`` seconds for the editor's scrub preview."""
        if not self.input_path or not os.path.exists(self.input_path):
            return None
        frame = os.path.join(tempfile.gettempdir(), "vidloc_scrub.png")
        subprocess.run([settings.FFMPEG, "-y", "-loglevel", "error",
                        "-ss", str(max(0.0, t)), "-i", self.input_path,
                        "-frames:v", "1", frame], check=False)
        if not os.path.exists(frame):
            return None
        from PySide6.QtGui import QPixmap
        return QPixmap(frame)

    def _on_run_removal(self) -> None:
        """«Прибрати»: run inpaint only (no translation) → watermark-free video."""
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Немає файлу", "Спочатку оберіть вхідне відео.")
            return
        hs = self.chk_hs.isChecked() and self._hs_region is not None
        if not (hs or self._regions):
            QMessageBox.information(
                self, "Видалення",
                "Позначте, що прибирати, і виділіть область на відео.")
            return
        job = self._build_job()
        job.translate.mode = TranslateMode.NONE   # removal only — no subs/dub
        stem = Path(self.input_path).stem
        job.output_path = str(
            settings.OUTPUT_DIR / f"{stem}_clean.{job.output.container.value}")

        self._removal_run = True
        self.log.clear()
        self.progress.setValue(0)
        self.btn_inpaint.setEnabled(False)
        self.btn_skip_removal.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker = JobWorker(job)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_stage.connect(self._on_stage)
        self.worker.sig_log.connect(self._on_log)
        self.worker.sig_done.connect(self._on_done)
        self.worker.sig_fail.connect(self._on_fail)
        self.worker.start()

    def _removal_done_dialog(self, out_path: str) -> None:
        """After removal: «Використати цей файл» / «Показати у Finder» / «Закрити».
        «Показати у Finder» must NOT close the window, so the user can still adopt
        the file afterwards — hence a custom QDialog instead of QMessageBox."""
        self._clean_output = out_path
        self.btn_skip_removal.setEnabled(self.chk_do_removal.isChecked())
        dlg = QDialog(self)
        dlg.setWindowTitle("Готово")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"Відео без вотермарок / субтитрів:\n{out_path}"))
        row = QHBoxLayout()
        btn_use = QPushButton("Використати цей файл")
        btn_use.setDefault(True)
        btn_reveal = QPushButton("Показати у Finder")
        btn_close = QPushButton("Закрити")
        row.addWidget(btn_use, 1)
        row.addWidget(btn_reveal)
        row.addWidget(btn_close)
        lay.addLayout(row)
        # Reveal opens Finder but keeps this dialog open on purpose.
        btn_reveal.clicked.connect(
            lambda: subprocess.run(["open", "-R", out_path], check=False))
        btn_use.clicked.connect(dlg.accept)
        btn_close.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._use_clean_file(out_path)

    def _use_clean_file(self, path: str) -> None:
        """Adopt the watermark-free video as input and lock the removal step."""
        self.input_path = path
        self.ed_input.setText(path)
        self.lbl_info.setText("Використовується очищене відео (без вотермарок / субтитрів)")
        self.chk_do_removal.setChecked(False)   # done → block becomes inactive
        # The scrub-preview proxy was prewarmed for the ORIGINAL video; the adopted
        # cleaned file is a different source (different proxy_path), so drop the stale
        # proxy and rebuild it — otherwise the subtitle editor would play the original.
        self._proxy_ready_path = ""
        self._prewarm_proxy()
        self._reveal_rest_after_removal()

    def _on_skip_removal(self) -> None:
        """«Пропустити»: skip removal, lock the block, open the rest."""
        self.chk_do_removal.setChecked(False)
        self._reveal_rest_after_removal()

    def _reveal_rest_after_removal(self) -> None:
        """Open translate / subtitle editor / dubbing / output and unlock «Старт»."""
        for sec in (self.sec_translate, self.sec_subedit, self.sec_tts, self.sec_output):
            sec.setVisible(True)
        self._can_start = True
        self.btn_start.setEnabled(True)
        self._suggest_output()

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
        self.sec_recog.setVisible(True)   # reveal «Розпізнавання» for picked files too
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
        self.sec_recog.setVisible(True)
        self._suggest_output()
        if self._recognize_after_dl:
            self._recognize_after_dl = False
            self._do_recognize()

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
        enc_preset, enc_hw = self.cmb_enc.currentData()   # encoding chosen at the removal step
        out = OutputConfig(
            container=Container(self.cmb_container.currentData()),
            codec=VideoCodec(self.cmb_codec.currentData()),
            height=(self.cmb_res.currentData() or None),
            crf=self.sld_crf.value(),
            fps=(self.cmb_fps.currentData() or None),
            audio_bitrate=self.cmb_abr.currentData(),
            hw_encode=enc_hw or self.chk_hw.isChecked(),
            preset=enc_preset,
        )
        tr = TranslateConfig(
            mode=TranslateMode(self.cmb_mode.currentData()),
            source=SourceLang(self.cmb_src.currentData()),
            asr_quality=ASRQuality(self.cmb_asr.currentData()),
            translate_quality=TranslateQuality(self.cmb_mt.currentData()),
            beam_size=self.sld_beam.value(),
            # subtitles are burned via the visual editor; captions (when authored)
            # are the source of truth for both burn-in and dubbing.
            subtitle_placement=SubtitlePlacement.BURN,
            subtitle_style=self._sub_default_style,
            captions=list(self._captions),
            keep_source_subtitles=self.chk_keepsrc.isChecked(),
            tts_engine=TTSEngine(self.cmb_tts.currentData()),
            tts_voice=self.cmb_voice.currentData(),
            tts_clone_reference=None,
            background_volume=self.sld_bg.value() / 100.0,
            tts_speed=self.sld_speed.value() / 100.0,
        )
        do_rm = self.chk_do_removal.isChecked()
        rm = RemovalConfig(
            remove_hardsubs=do_rm and self.chk_hs.isChecked() and self._hs_region is not None,
            hardsub_region=self._hs_region,
            quality=InpaintQuality(self.cmb_inpaint.currentData()),
            regions=list(self._regions) if do_rm else [],
            reuse=self.chk_reuse.isChecked(),
            use_master=self.chk_master.isChecked(),
        )
        return JobConfig(input_path=self.input_path,
                         output_path=self.ed_out.text(),
                         source_context=self.source_context,
                         workspace=self.workspace_path,
                         output=out, removal=rm, translate=tr)

    def _start(self) -> None:
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Немає файлу", "Спочатку оберіть вхідне відео.")
            return
        job = self._build_job()
        self._last_job = job
        self.log.clear()
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_remix.setEnabled(False)   # stale until this render finishes

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
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(self._can_start)
        self.lbl_stage.setText("Готово ✅")
        if self._removal_run:
            self._removal_run = False
            self.btn_inpaint.setEnabled(self.chk_do_removal.isChecked())
            self._removal_done_dialog(out_path)
            return
        self._arm_remix(out_path)
        box = QMessageBox(self)
        box.setWindowTitle("Готово")
        box.setText(f"Файл збережено:\n{out_path}")
        reveal = box.addButton("Показати у Finder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Закрити", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is reveal:
            subprocess.run(["open", "-R", out_path], check=False)

    def _on_fail(self, message: str) -> None:
        self._removal_run = False
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(self._can_start)
        self.btn_inpaint.setEnabled(self.chk_do_removal.isChecked())
        self.btn_skip_removal.setEnabled(self.chk_do_removal.isChecked())
        self.lbl_stage.setText("Помилка ⚠")
        QMessageBox.critical(self, "Помилка", message)

    # -- fast audio re-mix (retune original-under-dub without re-encoding) ---- #
    def _arm_remix(self, out_path: str) -> None:
        """After a dubbed render, remember the stems so the user can retune the
        original-under-dub level on this file in seconds. Needs the persisted dub
        (kept in the workspace) and the original source for its audio."""
        job = getattr(self, "_last_job", None)
        self._remix_ctx = None
        self.btn_remix.setEnabled(False)
        if not (job and job.needs_dubbing and self.workspace_path):
            return
        from ..workspace import Workspace
        dub = Workspace.at(self.workspace_path).dub_file
        if not dub.is_file():
            return
        self._remix_ctx = {"output": out_path, "source": job.input_path,
                           "dub": str(dub), "bitrate": job.output.audio_bitrate,
                           "container": job.output.container.value}
        self.btn_remix.setEnabled(True)

    def _remix(self) -> None:
        ctx = self._remix_ctx
        if not ctx or not os.path.exists(ctx["output"]):
            QMessageBox.information(
                self, "Немає що пересобирати",
                "Спершу зробіть відео з озвучкою — тоді можна змінювати гучність оригіналу.")
            return
        from ..media import ffmpeg
        try:
            dur = ffmpeg.probe(ctx["output"]).duration
        except Exception:  # noqa: BLE001
            dur = 0.0
        self.btn_remix.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_stage.setText("Пересобираю звук…")
        self.remix_worker = RemixWorker(
            ctx["output"], ctx["output"], dub_path=ctx["dub"], orig_src=ctx["source"],
            bg_volume=self.sld_bg.value() / 100.0, audio_bitrate=ctx["bitrate"],
            container=ctx["container"], duration=dur)
        self.remix_worker.sig_progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        self.remix_worker.sig_log.connect(self._on_log)
        self.remix_worker.sig_done.connect(self._on_remix_done)
        self.remix_worker.sig_fail.connect(self._on_remix_fail)
        self.remix_worker.start()

    def _on_remix_done(self, out_path: str) -> None:
        self.progress.setValue(1000)
        self.btn_start.setEnabled(self._can_start)
        self.btn_remix.setEnabled(True)
        lvl = self.sld_bg.value()
        self.lbl_stage.setText(
            f"Звук пересобрано ✅ (оригінал {'вимк.' if lvl == 0 else f'{lvl}%'})")
        box = QMessageBox(self)
        box.setWindowTitle("Готово")
        box.setText(f"Гучність оригіналу оновлено без перекодування:\n{out_path}")
        reveal = box.addButton("Показати у Finder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Закрити", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is reveal:
            subprocess.run(["open", "-R", out_path], check=False)

    def _on_remix_fail(self, message: str) -> None:
        self.btn_start.setEnabled(self._can_start)
        self.btn_remix.setEnabled(True)
        self.lbl_stage.setText("Помилка ⚠")
        QMessageBox.critical(self, "Помилка пересобирання звуку", message)
