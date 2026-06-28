"""QThread wrapper that runs a pipeline job and forwards progress as Qt signals."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..models import JobConfig
from ..pipeline import Cancelled, PipelineRunner, Reporter


class DownloadWorker(QThread):
    """Downloads a video from a URL (yt-dlp) off the UI thread."""

    sig_progress = Signal(float, str)
    sig_log = Signal(str)
    sig_done = Signal(str, str)   # local path, context string
    sig_fail = Signal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url

    def run(self) -> None:
        try:
            from ..media import fetch
            res = fetch.download(
                self.url,
                on_progress=lambda f, m: self.sig_progress.emit(f, m),
                on_log=lambda s: self.sig_log.emit(s),
            )
            self.sig_done.emit(res.path, res.context())
        except Exception as exc:  # noqa: BLE001
            self.sig_log.emit(traceback.format_exc())
            self.sig_fail.emit(f"{type(exc).__name__}: {exc}")


class SaveWorker(QThread):
    """Download-only: save a video/audio file to disk (no translation)."""

    sig_progress = Signal(float, str)
    sig_log = Signal(str)
    sig_done = Signal(str)   # saved path
    sig_fail = Signal(str)

    def __init__(self, url: str, dest_dir: str, quality: str, audio_only: bool, parent=None):
        super().__init__(parent)
        self.url = url
        self.dest_dir = dest_dir
        self.quality = quality
        self.audio_only = audio_only

    def run(self) -> None:
        try:
            from ..media import fetch
            path = fetch.save_media(
                self.url, self.dest_dir, self.quality, self.audio_only,
                on_progress=lambda f, m: self.sig_progress.emit(f, m),
                on_log=lambda s: self.sig_log.emit(s),
            )
            self.sig_done.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.sig_log.emit(traceback.format_exc())
            self.sig_fail.emit(f"{type(exc).__name__}: {exc}")


class JobWorker(QThread):
    sig_progress = Signal(float, str)   # overall fraction 0..1, message
    sig_stage = Signal(str)             # stage id
    sig_log = Signal(str)
    sig_done = Signal(str)              # output path
    sig_fail = Signal(str)             # error message

    def __init__(self, job: JobConfig, parent=None):
        super().__init__(parent)
        self.job = job
        self._reporter = Reporter(
            on_overall=lambda f, m: self.sig_progress.emit(f, m),
            on_log=lambda s: self.sig_log.emit(s),
            on_stage=lambda n: self.sig_stage.emit(n),
        )

    def cancel(self) -> None:
        self._reporter.cancel()

    def run(self) -> None:
        try:
            out = PipelineRunner(self.job, self._reporter).run()
            self.sig_done.emit(out)
        except Cancelled:
            self.sig_fail.emit("Скасовано користувачем.")
        except Exception as exc:  # noqa: BLE001 — surface everything to the UI
            self.sig_log.emit(traceback.format_exc())
            self.sig_fail.emit(f"{type(exc).__name__}: {exc}")
