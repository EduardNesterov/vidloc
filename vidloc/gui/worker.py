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


class RecognizeWorker(QThread):
    """Runs the recognition step (audio extract + ASR) for one model on its own
    thread and saves the transcript into the workspace cache. The pipeline later
    reuses that transcript at render time, so ASR happens exactly once."""

    sig_progress = Signal(float, str)
    sig_log = Signal(str)
    sig_done = Signal(int)        # number of recognized segments
    sig_fail = Signal(str)

    def __init__(self, video_path: str, workspace_root: str, asr_quality: str,
                 source_lang: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.workspace_root = workspace_root
        self.asr_quality = asr_quality   # 'fast' | 'balanced' | 'best'
        self.source_lang = source_lang   # 'auto' | 'en' | 'ru' | 'it'

    def run(self) -> None:
        try:
            from ..workspace import Workspace
            from ..models import ASRQuality, ASR_MODEL
            from ..media import ffmpeg
            from ..asr import transcriber

            ws = Workspace.at(self.workspace_root)
            cached = ws.load_transcript(self.asr_quality)
            if cached:
                self.sig_log.emit("Транскрипт уже в кеші — пропускаю розпізнавання.")
                self.sig_done.emit(len(cached.get("segments", [])))
                return

            self.sig_progress.emit(0.05, "Витягую аудіо…")
            wav = str(ws.audio_file)
            if not ws.audio_file.is_file():
                info = ffmpeg.probe(self.video_path)
                ffmpeg.extract_audio(
                    self.video_path, wav, duration=info.duration,
                    on_progress=lambda f: self.sig_progress.emit(0.05 + 0.15 * f, "Витягую аудіо…"),
                    on_log=lambda s: self.sig_log.emit(s))

            self.sig_progress.emit(0.25, "Розпізнавання мовлення…")
            model_repo = ASR_MODEL[ASRQuality(self.asr_quality)]
            self.sig_log.emit(f"ASR-модель: {model_repo}")
            res = transcriber.transcribe(
                wav, model_repo, language=self.source_lang,
                on_log=lambda s: self.sig_log.emit(s))
            segments = res["segments"]
            ws.save_transcript(self.asr_quality, language=res["language"],
                               segments=segments, model_repo=model_repo,
                               text=res.get("text", ""))
            self.sig_progress.emit(1.0, "Розпізнавання завершено")
            self.sig_done.emit(len(segments))
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
