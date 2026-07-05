"""QThread wrapper that runs a pipeline job and forwards progress as Qt signals."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..models import JobConfig
from ..pipeline import Cancelled, PipelineRunner, Reporter


class ProxyWorker(QThread):
    """Builds the editor scrub preview + the full-res inpaint master off the UI
    thread, in a single ffmpeg pass (one AV1 decode feeds both).

    If both are already cached and valid it emits ``sig_ready`` immediately;
    otherwise it renders to unique temp files and atomically renames them into the
    cache, so a half-written / clobbered file is never picked up as valid.
    """

    sig_progress = Signal(float)     # 0..1
    sig_ready = Signal(str)          # preview (scrub) path
    sig_fail = Signal(str)

    def __init__(self, src: str, parent=None):
        super().__init__(parent)
        self.src = src

    def run(self) -> None:
        import uuid

        try:
            from ..media import ffmpeg, proxy
            preview = proxy.proxy_path(self.src)
            master = proxy.master_path(self.src)
            if (preview.exists() and proxy.is_valid(str(preview))
                    and master.exists() and proxy.is_valid(str(master))):
                self.sig_ready.emit(str(preview))
                return
            dur = 0.0
            try:
                dur = ffmpeg.probe(self.src).duration
            except Exception:  # noqa: BLE001 — progress is best-effort
                pass
            # Unique temps per worker: prewarm (main window) and the editor may
            # build concurrently — a shared temp would clobber into garbage. Both
            # produce valid files; the atomic replace() lets last win.
            tag = uuid.uuid4().hex[:8]
            tmp_m = master.with_name(f"{master.stem}.part.{tag}.mp4")
            tmp_p = preview.with_name(f"{preview.stem}.part.{tag}.mp4")
            try:
                proxy.build(self.src, str(tmp_m), str(tmp_p), duration=dur,
                            on_progress=lambda f: self.sig_progress.emit(f or 0.0))
                tmp_m.replace(master)
                tmp_p.replace(preview)
            except Exception:
                tmp_m.unlink(missing_ok=True)
                tmp_p.unlink(missing_ok=True)
                raise
            self.sig_ready.emit(str(preview))
        except Exception as exc:  # noqa: BLE001
            self.sig_fail.emit(f"{type(exc).__name__}: {exc}")


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


class CookieCheckWorker(QThread):
    """Probes whether the chosen browser / cookies file has a logged-in Instagram
    session, off the UI thread (reading Chromium cookies can raise a Keychain prompt)."""

    sig_done = Signal(bool, str)   # ok, human message

    def __init__(self, browser: str, cookies_file: str, parent=None):
        super().__init__(parent)
        self.browser = browser
        self.cookies_file = cookies_file

    def run(self) -> None:
        try:
            from ..media import fetch
            ok, msg = fetch.check_cookies(self.browser, self.cookies_file)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        self.sig_done.emit(ok, msg)


class TranslateWorker(QThread):
    """«Підтягнути переклад»: load the cached ASR transcript and translate it to
    Ukrainian on its own thread, so the subtitle editor can seed captions from real
    translated lines. The result is cached in the workspace (per ASR model × MT
    quality) so re-opening the editor is instant."""

    sig_progress = Signal(float, str)
    sig_log = Signal(str)
    sig_done = Signal(list)       # translated segments (dicts with text_tr/start/end)
    sig_fail = Signal(str)

    def __init__(self, job: JobConfig, workspace_root: str, parent=None):
        super().__init__(parent)
        self.job = job
        self.workspace_root = workspace_root

    def run(self) -> None:
        try:
            from ..workspace import Workspace
            from ..translate.service import translate_for_job

            ws = Workspace.at(self.workspace_root)
            model = self.job.translate.asr_quality.value
            mt = self.job.translate.translate_quality.value

            cached_tr = ws.load_translation(model, mt)
            if cached_tr:
                self.sig_log.emit("♻ Переклад із кешу.")
                self.sig_done.emit(cached_tr["segments"])
                return

            transcript = ws.load_transcript(model)
            if not transcript:
                self.sig_fail.emit(
                    "Немає розпізнаного тексту. Спершу виконайте «Розпізнати».")
                return
            segments = transcript["segments"]
            lang = transcript.get("language") or self.job.translate.source.value

            self.sig_progress.emit(0.05, "Переклад на українську…")
            translated = translate_for_job(
                segments, lang, self.job,
                on_progress=lambda f: self.sig_progress.emit(
                    0.05 + 0.9 * (f or 0.0), "Переклад на українську…"),
                log=lambda s: self.sig_log.emit(s))
            ws.save_translation(model, mt, language=lang, segments=translated)
            self.sig_progress.emit(1.0, "Переклад завершено")
            self.sig_done.emit(translated)
        except Exception as exc:  # noqa: BLE001
            self.sig_log.emit(traceback.format_exc())
            self.sig_fail.emit(f"{type(exc).__name__}: {exc}")


class RemixWorker(QThread):
    """Retune the original-under-dub level on an already-rendered video, off the UI
    thread. Stream-copies the picture and rebuilds only the audio from the persisted
    stems (original track + cached dub) — seconds, no re-encode, no TTS."""

    sig_progress = Signal(float)   # 0..1
    sig_log = Signal(str)
    sig_done = Signal(str)         # output path
    sig_fail = Signal(str)

    def __init__(self, video: str, out: str, *, dub_path: str, orig_src: str,
                 bg_volume: float, audio_bitrate: str, container: str,
                 duration: float = 0.0, parent=None):
        super().__init__(parent)
        self.video = video
        self.out = out
        self.dub_path = dub_path
        self.orig_src = orig_src
        self.bg_volume = bg_volume
        self.audio_bitrate = audio_bitrate
        self.container = container
        self.duration = duration

    def run(self) -> None:
        try:
            from ..media import ffmpeg
            from ..models import Container
            ffmpeg.remix_audio(
                self.video, self.out, dub_path=self.dub_path, orig_src=self.orig_src,
                bg_volume=self.bg_volume, audio_bitrate=self.audio_bitrate,
                container=Container(self.container), duration=self.duration,
                on_progress=lambda f: self.sig_progress.emit(f or 0.0),
                on_log=lambda s: self.sig_log.emit(s))
            self.sig_done.emit(self.out)
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
