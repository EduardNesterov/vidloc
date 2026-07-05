"""End-to-end job orchestration: probe -> (inpaint) -> ASR -> translate ->
subtitles / dub -> final render.

The runner is GUI-agnostic: it talks to a :class:`Reporter` for progress and
logs, and checks ``reporter.cancelled`` at safe points.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List

from .. import settings
from ..media import ffmpeg
from ..models import (
    ASR_MODEL,
    JobConfig,
    SubtitlePlacement,
)
from .events import Cancelled, Reporter, StageTracker


def _captions_to_dub(captions) -> List[Dict]:
    """Build the dub source from editor captions. Captions sharing a non-None ``group``
    id were split from one sentence for display — re-join them into a single utterance
    spanning the group so the TTS is synthesized (and paced) as one phrase, not chopped
    into 3-word fragments. Ungrouped captions stay as individual units."""
    caps = [c for c in captions if (getattr(c, "text", "") or "").strip()]
    groups: Dict[int, list] = {}
    items: List[Dict] = []
    for c in caps:
        if c.group is None:
            items.append({"text": c.text, "text_tr": c.text,
                          "start": float(c.start), "end": float(c.end)})
        else:
            groups.setdefault(c.group, []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c.start)
        text = " ".join((m.text or "").strip() for m in members).strip()
        items.append({"text": text, "text_tr": text,
                      "start": float(members[0].start), "end": float(members[-1].end)})
    items.sort(key=lambda d: d["start"])
    return items


class PipelineRunner:
    def __init__(self, job: JobConfig, reporter: Reporter):
        self.job = job
        self.r = reporter

    # -- helpers --------------------------------------------------------- #
    def _check(self) -> None:
        if self.r.cancelled:
            raise Cancelled()

    @staticmethod
    def _nospace_copy(path: str) -> str:
        """Copy a file to a space-free temp path (ffmpeg filtergraph safety)."""
        tmp = tempfile.NamedTemporaryFile(suffix=Path(path).suffix, delete=False,
                                          dir=tempfile.gettempdir())
        tmp.close()
        shutil.copy(path, tmp.name)
        return tmp.name

    # -- main ------------------------------------------------------------ #
    def run(self) -> str:
        job, r = self.job, self.r
        self._check()

        info = ffmpeg.probe(job.input_path)
        r.log(f"Вхідне відео: {info.resolution}, {info.duration:.1f}s, "
              f"fps={info.fps:.2f}, audio={'так' if info.has_audio else 'ні'}")

        work = Path(settings.WORK_DIR) / f"job_{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True, exist_ok=True)

        # ---- stage plan (weights are rough wall-clock proportions) ----- #
        # Inpaint (GPU) and dubbing (CPU) overlap in one parallel "process" stage.
        stages: List[tuple] = []
        if job.needs_asr:
            stages += [("audio", 0.05), ("asr", 0.35)]
        # Editor captions already carry translated text → no translate stage on run.
        if (job.needs_subtitles or job.needs_dubbing) and not job.has_captions:
            stages.append(("translate", 0.15))
        if job.needs_inpaint or job.needs_dubbing:
            stages.append(("process", 0.40))
        stages.append(("render", 0.20))
        t = StageTracker(r, stages)

        src_video = job.input_path

        # ---- workspace cache: reuse the downloaded source + a transcript
        # already produced for this recognition model, so repeat runs skip
        # the download and the (slow) ASR pass entirely. ------------------ #
        ws = None
        model_key = job.translate.asr_quality.value
        if job.workspace:
            from ..workspace import Workspace
            ws = Workspace.at(job.workspace)
            # Prefer the explicit input (e.g. an adopted watermark-free file);
            # fall back to the cached Source copy only when it's missing.
            if not (job.input_path and Path(job.input_path).exists()):
                cached_src = ws.source_video()
                if cached_src:
                    src_video = str(cached_src)

        # ---- ASR ------------------------------------------------------- #
        segments: List[Dict] = []
        detected_lang = job.translate.source.value
        if job.needs_asr:
            cached = ws.load_transcript(model_key) if ws else None
            if cached:
                segments = cached["segments"]
                detected_lang = cached.get("language") or detected_lang
                r.log(f"♻ Транскрипт із кешу: {len(segments)} сегментів "
                      f"(модель «{model_key}») — пропускаю аудіо та розпізнавання.")
                t.start("audio", "Транскрипт із кешу…")
                t.done()
                t.start("asr", "Транскрипт із кешу…")
                t.done()
                self._check()
            else:
                t.start("audio", "Витягую аудіодоріжку…")
                if not info.has_audio:
                    raise RuntimeError("У відео немає аудіодоріжки — нема що розпізнавати.")
                wav = str(ws.audio_file) if ws else str(work / "audio16k.wav")
                if not (ws and ws.audio_file.is_file()):
                    ffmpeg.extract_audio(src_video, wav, duration=info.duration,
                                         on_progress=lambda f: t.sub(f), on_log=r.log)
                t.done()
                self._check()

                t.start("asr", "Розпізнавання мовлення…")
                asr_model = ASR_MODEL[job.translate.asr_quality]
                r.log(f"ASR-модель: {asr_model}")
                from ..asr import transcriber
                res = transcriber.transcribe(wav, asr_model,
                                             language=job.translate.source.value,
                                             on_log=r.log)
                segments = res["segments"]
                detected_lang = res["language"]
                r.log(f"Сегментів розпізнано: {len(segments)} | мова оригіналу: {detected_lang}")
                if ws:
                    ws.save_transcript(model_key, language=detected_lang,
                                       segments=segments, model_repo=asr_model,
                                       text=res.get("text", ""))
                    r.log(f"💾 Транскрипт збережено в кеш: {ws.model_dir(model_key)}")
                t.sub(1.0)
                t.done()
                self._check()

        # ---- 3. translate ---------------------------------------------- #
        if (job.needs_subtitles or job.needs_dubbing) and segments:
            t.start("translate", "Переклад на українську…")
            from ..translate.service import translate_for_job
            segments = translate_for_job(
                segments, detected_lang, job,
                on_progress=lambda f: t.sub(f), log=r.log)
            t.done()
            self._check()

        # ---- 4. subtitles file ----------------------------------------- #
        # Editor captions (when present) are the source of truth and always burn,
        # each positioned/styled per-caption. Otherwise fall back to the automatic
        # ASR→translate→rechunk path (CLI / no editor): styled ASS burn or soft SRT.
        sub_file = None
        sub_burn = False
        fontsdir = None
        style = job.translate.subtitle_style
        if job.needs_subtitles and job.has_captions:
            from ..subtitles.ass import write_ass_captions
            caps = job.translate.captions
            ass_path = str(work / "uk.ass")
            write_ass_captions(caps, ass_path, style, info.width, info.height)
            sub_file = self._nospace_copy(ass_path)
            sub_burn = True
            fontsdir = str(settings.FONTS_DIR)
            r.log(f"Субтитри з редактора: {len(caps)} реплік (вшивання).")
        elif job.needs_subtitles and segments:
            from ..subtitles.rechunk import rechunk
            sub_segs = rechunk(segments, style.max_chars)
            sub_burn = job.translate.subtitle_placement == SubtitlePlacement.BURN
            if sub_burn:
                from ..subtitles.ass import write_ass
                ass_path = str(work / "uk.ass")
                write_ass(sub_segs, ass_path, style, info.width, info.height)
                sub_file = self._nospace_copy(ass_path)
                fontsdir = str(settings.FONTS_DIR)
                r.log(f"Субтитри (стиль «{style.font}», {len(sub_segs)} реплік).")
            else:
                from ..subtitles.srt import write_srt
                srt_path = str(work / "uk.srt")
                write_srt(sub_segs, srt_path)
                sub_file = srt_path
                r.log(f"Субтитри ({len(sub_segs)} реплік): {srt_path}")

        # ---- inpaint (GPU)  ||  dubbing (CPU) — run in parallel -------- #
        # The voiceover speaks whatever the captions say (edit subtitles → change the
        # dub). Each caption is one synth unit laid onto the timeline at its start and
        # fit to its [start, end]. Captions split from one sentence (shared group id)
        # are re-joined into a single utterance so the tempo is not chopped. No captions
        # → the raw translated segments.
        dub_source = segments
        if job.has_captions:
            dub_source = _captions_to_dub(job.translate.captions)
        dub_path = None
        need_inp = job.needs_inpaint
        need_dub = job.needs_dubbing and bool(dub_source)
        if need_inp or need_dub:
            import threading
            from concurrent.futures import ThreadPoolExecutor

            t.start("process", "Відео та озвучка паралельно…")
            prog = {"inp": 0.0, "tts": 0.0}
            lock = threading.Lock()

            def _bump(key, f):
                with lock:
                    prog[key] = f
                    t.sub(max(prog["inp"], prog["tts"]))

            results: Dict[str, str] = {}
            errors: Dict[str, Exception] = {}

            def _inpaint_task():
                try:
                    from ..inpaint import inpaint_video
                    from ..media import proxy
                    cleaned = str(work / "cleaned.mp4")
                    # Decode from the fast H.264 master when available (≈10× faster
                    # than libaom AV1; the worker decodes the source twice per run).
                    inp_src = job.input_path
                    if getattr(job.removal, "use_master", True):
                        m = proxy.master_path(job.input_path)
                        if m.exists() and proxy.is_valid(str(m)):
                            inp_src = str(m)
                            r.log("Затирання: читаю з H.264-майстра (швидкий декод).")
                    inpaint_video(inp_src, cleaned, job.removal, info,
                                  output=job.output, reporter=r,
                                  on_progress=lambda f: _bump("inp", f))
                    results["cleaned"] = cleaned
                except Exception as exc:  # noqa: BLE001
                    errors["inp"] = exc

            def _dub_task():
                try:
                    from ..tts import dubbing
                    results["dub"] = dubbing.synthesize_dub(
                        dub_source, job, info, work, reporter=r,
                        on_progress=lambda f: _bump("tts", f))
                except Exception as exc:  # noqa: BLE001
                    errors["dub"] = exc

            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = []
                if need_inp:
                    futs.append(ex.submit(_inpaint_task))
                if need_dub:
                    futs.append(ex.submit(_dub_task))
                for fut in futs:
                    fut.result()

            if "dub" in errors:
                raise errors["dub"]
            if "inp" in errors:
                raise errors["inp"]
            if "cleaned" in results:
                src_video = results["cleaned"]
            dub_path = results.get("dub")
            # Persist the assembled dub so the GUI can retune the original-under-dub
            # level onto the finished video later without re-running TTS.
            if dub_path and ws:
                try:
                    ws.source.mkdir(parents=True, exist_ok=True)
                    shutil.copy(dub_path, ws.dub_file)
                except OSError as exc:  # noqa: BLE001 — caching is best-effort
                    r.log(f"Не вдалося зберегти дубляж у кеш: {exc}")
            t.done()
            self._check()

        # ---- 6. final render ------------------------------------------- #
        t.start("render", "Збирання підсумкового відео…")
        out = job.output_path
        if not out:
            stem = Path(job.input_path).stem
            out = str(settings.OUTPUT_DIR / f"{stem}_uk.{job.output.container.value}")

        ffmpeg.render_video(
            src_video, out, job.output,
            sub_path=sub_file, sub_burn=sub_burn, fontsdir=fontsdir,
            dub_path=dub_path, bg_volume=job.translate.background_volume,
            src_has_audio=info.has_audio, duration=info.duration,
            on_progress=lambda f: t.sub(f), on_log=r.log,
        )
        t.done()
        r.overall(1.0, "Готово")
        r.log(f"✅ Готово: {out}")
        return out
