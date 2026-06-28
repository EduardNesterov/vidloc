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
    LLM_TRANSLATE_MODEL,
    TRANSLATE_MODEL,
    JobConfig,
    SubtitlePlacement,
    TranslateQuality,
)
from .events import Cancelled, Reporter, StageTracker


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
        if (job.needs_subtitles or job.needs_dubbing):
            stages.append(("translate", 0.15))
        if job.needs_inpaint or job.needs_dubbing:
            stages.append(("process", 0.40))
        stages.append(("render", 0.20))
        t = StageTracker(r, stages)

        src_video = job.input_path

        # ---- ASR ------------------------------------------------------- #
        segments: List[Dict] = []
        detected_lang = job.translate.source.value
        if job.needs_asr:
            t.start("audio", "Витягую аудіодоріжку…")
            if not info.has_audio:
                raise RuntimeError("У відео немає аудіодоріжки — нема що розпізнавати.")
            wav = str(work / "audio16k.wav")
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
            t.sub(1.0)
            t.done()
            self._check()

        # ---- 3. translate ---------------------------------------------- #
        if (job.needs_subtitles or job.needs_dubbing) and segments:
            t.start("translate", "Переклад на українську…")
            tq = job.translate.translate_quality
            if tq == TranslateQuality.NATURAL:
                r.log(f"Переклад: локальна LLM ({LLM_TRANSLATE_MODEL})")
                try:
                    from ..translate import llm
                    segments = llm.translate_segments(
                        segments, detected_lang, on_progress=lambda f: t.sub(f),
                        extra_context=job.source_context)
                except Exception as exc:  # noqa: BLE001 — degrade gracefully
                    r.log(f"⚠ LLM-переклад недоступний ({exc}); відкочуюсь на NLLB.")
                    from ..translate import translator
                    segments = translator.translate_segments(
                        segments, detected_lang, TRANSLATE_MODEL[TranslateQuality.FAST],
                        beam_size=job.translate.beam_size, on_progress=lambda f: t.sub(f))
            else:
                tr_model = TRANSLATE_MODEL[tq]
                r.log(f"Модель перекладу: {tr_model}")
                from ..translate import translator
                segments = translator.translate_segments(
                    segments, detected_lang, tr_model,
                    beam_size=job.translate.beam_size, on_progress=lambda f: t.sub(f))
            t.done()
            self._check()

        # ---- 4. subtitles file (re-chunked; styled ASS if burned) ------ #
        sub_file = None
        sub_burn = False
        fontsdir = None
        if job.needs_subtitles and segments:
            style = job.translate.subtitle_style
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
        dub_path = None
        need_inp = job.needs_inpaint
        need_dub = job.needs_dubbing and bool(segments)
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
                    cleaned = str(work / "cleaned.mp4")
                    inpaint_video(job.input_path, cleaned, job.removal, info,
                                  output=job.output, reporter=r,
                                  on_progress=lambda f: _bump("inp", f))
                    results["cleaned"] = cleaned
                except Exception as exc:  # noqa: BLE001
                    errors["inp"] = exc

            def _dub_task():
                try:
                    from ..tts import dubbing
                    results["dub"] = dubbing.synthesize_dub(
                        segments, job, info, work, reporter=r,
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
            dub_path=dub_path, keep_background=job.translate.keep_background,
            src_has_audio=info.has_audio, duration=info.duration,
            on_progress=lambda f: t.sub(f), on_log=r.log,
        )
        t.done()
        r.overall(1.0, "Готово")
        r.log(f"✅ Готово: {out}")
        return out
