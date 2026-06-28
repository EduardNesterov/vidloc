"""Headless command-line runner — handy for testing the pipeline without the GUI.

Example:
    python -m vidloc.cli input.mp4 --source en --mode subtitles --subs burn
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from . import settings  # noqa: F401  (configures env before heavy imports)
from .models import (
    ASRQuality,
    Container,
    JobConfig,
    OutputConfig,
    SourceLang,
    SubtitlePlacement,
    TranslateConfig,
    TranslateMode,
    TranslateQuality,
    TTSEngine,
    VideoCodec,
)
from .pipeline import PipelineRunner, Reporter


def _console_reporter() -> Reporter:
    state = {"stage": ""}

    def on_stage(name: str) -> None:
        state["stage"] = name

    def on_overall(frac: float, msg: str) -> None:
        bar = int(frac * 30)
        sys.stdout.write(
            f"\r[{'#' * bar}{'.' * (30 - bar)}] {frac * 100:5.1f}% "
            f"{state['stage']:10} {msg[:42]:42}"
        )
        sys.stdout.flush()

    def on_log(text: str) -> None:
        sys.stdout.write("\n" + text.rstrip() + "\n")

    return Reporter(on_overall=on_overall, on_log=on_log, on_stage=on_stage)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser("vidloc")
    p.add_argument("input", nargs="?", default="")
    p.add_argument("--url", default="", help="download from YouTube/Instagram/TikTok first")
    p.add_argument("-o", "--output", default="")
    p.add_argument("--source", default="auto", choices=[e.value for e in SourceLang])
    p.add_argument("--mode", default="subtitles", choices=[e.value for e in TranslateMode])
    p.add_argument("--subs", default="soft", choices=["soft", "burn"])
    p.add_argument("--asr", default="balanced", choices=[e.value for e in ASRQuality])
    p.add_argument("--mt", default="natural", choices=[e.value for e in TranslateQuality])
    p.add_argument("--engine", default="", choices=["", "uktts", "piper", "coqui_uk"])
    p.add_argument("--voice", default="")
    p.add_argument("--tts-speed", type=float, default=1.0)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--container", default="mp4", choices=[e.value for e in Container])
    p.add_argument("--codec", default="h264", choices=[e.value for e in VideoCodec])
    p.add_argument("--crf", type=int, default=20)
    a = p.parse_args(argv)

    input_path = a.input
    source_context = ""
    if a.url:
        from .media import fetch
        print(f"Завантаження: {a.url}")
        res = fetch.download(a.url, on_log=lambda m: print(m))
        input_path = res.path
        source_context = res.context()
        print(f"Контекст: {source_context[:200]}\n")
    if not input_path:
        p.error("вкажіть відеофайл або --url")

    job = JobConfig(
        input_path=input_path,
        output_path=a.output,
        source_context=source_context,
        output=OutputConfig(
            container=Container(a.container), codec=VideoCodec(a.codec),
            height=a.height or None, crf=a.crf,
        ),
        translate=TranslateConfig(
            mode=TranslateMode(a.mode), source=SourceLang(a.source),
            asr_quality=ASRQuality(a.asr), translate_quality=TranslateQuality(a.mt),
            beam_size=a.beam,
            subtitle_placement=(SubtitlePlacement.BURN if a.subs == "burn"
                                else SubtitlePlacement.SOFT),
            tts_engine=TTSEngine(a.engine) if a.engine else TTSEngine.UKTTS,
            tts_voice=a.voice,
            tts_speed=a.tts_speed,
        ),
    )

    reporter = _console_reporter()
    t0 = time.time()
    out = PipelineRunner(job, reporter).run()
    print(f"\n\nГотово за {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
