"""Shared translation entry point for a :class:`JobConfig`.

The pipeline runner and the GUI's «Підтягнути переклад» both need to turn source
ASR segments into Ukrainian ones using the job's chosen quality (Aya / Dragoman /
NLLB), with the same graceful fallbacks. This module is the single implementation
so the two callers can never drift apart.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..models import (
    DRAGOMAN_TRANSLATE_MODEL,
    LLM_TRANSLATE_MODEL,
    TRANSLATE_MODEL,
    JobConfig,
    TranslateQuality,
)

ProgressCb = Optional[Callable[[float], None]]
LogCb = Optional[Callable[[str], None]]


def translate_for_job(segments: List[Dict], detected_lang: str, job: JobConfig,
                      on_progress: ProgressCb = None, log: LogCb = None) -> List[Dict]:
    """Return ``segments`` with a ``text_tr`` field, per the job's translate quality.

    Mirrors the runner's inline logic: Dragoman (EN→UK only, else Aya), Aya/NLLB with a
    fallback to NLLB-fast when the LLM is unavailable.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    tq = job.translate.translate_quality
    if tq == TranslateQuality.DRAGOMAN:
        if detected_lang == "en":
            _log(f"Переклад: Dragoman EN→UK ({DRAGOMAN_TRANSLATE_MODEL})")
            try:
                from . import dragoman
                return dragoman.translate_segments(
                    segments, detected_lang, on_progress=on_progress)
            except Exception as exc:  # noqa: BLE001
                _log(f"⚠ Dragoman недоступний ({exc}); відкочуюсь на Aya.")
                from . import llm
                return llm.translate_segments(
                    segments, detected_lang, on_progress=on_progress,
                    extra_context=job.source_context)
        _log(f"Dragoman підтримує лише EN→UK; мова «{detected_lang}» → Aya.")
        from . import llm
        return llm.translate_segments(
            segments, detected_lang, on_progress=on_progress,
            extra_context=job.source_context)

    if tq == TranslateQuality.NATURAL:
        _log(f"Переклад: локальна LLM ({LLM_TRANSLATE_MODEL})")
        try:
            from . import llm
            return llm.translate_segments(
                segments, detected_lang, on_progress=on_progress,
                extra_context=job.source_context)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            _log(f"⚠ LLM-переклад недоступний ({exc}); відкочуюсь на NLLB.")
            from . import translator
            return translator.translate_segments(
                segments, detected_lang, TRANSLATE_MODEL[TranslateQuality.FAST],
                beam_size=job.translate.beam_size, on_progress=on_progress)

    tr_model = TRANSLATE_MODEL[tq]
    _log(f"Модель перекладу: {tr_model}")
    from . import translator
    return translator.translate_segments(
        segments, detected_lang, tr_model,
        beam_size=job.translate.beam_size, on_progress=on_progress)
