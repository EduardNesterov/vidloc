"""Speech-to-text with word/segment timestamps via mlx-whisper (Metal-accelerated).

We feed Whisper a pre-decoded float32 waveform (extracted with our bundled
ffmpeg) rather than a file path, so the model never depends on ffmpeg being on
the system PATH.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def transcribe(audio_path: str, model_repo: str, language: Optional[str] = None,
               on_log=None) -> Dict:
    """Transcribe a 16 kHz mono wav. Returns ``{language, segments, text}``.

    ``segments`` is a list of ``{"start", "end", "text"}`` dicts.
    """
    import numpy as np
    import soundfile as sf
    import mlx_whisper

    audio, _sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    lang = None if language in (None, "", "auto") else language
    kwargs = dict(path_or_hf_repo=model_repo, word_timestamps=True, verbose=None)
    if lang:
        kwargs["language"] = lang

    result = mlx_whisper.transcribe(audio, **kwargs)

    segments: List[Dict] = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": text,
        })

    return {
        "language": result.get("language", lang or "en"),
        "segments": segments,
        "text": result.get("text", ""),
    }
