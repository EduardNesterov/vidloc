#!/usr/bin/env python
"""Piper TTS synthesis worker. Runs inside the isolated ``.venv-piper`` interpreter
(piper-tts + onnxruntime); must NOT import anything from ``vidloc``.

Spec JSON:
    {
      "model_path": "/path/uk_UA-ukrainian_tts-medium.onnx",
      "speaker_id": 0,
      "length_scale": 1.0,
      "items": [{"text": "...", "out": "/path/seg_0000.wav"}, ...]
    }

The Ukrainian voice is a ``phoneme_type: "text"`` model whose map holds only
LOWERCASE Cyrillic — so each line is lowercased before synthesis (otherwise
capital letters are dropped as "missing phonemes").
"""

from __future__ import annotations

import argparse
import json
import wave


def _write_silence(path: str, sr: int = 22050, dur: float = 0.25) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * dur))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    with open(ap.parse_args().spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    from piper import PiperVoice
    from piper.config import SynthesisConfig

    voice = PiperVoice.load(spec["model_path"])
    sid = int(spec.get("speaker_id", 0))
    length_scale = float(spec.get("length_scale", 1.0))
    items = spec["items"]
    n = len(items)

    # This voice was trained on STRESS-MARKED text (combining acute U+0301 is in its
    # phoneme map). Ukrainian orthography omits stress, so we must add it back or the
    # model guesses the stressed syllable wrong.
    stressify = None
    try:
        from ukrainian_word_stress import Stressifier, StressSymbol
        stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)
    except Exception as exc:  # noqa: BLE001
        print(f"WORKER warn: stress engine unavailable ({exc})", flush=True)
    print(f"WORKER piper speaker={sid} stress={'on' if stressify else 'off'}", flush=True)

    for i, it in enumerate(items):
        raw = (it.get("text") or "").strip()
        out = it["out"]
        if not raw:
            _write_silence(out)
            print(f"PROGRESS {i + 1}/{n}", flush=True)
            continue
        text = raw
        if stressify:
            try:
                text = stressify(raw)
            except Exception:  # noqa: BLE001
                text = raw
        text = text.lower()  # model map is lowercase-only (acute marks survive lower())
        try:
            with wave.open(out, "wb") as wf:
                voice.synthesize_wav(
                    text, wf, SynthesisConfig(speaker_id=sid, length_scale=length_scale))
        except Exception as exc:  # noqa: BLE001
            print(f"WORKER item {i} error: {exc}", flush=True)
            _write_silence(out)
        print(f"PROGRESS {i + 1}/{n}", flush=True)

    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
