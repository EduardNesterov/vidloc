#!/usr/bin/env python
"""ukrainian-tts (robinhad) synthesis worker. Runs in the isolated ``.venv-uktts``
interpreter; must NOT import anything from ``vidloc``.

Spec JSON:
    {
      "voice": "tetiana"|"mykyta"|"lada"|"dmytro"|"oleksa",
      "stress": "dictionary"|"model",
      "device": "cpu",
      "items": [{"text": "...", "out": "/path/seg_0000.wav"}, ...]
    }
``TTS.tts(text, voice, stress, output_fp)`` writes a wav to the file object.
Progress is emitted on stdout as ``PROGRESS i/N`` lines.
"""

from __future__ import annotations

import argparse
import json
import wave


def _write_silence(path: str, sr: int = 22050, dur: float = 0.25) -> None:
    nframes = int(sr * dur)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * nframes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    voice = spec.get("voice") or "tetiana"
    stress = spec.get("stress") or "dictionary"
    device = spec.get("device") or "cpu"
    items = spec["items"]

    from ukrainian_tts.tts import TTS

    print(f"WORKER ukrainian-tts voice={voice} stress={stress} device={device}", flush=True)
    tts = TTS(device=device)

    n = len(items)
    for i, it in enumerate(items):
        text = (it.get("text") or "").strip()
        out = it["out"]
        if not text:
            _write_silence(out)
            print(f"PROGRESS {i + 1}/{n}", flush=True)
            continue
        try:
            with open(out, "wb") as f:
                tts.tts(text, voice, stress, f)
        except Exception as exc:  # noqa: BLE001
            print(f"WORKER item {i} error: {exc}", flush=True)
            _write_silence(out)
        print(f"PROGRESS {i + 1}/{n}", flush=True)

    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
