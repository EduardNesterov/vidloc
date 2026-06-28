#!/usr/bin/env python
"""Coqui-TTS synthesis worker. Runs inside the isolated ``.venv-tts`` interpreter
(torch + coqui-tts); must NOT import anything from ``vidloc``.

Reads a JSON spec and writes one wav per item:
    {
      "backend": "coqui_uk" | "xtts",
      "model":   "tts_models/uk/mai/vits",   # optional override
      "language": "uk",                        # xtts only
      "speaker_wav": "/path/ref.wav" | null,   # xtts only (voice clone)
      "tts_home": "/path/models/tts",
      "items": [{"text": "...", "out": "/path/seg_0000.wav"}, ...]
    }
Progress is emitted on stdout as ``PROGRESS i/N`` lines.
"""

from __future__ import annotations

import argparse
import json
import os
import wave

DEFAULT_MODEL = {
    "coqui_uk": "tts_models/uk/mai/vits",
    "xtts": "tts_models/multilingual/multi-dataset/xtts_v2",
}


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

    backend = spec.get("backend", "coqui_uk")
    model = spec.get("model") or DEFAULT_MODEL.get(backend, DEFAULT_MODEL["coqui_uk"])
    language = spec.get("language") or "uk"
    speaker_wav = spec.get("speaker_wav") or None
    items = spec["items"]

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if spec.get("tts_home"):
        os.environ.setdefault("TTS_HOME", spec["tts_home"])

    import torch
    from TTS.api import TTS

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"WORKER backend={backend} model={model} device={device}", flush=True)
    tts = TTS(model)
    try:
        tts.to(device)
    except Exception as exc:  # noqa: BLE001
        print(f"WORKER warn: .to({device}) failed ({exc}); using cpu", flush=True)
        device = "cpu"
        tts.to("cpu")

    n = len(items)
    for i, it in enumerate(items):
        text = (it.get("text") or "").strip()
        out = it["out"]
        if not text:
            _write_silence(out)
            print(f"PROGRESS {i + 1}/{n}", flush=True)
            continue

        kwargs = {"text": text, "file_path": out}
        if backend == "xtts":
            kwargs["language"] = language
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
        try:
            tts.tts_to_file(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if device != "cpu":
                print(f"WORKER item {i} failed on {device} ({exc}); retry cpu", flush=True)
                device = "cpu"
                tts.to("cpu")
                try:
                    tts.tts_to_file(**kwargs)
                except Exception as exc2:  # noqa: BLE001
                    print(f"WORKER item {i} error: {exc2}", flush=True)
                    _write_silence(out)
            else:
                print(f"WORKER item {i} error: {exc}", flush=True)
                _write_silence(out)
        print(f"PROGRESS {i + 1}/{n}", flush=True)

    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
