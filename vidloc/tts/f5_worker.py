#!/usr/bin/env python
"""F5-TTS worker. Runs in isolated ``.venv-f5``.

Spec JSON:
    {
      "model": "F5TTS_v1_Base",
      "ref_audio": "/path/ref.wav",
      "ref_text": "reference transcript",
      "device": "mps",
      "items": [{"text": "...", "out": "/path/seg_0000.wav"}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path


def _write_silence(path: str, sr: int = 24000, dur: float = 0.25) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * dur))


def _patch_audio_load() -> None:
    """TorchAudio 2.9+ reads through TorchCodec, which needs Homebrew FFmpeg dylibs.
    The project ships a static ffmpeg, so read wav refs with soundfile instead."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _load(uri, *_, channels_first=True, **__):
        data, sr = sf.read(uri, dtype="float32", always_2d=True)
        if data.size == 0:
            data = np.zeros((1, 1), dtype=np.float32)
        x = torch.from_numpy(data.T if channels_first else data)
        return x, int(sr)

    torchaudio.load = _load
    try:
        import f5_tts.infer.utils_infer as utils_infer
        utils_infer.torchaudio.load = _load
    except Exception:
        pass


def _run_f5(model: str, ref_audio: str, ref_text: str, text: str,
            out: Path, device: str) -> None:
    _patch_audio_load()
    old_argv = sys.argv[:]
    sys.argv = [
        "f5-tts_infer-cli",
        "--model", model,
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", text,
        "--output_dir", str(out.parent),
        "--output_file", out.name,
        "--device", device,
    ]
    try:
        from f5_tts.infer.infer_cli import main as infer_main
        infer_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"f5-tts_infer-cli exited with {exc.code}") from exc
    finally:
        sys.argv = old_argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    root = Path(__file__).resolve().parents[2]
    cache = root / "models" / "f5" / "huggingface"
    os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vidloc_mpl_"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    ref_audio = spec.get("ref_audio") or ""
    ref_text = spec.get("ref_text") or ""
    if not ref_audio:
        print("WORKER error: F5-TTS needs ref_audio", flush=True)
        return 2

    model = spec.get("model") or "F5TTS_v1_Base"
    device = spec.get("device") or "mps"
    items = spec["items"]
    print(f"WORKER f5 model={model} device={device} ref={ref_audio}", flush=True)

    n = len(items)
    for i, it in enumerate(items):
        text = (it.get("text") or "").strip()
        out = Path(it["out"])
        if not text:
            _write_silence(str(out))
            print(f"PROGRESS {i + 1}/{n}", flush=True)
            continue
        try:
            _run_f5(model, ref_audio, ref_text, text, out, device)
        except Exception as exc:  # noqa: BLE001
            print(f"WORKER item {i} error: {exc}", flush=True)
            _write_silence(str(out))
        print(f"PROGRESS {i + 1}/{n}", flush=True)

    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
