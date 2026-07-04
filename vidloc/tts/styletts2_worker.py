#!/usr/bin/env python
"""StyleTTS2-Ukrainian worker. Runs in isolated ``.venv-styletts2``.

Spec JSON:
    {
      "model": "patriotyk/styletts2_ukrainian_multispeaker_hifigan",
      "voices_repo": "patriotyk/styletts2-ukrainian",
      "voice": "Петро Філяк",
      "speed": 1.0,
      "cache_dir": "/path/models/styletts2",
      "items": [{"text": "...", "out": "/path/seg_0000.wav"}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import wave
from pathlib import Path
from unicodedata import normalize


def _write_silence(path: str, sr: int = 24000, dur: float = 0.25) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * dur))


def _split(text: str) -> list[str]:
    text = re.sub(r"(\w+[^.,!:?\-])\n", r"\1. ", text).replace("\n", " ")
    parts = [p.strip() for p in re.split(r"(?<=[.?!:])\s+", text) if p.strip()]
    return parts or [text.strip()]


def _norm_text(text: str, stress_symbol: str) -> str:
    text = text.replace('"', "").replace("+", stress_symbol)
    text = normalize("NFKC", text)
    text = re.sub(r"[᠆‐‑‒–—―⁻₋−⸺⸻]", "-", text)
    if text and text[-1] not in ".?!:-":
        text += "."
    return re.sub(r" - ", ": ", text)


def _move_style(obj, device: str):
    try:
        import torch
        if torch.is_tensor(obj):
            return obj.to(device)
    except Exception:
        return obj
    if isinstance(obj, dict):
        return {k: _move_style(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_style(v, device) for v in obj)
    return obj


def _voice_path(repo: str, voice: str, cache_dir: str) -> Path:
    local = Path(cache_dir) / "voices" / f"{voice}.pt"
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo,
        repo_type="space",
        cache_dir=str(Path(cache_dir) / "hf"),
        allow_patterns=["voices/*.pt"],
    )
    path = Path(root) / "voices" / f"{voice}.pt"
    if path.exists():
        return path
    voices = sorted((Path(root) / "voices").glob("*.pt"))
    if voices:
        return voices[0]
    raise FileNotFoundError(f"StyleTTS2 voice not found: {voice}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    cache_dir = spec.get("cache_dir") or os.getcwd()
    os.environ.setdefault("HF_HOME", str(Path(cache_dir) / "huggingface"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import numpy as np
    import soundfile as sf
    import torch
    from ipa_uk import ipa
    from styletts2_inference.models import StyleTTS2
    from ukrainian_word_stress import Stressifier, StressSymbol

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model_repo = spec.get("model") or "patriotyk/styletts2_ukrainian_multispeaker_hifigan"
    try:
        model = StyleTTS2(hf_path=model_repo, device=device)
    except Exception as exc:  # noqa: BLE001
        fallback = "patriotyk/styletts2_ukrainian_multispeaker"
        print(f"WORKER warn: {model_repo} failed ({exc}); trying {fallback}", flush=True)
        model_repo = fallback
        model = StyleTTS2(hf_path=model_repo, device=device)

    voice_name = spec.get("voice") or "Петро Філяк"
    voices_repo = spec.get("voices_repo") or "patriotyk/styletts2-ukrainian"
    voice = _voice_path(voices_repo, voice_name, cache_dir)
    style = _move_style(torch.load(voice, map_location="cpu"), device)
    stressify = Stressifier()
    speed = float(spec.get("speed") or 1.0)
    items = spec["items"]

    print(
        f"WORKER styletts2 model={model_repo} voice={voice.stem} device={device}",
        flush=True,
    )
    n = len(items)
    for i, it in enumerate(items):
        text = (it.get("text") or "").strip()
        out = it["out"]
        if not text:
            _write_silence(out)
            print(f"PROGRESS {i + 1}/{n}", flush=True)
            continue
        try:
            wavs = []
            for part in _split(text):
                part = _norm_text(part, StressSymbol.CombiningAcuteAccent)
                part = stressify(part)
                phones = ipa(part)
                if not phones:
                    continue
                tokens = model.tokenizer.encode(phones)
                wavs.append(model(tokens, speed=speed, s_prev=style))
            if wavs:
                wav = torch.cat(wavs).detach().cpu().numpy().astype(np.float32)
                sf.write(out, wav, 24000)
            else:
                _write_silence(out)
        except Exception as exc:  # noqa: BLE001
            print(f"WORKER item {i} error: {exc}", flush=True)
            _write_silence(out)
        print(f"PROGRESS {i + 1}/{n}", flush=True)

    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
