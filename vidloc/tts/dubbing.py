"""Dubbing: synthesize Ukrainian speech per segment and fit each clip onto the
original timeline.

This module runs in the MAIN venv. The TTS models run in isolated venvs via
worker scripts invoked as subprocesses, so their (older / conflicting)
dependencies never touch the core environment:

  * STYLETTS2 -> StyleTTS2-Ukrainian     in ``.venv-styletts2`` via ``styletts2_worker.py``
  * UKTTS    -> ukrainian-tts (5 voices)  in ``.venv-uktts`` via ``uktts_worker.py``
  * COQUI_UK -> Coqui uk/mai/vits         in ``.venv-tts``   via ``coqui_worker.py``
  * F5_UK    -> F5-TTS voice clone        in ``.venv-f5``    via ``f5_worker.py``
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

from .. import settings
from ..models import TTS_VOICES, TTSEngine

SR = 24000  # assembly sample rate
MAX_SPEEDUP = 1.6  # cap on time-compression so the dub never sounds rushed
_HERE = Path(__file__).parent


def _engine_table() -> Dict[TTSEngine, dict]:
    return {
        TTSEngine.STYLETTS2_UK: {
            "python": settings.ROOT / ".venv-styletts2" / "bin" / "python",
            "worker": _HERE / "styletts2_worker.py",
            "cwd": settings.MODELS_DIR / "styletts2",
        },
        TTSEngine.UKTTS: {
            "python": settings.ROOT / ".venv-uktts" / "bin" / "python",
            "worker": _HERE / "uktts_worker.py",
            # ukrainian-tts downloads its model into the *current* dir — pin it
            "cwd": settings.MODELS_DIR / "uktts",
        },
        TTSEngine.PIPER: {
            "python": settings.ROOT / ".venv-piper" / "bin" / "python",
            "worker": _HERE / "piper_worker.py",
            "cwd": None,
        },
        TTSEngine.COQUI_UK: {
            "python": settings.ROOT / ".venv-tts" / "bin" / "python",
            "worker": _HERE / "coqui_worker.py",
            "cwd": settings.MODELS_DIR / "tts",
        },
        TTSEngine.F5_UK: {
            "python": settings.ROOT / ".venv-f5" / "bin" / "python",
            "worker": _HERE / "f5_worker.py",
            "cwd": settings.MODELS_DIR / "f5",
        },
    }


def _voice_id(job, engine: TTSEngine) -> str:
    valid = [vid for _, vid in TTS_VOICES.get(engine, [])]
    want = (job.translate.tts_voice or "").strip()
    if want in valid:
        return want
    for label, vid in TTS_VOICES.get(engine, []):
        if want == label:
            return vid
    return valid[0] if valid else "tetiana"


def _resolve_engine(job):
    """Return (engine, info) for the first available engine, preferring the
    user's choice, then falling back to whatever is installed."""
    table = _engine_table()
    order = [job.translate.tts_engine] + [e for e in TTSEngine if e != job.translate.tts_engine]
    for eng in order:
        info = table.get(eng)
        if info and Path(info["python"]).exists():
            return eng, info
    raise RuntimeError(
        "Жодне середовище озвучки не знайдено. Встановіть хоча б одне:\n"
        "  StyleTTS2 UA: uv venv .venv-styletts2 --python 3.11 && "
        "uv pip install --python .venv-styletts2/bin/python soundfile librosa scipy "
        "einops einops_exts munch pyyaml torch torchaudio transformers 'numpy<2' "
        "accelerate ctranslate2 sentencepiece huggingface-hub "
        "git+https://github.com/patriotyk/ukrainian-word-stress.git "
        "git+https://github.com/patriotyk/ipa-uk.git "
        "git+https://github.com/patriotyk/styletts2-inference.git\n"
        "  ukrainian-tts: uv venv .venv-uktts --python 3.11 && "
        "uv pip install --python .venv-uktts/bin/python ukrainian-tts\n"
        "  Piper UA:      uv venv .venv-piper --python 3.11 && "
        "uv pip install --python .venv-piper/bin/python piper-tts onnxruntime "
        "ukrainian-word-stress\n"
        "  Coqui UA:      uv venv .venv-tts --python 3.11 && "
        "uv pip install --python .venv-tts/bin/python coqui-tts torch torchaudio\n"
        "  F5-TTS clone: uv venv .venv-f5 --python 3.11 && "
        "uv pip install --python .venv-f5/bin/python f5-tts soundfile")


def _make_spec(engine: TTSEngine, job, items: List[dict],
               ref_audio: str = "", ref_text: str = "") -> dict:
    if engine == TTSEngine.STYLETTS2_UK:
        return {"backend": "styletts2_uk",
                "model": "patriotyk/styletts2_ukrainian_multispeaker_hifigan",
                "voices_repo": "patriotyk/styletts2-ukrainian",
                "voice": _voice_id(job, engine),
                "speed": 1.0,
                "cache_dir": str(settings.MODELS_DIR / "styletts2"),
                "items": items}
    if engine == TTSEngine.UKTTS:
        return {"backend": "uktts", "voice": _voice_id(job, engine),
                "stress": "dictionary", "device": "cpu", "items": items}
    if engine == TTSEngine.PIPER:
        return {"model_path": str(settings.MODELS_DIR / "piper"
                                  / "uk_UA-ukrainian_tts-medium.onnx"),
                "speaker_id": int(_voice_id(job, engine) or 0),
                "length_scale": 1.0, "items": items}
    if engine == TTSEngine.F5_UK:
        return {"backend": "f5_uk", "model": "F5TTS_v1_Base",
                "ref_audio": ref_audio, "ref_text": ref_text,
                "device": "mps", "items": items}
    return {"backend": "coqui_uk", "model": "tts_models/uk/mai/vits",
            "tts_home": str(settings.MODELS_DIR / "tts"), "items": items}


# --- timeline assembly helpers ------------------------------------------- #
def _run(args: List[str]) -> None:
    subprocess.run([str(a) for a in args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _wav_dur(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def _atempo_chain(factor: float) -> str:
    parts: List[str] = []
    f = factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def _fit_clip(src_wav: str, dst_wav: str, factor: float) -> None:
    args = [settings.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src_wav]
    if abs(factor - 1.0) > 0.02:
        args += ["-af", _atempo_chain(factor)]
    args += ["-ar", str(SR), "-ac", "1", dst_wav]
    _run(args)


def _reference_wav(segments: List[Dict], job, work) -> tuple[str, str]:
    """F5 prompt audio: explicit clone reference, or the first transcript segment."""
    explicit = (getattr(job.translate, "tts_clone_reference", None) or "").strip()
    if explicit:
        return explicit, ""
    cand = next((s for s in segments if (s.get("text") or "").strip()
                 and float(s.get("end", 0)) > float(s.get("start", 0))), None)
    if not cand:
        return "", ""
    start = max(0.0, float(cand["start"]))
    dur = min(10.0, max(1.0, float(cand["end"]) - start))
    out = str(Path(work) / "tts_ref.wav")
    args = [settings.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(dur), "-i", job.input_path,
            "-vn", "-ar", "24000", "-ac", "1", out]
    try:
        _run(args)
    except subprocess.CalledProcessError:
        return "", ""
    return out, (cand.get("text") or "").strip()


def _surface(line: str) -> bool:
    """Decide whether a worker stdout line is worth showing in the GUI log."""
    if line.startswith("WORKER") or "error" in line.lower() or "Download" in line:
        return True
    return False


def synthesize_dub(segments: List[Dict], job, info, work, reporter=None,
                   on_progress=None) -> str:
    import numpy as np
    import soundfile as sf

    log = reporter.log if reporter else (lambda m: None)
    engine, eng = _resolve_engine(job)
    if engine != job.translate.tts_engine:
        log(f"Рушій {job.translate.tts_engine.value} недоступний — використовую {engine.value}.")

    tdir = Path(work) / "tts"
    tdir.mkdir(parents=True, exist_ok=True)

    items = [
        {"text": (seg.get("text_tr") or seg.get("text") or ""),
         "src_text": (seg.get("text") or ""),
         "out": str(tdir / f"seg_{i:04d}.wav")}
        for i, seg in enumerate(segments)
    ]
    ref_audio = ref_text = ""
    if engine == TTSEngine.F5_UK:
        ref_audio, ref_text = _reference_wav(segments, job, work)
        if not ref_audio:
            raise RuntimeError("F5-TTS потребує reference-аудіо, але його не вдалося витягти.")
    spec = _make_spec(engine, job, items, ref_audio=ref_audio, ref_text=ref_text)
    spec_path = tdir / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    if "voice" in spec:
        voice_note = spec["voice"]
    elif "speaker_id" in spec:
        voice_note = f"#{spec['speaker_id']}"
    elif "ref_audio" in spec:
        voice_note = "клон з оригіналу"
    else:
        voice_note = "стандартний"
    log(f"Озвучення: {engine.value} / голос «{voice_note}» "
        f"(перший запуск завантажить модель).")

    cwd = eng.get("cwd")
    if cwd:
        Path(cwd).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(eng["python"]), str(eng["worker"]), "--spec", str(spec_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        cwd=str(cwd) if cwd else None)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS"):
            try:
                cur, total = line.split()[1].split("/")
                if on_progress:
                    on_progress(min(0.8, int(cur) / max(1, int(total)) * 0.8))
            except (ValueError, IndexError):
                pass
        elif line and _surface(line):
            log(line)
        if reporter and reporter.cancelled:
            proc.terminate()
            raise RuntimeError("Скасовано")
    if proc.wait() != 0:
        raise RuntimeError(f"Воркер озвучення завершився з кодом {proc.returncode}")

    # ---- lay each clip onto a silent timeline, speeding up over-long lines --
    last_end = max((s["end"] for s in segments), default=0.0)
    base = np.zeros(int((max(info.duration, last_end) + 1.0) * SR), dtype=np.float32)
    speed = max(0.5, min(1.5, getattr(job.translate, "tts_speed", 1.0) or 1.0))

    for i, seg in enumerate(segments):
        raw = str(tdir / f"seg_{i:04d}.wav")
        if not Path(raw).exists():
            continue
        dur = _wav_dur(raw)
        target = max(0.0, seg["end"] - seg["start"])
        # Match each line's pace to how fast the speaker said it in the original —
        # dynamic tempo that "breathes" with the video; only compress, never drag.
        factor = 1.0
        if target > 0.3 and dur > target * 1.02:
            factor = min(MAX_SPEEDUP, dur / target)
        factor = max(0.5, min(2.0, factor * speed))  # user slider nudges faster/slower
        fitted = str(tdir / f"fit_{i:04d}.wav")
        _fit_clip(raw, fitted, factor)

        clip, _ = sf.read(fitted, dtype="float32", always_2d=False)
        if getattr(clip, "ndim", 1) > 1:
            clip = clip.mean(axis=1)
        start = int(seg["start"] * SR)
        end = start + len(clip)
        if end > len(base):
            base = np.concatenate([base, np.zeros(end - len(base), dtype=np.float32)])
        base[start:end] += clip
        if on_progress:
            on_progress(min(0.99, 0.8 + 0.2 * (i + 1) / max(1, len(segments))))

    peak = float(np.max(np.abs(base))) if base.size else 0.0
    if peak > 1.0:
        base = base / peak * 0.98

    dub_path = str(Path(work) / "dub.wav")
    sf.write(dub_path, base, SR)
    log(f"Українську озвучку зібрано: {dub_path}")
    return dub_path
