"""Global paths and environment configuration.

Importing this module is cheap and side-effect-light: it only sets a few
environment defaults and resolves filesystem paths. Heavy ML imports happen
lazily inside the modules that need them, *after* these env vars are set, so
that model downloads are routed into the project's own ``models/`` folder and
the app stays fully self-contained.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = the directory that contains the `vidloc` package.
ROOT = Path(__file__).resolve().parent.parent

BIN_DIR = ROOT / "bin"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
MODELS_DIR = Path(os.environ.get("VIDLOC_MODELS_DIR", str(ROOT / "models")))
WORK_DIR = Path(os.environ.get("VIDLOC_WORK_DIR", str(ROOT / "work")))
OUTPUT_DIR = Path(os.environ.get("VIDLOC_OUTPUT_DIR", str(ROOT / "output")))

# Bundled static ffmpeg/ffprobe (Apple Silicon). Overridable via env.
FFMPEG = os.environ.get("VIDLOC_FFMPEG", str(BIN_DIR / "ffmpeg"))
FFPROBE = os.environ.get("VIDLOC_FFPROBE", str(BIN_DIR / "ffprobe"))

# All Hugging Face model downloads land here, inside the project.
HF_HOME = MODELS_DIR / "huggingface"


def configure_env() -> None:
    """Set environment defaults and make sure working folders exist."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
    # Let PyTorch fall back to CPU for the few ops not yet implemented on MPS,
    # instead of crashing — important for NLLB / XTTS on Apple Silicon.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Avoid tokenizers fork-warning noise inside Qt worker threads.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for d in (MODELS_DIR, WORK_DIR, OUTPUT_DIR, HF_HOME):
        d.mkdir(parents=True, exist_ok=True)


def torch_device() -> str:
    """Best available compute device for PyTorch on this machine."""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


configure_env()
