"""Quality tier: neural LaMa inpainting via an in-memory, batched worker
(``lama_worker.py`` in ``.venv-inpaint``).

This module (main venv) computes the crop / mask geometry and drives the worker.
``scale`` downscales the inpaint resolution for speed (compute is ~quadratic):
scale=1 = best, scale=2 = ~4x less compute.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import settings
from ..media.ffmpeg import video_encoder_args
from ..models import VideoCodec

INPAINT_PY = settings.ROOT / ".venv-inpaint" / "bin" / "python"
WORKER = Path(__file__).parent / "lama_worker.py"
MODEL = settings.MODELS_DIR / "inpaint" / "torch" / "hub" / "checkpoints" / "big-lama.pt"
MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
PAD = 48


def _even(n: int) -> int:
    n = int(n)
    return n - (n % 2)


def _ensure_model(log) -> None:
    if MODEL.exists():
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    log("Завантажую модель LaMa (~200 МБ, один раз)…")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, str(MODEL))


def lama_video(src, out, regions, info, output=None, scale: int = 2,
               reporter=None, on_progress=None) -> str:
    log = reporter.log if reporter else (lambda m: None)
    if not INPAINT_PY.exists():
        raise RuntimeError("Середовище інпейнту (.venv-inpaint) не знайдено.")
    _ensure_model(log)

    w, h = int(info.width), int(info.height)
    fps = info.fps or 30.0

    # union bounding box of all regions, + context padding
    px = [(int(round(rx * w)), int(round(ry * h)), int(round(rw * w)), int(round(rh * h)))
          for (rx, ry, rw, rh) in regions]
    cx = max(0, min(b[0] for b in px) - PAD)
    cy = max(0, min(b[1] for b in px) - PAD)
    cw = _even(min(w, max(b[0] + b[2] for b in px) + PAD) - cx)
    ch = _even(min(h, max(b[1] + b[3] for b in px) + PAD) - cy)
    cw = max(8, min(cw, w - cx))
    ch = max(8, min(ch, h - cy))
    boxes = [[bx - cx, by - cy, bw, bh] for (bx, by, bw, bh) in px]

    venc = video_encoder_args(output, crf=16, codec=VideoCodec.H264)
    total = max(1, round(info.duration * fps))
    log(f"Нейро-затирання {cw}×{ch} (масштаб ÷{scale}), ~{total} кадрів пачками в пам'яті…")

    cmd = [str(INPAINT_PY), str(WORKER),
           "--src", str(src), "--out", str(out), "--ffmpeg", settings.FFMPEG,
           "--model", str(MODEL), "--cx", str(cx), "--cy", str(cy),
           "--cw", str(cw), "--ch", str(ch), "--boxes", json.dumps(boxes),
           "--fps", str(fps), "--scale", str(scale), "--batch", "8",
           "--total", str(total), "--venc", json.dumps(venc)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS"):
            try:
                d, tt = line.split()[1].split("/")
                if on_progress:
                    on_progress(min(0.99, int(d) / max(1, int(tt))))
            except (ValueError, IndexError):
                pass
        elif line and ("ERROR" in line or "Traceback" in line or line.startswith("WORKER")):
            log(line)
        if reporter and reporter.cancelled:
            proc.terminate()
            raise RuntimeError("Скасовано")
    if proc.wait() != 0:
        raise RuntimeError(f"lama-воркер завершився з кодом {proc.returncode}")
    return str(out)
