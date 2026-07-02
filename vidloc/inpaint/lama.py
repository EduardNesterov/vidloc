"""Quality tier: neural LaMa inpainting via an in-memory, batched worker
(``lama_worker.py`` in ``.venv-inpaint``).

This module (main venv) computes the crop / mask geometry and drives the worker.
``scale`` downscales the inpaint resolution for speed (compute is ~quadratic):
scale=1 = best, scale=2 = ~4x less compute.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
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
               reporter=None, on_progress=None, reuse: bool = True,
               reuse_thresh: float = 0.02, reuse_keyint: int = 120) -> str:
    log = reporter.log if reporter else (lambda m: None)
    if not INPAINT_PY.exists():
        raise RuntimeError("Середовище інпейнту (.venv-inpaint) не знайдено.")
    _ensure_model(log)

    w, h = int(info.width), int(info.height)
    fps = info.fps or 30.0

    # pixel rect of every region; their union bbox (+ context padding) is the crop
    px = [(int(round(r.rect[0] * w)), int(round(r.rect[1] * h)),
           int(round(r.rect[2] * w)), int(round(r.rect[3] * h))) for r in regions]
    cx = max(0, min(b[0] for b in px) - PAD)
    cy = max(0, min(b[1] for b in px) - PAD)
    cw = _even(min(w, max(b[0] + b[2] for b in px) + PAD) - cx)
    ch = _even(min(h, max(b[1] + b[3] for b in px) + PAD) - cy)
    cw = max(8, min(cw, w - cx))
    ch = max(8, min(ch, h - cy))
    # each region's box is relative to the crop; intervals stay in seconds
    worker_regions = [
        {"box": [bx - cx, by - cy, bw, bh],
         "intervals": [[float(t0), float(t1)] for (t0, t1) in r.intervals],
         "kind": getattr(r, "kind", "object")}
        for r, (bx, by, bw, bh) in zip(regions, px)
    ]

    venc = video_encoder_args(output, crf=16, codec=VideoCodec.H264)
    total = max(1, round(info.duration * fps))
    log(f"Нейро-затирання {cw}×{ch} (масштаб ÷{scale}), {len(regions)} обл., "
        f"~{total} кадрів пачками в пам'яті…")

    cmd = [str(INPAINT_PY), str(WORKER),
           "--src", str(src), "--out", str(out), "--ffmpeg", settings.FFMPEG,
           "--model", str(MODEL), "--cx", str(cx), "--cy", str(cy),
           "--cw", str(cw), "--ch", str(ch), "--regions", json.dumps(worker_regions),
           "--fps", str(fps), "--scale", str(scale), "--batch", "8",
           "--total", str(total), "--venc", json.dumps(venc),
           "--reuse", "1" if reuse else "0",
           "--reuse-thresh", str(reuse_thresh), "--reuse-keyint", str(reuse_keyint)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=15)   # recent output, surfaced if the worker dies
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("PROGRESS"):
            tail.append(line)
        if line.startswith("PROGRESS"):
            try:
                d, tt = line.split()[1].split("/")
                if on_progress:
                    on_progress(min(0.99, int(d) / max(1, int(tt))))
            except (ValueError, IndexError):
                pass
        elif line.startswith("STAT"):
            try:
                kv = dict(tok.split("=") for tok in line.split()[1:])
                la, re_ = int(kv["lama_reg"]), int(kv["reuse_reg"])
                gpu, fr = int(kv["gpu_frames"]), int(kv["frames"])
                msg = f"LaMa-проходів: {gpu}/{fr} кадрів"
                if re_ > 0:      # reuse path — report how much background was reused
                    saved = 100 * re_ / max(1, la + re_)
                    msg += (f"; фон повторно використано у {re_} з {la + re_} "
                            f"обл.-кадрів ({saved:.0f}%)")
                log(msg)
            except (ValueError, KeyError):
                log(line)
        elif line and ("ERROR" in line or "Traceback" in line or line.startswith("WORKER")):
            log(line)
        if reporter and reporter.cancelled:
            proc.terminate()
            raise RuntimeError("Скасовано")
    if proc.wait() != 0:
        detail = " | ".join(t for t in tail if t)
        raise RuntimeError(
            f"lama-воркер завершився з кодом {proc.returncode}"
            + (f": {detail}" if detail else ""))
    return str(out)
