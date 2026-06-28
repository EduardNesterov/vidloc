"""Quality tier: neural LaMa inpainting via IOPaint — CROP-TO-REGION.

Only each region (+ context padding) is extracted and inpainted, not the whole
frame, then composited back with a single ffmpeg overlay pass. For a small corner
logo that is ~10x fewer pixels per frame -> minutes become seconds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from .. import settings
from ..media.ffmpeg import run_ffmpeg, video_encoder_args
from ..models import VideoCodec

IOPAINT = settings.ROOT / ".venv-inpaint" / "bin" / "iopaint"
MODEL_DIR = settings.MODELS_DIR / "inpaint"
PAD = 48  # context padding around the region (px) — LaMa needs surroundings to fill


def _even(n: int) -> int:
    n = int(n)
    return n - (n % 2)


def _crop_box(region, w: int, h: int):
    rx, ry, rw, rh = region
    bx = int(round(rx * w))
    by = int(round(ry * h))
    bw = int(round(rw * w))
    bh = int(round(rh * h))
    cx = max(0, bx - PAD)
    cy = max(0, by - PAD)
    cw = _even(min(w, bx + bw + PAD) - cx)
    ch = _even(min(h, by + bh + PAD) - cy)
    cw = max(8, min(cw, w - cx))
    ch = max(8, min(ch, h - cy))
    return {"cx": cx, "cy": cy, "cw": cw, "ch": ch,
            "mx": bx - cx, "my": by - cy, "mw": bw, "mh": bh}


def _make_mask(box, path: str) -> None:
    draw = (f"drawbox=x={box['mx']}:y={box['my']}:w={box['mw']}:h={box['mh']}"
            f":color=white@1.0:t=fill")
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={box['cw']}x{box['ch']}:d=1",
                "-vf", draw, "-frames:v", "1", str(path)])


def lama_video(src, out, regions, info, output=None, reporter=None, on_progress=None) -> str:
    log = reporter.log if reporter else (lambda m: None)
    if not IOPAINT.exists():
        raise RuntimeError("Середовище інпейнту (.venv-inpaint) не знайдено.")

    w, h = int(info.width), int(info.height)
    fps = info.fps or 30.0
    work = Path(settings.WORK_DIR) / "inpaint"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    boxes = [_crop_box(tuple(r), w, h) for r in regions]
    overlays: List[Tuple[Path, int, int]] = []

    for i, box in enumerate(boxes):
        cdir = work / f"crop{i}"
        odir = work / f"out{i}"
        cdir.mkdir()
        odir.mkdir()
        log(f"Область {i + 1}: затираю {box['cw']}×{box['ch']} замість {w}×{h} "
            f"(у {round(w * h / max(1, box['cw'] * box['ch']))}× менше пікселів).")
        run_ffmpeg(["-i", str(src), "-vf",
                    f"crop={box['cw']}:{box['ch']}:{box['cx']}:{box['cy']}",
                    str(cdir / "%06d.png")], total_duration=info.duration,
                   on_progress=(lambda f: on_progress(f * 0.15)) if (on_progress and i == 0) else None)
        mask = work / f"mask{i}.png"
        _make_mask(box, str(mask))

        cmd = [str(IOPAINT), "run", "--model", "lama", "--device", "mps",
               "--model-dir", str(MODEL_DIR), "--image", str(cdir),
               "--mask", str(mask), "--output", str(odir)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            ls = line.strip()
            if ls and (ls.lower().startswith(("error", "traceback")) or "Downloading" in ls):
                log(ls)
            if reporter and reporter.cancelled:
                proc.terminate()
                raise RuntimeError("Скасовано")
        if proc.wait() != 0:
            raise RuntimeError(f"iopaint завершився з кодом {proc.returncode}")
        overlays.append((odir, box["cx"], box["cy"]))
    if on_progress:
        on_progress(0.9)

    log("Вклеюю оброблені області назад…")
    args: List[str] = ["-i", str(src)]
    for (odir, _cx, _cy) in overlays:
        args += ["-framerate", str(fps), "-i", str(odir / "%06d.png")]
    cur = "[0:v]"
    fc = []
    for idx, (_odir, cx, cy) in enumerate(overlays, start=1):
        label = "[vout]" if idx == len(overlays) else f"[v{idx}]"
        fc.append(f"{cur}[{idx}:v]overlay={cx}:{cy}{label}")
        cur = label
    args += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "0:a:0?"]
    args += video_encoder_args(output, crf=16, codec=VideoCodec.H264)
    args += ["-c:a", "copy", str(out)]
    run_ffmpeg(args, total_duration=info.duration,
               on_progress=(lambda f: on_progress(0.9 + f * 0.1)) if on_progress else None)
    return str(out)
