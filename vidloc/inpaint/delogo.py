"""Fast tier: ffmpeg ``delogo`` — interpolates a rectangular region from its
border. Instant and dependency-free, but blurry. Good for faint corner logos.
"""

from __future__ import annotations

from ..media.ffmpeg import run_ffmpeg


def _px_one(rect, w: int, h: int):
    x, y, rw, rh = rect
    bx = int(round(x * w))
    by = int(round(y * h))
    bw = int(round(rw * w))
    bh = int(round(rh * h))
    # delogo needs a >=1px border inside the frame to sample from
    bx = max(1, min(bx, w - 3))
    by = max(1, min(by, h - 3))
    bw = max(1, min(bw, w - bx - 1))
    bh = max(1, min(bh, h - by - 1))
    return bx, by, bw, bh


def delogo_video(src, out, regions, w, h, output=None, duration: float = 0.0,
                 on_progress=None, on_log=None) -> str:
    """``regions`` is a list of TimedRegion; each box is time-gated with the
    ffmpeg ``delogo`` filter's ``enable`` expression (no intervals = always on)."""
    from ..media.ffmpeg import video_encoder_args
    from ..models import VideoCodec
    parts = []
    for reg in regions:
        bx, by, bw, bh = _px_one(reg.rect, int(w), int(h))
        f = f"delogo=x={bx}:y={by}:w={bw}:h={bh}"
        if reg.intervals:
            expr = "+".join(f"between(t,{float(t0)},{float(t1)})" for (t0, t1) in reg.intervals)
            f += f":enable='{expr}'"
        parts.append(f)
    vf = ",".join(parts)
    args = ["-i", str(src), "-vf", vf]
    args += video_encoder_args(output, crf=18, codec=VideoCodec.H264)
    args += ["-c:a", "copy", str(out)]
    run_ffmpeg(args, total_duration=duration, on_progress=on_progress, on_log=on_log)
    return str(out)
