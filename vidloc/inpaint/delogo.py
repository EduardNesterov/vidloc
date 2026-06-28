"""Fast tier: ffmpeg ``delogo`` — interpolates a rectangular region from its
border. Instant and dependency-free, but blurry. Good for faint corner logos.
"""

from __future__ import annotations

from typing import List, Tuple

from ..media.ffmpeg import run_ffmpeg


def _px(regions: List[Tuple[float, float, float, float]], w: int, h: int):
    out = []
    for (x, y, rw, rh) in regions:
        bx = int(round(x * w))
        by = int(round(y * h))
        bw = int(round(rw * w))
        bh = int(round(rh * h))
        # delogo needs a >=1px border inside the frame to sample from
        bx = max(1, min(bx, w - 3))
        by = max(1, min(by, h - 3))
        bw = max(1, min(bw, w - bx - 1))
        bh = max(1, min(bh, h - by - 1))
        out.append((bx, by, bw, bh))
    return out


def delogo_video(src, out, regions, w, h, output=None, duration: float = 0.0,
                 on_progress=None, on_log=None) -> str:
    from ..media.ffmpeg import video_encoder_args
    from ..models import VideoCodec
    px = _px([tuple(r) for r in regions], int(w), int(h))
    vf = ",".join(f"delogo=x={x}:y={y}:w={ww}:h={hh}" for (x, y, ww, hh) in px)
    args = ["-i", str(src), "-vf", vf]
    args += video_encoder_args(output, crf=18, codec=VideoCodec.H264)
    args += ["-c:a", "copy", str(out)]
    run_ffmpeg(args, total_duration=duration, on_progress=on_progress, on_log=on_log)
    return str(out)
