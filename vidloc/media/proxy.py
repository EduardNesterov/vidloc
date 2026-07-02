"""Editor preview proxy: a small, seek-friendly H.264 rendition of the source.

The region editor needs *instant* scrubbing (like a video editor). The raw
sources are usually AV1/VP9, whose software decode is too heavy for live
scrubbing, and Qt Multimedia can only hardware-decode H.264/HEVC. So we build a
downscaled, short-GOP H.264 proxy once and let ``QMediaPlayer`` seek inside it —
short keyframe interval means every seek lands on a nearby keyframe fast.

The proxy is cached on disk keyed by the source (path + mtime + size), so a
second open of the same video reuses it. ``load_pcm`` decodes the audio track to
an in-memory PCM buffer for jog ("shuttle") audio while dragging the slider.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import settings
from ..workspace import slugify
from . import ffmpeg

# Proxy render settings — 720p is plenty for region selection (regions are
# normalized), veryfast keeps build time low, g=15 + sc_threshold=0 forces a
# keyframe at least every 15 frames so seeks are near-instant.
PROXY_HEIGHT = 720
JOG_SR = 22050          # jog-audio sample rate (mono s16le)


def _cache_dir() -> Path:
    d = Path(settings.WORKSPACE_DIR) / ".preview"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _signature(src: str) -> str:
    """Stable id from the source so cache invalidates when the file changes."""
    try:
        st = Path(src).stat()
        return f"{int(st.st_mtime)}_{st.st_size}"
    except OSError:
        return "0_0"


def proxy_path(src: str) -> Path:
    """Deterministic cache path for ``src``'s 720p scrub preview (editor)."""
    stem = slugify(Path(src).stem)
    return _cache_dir() / f"{stem}_{_signature(src)}.mp4"


def master_path(src: str) -> Path:
    """Deterministic cache path for ``src``'s full-res H.264 working master.

    The AV1 sources decode ~10× slower than H.264 (no dav1d in the bundled ffmpeg,
    no AV1 hardware decode on M2); the inpaint worker decodes the source twice per
    run. Reading this near-lossless H.264 master instead makes those decodes cheap.
    """
    stem = slugify(Path(src).stem)
    return _cache_dir() / f"{stem}_{_signature(src)}.master.mp4"


def is_valid(path: str) -> bool:
    """True if ``path`` is a readable proxy with a real video stream.

    Guards against a half-written / clobbered cache file being reused forever
    (a bad proxy would silently fall the editor back to slow single-frame mode).
    """
    try:
        info = ffmpeg.probe(path)
        return info.has_video and info.duration > 0
    except Exception:  # noqa: BLE001
        return False


def build(src: str, master_dst: str, preview_dst: str, *, duration: float = 0.0,
          on_progress=None, on_log=None) -> None:
    """Render both derivations from ``src`` in a *single* ffmpeg pass.

    One AV1 decode feeds two encoders (the expensive part is the decode, so this
    is barely more than building the master alone):

      * ``master_dst`` — full-res, visually-lossless H.264 (CRF 16) + audio, for the
        inpaint worker to decode fast. Default GOP (decoded sequentially anyway).
      * ``preview_dst`` — 720p, short-GOP (fast seeks) H.264 + audio, for the editor
        scrub / jog / playback.
    """
    args = [
        "-i", src,
        # --- output 1: full-res master (near-lossless) ---
        "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", master_dst,
        # --- output 2: 720p seek-friendly preview ---
        "-map", "0:v:0", "-vf", f"scale=-2:{PROXY_HEIGHT}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", preview_dst,
    ]
    ffmpeg.run_ffmpeg(args, total_duration=duration,
                      on_progress=on_progress, on_log=on_log)


def load_pcm(src: str, sr: int = JOG_SR) -> bytes:
    """Decode the audio track to raw mono s16le PCM in memory (for jog audio).

    Returns empty bytes if the source has no audio or decoding fails — the caller
    treats that as "no jog audio available".
    """
    cmd = [
        settings.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "pipe:1",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=False)
        return out.stdout or b""
    except OSError:
        return b""
