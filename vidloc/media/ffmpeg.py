"""Thin, dependency-free wrappers around the bundled ffmpeg / ffprobe binaries.

These are deliberately subprocess-based (rather than PyAV) because the heavy
lifting — scaling, subtitle burn-in, audio ducking, container/codec choice — is
far easier to express as an ffmpeg filtergraph than through a bindings API.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from .. import settings
from ..models import Container, OutputConfig, VideoCodec

ProgressCb = Optional[Callable[[Optional[float]], None]]
LogCb = Optional[Callable[[str], None]]


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool
    v_codec: str = ""
    a_codec: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe(path: str) -> MediaInfo:
    cmd = [
        settings.FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(data.get("format", {}).get("duration") or 0.0)
    width = int(v["width"]) if v and "width" in v else 0
    height = int(v["height"]) if v and "height" in v else 0

    fps = 0.0
    if v:
        rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    return MediaInfo(
        duration=duration, width=width, height=height, fps=fps,
        has_audio=a is not None, has_video=v is not None,
        v_codec=(v or {}).get("codec_name", ""),
        a_codec=(a or {}).get("codec_name", ""),
    )


def run_ffmpeg(args: List[str], total_duration: float = 0.0,
               on_progress: ProgressCb = None, on_log: LogCb = None) -> None:
    """Run ffmpeg, streaming progress (0..1) via ``-progress pipe:1``."""
    cmd = [settings.FFMPEG, "-hide_banner", "-y",
           "-progress", "pipe:1", "-nostats", *[str(a) for a in args]]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us=") and total_duration > 0 and on_progress:
            try:
                us = int(line.split("=", 1)[1])
                on_progress(min(1.0, (us / 1_000_000) / total_duration))
            except ValueError:
                pass
        elif line == "progress=end" and on_progress:
            on_progress(1.0)
    err = proc.stderr.read() if proc.stderr else ""
    ret = proc.wait()
    if on_log and err:
        on_log(err[-4000:])
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with {ret}:\n{err[-2000:]}")


def extract_audio(src: str, out_wav: str, sr: int = 16000, mono: bool = True,
                  duration: float = 0.0, on_progress: ProgressCb = None,
                  on_log: LogCb = None) -> None:
    """Decode the audio track to PCM wav (16 kHz mono by default — Whisper-ready)."""
    args = [
        "-i", src, "-vn",
        "-ac", "1" if mono else "2", "-ar", str(sr),
        "-c:a", "pcm_s16le", out_wav,
    ]
    run_ffmpeg(args, total_duration=duration, on_progress=on_progress, on_log=on_log)


def _filter_path(p: str) -> str:
    """Quote/escape a filename for use inside an ffmpeg filtergraph value."""
    p = str(p).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return f"'{p}'"


def video_encoder_args(output=None, crf=None, codec=None) -> List[str]:
    """Video encoder flags — software (libx264/5, CRF) or VideoToolbox hardware
    (h264/hevc_videotoolbox, constant-quality). Shared by render and inpaint."""
    if codec is None:
        codec = output.codec if output else VideoCodec.H264
    if crf is None:
        crf = output.crf if output else 18
    is265 = codec == VideoCodec.H265
    hw = bool(getattr(output, "hw_encode", False)) if output else False
    if hw:
        enc = "hevc_videotoolbox" if is265 else "h264_videotoolbox"
        q = max(1, min(100, int(round(100 - (crf - 14) * 4))))  # CRF -> VT quality
        a = ["-c:v", enc, "-q:v", str(q), "-pix_fmt", "yuv420p"]
        if is265:
            a += ["-tag:v", "hvc1"]
        return a
    enc = "libx265" if is265 else "libx264"
    # `veryfast` is ~2x faster than `medium` on M2 with better size than VideoToolbox
    # (see benchmarks) — a much better default for re-encoding than the old `medium`.
    preset = getattr(output, "preset", "veryfast") if output else "veryfast"
    a = ["-c:v", enc, "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    if is265:
        a += ["-tag:v", "hvc1"]
    return a


def render_video(src: str, out: str, output: OutputConfig, *,
                 sub_path: Optional[str] = None, sub_burn: bool = False,
                 dub_path: Optional[str] = None, keep_background: bool = True,
                 src_has_audio: bool = True, duration: float = 0.0,
                 fontsdir: Optional[str] = None,
                 on_progress: ProgressCb = None, on_log: LogCb = None) -> None:
    """Assemble the final video from the source plus optional subtitles / dub.

    Handles scaling, subtitle burn-in or soft muxing, audio replacement or
    background-ducked mixing, and codec / container / quality selection.
    """
    inputs: List[str] = ["-i", src]
    idx = 1
    dub_idx = sub_idx = None
    if dub_path:
        inputs += ["-i", dub_path]
        dub_idx = idx
        idx += 1
    soft_sub = bool(sub_path) and not sub_burn
    if soft_sub:
        inputs += ["-i", sub_path]
        sub_idx = idx
        idx += 1

    # ---- video filter chain ------------------------------------------- #
    vf: List[str] = []
    if output.height:
        vf.append(f"scale=-2:{int(output.height)}")
    if sub_path and sub_burn:
        sub_arg = f"subtitles={_filter_path(sub_path)}"
        if fontsdir:
            sub_arg += f":fontsdir={_filter_path(fontsdir)}"
        vf.append(sub_arg)
    vfilter = ",".join(vf)

    args: List[str] = list(inputs)

    # ---- audio: replace, duck-mix, or passthrough --------------------- #
    use_complex = bool(dub_path) and keep_background and src_has_audio
    if use_complex:
        fc = (f"[0:a]volume=0.2[bg];"
              f"[bg][{dub_idx}:a]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        if vfilter:
            fc = f"[0:v]{vfilter}[vout];" + fc
            args += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
        else:
            args += ["-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]"]
    else:
        if vfilter:
            args += ["-vf", vfilter, "-map", "0:v:0"]
        else:
            args += ["-map", "0:v:0"]
        if dub_path:                       # replace audio with the dub
            args += ["-map", f"{dub_idx}:a:0"]
        elif src_has_audio:                # keep original audio
            args += ["-map", "0:a:0?"]
    if soft_sub:
        args += ["-map", f"{sub_idx}:0"]

    # ---- video codec -------------------------------------------------- #
    # If nothing transforms the video (no scale / burn-in / fps change) and the source
    # already carries the target codec, stream-copy instead of re-encoding. This makes a
    # removal-only job (cleaned.mp4 is already H.264) a fast remux, not a 2nd full encode.
    target_vc = "hevc" if output.codec == VideoCodec.H265 else "h264"
    try:
        src_vc = probe(src).v_codec
    except Exception:  # noqa: BLE001
        src_vc = ""
    can_copy_v = (not vfilter and not use_complex and not output.fps
                  and (output.codec == VideoCodec.KEEP or src_vc == target_vc))
    if can_copy_v:
        args += ["-c:v", "copy"]
    else:
        args += video_encoder_args(output)
    if output.fps:
        args += ["-r", str(output.fps)]

    # ---- audio codec -------------------------------------------------- #
    if dub_path or src_has_audio:
        acodec = "libopus" if output.container == Container.WEBM else "aac"
        args += ["-c:a", acodec, "-b:a", output.audio_bitrate]

    # ---- subtitle codec (soft) --------------------------------------- #
    if soft_sub:
        scodec = "mov_text" if output.container in (Container.MP4, Container.MOV) else "srt"
        args += ["-c:s", scodec, "-metadata:s:s:0", "language=ukr"]

    args += ["-movflags", "+faststart", out] if output.container in (
        Container.MP4, Container.MOV) else [out]

    run_ffmpeg(args, total_duration=duration, on_progress=on_progress, on_log=on_log)
