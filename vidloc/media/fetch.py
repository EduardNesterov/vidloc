"""Fetch videos from URLs (YouTube / Instagram / TikTok / 1000+ sites) with yt-dlp.

Besides downloading the file, we pull the title / description / tags and turn them
into a compact context string — the single best signal for disambiguating word
senses during translation (e.g. a "$10,000 DRIFT CHALLENGE" title makes "drift"
unambiguous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .. import settings

ProgressCb = Optional[Callable[[float, str], None]]
LogCb = Optional[Callable[[str], None]]


@dataclass
class FetchResult:
    path: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    uploader: str = ""
    webpage_url: str = ""

    def context(self) -> str:
        """Compact context string for the translator (kept short)."""
        parts: List[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.uploader:
            parts.append(f"Channel: {self.uploader}")
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags[:12]))
        if self.description:
            parts.append("Description: " + " ".join(self.description.split())[:400])
        return " | ".join(parts)


def is_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _result_from_info(info: Optional[dict], path: str) -> FetchResult:
    if not info:
        return FetchResult(path=path)
    tags = info.get("tags") or info.get("categories") or []
    if not isinstance(tags, list):
        tags = []
    return FetchResult(
        path=path,
        title=info.get("title") or "",
        description=info.get("description") or "",
        tags=[str(t) for t in tags][:20],
        uploader=info.get("uploader") or info.get("channel") or "",
        webpage_url=info.get("webpage_url") or "",
    )


def fetch_info(url: str) -> FetchResult:
    """Metadata only, no download — quick, for previewing the detected context."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return _result_from_info(info, path="")


QUALITY_FORMAT = {
    "best": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "1080": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
    "720": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b",
    "480": "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/b",
}


def _make_hook(on_progress: ProgressCb):
    def hook(d):
        if not on_progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            on_progress(min(0.99, got / total) if total else 0.0, "Завантаження…")
        elif d.get("status") == "finished":
            on_progress(1.0, "Оброблення…")
    return hook


def _final_path(ydl, info: dict, audio_only: bool) -> str:
    rds = info.get("requested_downloads") or []
    if rds and rds[0].get("filepath"):
        p = rds[0]["filepath"]
    else:
        p = ydl.prepare_filename(info)
    if audio_only:
        mp3 = str(Path(p).with_suffix(".mp3"))
        return mp3 if Path(mp3).exists() else p
    if not Path(p).exists():
        mp4 = str(Path(p).with_suffix(".mp4"))
        if Path(mp4).exists():
            return mp4
    return p


def save_media(url: str, dest_dir: str, quality: str = "best", audio_only: bool = False,
               on_progress: ProgressCb = None, on_log: LogCb = None) -> str:
    """Download-only: save the video (or mp3 audio) into ``dest_dir`` at the chosen
    quality. Returns the saved file path."""
    import yt_dlp

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(dest / "%(title).180B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(settings.BIN_DIR),
        "progress_hooks": [_make_hook(on_progress)],
    }
    if audio_only:
        opts["format"] = "ba/b"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192",
        }]
    else:
        opts["format"] = QUALITY_FORMAT.get(quality, QUALITY_FORMAT["best"])
        opts["merge_output_format"] = "mp4"

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = _final_path(ydl, info, audio_only)
    if on_log:
        on_log(f"Збережено: {path}")
    return path


def download(url: str, on_progress: ProgressCb = None, on_log: LogCb = None) -> FetchResult:
    """Download best mp4 into work/downloads and return its path + metadata."""
    import yt_dlp

    outdir = Path(settings.WORK_DIR) / "downloads"
    outdir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if not on_progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            on_progress(min(0.99, got / total) if total else 0.0, "Завантаження відео…")
        elif d.get("status") == "finished":
            on_progress(1.0, "Оброблення…")

    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(settings.BIN_DIR),  # use our bundled ffmpeg to merge
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        if not Path(path).exists():
            alt = Path(path).with_suffix(".mp4")
            if alt.exists():
                path = str(alt)
    if on_log:
        on_log(f"Завантажено: {path}")
    return _result_from_info(info, path=str(path))
