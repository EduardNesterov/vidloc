"""Per-video workspace: a persistent on-disk cache so a repeat run of the same
video (and recognition model) skips re-downloading and re-transcribing.

Layout::

    workspace/<slug>_<YYYY-MM-DD>/
        Source/                     # shared across recognition models
            <video>                 # the file the pipeline actually reads
            audio16k.wav            # extracted 16 kHz mono ASR input (cache)
            meta.json               # url / title / description / probe info
        Text Analysis/
            <model>/                # one folder per recognition setting
                transcript.json     # {model, language, segments, text, ...}
                transcript.txt      # human-readable transcript

The ``Source/`` folder is keyed to the *video* (downloaded once); each
recognition model gets its own subfolder under ``Text Analysis/`` so switching
models keeps every transcript and never re-downloads the video.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import settings

SCHEMA = 1
_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv"}
_SLUG_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЇїІіЄєҐґ._-]+")


def slugify(name: str, *, limit: int = 80) -> str:
    """Filesystem-safe folder fragment (keeps Latin/Cyrillic word chars)."""
    cleaned = _SLUG_RE.sub("_", (name or "").strip()).strip("._")
    return (cleaned or "video")[:limit]


@dataclass
class Workspace:
    """Handle to one ``workspace/<slug>_<date>/`` directory."""

    root: Path
    source: Path
    analysis: Path

    @classmethod
    def at(cls, root) -> "Workspace":
        """Attach to an existing workspace directory by its root path."""
        root = Path(root)
        return cls(root=root, source=root / "Source",
                   analysis=root / "Text Analysis")

    # -- source (shared per video) ------------------------------------- #
    @property
    def meta_file(self) -> Path:
        return self.source / "meta.json"

    @property
    def audio_file(self) -> Path:
        return self.source / "audio16k.wav"

    def source_video(self) -> Optional[Path]:
        """First video file already placed in ``Source/`` (if any)."""
        if not self.source.is_dir():
            return None
        for p in sorted(self.source.iterdir()):
            if p.is_file() and p.suffix.lower() in _VIDEO_EXT:
                return p
        return None

    def write_meta(self, meta: Dict) -> None:
        self.source.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    def read_meta(self) -> Dict:
        if self.meta_file.is_file():
            try:
                return json.loads(self.meta_file.read_text("utf-8"))
            except Exception:  # noqa: BLE001 — corrupt cache shouldn't crash
                return {}
        return {}

    # -- per-model analysis -------------------------------------------- #
    def model_dir(self, model: str) -> Path:
        return self.analysis / slugify(model)

    def has_model(self, model: str) -> bool:
        """True if this recognition setting already has a folder."""
        return self.model_dir(model).is_dir()

    def transcript_file(self, model: str) -> Path:
        return self.model_dir(model) / "transcript.json"

    def create_model_dir(self, model: str) -> Path:
        d = self.model_dir(model)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_transcript(self, model: str) -> Optional[Dict]:
        """Cached ASR result for this model, or None if absent/unreadable."""
        f = self.transcript_file(model)
        if not f.is_file():
            return None
        try:
            data = json.loads(f.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return data if data.get("segments") else None

    def save_transcript(self, model: str, *, language: str,
                        segments: List[Dict], model_repo: str = "",
                        text: str = "") -> Path:
        d = self.create_model_dir(model)
        payload = {
            "schema": SCHEMA,
            "model": model,
            "model_repo": model_repo,
            "language": language,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "segments": segments,
            "text": text,
        }
        (d / "transcript.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        (d / "transcript.txt").write_text(
            "\n".join(s.get("text", "") for s in segments), "utf-8")
        return d / "transcript.json"


def resolve(video_name: str, *, create: bool = True) -> Workspace:
    """Return the workspace for ``video_name``.

    Reuses an existing ``<slug>_*`` directory for the same video (any date) so
    the source/transcripts persist across days; otherwise names a fresh one
    ``<slug>_<today>``.
    """
    base = Path(settings.WORKSPACE_DIR)
    slug = slugify(Path(video_name).stem)
    existing = sorted(base.glob(f"{slug}_*")) if base.is_dir() else []
    root = existing[0] if existing else base / f"{slug}_{_dt.date.today().isoformat()}"
    ws = Workspace(root=root, source=root / "Source",
                   analysis=root / "Text Analysis")
    if create:
        ws.source.mkdir(parents=True, exist_ok=True)
        ws.analysis.mkdir(parents=True, exist_ok=True)
    return ws
