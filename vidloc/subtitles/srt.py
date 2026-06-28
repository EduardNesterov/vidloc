"""Write translated segments out as SubRip (.srt) subtitle files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _ts(t: float, sep: str = ",") -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(segments: List[Dict], path: str, field: str = "text_tr") -> str:
    blocks: List[str] = []
    n = 0
    for seg in segments:
        text = (seg.get(field) or seg.get("text") or "").strip()
        if not text:
            continue
        n += 1
        blocks.append(str(n))
        blocks.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        blocks.append(text)
        blocks.append("")
    Path(path).write_text("\n".join(blocks), encoding="utf-8")
    return str(path)
