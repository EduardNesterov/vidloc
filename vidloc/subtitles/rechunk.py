"""Split translated segments into short display chunks (TikTok/Reels style).

Translation reorders words, so we don't have word-level timing for the target
text. Instead we split each segment's translated text into <= ``max_chars`` pieces
and hand out the segment's time slice proportionally to each piece's length.
This is display-only — dubbing still uses the full segments.
"""

from __future__ import annotations

import math
from typing import Dict, List


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split into ~equal pieces at word boundaries (balanced, so we don't leave a
    tiny one-word trailer like «вогонь.» on its own line)."""
    words = text.split()
    if not words:
        return []
    if len(text) <= max_chars:
        return [text]
    n = max(2, math.ceil(len(text) / max_chars))
    target = len(text) / n
    chunks: List[str] = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if cur and len(cur) >= target and len(chunks) < n - 1:
            chunks.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


def rechunk(segments: List[Dict], max_chars: int = 28, field: str = "text_tr") -> List[Dict]:
    """Return new, smaller segments. ``max_chars <= 0`` returns the input unchanged."""
    if max_chars <= 0:
        return segments
    out: List[Dict] = []
    for seg in segments:
        text = (seg.get(field) or seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        dur = max(0.0, end - start)
        pieces = _split_text(text, max_chars)
        if len(pieces) <= 1:
            out.append({**seg, "start": start, "end": end, field: text})
            continue
        total = sum(len(p) for p in pieces) or 1
        t = start
        for p in pieces:
            slice_dur = dur * (len(p) / total)
            out.append({"start": t, "end": t + slice_dur, field: p, "text": p})
            t += slice_dur
    return out
