"""Write styled subtitles as ASS (Advanced SubStation Alpha) for burn-in.

ASS gives full control over font, size, outline, and on-screen position — which
SRT/soft subtitles can't. Used for the "TikTok/Reels" look: big bold centered
captions in a nice font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..models import SubtitlePosition, SubtitleStyle


def _ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _text(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
            .replace("\n", " ").strip())


def _ass_color(hex_color: str) -> str:
    """#RRGGBB -> ASS &HAABBGGRR (opaque, alpha 00)."""
    h = (hex_color or "").lstrip("#")
    if len(h) == 6:
        try:
            int(h, 16)
            return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()
        except ValueError:
            pass
    return "&H00FFFFFF"


def write_ass(chunks: List[Dict], path: str, style: SubtitleStyle,
              video_w: int, video_h: int, field: str = "text_tr") -> str:
    w = int(video_w) or 1280
    h = int(video_h) or 720
    fontsize = max(20, round(h * style.font_size_pct / 100.0))
    outline = max(2, round(fontsize * 0.11))
    shadow = max(0, round(fontsize * 0.04))
    align = 5 if style.position == SubtitlePosition.CENTER else 2  # 5=mid, 2=bottom
    marginv = 0 if align == 5 else round(h * 0.08)
    marginlr = round(w * 0.06)
    primary_c = _ass_color(style.primary_color)
    outline_c = _ass_color(style.outline_color)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{fontsize},{primary_c},&H000000FF,{outline_c},"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{align},"
        f"{marginlr},{marginlr},{marginv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for c in chunks:
        txt = _text(c.get(field) or c.get("text") or "")
        if not txt:
            continue
        if style.uppercase:
            txt = txt.upper()
        lines.append(f"Dialogue: 0,{_ts(c['start'])},{_ts(c['end'])},Default,,0,0,0,,{txt}")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
