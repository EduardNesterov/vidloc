"""Write styled subtitles as ASS (Advanced SubStation Alpha) for burn-in.

ASS gives full control over font, size, outline, and on-screen position — which
SRT/soft subtitles can't. Used for the "TikTok/Reels" look: big bold centered
captions in a nice font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..models import Caption, SubtitlePosition, SubtitleStyle

# A caption box is taller than its text (padding / room for a 2nd line), so the font
# is a fraction of the box height. 0.5 makes the default box (~0.14 h) render at the
# global 7 % default; the editor preview uses the same factor so WYSIWYG holds.
BOX_FONT_FACTOR = 0.5


def caption_font_px(cap: Caption, video_h: int, default_style: SubtitleStyle) -> int:
    """Font size in video pixels for a caption: explicit % override, else box height."""
    if cap.font_size_pct is not None:
        return max(12, round(video_h * cap.font_size_pct / 100.0))
    return max(12, round(cap.rect[3] * video_h * BOX_FONT_FACTOR))


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


def _ass_c(hex_color: str) -> str:
    """#RRGGBB -> ASS inline colour override token &Hbbggrr& (no alpha)."""
    h = (hex_color or "").lstrip("#")
    if len(h) == 6:
        try:
            int(h, 16)
            return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()
        except ValueError:
            pass
    return "&HFFFFFF&"


def _text_ml(s: str) -> str:
    """Escape caption text but keep user line breaks as ASS hard breaks (\\N)."""
    s = (s or "").replace("\\", "").replace("{", "(").replace("}", ")")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\\N".join(line.strip() for line in s.split("\n")).strip()


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


def write_ass_captions(captions: List[Caption], path: str, default_style: SubtitleStyle,
                       video_w: int, video_h: int) -> str:
    """Render editor-authored captions to a burnable ASS file.

    Each caption is one ``Dialogue`` line positioned by its ``rect`` (via ``\\pos`` at
    the box centre, anchored ``\\an5``) with per-caption inline style overrides layered
    over ``default_style``. The box height sets the font size unless the caption
    overrides ``font_size_pct``, so stretching a box vertically scales its text.
    """
    w = int(video_w) or 1280
    h = int(video_h) or 720
    base_font = default_style.font
    base_size = max(20, round(h * default_style.font_size_pct / 100.0))
    def_primary = _ass_color(default_style.primary_color)
    def_outline = _ass_color(default_style.outline_color)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{base_font},{base_size},{def_primary},&H000000FF,{def_outline},"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(2, round(base_size * 0.11))},{max(0, round(base_size * 0.04))},5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for c in captions:
        txt = _text_ml(c.text)
        if not txt:
            continue
        upper = default_style.uppercase if c.uppercase is None else c.uppercase
        if upper:
            txt = txt.upper()

        x, y, rw, rh = c.rect
        cx = int(round((x + rw / 2.0) * w))
        cy = int(round((y + rh / 2.0) * h))
        fs = caption_font_px(c, h, default_style)
        bord = c.outline_w if c.outline_w is not None else max(2, round(fs * 0.11))
        an = c.align or 5

        ov = [f"\\an{an}", f"\\pos({cx},{cy})", f"\\fs{fs}", f"\\bord{bord}"]
        ov.append(f"\\fn{c.font or base_font}")
        if c.bold is not None:
            ov.append("\\b1" if c.bold else "\\b0")
        if c.primary_color is not None:
            ov.append(f"\\c{_ass_c(c.primary_color)}")
        if c.outline_color is not None:
            ov.append(f"\\3c{_ass_c(c.outline_color)}")
        tag = "{" + "".join(ov) + "}"
        lines.append(
            f"Dialogue: 0,{_ts(c.start)},{_ts(c.end)},Default,,0,0,0,,{tag}{txt}")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
