"""Register the bundled subtitle fonts with Qt and list their families.

The fonts in ``assets/fonts/`` were only ever handed to libass (burn-in) via a
``fontsdir``; Qt never knew them, so the editor preview silently fell back to a
default face and picking a font did nothing. Registering them with
``QFontDatabase`` fixes the preview, and the family names (name ID 1) double as the
strings written into the ASS ``Fontname`` / ``\\fn`` — libass matches the same table,
so what you preview is what gets burned.
"""

from __future__ import annotations

import struct
from typing import List, Optional

from .. import settings

_loaded: Optional[List[str]] = None


def _font_files():
    d = settings.FONTS_DIR
    if not d.is_dir():
        return []
    return sorted([*d.glob("*.ttf"), *d.glob("*.otf"), *d.glob("*.ttc")])


def _family_from_name_table(path: str) -> Optional[str]:
    """Read the font family (name ID 1) straight from the TTF ``name`` table — the
    same name libass resolves. Used as a fallback when Qt can't report families
    (e.g. the offscreen platform)."""
    try:
        with open(path, "rb") as fh:
            d = fh.read()
        num = struct.unpack(">H", d[4:6])[0]
        off, name_off = 12, None
        for _ in range(num):
            if d[off:off + 4] == b"name":
                name_off = struct.unpack(">I", d[off + 8:off + 12])[0]
            off += 16
        if name_off is None:
            return None
        count, str_off = struct.unpack(">HH", d[name_off + 2:name_off + 6])
        found = None
        for i in range(count):
            pid, eid, lid, nid, ln, o = struct.unpack(
                ">HHHHHH", d[name_off + 6 + i * 12:name_off + 6 + i * 12 + 12])
            if nid != 1:
                continue
            raw = d[name_off + str_off + o:name_off + str_off + o + ln]
            try:
                txt = raw.decode("utf-16-be") if pid in (0, 3) else raw.decode("latin-1")
            except UnicodeDecodeError:
                continue
            if txt:
                found = txt
                if pid == 3:      # prefer the Windows record
                    break
        return found
    except OSError:
        return None


def load() -> List[str]:
    """Register every bundled font with Qt (idempotent) and return the family names,
    Cyrillic-capable first then alphabetical. Must run after a QApplication exists."""
    global _loaded
    if _loaded is not None:
        return _loaded
    from PySide6.QtGui import QFontDatabase

    families = set()
    for f in _font_files():
        fid = QFontDatabase.addApplicationFont(str(f))
        names = QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
        if not names:
            nm = _family_from_name_table(str(f))
            names = [nm] if nm else []
        families.update(n for n in names if n)

    def is_cyrillic(fam: str) -> bool:
        try:
            ws = QFontDatabase.writingSystems(fam)
            return QFontDatabase.WritingSystem.Cyrillic in ws
        except Exception:  # noqa: BLE001 — best effort; bundled library is all-Cyrillic
            return True

    _loaded = sorted(families, key=lambda f: (not is_cyrillic(f), f.lower()))
    return _loaded
