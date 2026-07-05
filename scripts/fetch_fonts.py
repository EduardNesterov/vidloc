#!/usr/bin/env python3
"""Fetch a curated, Cyrillic-first font library into ``assets/fonts/``.

All families are OFL (SIL Open Font License) from the google/fonts repo. Every
output file is normalized to a **unique, self-consistent family name** (name ID
1 == full name, subfamily "Regular") so that Qt (editor preview) and libass
(burn-in) resolve the exact same file — no ambiguous multi-weight family
matching. Variable families are instanced to a fixed weight first (full glyph
coverage, incl. Cyrillic, is preserved).

Run:  uv run --with fonttools python scripts/fetch_fonts.py
"""

from __future__ import annotations

import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

DEST = Path(__file__).resolve().parent.parent / "assets" / "fonts"
RAW = "https://github.com/google/fonts/raw/main/ofl/{slug}/{file}"

# kind, slug, remote_file, weight, family_label, out_name
#   dl    = download a static weight verbatim, then normalize the name
#   var   = download a variable font, instance to `weight`, then normalize
#   local = normalize an already-present file in place (no download)
SPECS = [
    ("dl", "firasans", "FiraSans-Bold.ttf", None, "Fira Sans", "FiraSans-Bold.ttf"),
    ("dl", "firasans", "FiraSans-ExtraBold.ttf", None, "Fira Sans ExtraBold",
     "FiraSans-ExtraBold.ttf"),
    ("dl", "firasans", "FiraSans-Black.ttf", None, "Fira Sans Black", "FiraSans-Black.ttf"),
    ("dl", "ptsans", "PT_Sans-Web-Bold.ttf", None, "PT Sans", "PTSans-Bold.ttf"),
    ("dl", "ptserif", "PT_Serif-Web-Bold.ttf", None, "PT Serif", "PTSerif-Bold.ttf"),
    ("dl", "russoone", "RussoOne-Regular.ttf", None, "Russo One", "RussoOne.ttf"),
    ("var", "roboto", "Roboto[wdth,wght].ttf", 700, "Roboto", "Roboto-Bold.ttf"),
    ("var", "roboto", "Roboto[wdth,wght].ttf", 900, "Roboto Black", "Roboto-Black.ttf"),
    ("var", "oswald", "Oswald[wght].ttf", 600, "Oswald", "Oswald-SemiBold.ttf"),
    ("var", "rubik", "Rubik[wght].ttf", 700, "Rubik", "Rubik-Bold.ttf"),
    ("var", "rubik", "Rubik[wght].ttf", 900, "Rubik Black", "Rubik-Black.ttf"),
    ("var", "opensans", "OpenSans[wdth,wght].ttf", 700, "Open Sans", "OpenSans-Bold.ttf"),
    ("var", "opensans", "OpenSans[wdth,wght].ttf", 800, "Open Sans ExtraBold",
     "OpenSans-ExtraBold.ttf"),
    ("var", "golostext", "GolosText[wght].ttf", 600, "Golos Text", "GolosText-SemiBold.ttf"),
    ("var", "golostext", "GolosText[wght].ttf", 800, "Golos Text ExtraBold",
     "GolosText-ExtraBold.ttf"),
    ("var", "manrope", "Manrope[wght].ttf", 700, "Manrope", "Manrope-Bold.ttf"),
    ("var", "manrope", "Manrope[wght].ttf", 800, "Manrope ExtraBold", "Manrope-ExtraBold.ttf"),
    ("var", "inter", "Inter[opsz,wght].ttf", 700, "Inter", "Inter-Bold.ttf"),
    ("var", "inter", "Inter[opsz,wght].ttf", 900, "Inter Black", "Inter-Black.ttf"),
    ("var", "nunito", "Nunito[wght].ttf", 800, "Nunito ExtraBold", "Nunito-ExtraBold.ttf"),
    ("var", "unbounded", "Unbounded[wght].ttf", 700, "Unbounded", "Unbounded-Bold.ttf"),
    ("var", "unbounded", "Unbounded[wght].ttf", 900, "Unbounded Black", "Unbounded-Black.ttf"),
    ("local", None, None, None, "Montserrat", "Montserrat-Bold.ttf"),
    ("local", None, None, None, "Montserrat ExtraBold", "Montserrat-ExtraBold.ttf"),
    ("local", None, None, None, "Montserrat Black", "Montserrat-Black.ttf"),
]
# Thin weights previously fetched that we no longer ship (subtitles want bold+).
STALE = ["FiraSans-Regular.ttf", "PTSans-Regular.ttf", "PTSerif-Regular.ttf"]

_cache: dict = {}


def _dl(slug: str, file: str) -> bytes:
    key = (slug, file)
    if key not in _cache:
        url = RAW.format(slug=slug, file=urllib.parse.quote(file))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        _cache[key] = urllib.request.urlopen(req, timeout=90).read()
    return _cache[key]


def _set_family(font: TTFont, label: str) -> None:
    """Make the file self-describe as a single family ``label`` (Regular) across all
    name records, so Qt and libass resolve it identically."""
    name = font["name"]
    ps = label.replace(" ", "")
    for pid, eid, lid in ((3, 1, 0x409), (1, 0, 0)):
        name.setName(label, 1, pid, eid, lid)
        name.setName("Regular", 2, pid, eid, lid)
        name.setName(label, 4, pid, eid, lid)
        name.setName(ps, 6, pid, eid, lid)
        name.setName(label, 16, pid, eid, lid)
        name.setName("Regular", 17, pid, eid, lid)


def _process(spec) -> None:
    kind, slug, file, weight, label, out = spec
    if kind == "local":
        font = TTFont(str(DEST / out))
    else:
        font = TTFont(io.BytesIO(_dl(slug, file)))
        if kind == "var":
            axes = {a.axisTag: a.defaultValue for a in font["fvar"].axes}
            axes["wght"] = weight
            instantiateVariableFont(font, axes, inplace=True)
    _set_family(font, label)
    font.save(str(DEST / out))


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for spec in SPECS:
        out, label = spec[5], spec[4]
        try:
            _process(spec)
            print(f"  ✓ {out:28} ({label})")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {out}: {e}")
    for s in STALE:
        p = DEST / s
        if p.exists():
            p.unlink()
            print(f"  – removed stale {s}")
    print(f"\nDone: {ok}/{len(SPECS)} fonts in {DEST}")
    return 0 if ok == len(SPECS) else 1


if __name__ == "__main__":
    sys.exit(main())
