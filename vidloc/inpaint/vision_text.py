"""Text detection via Apple Vision (runs on the Neural Engine), fully offline.

Used to build a *tight* mask of the actual text inside a user-marked region, so LaMa
erases only the glyphs and not the whole rectangle (which destroys the background).

Standalone by design: no ``vidloc`` imports, so ``lama_worker.py`` (a different venv)
can import it as a sibling module. Requires ``pyobjc`` (Vision + Quartz) in that venv;
import is lazy and ``available()`` lets the caller degrade gracefully when it is not.
"""

from __future__ import annotations

import cv2
import numpy as np


def _fill_small_holes(m: np.ndarray, max_frac: float = 0.02) -> np.ndarray:
    """Fill enclosed background pockets smaller than ``max_frac`` of the box (bite-marks
    at stroke joins, letter counters) so glyphs erase solidly — but keep the large gaps
    *between* words (those touch the box border, so they are never filled)."""
    inv = (cv2.bitwise_not(m) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    area = m.shape[0] * m.shape[1]
    out = m.copy()
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        touches = x == 0 or y == 0 or x + bw >= m.shape[1] or y + bh >= m.shape[0]
        if not touches and a < max_frac * area:
            out[lab == i] = 255
    return out


def glyph_mask(box_rgb: np.ndarray, se_frac: float = 0.5, grow_frac: float = 0.05,
               color: bool = True) -> np.ndarray:
    """Segment glyph *strokes* (filled, not just outlined) inside a detected text box,
    so we erase the letters but keep the background between them and inside the holes of
    e/a/o etc.

    Morphological top-hat / black-hat: an opening/closing with a structuring element
    wider than the stroke removes the text, so ``g - opening`` (bright text) and
    ``closing - g`` (dark text) return the strokes *solidly* — unlike an edge/high-pass
    response, which only fires at stroke borders and leaves the interior (and the text)
    visible. Polarity-agnostic (max of both).

    ``color`` runs the response over L*a*b* (not just luma) so text that differs from the
    background in *hue* but not brightness is still caught (multi-colored captions).
    ``grow_frac`` dilates the mask by a fraction of the text height so the glyph *edge*
    and its outline are covered too — otherwise a thin rim of the stroke survives inpaint
    and leaves a ghost outline (the classic failure on bold, thickly-outlined text).
    Returns a uint8 {0,1} mask, box-sized."""
    h, w = box_rgb.shape[:2]
    if h < 4 or w < 4:
        return np.ones((h, w), np.uint8)     # too small to segment -> treat all as text
    if color:
        lab = cv2.cvtColor(box_rgb, cv2.COLOR_RGB2LAB)
        chans = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]
    else:
        chans = [cv2.cvtColor(box_rgb, cv2.COLOR_RGB2GRAY)]
    k = max(3, int(round(h * se_frac)) | 1)  # odd; wider than a stroke, so strokes fill
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = np.zeros((h, w), np.uint8)
    for ch in chans:
        resp = cv2.max(resp, cv2.max(cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, se),
                                     cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, se)))
    _, m = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = _fill_small_holes(m)
    if grow_frac > 0:                        # cover the stroke edge + outline (no ghost rim)
        g = max(1, int(round(h * grow_frac)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    return (m > 0).astype(np.uint8)


def caption_mask(box_rgb: np.ndarray, se_frac: float = 0.30, close_frac: float = 0.08,
                 grow_frac: float = 0.06, color: bool = True, clahe: bool = True,
                 contrast_k: float = 0.28, contrast_win: float = 0.5,
                 min_h_frac: float = 0.12, max_h_frac: float = 0.95,
                 max_w_frac: float = 0.98) -> np.ndarray:
    """OCR-free caption mask for a user-marked subtitle band — covers the *whole text*
    (word blobs), no text recognition needed.

    For stylized / low-contrast / karaoke captions Apple Vision often detects nothing
    (see ``detect_text_boxes``), so we segment by image structure instead: bright and/or
    coloured strokes that are locally sharp, merged into word blobs, then filtered by
    geometry to drop background. Since the caller marks the band tightly, the goal is a
    solid cover of the text (LaMa still only inpaints the text-shaped area, not a
    rectangle, so it stays clean). Returns a uint8 {0,1} mask, box-sized.

    ``se_frac`` sizes the top-hat kernel (fraction of box height — wider than a stroke).
    ``close_frac`` merges strokes into word blobs. ``grow_frac`` dilates to cover the
    stroke edge / outline. ``color`` also fires on the red "active" karaoke word. ``clahe``
    equalizes local contrast first, which pulls faint low-contrast text (white on a washed
    background) above the threshold *and* suppresses smooth background gradients — measured
    to both raise text coverage and cut false positives.

    ``contrast_k`` adds a *local-adaptive* ("contrast points") channel: a pixel is text if
    it deviates from its local mean by more than ``contrast_k``·local_std. A global Otsu
    picks one threshold for the whole box, so a faint word next to bright text drops out;
    the local test catches it (measured to lift the faint "гони" frame ~3.5×). It is
    unioned with the morphological response, so frames Otsu already handled are unaffected.
    ``0`` disables it. The ``*_frac`` geometry bounds keep letter-sized components."""
    h, w = box_rgb.shape[:2]
    if h < 8 or w < 8:
        return np.ones((h, w), np.uint8)
    lab = cv2.cvtColor(box_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    if clahe:
        L = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(L)
    k = max(3, int(round(h * se_frac)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.max(cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se),    # bright (white) strokes
                   cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se))  # dark strokes on light bg
    if color:
        redness = np.clip(lab[:, :, 1].astype(np.int16) - 128, 0, 127).astype(np.uint8) * 2
        resp = cv2.max(resp, cv2.morphologyEx(redness, cv2.MORPH_TOPHAT, se))  # red active word
    _, m = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def _blobs(binm):
        """Close strokes into word blobs, fill enclosed holes, keep letter/word-sized
        components, drop thin/huge background, then grow to cover stroke edges."""
        ck = max(3, int(round(h * close_frac)) | 1)
        bm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
        bm = _fill_small_holes(bm)
        nn, l2, st, _ = cv2.connectedComponentsWithStats((bm > 0).astype(np.uint8), 8)
        out = np.zeros((h, w), np.uint8)
        for i in range(1, nn):
            _, _, cw, chh, area = st[i]
            if (min_h_frac * h <= chh <= max_h_frac * h and cw <= max_w_frac * w
                    and area >= 0.0006 * h * w):
                out[l2 == i] = 255
        if grow_frac > 0:
            g = max(1, int(round(h * grow_frac)))
            out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        return out

    # Finish EACH response channel through the geometry filter *separately*, then union —
    # unioning raw pixels first would let contrast-points noise bridge a text blob into the
    # background, so the merged component fails the size filter and the text is dropped.
    keep = _blobs(m)
    if contrast_k > 0:                       # local-adaptive "contrast points" (faint text)
        gl = lab[:, :, 0].astype(np.float32)
        cwin = max(3, int(round(h * contrast_win)) | 1)
        lm = cv2.boxFilter(gl, -1, (cwin, cwin))
        lstd = np.sqrt(np.maximum(cv2.boxFilter(gl * gl, -1, (cwin, cwin)) - lm * lm, 0))
        m_lc = ((np.abs(gl - lm) > contrast_k * lstd + 4) & (lstd > 3)).astype(np.uint8) * 255
        keep = cv2.max(keep, _blobs(m_lc))
    return (keep > 0).astype(np.uint8)


def available() -> bool:
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
        return True
    except Exception:
        return False


def _cgimage_from_rgb(rgb: np.ndarray):
    """Build a CGImage straight from an in-memory RGB frame (no temp file)."""
    import Quartz
    h, w = rgb.shape[:2]
    rgba = np.ascontiguousarray(np.dstack([rgb, np.full((h, w), 255, np.uint8)]))
    data = rgba.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    bitmap = Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrderDefault
    img = Quartz.CGImageCreate(w, h, 8, 32, w * 4, cs, bitmap, provider,
                               None, False, Quartz.kCGRenderingIntentDefault)
    return img, data   # keep ``data`` alive until the request has run


def detect_text_boxes(rgb: np.ndarray, languages: list[str] | None = None,
                      revision: int = 0, min_text_height: float = 0.0,
                      ) -> list[tuple[int, int, int, int]]:
    """Detect text in an RGB uint8 frame. Returns pixel boxes ``(x, y, w, h)`` with a
    top-left origin (Vision reports normalized, bottom-left; we convert).

    ``languages`` sets the scripts Vision looks for (default is English-only, so
    Cyrillic / accented text is under-detected — pass e.g. ``["uk-UA","ru-RU"]``).
    ``revision`` picks the recognizer model (3 = newest, best) when supported.
    ``min_text_height`` (fraction of frame height, 0..1) lets small text / digits
    through — 0 keeps Vision's default."""
    import Vision
    h, w = rgb.shape[:2]
    img, _data = _cgimage_from_rgb(rgb)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    if revision:
        try:
            if revision in list(Vision.VNRecognizeTextRequest.supportedRevisions()):
                req.setRevision_(revision)
        except Exception:  # noqa: BLE001 — revision is best-effort
            pass
    req.setRecognitionLevel_(1)          # 1 = accurate
    req.setUsesLanguageCorrection_(False)
    if languages:
        req.setRecognitionLanguages_(languages)
    if min_text_height > 0:
        req.setMinimumTextHeight_(float(min_text_height))
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, _err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    boxes = []
    for o in (req.results() or []):
        bb = o.boundingBox()
        px = int(round(bb.origin.x * w))
        py = int(round((1.0 - bb.origin.y - bb.size.height) * h))
        pw = int(round(bb.size.width * w))
        ph = int(round(bb.size.height * h))
        boxes.append((px, py, pw, ph))
    return boxes


if __name__ == "__main__":   # tiny self-test: python vision_text.py frame.png
    import sys

    import cv2
    bgr = cv2.imread(sys.argv[1])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print("Vision available:", available())
    for b in detect_text_boxes(rgb):
        print(b)
