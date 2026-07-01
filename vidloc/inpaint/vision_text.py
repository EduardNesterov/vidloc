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


def glyph_mask(box_rgb: np.ndarray, se_frac: float = 0.5) -> np.ndarray:
    """Segment glyph *strokes* (filled, not just outlined) inside a detected text box,
    so we erase the letters but keep the background between them and inside the holes of
    e/a/o etc.

    Morphological top-hat / black-hat: an opening/closing with a structuring element
    wider than the stroke removes the text, so ``g - opening`` (bright text) and
    ``closing - g`` (dark text) return the strokes *solidly* — unlike an edge/high-pass
    response, which only fires at stroke borders and leaves the interior (and the text)
    visible. Polarity-agnostic (max of both). Returns a uint8 {0,1} mask, box-sized."""
    h, w = box_rgb.shape[:2]
    if h < 4 or w < 4:
        return np.ones((h, w), np.uint8)     # too small to segment -> treat all as text
    g = cv2.cvtColor(box_rgb, cv2.COLOR_RGB2GRAY)
    k = max(3, int(round(h * se_frac)) | 1)  # odd; wider than a stroke, so strokes fill
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se))
    _, m = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return (m > 0).astype(np.uint8)


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


def detect_text_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect text in an RGB uint8 frame. Returns pixel boxes ``(x, y, w, h)`` with a
    top-left origin (Vision reports normalized, bottom-left; we convert)."""
    import Vision
    h, w = rgb.shape[:2]
    img, _data = _cgimage_from_rgb(rgb)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(1)          # 1 = accurate
    req.setUsesLanguageCorrection_(False)
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
