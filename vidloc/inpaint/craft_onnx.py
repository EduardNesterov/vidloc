"""CRAFT text detector via ONNX Runtime + CoreML — offline, no torch.

Replaces Apple Vision for ``"text"`` regions: CRAFT (Character Region Awareness)
is far more robust on stylized / low-contrast / Cyrillic / karaoke captions that
Vision's OCR misses entirely. Variant A: we threshold CRAFT's character-region
**heatmap** directly into a mask (no box extraction / recognition needed).

Standalone by design (no ``vidloc`` imports), so ``lama_worker.py`` (a different
venv) imports it as a sibling. Needs ``onnxruntime`` in ``.venv-inpaint`` and the
exported model (``models/craft/craft.onnx``); the worker passes the model path.
Re-create the model with ``scripts/export_craft_onnx.py`` if it is missing.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0     # ImageNet, CRAFT preproc
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
_SESSION = None
_SESSION_PATH = None


def available(model_path: str) -> bool:
    """True if onnxruntime is importable and the CRAFT model exists on disk."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return bool(model_path) and os.path.exists(model_path)


def load(model_path: str):
    """Lazily build (and cache) the ORT session, preferring the CoreML EP."""
    global _SESSION, _SESSION_PATH
    if _SESSION is not None and _SESSION_PATH == model_path:
        return _SESSION
    import onnxruntime as ort
    avail = ort.get_available_providers()
    provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]
    _SESSION = ort.InferenceSession(model_path, providers=provs or None)
    _SESSION_PATH = model_path
    return _SESSION


def heatmap(box_rgb: np.ndarray, model_path: str, *, mag: float = 1.5,
            max_side: int = 1280) -> np.ndarray:
    """CRAFT character-region score for ``box_rgb`` (RGB uint8), resized back to the
    box. Small crops are upscaled by ``mag`` (helps small text); large ones are capped
    at ``max_side``. CRAFT is fully convolutional, so any 32-multiple size works."""
    sess = load(model_path)
    h, w = box_rgb.shape[:2]
    scale = min(max_side / max(h, w), mag)
    dh = max(32, int(round(h * scale / 32)) * 32)
    dw = max(32, int(round(w * scale / 32)) * 32)
    img = cv2.resize(box_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = (img - _MEAN) / _STD
    x = img.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {"input": x})[0]          # [1, dh/2, dw/2, 2] (text, link)
    return cv2.resize(out[0, :, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)


def text_mask(box_rgb: np.ndarray, model_path: str, *, thresh: float = 0.4,
              close_frac: float = 0.06, grow_frac: float = 0.05) -> np.ndarray:
    """Threshold CRAFT's heatmap into a text mask, then close per-character blobs into
    solid text and grow the edge (so the whole glyph is covered). Returns uint8 {0,1}
    box-sized; empty if no text region fires."""
    h, w = box_rgb.shape[:2]
    if h < 8 or w < 8:
        return np.zeros((h, w), np.uint8)
    m = (heatmap(box_rgb, model_path) > thresh).astype(np.uint8) * 255
    if not m.any():
        return np.zeros((h, w), np.uint8)
    if close_frac > 0:
        ck = max(3, int(round(h * close_frac)) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    if grow_frac > 0:
        g = max(1, int(round(h * grow_frac)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    return (m > 0).astype(np.uint8)
