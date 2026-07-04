#!/usr/bin/env python3
"""Provision the CRAFT text detector as ONNX for the inpaint worker.

CRAFT replaces Apple Vision for ``"text"`` regions (robust on stylized / low-contrast
/ Cyrillic captions). At runtime the worker needs only ``onnxruntime`` (in
``.venv-inpaint``) and ``models/craft/craft.onnx`` — no torch. This one-off script
downloads the CRAFT weights (via EasyOCR) and exports them to ONNX.

Run it in a throwaway venv that has torch + easyocr (they conflict with the LaMa
torch, so keep them out of ``.venv-inpaint``)::

    uv venv .venv-craft --python 3.11
    uv pip install --python .venv-craft/bin/python easyocr onnx
    .venv-craft/bin/python scripts/export_craft_onnx.py

The EasyOCR CRAFT weights download from the JAIDED servers on first use and are
cached; the export itself is offline after that. Output: ``models/craft/craft.onnx``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "craft" / "craft.onnx"


def main() -> int:
    try:
        import easyocr
        import torch
    except ImportError as exc:
        print(f"Need torch + easyocr in this venv: {exc}\n"
              "See the module docstring for the throwaway-venv setup.", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)   # downloads CRAFT weights
    craft = reader.detector.eval()

    class _TextScore(torch.nn.Module):
        """CRAFT returns (score[n,H/2,W/2,2], feature); we export only the score maps."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    dummy = torch.randn(1, 3, 480, 640)
    torch.onnx.export(
        _TextScore(craft), dummy, str(OUT), dynamo=False,
        input_names=["input"], output_names=["score"],
        dynamic_axes={"input": {0: "n", 2: "h", 3: "w"},
                      "score": {0: "n", 1: "h2", 2: "w2"}},
        opset_version=17, do_constant_folding=True)
    print(f"exported {OUT} ({OUT.stat().st_size // 1024 // 1024} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
