"""Sentence-level English -> Ukrainian translation with Dragoman (MLX).

Dragoman is a Mistral-7B LoRA merged and quantized for Apple Silicon as
``lang-uk/dragoman-4bit``. It is specialized for EN->UK sentence translation;
other language pairs should be routed to Aya/NLLB by the caller.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from ..models import DRAGOMAN_TRANSLATE_MODEL

ProgressCb = Optional[Callable[[float], None]]

_CACHE: dict = {}
_INST_RE = re.compile(r"\[/?INST\]", re.IGNORECASE)


def _load(repo: str):
    if repo not in _CACHE:
        from mlx_lm import load
        _CACHE[repo] = load(repo)
    return _CACHE[repo]


def _gen(model, tok, prompt: str) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    return generate(model, tok, prompt, max_tokens=160,
                    sampler=make_sampler(temp=0.0), verbose=False)


def _clean(text: str, src: str) -> str:
    text = _INST_RE.sub("", text).strip()
    if text.startswith(src):
        text = text[len(src):].strip()
    text = re.sub(r"^\s*[:\-–—]\s*", "", text)
    return text.strip().strip('"').strip("«»").strip()


def translate_segments(segments: List[Dict], src_lang: str,
                       repo: str = DRAGOMAN_TRANSLATE_MODEL,
                       on_progress: ProgressCb = None) -> List[Dict]:
    if src_lang != "en":
        raise ValueError("Dragoman supports only EN->UK")

    model, tok = _load(repo)
    result: List[Dict] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        src = (seg.get("text") or "").strip()
        if src:
            prompt = f"[INST] {src} [/INST]"
            tr = _clean(_gen(model, tok, prompt), src)
        else:
            tr = ""
        new = dict(seg)
        new["text_tr"] = tr or src
        result.append(new)
        if on_progress:
            on_progress(min(1.0, (i + 1) / max(1, n)))
    return result
