"""Text translation into Ukrainian with NLLB-200 (runs on MPS / CPU).

Models are cached in-process so repeated jobs in one GUI session don't reload
weights. Translation is batched for throughput.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .. import settings
from ..models import NLLB_LANG, TARGET_LANG

ProgressCb = Optional[Callable[[float], None]]

_CACHE: dict = {}


def _load(model_repo: str):
    cached = _CACHE.get(model_repo)
    if cached is not None:
        return cached
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = settings.torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_repo).to(device).eval()
    _CACHE[model_repo] = (tokenizer, model, device)
    return _CACHE[model_repo]


def translate_texts(texts: List[str], src_code: str, tgt_code: str, model_repo: str,
                    beam_size: int = 5, batch_size: int = 8,
                    on_progress: ProgressCb = None) -> List[str]:
    import torch

    tokenizer, model, device = _load(model_repo)
    tokenizer.src_lang = src_code
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    out: List[str] = []
    total = max(1, len(texts))
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else "…" for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_length=512,
                num_beams=max(1, beam_size),
                no_repeat_ngram_size=3,
            )
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        if on_progress:
            on_progress(min(1.0, (i + batch_size) / total))
    return out


def translate_segments(segments: List[Dict], src_lang: str, model_repo: str,
                       tgt_lang: str = TARGET_LANG, beam_size: int = 5,
                       on_progress: ProgressCb = None) -> List[Dict]:
    """Translate the ``text`` of each segment, adding a ``text_tr`` field."""
    src_code = NLLB_LANG.get(src_lang, "eng_Latn")
    tgt_code = NLLB_LANG.get(tgt_lang, "ukr_Cyrl")
    texts = [s.get("text", "").strip() for s in segments]
    translated = translate_texts(texts, src_code, tgt_code, model_repo,
                                 beam_size=beam_size, on_progress=on_progress)
    result: List[Dict] = []
    for seg, tr in zip(segments, translated):
        new = dict(seg)
        new["text_tr"] = tr.strip()
        result.append(new)
    return result
