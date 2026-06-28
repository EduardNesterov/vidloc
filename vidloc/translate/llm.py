"""Natural, idiomatic translation into Ukrainian with a local LLM (MLX).

Unlike NLLB (literal, sentence-level), an instruction-tuned LLM translates by
*meaning* — keeping slang, tone/register, and staying concise enough for
dubbing. Runs fully locally on Apple Silicon via mlx-lm (Metal).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from ..models import LLM_TRANSLATE_MODEL

ProgressCb = Optional[Callable[[float], None]]

_CACHE: dict = {}

_SRC_NAME = {
    "en": "English", "ru": "Russian", "it": "Italian", "uk": "Ukrainian",
    "pl": "Polish", "de": "German", "fr": "French", "es": "Spanish",
}

_SYSTEM = (
    "You are a Ukrainian video localizer who writes the way young Ukrainian YouTubers "
    "and streamers actually talk — informal, punchy, modern. Translate by MEANING, "
    "never word for word. For casual or slangy source, use natural Ukrainian slang and "
    "colloquialisms WHERE THEY FIT (e.g. «тачка», «капець», «зуб даю», «погнали», "
    "«кайф», «жесть», «крутяк», «чел», «підписка», «бабки») — but NEVER force a slang "
    "word where it does not belong; for formal source stay formal. Use correct, living "
    "Ukrainian — never russianisms or invented words. Use the provided VIDEO CONTEXT to "
    "pick the correct sense of ambiguous words (e.g. in a car video 'drift' = "
    "«дрифт»/«продрифтувати», NOT «пливти»; 'whip' = «тачка», not a whip). Keep the "
    "MEANING faithful and numbers, names, brands and amounts EXACT ('ten grand' = "
    "«десять штук», not «баксів»). Keep each line short and easy to say aloud, close in "
    "length to the source.\n"
    "Style examples:\n"
    "EN: This whip is so clean, no cap. -> UK: Ця тачка просто вогонь, зуб даю.\n"
    "EN: Drift through that wall to win. -> UK: Продрифтуй крізь ту стіну, щоб виграти.\n"
    "EN: Let's bounce, I'm dead tired. -> UK: Погнали звідси, я капець втомився.\n"
    "Output ONLY the Ukrainian translation, no notes or quotes."
)

_NUM_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$")


def _load(repo: str):
    if repo not in _CACHE:
        from mlx_lm import load
        _CACHE[repo] = load(repo)
    return _CACHE[repo]


def _gen(model, tok, messages: List[dict], max_tokens: int = 800, temp: float = 0.4) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temp), verbose=False)


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip("«»").strip()
    s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)  # drop echoed numbering
    return s.strip()


def _video_context(model, tok, texts: List[str]) -> str:
    """One short English sentence describing the video's topic/domain, used to
    disambiguate word senses (e.g. car 'drift' vs water 'drift')."""
    joined = " ".join(t for t in texts if t).strip()[:3000]
    if not joined:
        return ""
    msgs = [
        {"role": "system", "content":
            "Read this video transcript and reply with ONE short English sentence: what "
            "the video is about and its domain (e.g. car drifting, cooking, gaming), so a "
            "translator can pick correct word senses. One sentence, nothing else."},
        {"role": "user", "content": joined},
    ]
    try:
        return _gen(model, tok, msgs, max_tokens=80, temp=0.2).strip().replace("\n", " ")
    except Exception:  # noqa: BLE001
        return ""


def _translate_one(model, tok, text: str, src_name: str, context: str = "") -> str:
    ctx = f"Video context (for correct word senses): {context}\n" if context else ""
    msgs = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content":
            ctx + f"Translate this {src_name} line into natural spoken Ukrainian:\n{text}"},
    ]
    return _clean(_gen(model, tok, msgs, max_tokens=256))


def translate_segments(segments: List[Dict], src_lang: str,
                       repo: str = LLM_TRANSLATE_MODEL,
                       on_progress: ProgressCb = None, chunk: int = 10,
                       extra_context: str = "") -> List[Dict]:
    model, tok = _load(repo)
    src_name = _SRC_NAME.get(src_lang, "the source language")
    texts = [(s.get("text") or "").strip() for s in segments]
    out_tr: List[str] = [""] * len(texts)

    # Prefer caller-supplied context (e.g. video title/description from the URL);
    # otherwise infer the topic from the transcript itself.
    context = extra_context.strip() or _video_context(model, tok, texts)
    ctx_line = (f"Video context (use it to pick correct word senses): {context}\n\n"
                if context else "")

    n = len(texts)
    done = 0
    for i in range(0, n, chunk):
        block = texts[i:i + chunk]
        numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(block))
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": ctx_line + (
                f"Translate these {len(block)} {src_name} subtitle lines into natural, "
                f"idiomatic spoken Ukrainian. Return EXACTLY {len(block)} lines, each "
                f"starting with its number and a period, in the same order. Nothing "
                f"else.\n\n{numbered}")},
        ]
        resp = _gen(model, tok, msgs, max_tokens=min(2048, 120 * len(block) + 200))

        parsed: Dict[int, str] = {}
        for line in resp.splitlines():
            m = _NUM_RE.match(line)
            if m:
                parsed[int(m.group(1))] = m.group(2).strip().strip('"').strip("«»")

        if len(parsed) >= len(block):
            for j in range(len(block)):
                out_tr[i + j] = (parsed.get(j + 1) or "").strip()
        else:
            for j, t in enumerate(block):  # robust per-line fallback
                out_tr[i + j] = _translate_one(model, tok, t, src_name, context) if t else ""

        done += len(block)
        if on_progress:
            on_progress(min(1.0, done / max(1, n)))

    result: List[Dict] = []
    for seg, tr in zip(segments, out_tr):
        new = dict(seg)
        new["text_tr"] = (tr or seg.get("text", "")).strip()
        result.append(new)
    return result
