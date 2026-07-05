"""Small persisted user-preferences store (JSON in ``~/.vidloc/config.json``).

Unlike :mod:`vidloc.settings` (paths/env, read once at import), this holds choices
the user makes in the GUI at runtime — currently which browser to read Instagram
cookies from. Read on demand so a change in the GUI takes effect without a restart.
"""

from __future__ import annotations

import json
from typing import Any

from . import settings


def _path():
    return settings.CONFIG_DIR / "config.json"


def load() -> dict:
    try:
        return json.loads(_path().read_text("utf-8"))
    except Exception:  # noqa: BLE001 — missing / corrupt config is not fatal
        return {}


def save(cfg: dict) -> None:
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


def get(key: str, default: str = "") -> str:
    val = load().get(key, default)
    return str(val) if val is not None else default


def set(key: str, value: Any) -> None:  # noqa: A001 — mirror dict.set ergonomics
    cfg = load()
    cfg[key] = value
    save(cfg)
