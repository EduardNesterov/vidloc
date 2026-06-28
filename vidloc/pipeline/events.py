"""Progress / logging plumbing shared between the pipeline and the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


class Cancelled(Exception):
    """Raised inside the pipeline when the user requests cancellation."""


@dataclass
class Reporter:
    on_overall: Optional[Callable[[float, str], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    on_stage: Optional[Callable[[str], None]] = None
    _cancelled: bool = False

    def log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    def stage(self, name: str) -> None:
        if self.on_stage:
            self.on_stage(name)

    def overall(self, fraction: float, msg: str = "") -> None:
        if self.on_overall:
            self.on_overall(max(0.0, min(1.0, fraction)), msg)

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class StageTracker:
    """Maps per-stage progress (0..1) onto an overall 0..1 bar by weight."""

    def __init__(self, reporter: Reporter, stages: List[Tuple[str, float]]):
        self.r = reporter
        total = sum(w for _, w in stages) or 1.0
        self.order = [n for n, _ in stages]
        self.weight = {n: w / total for n, w in stages}
        self._base = 0.0
        self._cur = 0.0

    def start(self, name: str, msg: str = "") -> None:
        i = self.order.index(name)
        self._base = sum(self.weight[n] for n in self.order[:i])
        self._cur = self.weight[name]
        self.r.stage(name)
        self.r.overall(self._base, msg)

    def sub(self, fraction: float, msg: str = "") -> None:
        self.r.overall(self._base + self._cur * max(0.0, min(1.0, fraction)), msg)

    def done(self) -> None:
        self.r.overall(self._base + self._cur)
