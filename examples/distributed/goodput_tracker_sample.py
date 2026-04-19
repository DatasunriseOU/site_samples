"""Training goodput tracker excerpt.

This example shows the donor-backed runtime metric used to separate useful step
time from compilation, checkpointing, evaluation, and input stalls. It exists
so an optimization pass can answer whether a run got faster for the right
reason.

The problem it solves is misleading step-time summaries. A short hot step can
still hide a slow run if compilation or data loading dominates wall time.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from enum import Enum


class GoodputEvent(Enum):
    JOB = "job"
    TPU_INIT = "tpu_init"
    TRAINING_PREPARATION = "training_preparation"
    DATA_LOADING = "data_loading"
    STEP = "step"
    CHECKPOINT = "checkpoint"
    EVAL = "eval"
    COMPILATION = "compilation"


_SPAN_EVENTS = {
    GoodputEvent.DATA_LOADING: "data_loading",
    GoodputEvent.STEP: "step",
    GoodputEvent.CHECKPOINT: "checkpoint",
    GoodputEvent.EVAL: "eval",
    GoodputEvent.COMPILATION: "compilation",
}


class GoodputTracker:
    """Trimmed donor implementation for wall-time accounting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._milestones: dict[str, float] = {}
        self._durations: dict[str, float] = {
            "step": 0.0,
            "checkpoint": 0.0,
            "eval": 0.0,
            "compilation": 0.0,
            "data_loading": 0.0,
        }
        self._step_count = 0

    def record(self, milestone: str) -> None:
        with self._lock:
            self._milestones[milestone] = time.monotonic()

    def record_event(self, event: GoodputEvent, phase: str = "point") -> None:
        if phase not in {"start", "end", "point"}:
            raise ValueError(f"Unsupported phase={phase!r}")
        suffix = {"start": "_start", "end": "_end", "point": ""}[phase]
        self.record(f"{event.value}{suffix}")

    @contextmanager
    def span(self, category: str):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self._durations[category] = self._durations.get(category, 0.0) + elapsed
                if category == "step":
                    self._step_count += 1

    def compute_goodput(self) -> float:
        with self._lock:
            wall = self._wall_time()
            if wall <= 0 or self._step_count == 0:
                return 0.0
            return min(1.0, self._durations["step"] / wall)

    def compute_badput_breakdown(self) -> dict[str, float]:
        with self._lock:
            wall = self._wall_time()
            result: dict[str, float] = {}
            accounted = 0.0
            for category in ("step", "checkpoint", "eval", "compilation", "data_loading"):
                duration = self._durations.get(category, 0.0)
                result[category] = duration
                accounted += duration
            result["idle"] = max(0.0, wall - accounted)
            result["wall_time"] = wall
            return result

    def summary(self) -> dict[str, float | int | dict[str, float]]:
        with self._lock:
            wall = self._wall_time()
            step_time = self._durations["step"]
            goodput = min(1.0, step_time / wall) if wall > 0 and self._step_count > 0 else 0.0
            return {
                "goodput": round(goodput, 4),
                "wall_time_s": round(wall, 2),
                "step_time_s": round(step_time, 2),
                "step_count": self._step_count,
                "avg_step_s": round(step_time / self._step_count, 4) if self._step_count > 0 else 0.0,
            }

    def _wall_time(self) -> float:
        if not self._milestones:
            return 0.0
        started = self._milestones.get("job_start", min(self._milestones.values()))
        return time.monotonic() - started


@contextmanager
def maybe_record_goodput(tracker: GoodputTracker | None, event: GoodputEvent):
    """Donor-backed event wrapper for code that may or may not enable tracking."""

    if tracker is None:
        yield
        return
    category = _SPAN_EVENTS.get(event)
    if category is not None:
        with tracker.span(category):
            yield
        return
    tracker.record_event(event, "start")
    try:
        yield
    finally:
        tracker.record_event(event, "end")
