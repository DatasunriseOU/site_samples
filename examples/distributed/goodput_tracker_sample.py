"""Goodput tracker sample.

This example shows how to measure how much wall time is spent doing real
training work versus setup, data loading, compilation, evaluation, or
checkpointing. The problem it solves is that step time alone hides where a
training run is actually wasting time.
"""

from __future__ import annotations

from enum import Enum


class GoodputEvent(Enum):
    JOB = "job"
    DATA_LOADING = "data_loading"
    STEP = "step"
    CHECKPOINT = "checkpoint"
    EVAL = "eval"
    COMPILATION = "compilation"


def compute_goodput(*, step_time: float, wall_time: float) -> float:
    """Mirror the donor's central goodput metric: step time over wall time."""
    if wall_time <= 0:
        return 0.0
    return min(1.0, max(0.0, step_time / wall_time))


def compute_badput_breakdown(
    *,
    wall_time: float,
    step_time: float,
    checkpoint_time: float = 0.0,
    eval_time: float = 0.0,
    compilation_time: float = 0.0,
    data_loading_time: float = 0.0,
) -> dict[str, float]:
    """Return donor-style overhead buckets plus residual idle time."""
    breakdown = {
        "step": step_time,
        "checkpoint": checkpoint_time,
        "eval": eval_time,
        "compilation": compilation_time,
        "data_loading": data_loading_time,
    }
    accounted = sum(breakdown.values())
    breakdown["idle"] = max(0.0, wall_time - accounted)
    breakdown["wall_time"] = wall_time
    return breakdown


def explain_goodput() -> tuple[str, ...]:
    return (
        "goodput asks how much of the run is actual optimizer-step work",
        "badput buckets explain where the lost time went",
        "the split helps decide whether to optimize kernels, input pipeline, compilation, or checkpointing",
    )
