"""DualPipe schedule sample.

This example shows why DualPipe exists: it overlaps forward and backward work
from different microbatches so pipeline bubbles shrink. The problem it solves
is poor device utilization in plain sequential pipeline schedules.
"""

from __future__ import annotations


def dualpipe_loss_scale(grad_accum_steps: int, num_chunks: int) -> float:
    """Mirror the MegaCpp POC scale rule for auxiliary losses under DualPipe."""
    if grad_accum_steps < 1 or num_chunks < 1:
        raise ValueError("grad_accum_steps and num_chunks must be >= 1")
    return 1.0 / (grad_accum_steps * num_chunks)


def describe_dualpipe_overlap() -> tuple[str, ...]:
    return (
        "run forward work from one microbatch while backward runs for another",
        "reduce the startup and drain bubbles of plain pipeline parallelism",
        "pay for the overlap with more schedule complexity and stricter scaling rules",
    )
