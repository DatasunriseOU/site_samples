"""CPU optimizer offload sample.

This example shows how to move part of AdamW-owned state to CPU when GPU memory
is too tight. The problem it solves is optimizer-state pressure that blocks a
larger batch size or model size even when forward/backward still fits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadPlan:
    gpu_params: int
    cpu_params: int
    offload_fraction: float


def split_params_largest_first(param_numels: list[int], offload_fraction: float) -> OffloadPlan:
    """Mirror the donor's largest-first offload heuristic for AdamW params."""
    if not 0.0 <= offload_fraction <= 1.0:
        raise ValueError("offload_fraction must be in [0, 1]")
    total = sum(param_numels)
    target = int(total * offload_fraction)
    moved = 0
    for n in sorted(param_numels, reverse=True):
        if moved >= target:
            break
        moved += n
    return OffloadPlan(gpu_params=total - moved, cpu_params=moved, offload_fraction=offload_fraction)


def explain_optimizer_offload() -> tuple[str, ...]:
    return (
        "keep the main forward and backward path on GPU",
        "move a selected fraction of AdamW-owned parameters and optimizer state to CPU",
        "save GPU memory at the cost of extra host-device traffic during optimizer.step",
    )
