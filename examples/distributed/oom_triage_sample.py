"""Public OOM triage helper using MegaCpp POC-backed memory buckets.

The bucket names are adapted from the internal memory estimator and the
residual split used by the internal memory debug helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBreakdown:
    """Per-device memory buckets aligned to the MegaCpp POC estimator."""

    params_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_gb: float = 0.0
    activations_gb: float = 0.0
    moe_routing_gb: float = 0.0
    feature_activations_gb: float = 0.0
    runtime_reserved_gb: float = 0.0
    overhead_gb: float = 0.0


def estimate_memory_budget(breakdown: MemoryBreakdown) -> float:
    return round(
        breakdown.params_gb
        + breakdown.gradients_gb
        + breakdown.optimizer_gb
        + breakdown.activations_gb
        + breakdown.moe_routing_gb
        + breakdown.feature_activations_gb
        + breakdown.runtime_reserved_gb
        + breakdown.overhead_gb,
        2,
    )


def dominant_pressure(breakdown: MemoryBreakdown) -> str:
    buckets = {
        "parameters": breakdown.params_gb,
        "gradients": breakdown.gradients_gb,
        "optimizer_state": breakdown.optimizer_gb,
        "activations": breakdown.activations_gb,
        "moe_routing": breakdown.moe_routing_gb,
        "feature_activations": breakdown.feature_activations_gb,
        "runtime_reserved": breakdown.runtime_reserved_gb,
        "overhead": breakdown.overhead_gb,
    }
    return max(buckets, key=buckets.get)


def residual_runtime_bytes(*, allocated_bytes: int, param_bytes: int, grad_bytes: int, buffer_bytes: int) -> int:
    """Mirror the MegaCpp POC debug split between model-state memory and everything else."""

    residual = allocated_bytes - (param_bytes + grad_bytes + buffer_bytes)
    return max(residual, 0)
