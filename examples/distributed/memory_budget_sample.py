"""Memory budget sample.

This example shows how to split GPU memory into parameters, optimizer state,
activations, and residual overhead. The problem it solves is that OOMs are hard
to reason about if everything is treated as one opaque number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBreakdown:
    params_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float = 0.0


def estimate_total_memory(breakdown: MemoryBreakdown) -> float:
    """Mirror the MegaCpp POC budget arithmetic used in memory planning helpers."""
    return (
        breakdown.params_gb
        + breakdown.optimizer_gb
        + breakdown.activations_gb
        + breakdown.overhead_gb
    )


def estimate_fsdp_memory(
    *,
    total_param_gb: float,
    total_optimizer_gb: float,
    activations_per_dbs_gb: float,
    dbs: int,
    seq_len: int,
    dp: int,
    tp: int,
    fsdp_enabled: bool,
    base_seq_len: int = 4096,
    overhead_gb: float = 1.5,
) -> float:
    """Public-safe excerpt of MegaCpp POC memory planning math for FSDP/TP runs."""
    sharding = max(dp, 1) if fsdp_enabled and dp > 1 else 1
    params_gb = total_param_gb / sharding
    optimizer_gb = total_optimizer_gb / sharding
    activations_gb = dbs * (seq_len / base_seq_len) * activations_per_dbs_gb / max(tp, 1)
    return params_gb + optimizer_gb + activations_gb + overhead_gb


def explain_memory_budget() -> tuple[str, ...]:
    return (
        "FSDP usually shards parameters and optimizer state across data-parallel ranks",
        "activations are a different bucket and mostly scale with batch size and sequence length",
        "separating the buckets makes it clearer whether an OOM is model-state pressure or activation pressure",
    )
