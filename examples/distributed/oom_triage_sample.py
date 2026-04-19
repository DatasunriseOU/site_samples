"""Public OOM triage helper using the real distributed-memory buckets."""

from __future__ import annotations


def estimate_memory_budget(
    *, params_gb: float, activations_gb: float, optimizer_gb: float, comm_buffers_gb: float, ep_temp_gb: float
) -> float:
    return round(params_gb + activations_gb + optimizer_gb + comm_buffers_gb + ep_temp_gb, 2)


def dominant_pressure(
    *, params_gb: float, activations_gb: float, optimizer_gb: float, comm_buffers_gb: float, ep_temp_gb: float
) -> str:
    buckets = {
        "parameters": params_gb,
        "activations": activations_gb,
        "optimizer_state": optimizer_gb,
        "communication_buffers": comm_buffers_gb,
        "ep_temporaries": ep_temp_gb,
    }
    return max(buckets, key=buckets.get)
