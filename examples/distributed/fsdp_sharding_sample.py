"""Public-safe FSDP/meta-init memory sketch grounded in shard-local accounting."""

from __future__ import annotations


def estimate_rank_memory(
    *,
    total_params_billion: float,
    world_size: int,
    bytes_per_param: int = 2,
    optimizer_multiplier: float = 4.0,
    activation_gb: float = 0.0,
    comm_buffer_gb: float = 0.0,
) -> dict[str, float]:
    param_bytes = total_params_billion * 1_000_000_000 * bytes_per_param
    param_shard_gb = param_bytes / world_size / (1024**3)
    optimizer_gb = param_shard_gb * optimizer_multiplier
    total_gb = param_shard_gb + optimizer_gb + activation_gb + comm_buffer_gb
    return {
        "meta_init": 1.0,
        "param_shard_gb": round(param_shard_gb, 2),
        "optimizer_state_gb": round(optimizer_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "comm_buffer_gb": round(comm_buffer_gb, 2),
        "total_rank_gb": round(total_gb, 2),
    }


def explain_fsdp_bootstrap() -> list[str]:
    return [
        "Construct on meta or empty storage first, then materialize shard-local parameters.",
        "FSDP cuts parameter residency per rank, but activations and buckets can still dominate memory.",
        "A small model can still OOM when checkpointing, sequence length, or bucket size keeps too much state alive.",
    ]
