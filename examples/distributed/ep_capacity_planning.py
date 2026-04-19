"""Public-safe EP capacity planning excerpt from donor parallelism checks."""

from __future__ import annotations


def validate_ep_degree(*, routed_experts: int, ep_size: int, moe_enabled: bool = True) -> None:
    """Mirror donor hard checks for expert-parallel splits."""
    if ep_size <= 1:
        return
    if not moe_enabled:
        raise ValueError(f"ep={ep_size} > 1 requires moe_enabled=True in model config.")
    if routed_experts % ep_size != 0:
        raise ValueError(
            f"moe_n_routed_experts={routed_experts} not divisible by "
            f"ep={ep_size}. Expert parallelism requires even expert splits."
        )


def estimate_memory_per_device(
    *,
    pp: int,
    dp: int,
    tp: int,
    total_params: int,
    bytes_per_param: float = 2.0,
    optimizer_multiplier: float = 10.0,
) -> dict[str, float]:
    """Simplified donor-based per-device memory estimate under 3D parallelism."""
    params_per_stage = total_params / max(pp, 1)

    tp_shardable_frac = 0.65
    tp_reduction = 1.0 - tp_shardable_frac * (1.0 - 1.0 / max(tp, 1))
    params_per_device = params_per_stage * tp_reduction

    fsdp_reduction = 1.0 / max(dp, 1)
    param_bytes = params_per_device * bytes_per_param * fsdp_reduction
    opt_bytes = params_per_device * optimizer_multiplier * fsdp_reduction

    param_gb = param_bytes / (1024**3)
    opt_gb = opt_bytes / (1024**3)
    return {
        "params_gb": param_gb,
        "optimizer_gb": opt_gb,
        "total_gb": param_gb + opt_gb,
    }


def plan_ep_capacity(
    *,
    tokens: int,
    top_k: int,
    routed_experts: int,
    ep_size: int,
    total_params: int,
    pp: int = 1,
    dp: int = 1,
    tp: int = 1,
) -> dict[str, int | float]:
    """Combine donor EP divisibility checks with donor memory heuristics."""
    validate_ep_degree(routed_experts=routed_experts, ep_size=ep_size)
    memory = estimate_memory_per_device(pp=pp, dp=dp, tp=tp, total_params=total_params)
    return {
        "routed_tokens": tokens * top_k,
        "experts_per_rank": routed_experts // max(ep_size, 1),
        **memory,
    }
