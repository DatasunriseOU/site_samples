"""Public-safe expert-parallel capacity sketch grounded in routed-token planning."""

from __future__ import annotations

from math import ceil


def capacity_per_expert(tokens: int, experts: int, top_k: int, capacity_factor: float = 1.0) -> int:
    routed_tokens = tokens * top_k
    return ceil((routed_tokens / experts) * capacity_factor)


def routing_summary(
    *,
    tokens: int,
    experts: int,
    top_k: int,
    shared_expert: bool = True,
    router_dtype: str = "fp32",
) -> dict[str, int | bool | str]:
    return {
        "tokens": tokens,
        "experts": experts,
        "top_k": top_k,
        "shared_expert": shared_expert,
        "router_dtype": router_dtype,
        "capacity_per_expert": capacity_per_expert(tokens, experts, top_k),
        "fixed_buffer_dispatch": True,
    }
