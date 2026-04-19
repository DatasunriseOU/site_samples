"""Public-safe expert-parallel routing planning excerpt."""

from __future__ import annotations

from math import ceil


def routed_tokens_per_step(tokens: int, top_k: int) -> int:
    """Count how many token-to-expert assignments the router emits per step."""
    return tokens * top_k


def capacity_per_expert(
    tokens: int,
    experts: int,
    top_k: int,
    capacity_factor: float = 1.0,
) -> int:
    """Estimate per-expert routing capacity for a fixed-size dispatch buffer."""
    routed_tokens = routed_tokens_per_step(tokens, top_k)
    return ceil((routed_tokens / experts) * capacity_factor)


def routing_summary(
    *,
    tokens: int,
    routed_experts: int,
    top_k: int,
    shared_experts: int = 1,
    routing_mode: str = "token_choice",
    capacity_factor: float = 1.0,
) -> dict[str, int | str]:
    """Summarize routed/shared expert planning numbers for one training step."""
    router_outputs = routed_experts + 1 if routing_mode == "token_choice" else routed_experts
    return {
        "tokens": tokens,
        "top_k": top_k,
        "routed_experts": routed_experts,
        "shared_experts": shared_experts,
        "router_outputs": router_outputs,
        "routed_tokens": routed_tokens_per_step(tokens, top_k),
        "capacity_per_expert": capacity_per_expert(
            tokens,
            routed_experts,
            top_k,
            capacity_factor,
        ),
        "routing_mode": routing_mode,
    }
