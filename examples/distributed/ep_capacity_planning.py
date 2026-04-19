"""Public-safe expert-parallel capacity planning helper."""

from math import ceil


def plan_ep_capacity(tokens: int, top_k: int, experts: int, capacity_factor: float = 1.0) -> dict[str, int]:
    routed = tokens * top_k
    per_expert = ceil((routed / experts) * capacity_factor)
    return {
        "routed_tokens": routed,
        "per_expert_capacity": per_expert,
    }
