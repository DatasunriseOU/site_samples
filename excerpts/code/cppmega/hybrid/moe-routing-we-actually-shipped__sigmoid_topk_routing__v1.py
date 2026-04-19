"""Public excerpt.

Source: MegaCpp routing notes distilled into a minimal public-safe sample
Purpose: show sigmoid token-choice routing with null-slot masking and real-expert renormalization
Edited for clarity.
"""

from math import exp


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def topk_token_choice(router_logits: list[float], top_k: int) -> list[dict[str, float]]:
    scored = [
        {"expert_index": expert_index, "score": sigmoid(logit)}
        for expert_index, logit in enumerate(router_logits)
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def renormalize_real_expert_weights(
    routed: list[dict[str, float]],
    real_expert_count: int,
) -> list[dict[str, float]]:
    real_weight_sum = sum(
        item["score"] for item in routed if item["expert_index"] < real_expert_count
    )
    if real_weight_sum <= 0.0:
        return []

    normalized = []
    for item in routed:
        if item["expert_index"] >= real_expert_count:
            continue
        normalized.append(
            {
                "expert_index": item["expert_index"],
                "weight": item["score"] / real_weight_sum,
            }
        )
    return normalized


def route_token(
    router_logits: list[float],
    *,
    n_real_experts: int,
    top_k: int,
    null_rho: float,
) -> dict[str, object]:
    null_slots = int(top_k * null_rho / max(1e-9, 1.0 - null_rho))
    max_choices = top_k + null_slots
    routed = topk_token_choice(router_logits, top_k=max_choices)
    return {
        "selected": routed,
        "real_expert_weights": renormalize_real_expert_weights(
            routed,
            real_expert_count=n_real_experts,
        ),
        "shared_expert_always_on": True,
    }
