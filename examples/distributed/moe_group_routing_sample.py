"""Two-stage group routing sample for MegaCpp specialist experts.

This example shows how the MegaCpp POC narrows routing before top-k expert
selection. The problem it solves is expert churn: without a group filter, one
token can spray probability mass across too many experts and reduce
specialization.
"""

from __future__ import annotations

import torch


def apply_group_routing_mask(
    real_logits: torch.Tensor,
    n_routing_groups: int,
    topk_routing_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only the strongest routing groups before expert top-k.

    Mirrors the MegaCpp POC helper derived from the token-choice MoE path.
    Experts are partitioned into equal groups. Each token first picks the best
    groups, scored by the sum of the top-2 logits inside each group.
    """
    if real_logits.dim() != 2:
        raise ValueError("real_logits must be [tokens, experts]")
    if n_routing_groups <= 0:
        raise ValueError("n_routing_groups must be positive")
    bt, n_experts = real_logits.shape
    if n_experts % n_routing_groups != 0:
        raise ValueError("n_experts must be divisible by n_routing_groups")

    experts_per_group = n_experts // n_routing_groups
    grouped = real_logits.view(bt, n_routing_groups, experts_per_group)
    intra_k = min(2, experts_per_group)
    top2_vals, _ = torch.topk(grouped, intra_k, dim=-1)
    group_scores = top2_vals.sum(dim=-1)
    _, top_groups = torch.topk(group_scores, min(topk_routing_group, n_routing_groups), dim=-1)

    group_mask = torch.zeros((bt, n_routing_groups), dtype=real_logits.dtype, device=real_logits.device)
    group_mask.scatter_(1, top_groups, 1.0)
    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(bt, n_experts) > 0.5
    masked = torch.where(
        expert_mask,
        real_logits,
        torch.tensor(float("-inf"), dtype=real_logits.dtype, device=real_logits.device),
    )
    return masked, top_groups
