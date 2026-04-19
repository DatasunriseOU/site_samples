"""FlexiDepth loss and skip-statistics sample.

What it is: a public-safe excerpt of the MegaCpp POC FlexiDepth bookkeeping
surface around skip loss and monitoring.

Why it exists: FlexiDepth needs more than a router mask. Training has to track
how many layers tokens use, how often tokens skip, and how the skip loss mixes
with the language-model objective.

What problem it solves: it gives one stable contract for skip loss and logging
instead of deriving skip-rate numbers from unrelated tensors later.
"""

from __future__ import annotations

import torch


def flexidepth_skip_loss_sample(all_scores: list[torch.Tensor], like: torch.Tensor | None = None) -> torch.Tensor:
    if not all_scores:
        if like is not None:
            return like.new_zeros(())
        return torch.zeros(())
    stacked = torch.stack(all_scores, dim=0)
    per_token_sum = stacked.sum(dim=0).squeeze(-1)
    return (per_token_sum**2).mean()


def flexidepth_total_loss_sample(
    lm_loss: torch.Tensor,
    all_scores: list[torch.Tensor],
    alpha: float = 1e-3,
) -> torch.Tensor:
    return lm_loss + alpha * flexidepth_skip_loss_sample(all_scores, like=lm_loss)


def flexidepth_skip_stats_sample(
    all_scores: list[torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, float | list[float]]:
    if not all_scores:
        return {
            "mean_layers_used": 0.0,
            "skip_rate": 0.0,
            "per_layer_skip_rate": [],
            "mean_gate_score": 0.0,
        }
    stacked = torch.stack(all_scores, dim=0)
    masks = (stacked > threshold).float()
    layers_per_token = masks.sum(dim=0).squeeze(-1)
    return {
        "mean_layers_used": layers_per_token.mean().item(),
        "skip_rate": 1.0 - masks.mean().item(),
        "per_layer_skip_rate": [1.0 - masks[i].mean().item() for i in range(stacked.shape[0])],
        "mean_gate_score": stacked.mean().item(),
    }
