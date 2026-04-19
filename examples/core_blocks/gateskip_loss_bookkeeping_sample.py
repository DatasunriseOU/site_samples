"""GateSkip loss and budget bookkeeping sample.

What it is: a public-safe excerpt of the MegaCpp POC training-side GateSkip
loss bookkeeping.

Why it exists: GateSkip is not only a gate module. Training also needs a stable
way to combine CE loss, sparsity pressure, and the linearly decaying token
budget used later for hard inference masks.

What problem it solves: it makes the training control surface explicit so gate
values are not logged and regularized in ad hoc ways across layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GateSkipBudgetConfigSample:
    n_embd: int
    sparsity_lambda: float = 0.1
    budget_start: float = 1.0
    budget_end: float = 0.8
    budget_warmup_steps: int = 1000


def gateskip_sparsity_loss_sample(gate_infos: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    if not gate_infos:
        return torch.zeros(())
    gate_dtype = gate_infos[0]["attn_gate"].dtype
    gate_device = gate_infos[0]["attn_gate"].device
    total = torch.zeros(1, dtype=gate_dtype, device=gate_device).squeeze()
    for info in gate_infos:
        total = total + info["attn_gate"].pow(2).mean()
        total = total + info["mlp_gate"].pow(2).mean()
    return total / (2 * len(gate_infos))


def gateskip_total_loss_sample(
    ce_loss: torch.Tensor,
    gate_infos: list[dict[str, torch.Tensor]],
    config: GateSkipBudgetConfigSample,
) -> torch.Tensor:
    if not gate_infos or config.sparsity_lambda == 0.0:
        return ce_loss
    sparsity = gateskip_sparsity_loss_sample(gate_infos)
    return ce_loss + config.sparsity_lambda * sparsity


def compute_gateskip_budget_sample(step: int, config: GateSkipBudgetConfigSample) -> float:
    if config.budget_warmup_steps <= 0:
        return config.budget_end
    progress = min(1.0, step / config.budget_warmup_steps)
    budget = config.budget_start - (config.budget_start - config.budget_end) * progress
    return max(0.01, budget)


def summarize_gateskip_bookkeeping_sample(
    *,
    ce_loss: torch.Tensor,
    gate_infos: list[dict[str, torch.Tensor]],
    step: int,
    config: GateSkipBudgetConfigSample,
) -> dict[str, float | torch.Tensor]:
    return {
        "budget": compute_gateskip_budget_sample(step, config),
        "sparsity_loss": gateskip_sparsity_loss_sample(gate_infos),
        "total_loss": gateskip_total_loss_sample(ce_loss, gate_infos, config),
    }
