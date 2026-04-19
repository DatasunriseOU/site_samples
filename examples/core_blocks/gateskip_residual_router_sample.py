"""GateSkip residual gate sample.

What it is: a public-safe excerpt of the MegaCpp POC residual gating surface
used for token-wise layer skipping.

Why it exists: some tokens do not need the full branch output at every layer,
but the runtime still needs a differentiable training path and a simple hard
decision at inference.

What problem it solves: it turns "skip this layer for some tokens" into a
small residual gate instead of a shape-changing gather/scatter path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GateSkipConfigSample:
    n_embd: int
    scalar_gate: bool = True
    initial_bias: float = 5.0
    sparsity_weight: float = 0.1


class GateSkipGateSample(nn.Module):
    def __init__(self, config: GateSkipConfigSample) -> None:
        super().__init__()
        out_dim = 1 if config.scalar_gate else config.n_embd
        self.proj = nn.Linear(config.n_embd, out_dim, bias=True)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.proj.bias, config.initial_bias)

    def forward(self, residual_input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(residual_input))


def apply_gateskip_residual(
    residual_input: torch.Tensor,
    branch_output: torch.Tensor,
    gate: GateSkipGateSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_values = gate(residual_input).to(dtype=branch_output.dtype)
    mixed = residual_input + branch_output * gate_values
    return mixed, gate_values


def gateskip_sparsity_penalty(gate_values: torch.Tensor, weight: float) -> torch.Tensor:
    return gate_values.square().mean() * weight


def gateskip_inference_mask(gate_values: torch.Tensor, budget: float) -> torch.Tensor:
    token_scores = gate_values.mean(dim=-1)
    threshold = torch.quantile(token_scores.float(), max(0.0, 1.0 - budget))
    return (token_scores > threshold).to(dtype=gate_values.dtype)
