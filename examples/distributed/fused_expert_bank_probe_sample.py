"""Grouped expert-bank probe sample.

This example is adapted from the MegaCpp POC grouped expert-bank probe. The
problem it solves is simple: once routing is fixed, the next bottleneck is
expert compute throughput.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_expert_dense_step(
    x: torch.Tensor,
    w_fc: torch.Tensor,
    w_proj: torch.Tensor,
    *,
    activation: str,
    w_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small public-safe grouped expert-bank step.

    Shapes follow the real MegaCpp POC probe surface:
    - x: [experts, tokens_per_expert, model_dim]
    - w_fc: [experts, model_dim, hidden_dim]
    - w_proj: [experts, hidden_dim, model_dim]
    - w_gate: [experts, model_dim, hidden_dim] for SwiGLU
    """
    hidden = torch.einsum("etd,edh->eth", x, w_fc)
    if activation == "swiglu":
        if w_gate is None:
            raise ValueError("w_gate is required for swiglu")
        gate = torch.einsum("etd,edh->eth", x, w_gate)
        hidden = F.silu(gate) * hidden
    elif activation == "relu2":
        hidden = F.relu(hidden).square()
    else:
        hidden = F.gelu(hidden)
    return torch.einsum("eth,ehd->etd", hidden, w_proj)
