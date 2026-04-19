"""ReDo dormancy and recycle excerpt.

This example shows how ReDo finds dormant neurons and selectively reinitializes
them. The problem it solves is dead or nearly-dead MLP units that stop learning
and waste capacity during long training runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ReDoDiagnostics:
    """Track per-neuron activity so dormant units can be identified later."""

    stats: Dict[str, torch.Tensor] = field(default_factory=dict)

    def get_dormant_ratio(self, tau: float = 0.025) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for name, stats in self.stats.items():
            layer_mean = stats.mean().clamp(min=1e-8)
            scores = stats / layer_mean
            n_dormant = (scores < tau).sum().item()
            ratios[name] = n_dormant / stats.numel()
        return ratios


def dormant_ratio(stats: torch.Tensor, *, tau: float = 0.025) -> float:
    """Return the fraction of units whose normalized activity falls below tau."""
    layer_mean = stats.mean().clamp(min=1e-8)
    scores = stats / layer_mean
    n_dormant = (scores < tau).sum().item()
    return n_dormant / stats.numel()


@torch.no_grad()
def recycle_core(
    fc_in: nn.Module,
    fc_out: nn.Module,
    stats: torch.Tensor,
    tau: float = 0.025,
    redo_stats: Optional[Dict[str, torch.Tensor]] = None,
    name: Optional[str] = None,
) -> int:
    """Reinitialize dormant rows in fc_in/fc_out while preserving active ones."""
    layer_mean = stats.mean().clamp(min=1e-8)
    is_dormant = (stats / layer_mean) < tau
    n_dormant = is_dormant.sum().item()
    if n_dormant == 0:
        return 0

    in_weight = fc_in.weight
    std_in = in_weight.std()
    mask_in = is_dormant.unsqueeze(1).expand_as(in_weight)
    new_in = torch.randn_like(in_weight) * std_in
    in_weight.copy_(torch.where(mask_in, new_in, in_weight))

    if getattr(fc_in, "bias", None) is not None:
        fc_in.bias.copy_(torch.where(is_dormant, torch.zeros_like(fc_in.bias), fc_in.bias))

    out_weight = fc_out.weight
    std_out = out_weight.std()
    mask_out = is_dormant.unsqueeze(0).expand_as(out_weight)
    new_out = torch.randn_like(out_weight) * (std_out * 0.1)
    out_weight.copy_(torch.where(mask_out, new_out, out_weight))

    if redo_stats is not None and name is not None:
        redo_stats[name] = torch.where(is_dormant, layer_mean, stats)

    return int(n_dormant)
