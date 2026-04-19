"""MoE auxiliary-loss collection sample.

This example mirrors the MegaCpp POC post-forward loss collection. The problem
it solves is training drift: routing quality and router stability need extra
signals beyond the language-model loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutedLayerLossState:
    last_aux_loss: torch.Tensor
    last_z_loss: torch.Tensor


def collect_moe_losses(
    layers: list[RoutedLayerLossState],
    aux_loss_weight: float = 0.01,
    z_loss_weight: float = 1e-3,
) -> torch.Tensor:
    """Sum the stored load-balancing and router z-loss terms.

    The weights match the public MegaCpp POC defaults taken from the real MoE
    module configuration.
    """
    if not layers:
        return torch.tensor(0.0)
    total = torch.zeros((), device=layers[0].last_aux_loss.device)
    for layer in layers:
        total = total + aux_loss_weight * layer.last_aux_loss
        total = total + z_loss_weight * layer.last_z_loss
    return total
