"""Selective FIRE optimizer-state reset excerpt.

This example shows how FIRE clears optimizer state only for weights that were
just orthogonalized. The problem it solves is stale AdamW or Muon momentum on
freshly reset weights, which can otherwise push them back toward the old state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

@torch.no_grad()
def reset_optimizer_states_for_fired_params(
    optimizers: list[torch.optim.Optimizer],
    modified_params: set[nn.Parameter],
) -> int:
    """Reset optimizer slots only for parameters modified by FIRE."""
    reset_count = 0
    for opt in optimizers:
        keys_to_pop: list[nn.Parameter] = []
        for p in modified_params:
            if p in opt.state:
                keys_to_pop.append(p)
        for p in keys_to_pop:
            opt.state.pop(p)
            reset_count += 1
    return reset_count
