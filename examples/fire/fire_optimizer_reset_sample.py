"""Selective FIRE optimizer-state reset excerpt."""

from __future__ import annotations

import torch
import torch.nn as nn

@torch.no_grad()
def reset_optimizer_states_for_fired_params(
    optimizers: list[torch.optim.Optimizer],
    modified_params: set[nn.Parameter],
) -> int:
    """Selectively reset optimizer states for parameters that were FIRE'd."""
    reset_count = 0
    for opt in optimizers:
        keys_to_pop = []
        for p in modified_params:
            if p in opt.state:
                keys_to_pop.append(p)
        for p in keys_to_pop:
            opt.state.pop(p)
            reset_count += 1
    return reset_count
