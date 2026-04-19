"""Shared-block MTP sample.

What it is: a public-safe excerpt of the MegaCpp POC multi-token prediction
head that reuses one transformer block across multiple prediction depths.

Why it exists: FastMTP keeps parameter growth under control by sharing the same
block K times instead of allocating one extra head per future token.

What problem it solves: it preserves static shapes for TPU and compile-friendly
training while still predicting several future tokens from one main forward
pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def compute_step_weights(depth: int, decay: float) -> torch.Tensor:
    raw = torch.tensor([decay**k for k in range(depth)], dtype=torch.float32)
    return raw / raw.sum()


def roll_and_mask_targets(x: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    tail = x.new_full((x.shape[0], 1), ignore_index)
    return torch.cat((x[:, 1:], tail), dim=1)


def roll_and_mask_ids(x: torch.Tensor) -> torch.Tensor:
    tail = x.new_zeros((x.shape[0], 1))
    return torch.cat((x[:, 1:], tail), dim=1)


@dataclass(frozen=True)
class MTPPlan:
    depth: int
    decay: float
    recompute: bool
    learnable_norm: bool
    uses_shared_block: bool = True


def summarize_mtp_plan(config: object) -> dict[str, object]:
    depth = int(getattr(config, "mtp_depth", 3))
    decay = float(getattr(config, "mtp_decay", 0.6))
    recompute = bool(getattr(config, "mtp_recompute", True))
    learnable_norm = bool(getattr(config, "mtp_learnable_norm", False))
    plan = MTPPlan(
        depth=depth,
        decay=decay,
        recompute=recompute,
        learnable_norm=learnable_norm,
    )
    return {
        "depth": plan.depth,
        "step_weights": compute_step_weights(plan.depth, plan.decay).tolist(),
        "recompute": plan.recompute,
        "learnable_norm": plan.learnable_norm,
        "uses_shared_block": plan.uses_shared_block,
        "plain_block_contract": {
            "engram": False,
            "mhc": False,
            "dsa": False,
            "mamba": False,
            "moe": False,
            "mod": False,
        },
    }
