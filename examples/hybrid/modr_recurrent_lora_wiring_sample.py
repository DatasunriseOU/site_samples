"""MoDr recurrent + LoRA branch wiring sample.

What it is: a public-safe receipt for how a recurrent shared block can be
paired with several lightweight LoRA branches and selected at each recurrence
step.

Why it exists: MoDr does not allocate a separate full block per route. It keeps
one shared recurrent core and uses small branch deltas to explore different
reasoning paths.

What problem it solves: it shows how recurrent depth and branch diversity can
be combined without paying full-model parameter cost for every branch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoDrBranchConfig:
    num_recurrent_layers: int
    num_branches: int
    lora_rank: int
    lora_alpha: float
    lora_target: str
    num_branches_per_token: int


def build_modr_branch_wiring(config: object) -> dict[str, object]:
    """Return the recurrent-core and LoRA-branch contract used by MoDr."""

    branch_cfg = MoDrBranchConfig(
        num_recurrent_layers=int(getattr(config, "n_recurrent_layers", 4)),
        num_branches=int(getattr(config, "num_branches", 3)),
        lora_rank=int(getattr(config, "lora_r", 16)),
        lora_alpha=float(getattr(config, "lora_alpha", 16.0)),
        lora_target=str(getattr(config, "lora_target", "qkv,proj")),
        num_branches_per_token=int(getattr(config, "num_branches_per_tok", 1)),
    )
    targets = [item.strip() for item in branch_cfg.lora_target.split(",") if item.strip()]
    return {
        "recurrent_core": {
            "shared_layers": branch_cfg.num_recurrent_layers,
            "state_update": "shared core iterated across recurrence steps",
            "branch_application": "LoRA deltas are applied on top of the shared block outputs",
        },
        "branches": {
            "num_branches": branch_cfg.num_branches,
            "targets": targets,
            "lora_rank": branch_cfg.lora_rank,
            "lora_alpha": branch_cfg.lora_alpha,
            "topk_branches_per_token": branch_cfg.num_branches_per_token,
        },
        "notes": {
            "parameter_story": "branch diversity comes from low-rank deltas rather than full duplicate blocks",
            "routing_story": "each recurrence step chooses which branch delta modulates the shared recurrent block",
        },
    }
