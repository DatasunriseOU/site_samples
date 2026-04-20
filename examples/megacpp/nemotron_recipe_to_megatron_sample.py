"""MegaCpp public example: express a Nemotron-style recipe as Megatron-native args.

What this solves in simple words:
- recipe objects are easier to author than giant shell launchers;
- a public sample should show which parts become native Megatron flags and which
  parts remain local feature notes.
"""

from __future__ import annotations


def build_nemotron_recipe_to_megatron_args() -> dict[str, object]:
    return {
        "pattern": "AEMEAEMEAEMR",
        "num_layers": 52,
        "moe_enabled": True,
        "moe_top_k": 4,
        "use_mla": True,
        "use_mtp": True,
        "native_flags": [
            "--sequence-parallel",
            "--tensor-model-parallel-size=2",
            "--pipeline-model-parallel-size=4",
        ],
        "custom_notes": [
            "custom mamba and recurrent seams stay outside native Megatron flags",
            "recipe-level CUDA graph policy remains a separate launch concern",
        ],
    }
