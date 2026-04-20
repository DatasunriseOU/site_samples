"""MegaCpp public example: hybrid plan translation into Megatron-native roles.

What this solves in simple words:
- a hybrid pattern is only useful if it becomes a plan the runtime can actually
  execute;
- this sample keeps the plan explicit instead of hiding it inside one launcher.
"""

from __future__ import annotations


def build_megatron_plan() -> dict[str, object]:
    return {
        "pattern": "AEMEAEMEAEMR",
        "expanded_roles": ["attention", "moe", "mamba", "moe"] * 3 + ["mamba", "recurrent_tail"],
        "mtp_depths": 0,
        "fail_closed": True,
    }
