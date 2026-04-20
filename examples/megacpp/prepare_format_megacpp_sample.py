"""MegaCpp public example: format and split Megatron-ready data with public naming rules.

What this solves in simple words:
- a thin wrapper can encode naming, shard, and split policy without dragging in
  the whole low-level converter each time;
- this makes data-prep outputs easier to reason about and document.
"""

from __future__ import annotations


def build_prepare_format_contract() -> dict[str, object]:
    return {
        "train_split": "explicit",
        "valid_split": "explicit",
        "naming_policy": "public-safe MegaCpp dataset names",
        "artifact_family": "Megatron-ready indexed dataset",
    }
