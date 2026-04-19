"""Block taxonomy sample for the MegaCpp POC stack.

What it is: a compact description of the main block families used in the model
stack.

Why it exists: the runtime mixes several block types, and the short names can be
confusing without an explicit map.

What problem it solves: it gives one place that explains what each block family
does and where it sits in the model.
"""

from __future__ import annotations


def describe_block_taxonomy() -> dict[str, dict[str, object]]:
    return {
        "ablock": {
            "role": "attention-side feature block",
            "main_surfaces": ["attention", "engram", "mHC", "sparse branches"],
            "where_used": "selected Nemotron-style attention layers",
        },
        "mblock": {
            "role": "state-space block",
            "main_surfaces": ["Mamba path", "residual mixing", "compile-sensitive scans"],
            "where_used": "layers that use the M symbol in the NAM pattern",
        },
        "eblock": {
            "role": "expert or feed-forward block",
            "main_surfaces": ["MoE", "dense FFN", "aux-loss collection"],
            "where_used": "post-attention expert/feed-forward stages",
        },
        "cblock": {
            "role": "concept retrieval block",
            "main_surfaces": ["concept bank", "cross-attention over concepts"],
            "where_used": "optional concept-augmented layers",
        },
        "rblock": {
            "role": "recurrent-style residual block family",
            "main_surfaces": ["M2RNN-style recurrence", "residual plumbing"],
            "where_used": "recurrent experiments and mixed Mamba-recurrent lanes",
        },
    }
