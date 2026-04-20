"""MegaCpp public example: Nemotron-style NAM56R recipe contract.

What this solves in simple words:
- one recipe object can describe model shape, precision mode, parallelism mode,
  and launcher intent without forcing the user to read one giant shell command;
- the recipe stays import-safe away from CUDA machines and lowers later.
"""

from __future__ import annotations


def build_recipe_contract() -> dict[str, object]:
    return {
        "pattern": "AEMEAEMEAEMR",
        "depth": 52,
        "hidden_size": 3584,
        "ffn_hidden_size": 18944,
        "num_attention_heads": 56,
        "num_query_groups": 8,
        "vocab_size": 65536,
        "precision_modes": ["bf16", "fp8"],
        "parallelism_modes": ["nemo_native", "author_dp"],
        "why_it_exists": "keep model intent and launch intent together before lowering",
    }
