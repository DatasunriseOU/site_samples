"""Hybrid block taxonomy sample.

What it is: a public-safe map from NAM-style block letters to the real layer
surfaces they select in the MegaCpp POC stack.

Why it exists: users see short block labels in recipes and docs, but those
labels hide several different sequence mixers and optional side branches.

What problem it solves: it gives a concrete dictionary for the block names so
readers can tell which layers are attention blocks, Mamba blocks, DeltaNet
blocks, routed-expert blocks, or residual-only tail blocks.
"""

from __future__ import annotations


def hybrid_block_taxonomy() -> dict[str, dict[str, object]]:
    return {
        "ablock": {
            "symbol": "A",
            "core_mixer": "attention",
            "optional_branches": ["sparse_attention", "engram", "mHC"],
            "used_for": "standard causal or sparse-attention layers",
        },
        "mblock": {
            "symbol": "M",
            "core_mixer": "mamba",
            "optional_branches": [],
            "used_for": "state-space mixer layers from the M-pattern",
        },
        "dblock": {
            "symbol": "D",
            "core_mixer": "gated_deltanet",
            "optional_branches": [],
            "used_for": "recurrent DeltaNet-style layers selected by the same pattern",
        },
        "eblock": {
            "symbol": "E",
            "core_mixer": "expert_mlp",
            "optional_branches": ["shared_expert", "routed_experts"],
            "used_for": "MoE layers with routed and shared expert paths",
        },
        "rblock": {
            "symbol": "R",
            "core_mixer": "mamba_tail_variant",
            "optional_branches": [],
            "used_for": "recipe-level tail symbol that still maps to the Mamba-backed lane in the public sample pack",
        },
        "cblock": {
            "symbol": "C",
            "core_mixer": "compile_or_capture_scope",
            "optional_branches": ["regional_compile", "cuda_graphs"],
            "used_for": "runtime-facing block grouping in public docs rather than a distinct math layer",
        },
    }
