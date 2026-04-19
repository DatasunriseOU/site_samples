"""NAM56R feature placement receipt.

What it is: a public-safe snapshot of where the main NAM56R feature families are
placed across the real block stack.

Why it exists: feature names like Engram, DSA, mHC, MTP, MoD, and n-gram hash
are easy to list without saying where they actually attach.

What problem it solves: it ties each feature to the block family or layer ranks
where it is expected to run, so the public model description stays concrete.
"""

from __future__ import annotations


def build_nam56r_feature_receipt() -> dict[str, object]:
    return {
        "pattern": "AEMEAEMEAEMR",
        "depth": 52,
        "global_features": {
            "ngram_hash": {
                "placement": "input side before the main block stack",
                "notes": ["CPU offload is supported", "used as cheap local motif enrichment"],
            },
            "structure": {
                "placement": "training data and loader contracts",
                "notes": ["enriched parquet metadata", "tree-aware FFN and structure features"],
            },
            "mtp": {
                "placement": "training objective suffix over the main stack",
                "notes": ["extra prediction depths", "separate weighted auxiliary loss"],
            },
        },
        "block_family_features": {
            "A-block": {
                "features": ["MLA", "selected DSA ranks", "Engram", "mHC"],
                "placement": {
                    "engram_layers": "first A-layer ranks",
                    "dsa_a_layer_ranks": "selected later A-layer ranks",
                    "mhc_layers": "first A-layer ranks with multi-stream mixing",
                },
            },
            "E-block": {
                "features": ["MoE", "MoD"],
                "placement": {
                    "moe": "all E-blocks",
                    "mod": "E-block routing/computation budget control",
                },
            },
            "M-block": {
                "features": ["Mamba-3", "MIMO"],
                "placement": {
                    "mamba": "all M-blocks",
                    "mimo": "enabled through Mamba stack configuration",
                },
            },
            "R-block": {
                "features": ["M2RNN"],
                "placement": {
                    "custom_r_indices": "explicit R-layer index list",
                },
            },
        },
    }
