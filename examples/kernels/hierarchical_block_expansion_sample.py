"""Hierarchical block expansion sample.

What it is: a public-safe receipt for the coarse-to-fine sparse Pallas scoring
design.

Why it exists: exhaustive block scoring is too expensive at very long context,
so the planner first scores coarse meta-blocks and only expands the children of
the best meta-blocks.

What problem it solves: it makes the hierarchical refinement contract explicit
so readers can see how the runtime trades a small recall risk for a large speed
gain.
"""

from __future__ import annotations


def compress_keys_coarse(*, num_compressed_blocks: int, coarse_ratio: int) -> int:
    if coarse_ratio <= 0:
        raise ValueError("coarse_ratio must be > 0")
    return (int(num_compressed_blocks) + int(coarse_ratio) - 1) // int(coarse_ratio)


def build_hierarchical_scoring_plan(*, num_compressed_blocks: int, coarse_ratio: int = 8, coarse_topc: int = 16, fine_children_per_meta: int = 8, final_topk: int = 8) -> dict[str, object]:
    meta_blocks = compress_keys_coarse(
        num_compressed_blocks=num_compressed_blocks,
        coarse_ratio=coarse_ratio,
    )
    fine_candidates = int(coarse_topc) * int(fine_children_per_meta)
    return {
        "coarse_pass": {
            "meta_blocks": meta_blocks,
            "coarse_ratio": int(coarse_ratio),
            "top_meta_blocks": int(coarse_topc),
        },
        "fine_pass": {
            "children_per_meta_block": int(fine_children_per_meta),
            "fine_candidates": fine_candidates,
            "final_topk": int(final_topk),
        },
        "rule": "score all coarse meta-blocks, expand children of top meta-blocks, then take final top-k",
    }
