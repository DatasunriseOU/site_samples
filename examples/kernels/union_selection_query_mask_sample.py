"""Clustered sparse union-selection stage sample.

What it is: a public-safe excerpt of the Phase-2 union-selection contract used
before the clustered sparse TPU attention kernel runs.

Why it exists: the sparse pipeline first selects candidate blocks, then has to
deduplicate them into compact union maps that the Phase-3 kernel can actually
consume.

What problem it solves: it makes the query-tile union contract explicit so the
runtime can validate shapes, carry exact legality masks forward, and expose a
debuggable dense mask for tests without changing the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelConfig:
    query_tile_size: int
    num_kv_heads: int
    top_n: int


def validate_selected_block_shape(
    *,
    selected_shape: tuple[int, int, int, int],
    cfg: KernelConfig,
) -> dict[str, int]:
    batch_size, query_tokens, kv_heads, top_n = selected_shape
    if kv_heads != cfg.num_kv_heads:
        raise ValueError(f"KV heads mismatch: got {kv_heads}, expected {cfg.num_kv_heads}")
    if top_n != cfg.top_n:
        raise ValueError(f"top_n mismatch: got {top_n}, expected {cfg.top_n}")
    if query_tokens % cfg.query_tile_size != 0:
        raise ValueError(
            f"T_q={query_tokens} not divisible by query_tile_size={cfg.query_tile_size}"
        )
    return {
        "batch_size": batch_size,
        "query_tokens": query_tokens,
        "num_query_tiles": query_tokens // cfg.query_tile_size,
        "kv_heads": kv_heads,
        "top_n": top_n,
    }


def summarize_union_stage(
    *,
    selected_shape: tuple[int, int, int, int],
    seq_len: int,
    cfg: KernelConfig,
    has_selected_valid: bool,
    uses_bitmap: bool,
) -> dict[str, object]:
    shape_info = validate_selected_block_shape(selected_shape=selected_shape, cfg=cfg)
    return {
        "selected_contract": shape_info,
        "sequence_length": seq_len,
        "selected_valid_mask": has_selected_valid,
        "union_mode": "bitmap" if uses_bitmap else "sort",
        "phase2_outputs": [
            "union_block_indices",
            "union_sizes",
            "query_union_indices",
            "query_union_valid",
        ],
        "debug_surface": "materialize_query_union_mask can expand compact union data into a dense legality mask for tests",
    }
