"""Clustered sparse TPU forward-cache contract sample.

What it is: a public-safe receipt of the cached three-phase forward builder
used by the clustered sparse TPU path.

Why it exists: the bridge reuses compiled JAX functions by caching on static
config and exact-mask semantics, while keeping per-batch tensors like `doc_ids`
out of the cache key.

What problem it solves: it prevents TPU recompilation churn and avoids silently
capturing runtime tensors inside supposedly reusable sparse-Pallas closures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactMaskKey:
    window_size: tuple[int, int]
    local_window: int
    has_doc_ids: bool
    has_valid_token_counts: bool


def make_exact_mask_contract_static_key(mask_contract: dict | None) -> ExactMaskKey | None:
    if not isinstance(mask_contract, dict):
        return None
    return ExactMaskKey(
        window_size=tuple(mask_contract.get("window_size", (-1, 0))),
        local_window=int(mask_contract.get("local_window", 0) or 0),
        has_doc_ids=mask_contract.get("doc_ids") is not None,
        has_valid_token_counts=mask_contract.get("valid_token_counts") is not None,
    )


def rebuild_exact_mask_contract_runtime(
    mask_key: ExactMaskKey | None,
    *,
    doc_ids_present: bool,
    valid_token_counts_present: bool,
) -> dict | None:
    if mask_key is None:
        return None
    return {
        "window_size": mask_key.window_size,
        "local_window": mask_key.local_window,
        "doc_ids": "runtime doc_ids" if mask_key.has_doc_ids and doc_ids_present else None,
        "valid_token_counts": (
            "runtime valid_token_counts"
            if mask_key.has_valid_token_counts and valid_token_counts_present
            else None
        ),
    }


def describe_clustered_sparse_forward_cache(
    *,
    use_fused_scoring: bool,
    use_pallas: bool,
    seq_len: int,
    has_doc_ids: bool,
    has_valid_token_counts: bool,
) -> dict[str, object]:
    return {
        "cache_key": {
            "use_fused_scoring": use_fused_scoring,
            "use_pallas": use_pallas,
            "seq_len": seq_len,
            "exact_mask_semantics": {
                "has_doc_ids": has_doc_ids,
                "has_valid_token_counts": has_valid_token_counts,
            },
        },
        "phase_split": {
            "phase1": "importance scoring",
            "phase2": "union selection",
            "phase3": "sparse attention",
        },
        "runtime_rule": "static mask semantics are cached, per-batch doc_ids and valid_token_counts are rebuilt at call time",
        "goal": "stable call_jax caching without capturing batch-specific tensors in TPU sparse closures",
    }
