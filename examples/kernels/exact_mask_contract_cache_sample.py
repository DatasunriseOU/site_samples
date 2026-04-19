"""Exact-mask contract cache-key sample.

What it is: a public-safe excerpt of the MegaCpp POC static cache key used for
clustered sparse exact-mask contracts.

Why it exists: the runtime payload can carry per-batch tensors such as
`doc_ids` and `valid_token_counts`, but cached JAX/Pallas closures must only be
keyed by static shape and semantic knobs.

What problem it solves: it prevents recompilation and cache pollution caused by
capturing batch-specific tensors inside the kernel-closure key.
"""

from __future__ import annotations


def make_exact_mask_contract_static_key(mask_contract: dict[str, object] | None) -> tuple | None:
    if not isinstance(mask_contract, dict):
        return None
    return (
        tuple(mask_contract.get("window_size", (-1, 0))),
        int(mask_contract.get("local_window", 0) or 0),
        bool(mask_contract.get("doc_ids") is not None),
        bool(mask_contract.get("valid_token_counts") is not None),
    )


def rebuild_exact_mask_contract_runtime(
    mask_key: tuple | None,
    *,
    doc_ids: object = None,
    valid_token_counts: object = None,
) -> dict[str, object] | None:
    if mask_key is None:
        return None
    window_size, local_window, has_doc_ids, has_valid_token_counts = mask_key
    return {
        "window_size": tuple(window_size),
        "local_window": int(local_window),
        "doc_ids": doc_ids if has_doc_ids else None,
        "valid_token_counts": valid_token_counts if has_valid_token_counts else None,
    }
