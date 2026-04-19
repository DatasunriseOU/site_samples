"""Clustered sparse TPU JAX/Pallas interop surface.

This example shows the helper layer around the experimental clustered-sparse
TPU path. It exists to keep JAX/Pallas dependency loading, exact mask metadata,
and `call_jax` probing confined to one place instead of leaking throughout the
model code.

The donor implementation improved this path by normalizing local-window and
document-mask contracts before phase selection, which reduced shape drift across
the importance-scoring and sparse-attention kernels.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable


MASK_CONTRACT_PARAM_NAMES = (
    "exact_mask_contract",
    "mask_contract",
    "attention_mask_contract",
    "mask_info",
    "mask_meta",
    "exact_mask",
)
WINDOW_PARAM_NAMES = ("window_size", "local_window")
DOC_IDS_PARAM_NAMES = (
    "doc_ids",
    "q_doc_ids",
    "kv_doc_ids",
    "segment_ids",
    "q_segment_ids",
    "kv_segment_ids",
)
VALID_PREFIX_PARAM_NAMES = (
    "valid_token_counts",
    "q_valid_token_counts",
    "kv_valid_token_counts",
    "attention_validity",
)


def ensure_jax_modules() -> tuple[object, object]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    return jax, jnp


def expected_sparse_pallas_symbols() -> tuple[str, ...]:
    """Return the donor-side sparse kernel symbols the TPU bridge expects."""

    return (
        "KernelConfig",
        "importance_scoring_pipeline",
        "importance_scoring_pipeline_fused",
        "union_selection_pipeline",
        "sparse_attention",
        "UnionMaps",
    )


def ensure_sparse_pallas_loader() -> tuple[str, tuple[str, ...]]:
    """Keep the import contract explicit without depending on private packages here."""

    return ("experiments.sparse_pallas", expected_sparse_pallas_symbols())


def ensure_call_jax() -> Callable | None:
    try:
        xla_builder = importlib.import_module("torch_xla.core.xla_builder")
    except ImportError:
        return None
    return getattr(xla_builder, "call_jax")


def phase_supports_local_window(explicit_kwargs: set[str]) -> bool:
    return any(name in explicit_kwargs for name in MASK_CONTRACT_PARAM_NAMES) or any(
        name in explicit_kwargs for name in WINDOW_PARAM_NAMES
    )


def phase_supports_doc_ids(explicit_kwargs: set[str]) -> bool:
    return any(name in explicit_kwargs for name in MASK_CONTRACT_PARAM_NAMES) or any(
        name in explicit_kwargs for name in DOC_IDS_PARAM_NAMES
    )


def normalize_local_window(window_size: object) -> int:
    if isinstance(window_size, tuple):
        if len(window_size) != 2:
            raise TypeError("window_size must be a 2-tuple")
        return int(window_size[0])
    if window_size is None:
        return 0
    return int(window_size)


def mask_contract_summary(mask_contract: dict[str, object]) -> dict[str, object]:
    """Mirror the donor-side exact-mask summary used before phase dispatch."""

    return {
        "window_size": normalize_local_window(mask_contract.get("window_size")),
        "local_window": int(mask_contract.get("local_window", 0) or 0),
        "has_doc_ids": bool(mask_contract.get("has_doc_ids", False)),
        "has_valid_token_counts": bool(mask_contract.get("has_valid_token_counts", False)),
    }


def choose_clustered_sparse_backend(*, torch_xla_active: bool, jax_kernel_available: bool) -> str:
    if torch_xla_active and jax_kernel_available:
        return "torch_xla_plus_call_jax"
    if torch_xla_active:
        return "xla_dense_fallback"
    return "non_xla_fallback"


def interop_summary() -> list[str]:
    return [
        "Confine JAX/Pallas loading to the clustered-sparse helper boundary.",
        "Normalize local-window, segment-id, and valid-prefix metadata before phase selection.",
        "Prefer the TPU bridge only when both torch_xla and the sparse JAX kernels are available.",
    ]
