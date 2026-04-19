"""Clustered sparse TPU interop sample.

This example shows how the TPU sparse-attention path keeps PyTorch as the main
frontend but can hand a hot scoring kernel to JAX/Pallas when the specialized
kernel is available. The point is to isolate JAX interop to a narrow surface
instead of making the whole model depend on a second execution frontend.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable


def ensure_jax_modules() -> tuple[object, object]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    return jax, jnp


def expected_sparse_pallas_symbols() -> tuple[str, ...]:
    """Return the donor-side JAX/Pallas symbols the bridge expects.

    The public sample keeps the contract shape without importing any internal
    experiment package directly.
    """

    return (
        "KernelConfig",
        "importance_scoring_pipeline",
        "importance_scoring_pipeline_fused",
        "union_selection_pipeline",
        "sparse_attention",
        "UnionMaps",
    )


def ensure_call_jax() -> Callable | None:
    try:
        xla_builder = importlib.import_module("torch_xla.core.xla_builder")
    except ImportError:
        return None
    return getattr(xla_builder, "call_jax")


def choose_clustered_sparse_backend(*, torch_xla_active: bool, jax_kernel_available: bool) -> str:
    if torch_xla_active and jax_kernel_available:
        return "torch_xla_plus_call_jax"
    if torch_xla_active:
        return "xla_dense_fallback"
    return "non_xla_fallback"
