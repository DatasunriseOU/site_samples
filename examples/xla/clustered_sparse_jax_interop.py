"""Clustered sparse JAX interop contract sample.

What it is: a MegaCpp POC-based excerpt of the narrow helper layer that prepares a
clustered sparse TPU path to call JAX/Pallas kernels from a PyTorch model.
Why it exists: the sparse pipeline evolved across multiple kernel signatures,
so the wrapper had to detect which exact-mask, doc-id, and valid-prefix kwargs
were actually supported instead of assuming every phase accepted the same API.
What problem it solves: it keeps the bridge tolerant to narrow interop changes
without silently claiming support for mask contracts that the downstream kernel
does not really implement.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType


def _import_optional_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _supports_kwarg(fn, kwarg: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return kwarg in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _explicit_param_names(fn) -> set[str]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, param in params.items()
        if param.kind != inspect.Parameter.VAR_KEYWORD
    }


def _first_explicit_kwarg(fn, names: tuple[str, ...]) -> str | None:
    params = _explicit_param_names(fn)
    for name in names:
        if name in params:
            return name
    return None


def _call_with_optional_kwargs(fn, *args, **kwargs):
    filtered = {
        key: value for key, value in kwargs.items() if _supports_kwarg(fn, key)
    }
    return fn(*args, **filtered)


def _ensure_jax():
    """Lazy-load JAX and return (jax, jnp) or raise ImportError."""
    try:
        import jax

        jnp = _import_optional_module("jax.numpy")
        return jax, jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for the clustered sparse TPU bridge. "
            "Install jax and jaxlib for this path."
        ) from exc


def _ensure_sparse_pallas():
    """Lazy-load sparse-pallas pipeline functions or raise ImportError."""
    raise ImportError(
        "The MegaCpp POC bridge expected a sparse-pallas package on the Python path. "
        "This public sample documents the contract only and does not ship that package."
    )


def _ensure_call_jax():
    """Lazy-load call_jax from torch_xla or return None for fallback paths."""
    try:
        xla_builder = _import_optional_module("torch_xla.core.xla_builder")
        return getattr(xla_builder, "call_jax")
    except ImportError:
        return None


_MASK_CONTRACT_PARAM_NAMES = (
    "exact_mask_contract",
    "mask_contract",
    "attention_mask_contract",
    "mask_info",
    "mask_meta",
    "exact_mask",
)
_WINDOW_PARAM_NAMES = ("window_size", "local_window")
_DOC_IDS_PARAM_NAMES = (
    "doc_ids",
    "q_doc_ids",
    "kv_doc_ids",
    "segment_ids",
    "q_segment_ids",
    "kv_segment_ids",
)
_VALID_PREFIX_PARAM_NAMES = (
    "valid_token_counts",
    "q_valid_token_counts",
    "kv_valid_token_counts",
    "attention_validity",
)


def _phase_supports_local_window(fn) -> bool:
    return _first_explicit_kwarg(fn, _MASK_CONTRACT_PARAM_NAMES) is not None or (
        _first_explicit_kwarg(fn, _WINDOW_PARAM_NAMES) is not None
    )


def _phase_supports_doc_ids(fn) -> bool:
    return _first_explicit_kwarg(fn, _MASK_CONTRACT_PARAM_NAMES) is not None or (
        _first_explicit_kwarg(fn, _DOC_IDS_PARAM_NAMES) is not None
    )


def _phase_supports_valid_prefix(fn) -> bool:
    return _first_explicit_kwarg(fn, _MASK_CONTRACT_PARAM_NAMES) is not None or (
        _first_explicit_kwarg(fn, _VALID_PREFIX_PARAM_NAMES) is not None
    )


def _normalize_local_window(window_size) -> int:
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        return 0
    left, right = window_size
    if right is not None and int(right) > 0:
        return max(int(left), 0) if left is not None else 0
    if left is None:
        return 0
    return max(int(left), 0)


@dataclass(frozen=True)
class ThreePhaseExactMaskSupport:
    local_window: bool = False
    general_doc_ids: bool = False
    valid_prefix: bool = False
    phase1_has_contract: bool = False
    phase2_has_contract: bool = False
    phase3_has_contract: bool = False


def get_three_phase_exact_mask_support(*, use_fused_scoring: bool) -> ThreePhaseExactMaskSupport:
    """Inspect sparse-phase signatures and only trust explicit support."""
    try:
        sparse_pallas = _ensure_sparse_pallas()
    except ImportError:
        return ThreePhaseExactMaskSupport()

    importance_scoring_pipeline = sparse_pallas[1]
    importance_scoring_pipeline_fused = sparse_pallas[2]
    union_selection_pipeline = sparse_pallas[3]
    sparse_attention_fn = sparse_pallas[4]
    phase1_fn = (
        importance_scoring_pipeline_fused
        if use_fused_scoring
        else importance_scoring_pipeline
    )

    phase1_has_contract = _first_explicit_kwarg(phase1_fn, _MASK_CONTRACT_PARAM_NAMES) is not None
    phase2_has_contract = _first_explicit_kwarg(
        union_selection_pipeline, _MASK_CONTRACT_PARAM_NAMES
    ) is not None
    phase3_has_contract = _first_explicit_kwarg(
        sparse_attention_fn, _MASK_CONTRACT_PARAM_NAMES
    ) is not None

    return ThreePhaseExactMaskSupport(
        local_window=_phase_supports_local_window(phase1_fn)
        and _phase_supports_local_window(sparse_attention_fn),
        general_doc_ids=_phase_supports_doc_ids(phase1_fn)
        and _phase_supports_doc_ids(sparse_attention_fn),
        valid_prefix=_phase_supports_valid_prefix(phase1_fn)
        and _phase_supports_valid_prefix(sparse_attention_fn),
        phase1_has_contract=phase1_has_contract,
        phase2_has_contract=phase2_has_contract,
        phase3_has_contract=phase3_has_contract,
    )


def is_effectively_global_causal_window(window_size, seq_len: int) -> bool:
    """Treat only full-context causal windows as sparse-path-safe."""
    if window_size is None:
        return True
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        return False
    left, right = window_size
    full_left_context = (
        left is None or int(left) < 0 or int(left) >= max(int(seq_len) - 1, 0)
    )
    no_future_window = right is None or int(right) <= 0
    return full_left_context and no_future_window
