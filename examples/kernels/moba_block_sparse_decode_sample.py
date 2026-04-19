"""MoBA block-sparse decode sample.

What it is: a public-safe receipt for the MegaCpp POC blockized sparse decode
backend resolver.

Why it exists: the runtime can request a FLASH-style FlexAttention backend for
block-sparse execution and still legitimately fall back to Triton if the narrow
sparse kernel path is unavailable.

What problem it solves: it separates requested backend from actual runtime
backend so sparse receipts do not accidentally claim dense FA4 execution.
"""

from __future__ import annotations


def normalize_requested_flex_backend(requested_backend: str | None) -> str:
    if requested_backend in {None, "", "auto"}:
        return "triton"
    value = str(requested_backend).strip().lower()
    if value not in {"triton", "flash"}:
        raise ValueError(f"unsupported block-sparse backend {requested_backend!r}")
    return value


def resolve_block_sparse_execution_backend(
    *,
    requested_backend: str | None,
    flash_runtime_available: bool,
) -> tuple[str, str, str]:
    requested = normalize_requested_flex_backend(requested_backend)
    if requested == "flash" and not flash_runtime_available:
        return requested, "triton", "flash_cute_runtime_unavailable"
    return requested, requested, ""


def build_block_sparse_runtime_telemetry(
    *,
    requested_backend: str | None,
    flash_runtime_available: bool,
) -> dict[str, object]:
    requested, actual, fallback_reason = resolve_block_sparse_execution_backend(
        requested_backend=requested_backend,
        flash_runtime_available=flash_runtime_available,
    )
    return {
        "semantic_branch": "block_sparse",
        "router_type": "moba",
        "requested_backend": requested,
        "requested_backend_source": "runtime_contract",
        "actual_backend": actual,
        "runtime_mode": "blockized_sparse_cuda",
        "fallback_reason": fallback_reason,
        "note": (
            "The sparse semantics do not change when the execution backend falls "
            "back from FLASH-style FlexAttention to Triton."
        ),
    }
