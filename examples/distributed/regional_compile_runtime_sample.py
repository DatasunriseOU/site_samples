"""Regional compile runtime sample.

This example shows two compile-safety patterns used in the hot training path:
explicit-keyword wrappers and pre-resolved kwargs. It exists because Python's
dynamic call machinery causes graph breaks inside `regional_compile` regions.
The problem it solves is avoiding per-microbatch recompiles that destroy GPU
throughput.
"""

from __future__ import annotations

from typing import Any


def build_flash_call_kwargs(
    *,
    window_size: tuple[int, int],
    doc_ids: object,
    softcap: float,
    attention_validity: object | None,
    flash_accepts_attention_validity: bool,
) -> dict[str, Any]:
    """Pre-resolve optional kwargs before the donor flash-attention call.

    The donor comment explains why this matters: conditionally building a dict
    and then unpacking it with `**kwargs` triggers a `CALL_FUNCTION_EX` graph
    break under `regional_compile`. That caused per-step recompiles on a live
    NAM56R-class lane, so the hot path resolves optional arguments before the
    call site.
    """

    call_kwargs: dict[str, Any] = {
        "causal": True,
        "window_size": window_size,
        "doc_ids": doc_ids,
        "softcap": softcap,
    }
    if attention_validity is not None and flash_accepts_attention_validity:
        call_kwargs["attention_validity"] = attention_validity
    return call_kwargs


def checkpoint_wrapper_contract() -> tuple[str, ...]:
    """Return the donor's compile contract for recompute wrappers.

    The donor replaces a cold `_maybe_recompute(fn, recompute, *args, **kwargs)`
    wrapper with an explicit-keyword helper because variadic argument unpacking
    causes a Dynamo `CALL_FUNCTION_EX` inside the compiled region.
    """

    return (
        "use named arguments for hot recompute wrappers inside regional_compile",
        "avoid variadic *args/**kwargs in the inner compiled path",
        "compile the downstream block, not the Python call-shape builder",
        "this keeps per-microbatch execution on one traced contract",
    )


def dynamic_batch_expectation() -> tuple[str, ...]:
    """Mirror the donor test expectation for dynamic regional compile lanes."""

    return (
        "compile once at a small batch size",
        "increase batch size on the same dynamic lane",
        "expect zero recompilations when the dynamic contract is compatible",
    )
