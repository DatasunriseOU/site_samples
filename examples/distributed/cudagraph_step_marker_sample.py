"""CUDA graph step-marker excerpt.

This example captures the MegaCpp POC runtime seam around `cudagraph_mark_step_begin`.
It exists because CUDA graph replay only stays correct when the runtime marks
the start of each new step before replayed work begins.

The problem it solves is graph reuse across step boundaries. Without the marker,
stateful step transitions can become unsafe or silently fall back.
"""

from __future__ import annotations

import torch


def begin_cudagraph_step(*, startup_trace: bool = False) -> None:
    """Public-safe excerpt of the MegaCpp POC step-boundary helper."""

    if startup_trace:
        print("[startup] gpt.forward cudagraph_mark_step_begin_enter", flush=True)
    torch.compiler.cudagraph_mark_step_begin()
    if startup_trace:
        print("[startup] gpt.forward cudagraph_mark_step_begin_done", flush=True)


def cudagraph_notes() -> tuple[str, ...]:
    return (
        "The marker belongs at the forward-step boundary, not deep inside a submodule.",
        "The MegaCpp POC keeps optional startup tracing so graph activation can be verified during bring-up.",
        "This helper is about correctness first; any speed win only matters if replay stays valid.",
    )
