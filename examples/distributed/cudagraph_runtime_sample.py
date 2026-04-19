"""CUDA Graph runtime sample.

This example shows how the training forward path marks the start of a new CUDA
graph step before running the model body. It exists because compiled GPU runs
can hit aliasing and replay issues when each step is not cleanly delimited. The
problem it solves is reducing Python dispatch overhead while keeping repeated
training steps graph-safe.
"""

from __future__ import annotations

import os

import torch


def maybe_mark_cudagraph_step_begin(*, training: bool, device: torch.device) -> bool:
    """Mirror the MegaCpp POC guard before calling cudagraph_mark_step_begin().

    The MegaCpp POC calls this at the start of `GPT.forward()` on CUDA training paths.
    Graph capture helps when repeated steps are shape-stable, but the runtime
    also needs a clear step boundary so replay does not observe stale aliases
    from patterns like `x = x + f(x)`.
    """

    if not training or device.type != "cuda":
        return False
    torch.compiler.cudagraph_mark_step_begin()
    return True


def apply_cudagraph_env_defaults(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return the MegaCpp POC's CUDA-graph-related environment defaults.

    Grounded MegaCpp POC note: the training launcher sets these defaults because
    Inductor can capture per-block compiled kernels as CUDA graphs. The MegaCpp POC
    comment records a measured `+4.5% throughput on bench3 H200:8 d20 FSDP`
    when `TORCHINDUCTOR_TRITON_CUDAGRAPHS=1` is active together with
    `TORCH_COMPILE_CUDAGRAPH_TREES=1`.
    """

    env = dict(os.environ if environ is None else environ)
    env.setdefault("TORCHINDUCTOR_TRITON_CUDAGRAPHS", "1")
    env.setdefault("TORCH_COMPILE_CUDAGRAPH_TREES", "1")
    return env


def cudagraph_constraints() -> tuple[str, ...]:
    """Summarize the grounded constraints that make CUDA graphs useful here."""

    return (
        "graphs help most when the hot training step repeats with stable shapes",
        "the forward path must mark a fresh graph step on CUDA training runs",
        "graph capture reduces Python dispatch overhead but tightens aliasing expectations",
        "shape churn and eager-only side paths reduce the value of graph capture",
    )
