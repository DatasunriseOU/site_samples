"""Mamba scan compile wrapper sample.

This example shows how a Triton-backed scan kernel was wrapped so the rest of
the block could still live inside ``torch.compile``. It exists because the raw
scan autograd path touched storage details that Dynamo/FakeTensor could not
model safely.

What problem it solves: it makes the scan opaque, provides fake outputs with
the right shapes, and keeps the surrounding compiled block from falling back to
eager just because one custom kernel has compile-hostile internals.
"""

from __future__ import annotations

import math

import torch


def normalize_cuda_backward_operands(
    dout: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    force_cuda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror the MegaCpp POC dtype-normalization rule for backward-only operands."""
    if not (force_cuda or dout.device.type == "cuda"):
        return A, B, C
    if dout.dtype not in (torch.float16, torch.bfloat16):
        return A, B, C
    scan_dtype = dout.dtype
    B_bwd = B.to(scan_dtype) if B.is_floating_point() and B.dtype != scan_dtype else B
    C_bwd = C.to(scan_dtype) if C.is_floating_point() and C.dtype != scan_dtype else C
    return A, B_bwd, C_bwd


def mamba_scan_fake_contract(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the MegaCpp POC fake-kernel output shapes used by compile."""
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[3]
    nchunks = math.ceil(seqlen / chunk_size)
    out = x.new_empty(batch, seqlen, nheads, headdim)
    out_x = x.new_empty(0)
    dt_out = dt.new_empty(batch, nheads, nchunks, chunk_size)
    dA_cumsum = dt.new_empty(batch, nheads, nchunks, chunk_size)
    states = x.new_empty(batch, nchunks, nheads, headdim, dstate)
    final_states = x.new_empty(batch, nheads, headdim, dstate)
    return out, out_x, dt_out, dA_cumsum, states, final_states


def compile_wrapper_notes() -> tuple[str, ...]:
    """Summarize the MegaCpp POC compile contract for the wrapper."""
    return (
        "keep the custom scan opaque to torch.compile",
        "provide fake outputs with exact runtime shapes",
        "normalize only transient backward operands instead of changing model weights",
        "force fake and real outputs to agree on contiguity when compile validates strides",
    )
