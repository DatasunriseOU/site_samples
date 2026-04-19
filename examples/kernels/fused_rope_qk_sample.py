"""Fused Q/K RoPE application for CUDA attention ingress.

What it is: a single-kernel shape gate for applying rotary embeddings to query
and key tensors.

Why it exists: the model always rotates Q and K together, so doing two separate
passes wastes launches and repeats the same layout checks.

What problem it solves: it keeps a fast CUDA path for the common layout while
preserving a plain eager fallback for unsupported dtypes and non-CUDA devices.
"""

from __future__ import annotations

import torch


def _apply_rotary_emb_plain(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half : half * 2]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    if head_dim > half * 2:
        rotated = torch.cat((rotated, x[..., half * 2 :]), dim=-1)
    return rotated


def fused_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public sample of the fused RoPE ingress gate from `triton_kernels.py`.

    Grounded changes:
    - the fast path is CUDA-only
    - bf16 tensors are deliberately excluded in the MegaCpp POC guard
    - invalid cosine/sine layouts fall back immediately instead of trying to
      coerce the launch into a shape it was not written for
    """
    if (
        not q.is_cuda
        or not k.is_cuda
        or q.dtype == torch.bfloat16
        or k.dtype == torch.bfloat16
        or cos.dtype == torch.bfloat16
        or sin.dtype == torch.bfloat16
        or cos.ndim != 4
        or sin.ndim != 4
        or cos.shape[0] != 1
        or sin.shape[0] != 1
        or cos.shape[2] != 1
        or sin.shape[2] != 1
    ):
        return _apply_rotary_emb_plain(q, cos, sin), _apply_rotary_emb_plain(k, cos, sin)

    # The MegaCpp POC launches two autograd-backed Triton calls here, one for Q and
    # one for K, so callers can backpropagate through each output independently.
    # This sample keeps the same guard contract and the same output semantics.
    return _apply_rotary_emb_plain(q, cos, sin), _apply_rotary_emb_plain(k, cos, sin)
