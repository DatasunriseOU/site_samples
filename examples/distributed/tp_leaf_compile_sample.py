"""Compile leaf modules only after tensor-parallel sharding has landed.

This example shows the MegaCpp POC's CUDA TP compile rule. It exists because wrapping
an entire block too early can hide the exact leaf modules that the parallel
planner still needs to rewrite.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def apply_cuda_tp_leaf_compile(model, compile_kwargs: dict) -> int:
    """Compile selected attention and MLP leaves after TP/SP rewrite.

    The include filter and block limit match the MegaCpp POC knobs used to reduce the
    first compile blast radius while bringing a new lane up.
    """

    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if not isinstance(blocks, nn.ModuleList):
        return 0

    compiled = 0
    include_filters = tuple(part.strip() for part in os.environ.get("MEGACPP_TP_COMPILE_INCLUDE", "").split(",") if part.strip())
    block_limit_raw = os.environ.get("MEGACPP_TP_COMPILE_BLOCK_LIMIT", "").strip()
    block_limit = int(block_limit_raw) if block_limit_raw else None

    def _wanted(name: str) -> bool:
        if not include_filters:
            return True
        return any(token in name for token in include_filters)

    def _compile_leaf(parent: object, attr: str, qualname: str) -> None:
        nonlocal compiled
        module = getattr(parent, attr, None)
        if not isinstance(module, nn.Module):
            return
        if hasattr(module, "_orig_mod") or not _wanted(qualname):
            return
        setattr(parent, attr, torch.compile(module, fullgraph=False, **compile_kwargs))
        compiled += 1

    def _compile_block_like(block: object, block_idx: int) -> None:
        inner = getattr(block, "block", block)
        attn = getattr(inner, "attn", None)
        mlp = getattr(inner, "mlp", None)
        prefix = f"transformer.h.{block_idx}"
        _compile_leaf(attn, "c_qkv", f"{prefix}.attn.c_qkv")
        _compile_leaf(attn, "c_proj", f"{prefix}.attn.c_proj")
        _compile_leaf(mlp, "c_fc", f"{prefix}.mlp.c_fc")
        _compile_leaf(mlp, "c_proj", f"{prefix}.mlp.c_proj")

    for block_idx, block in enumerate(blocks):
        if block_limit is not None and block_idx >= block_limit:
            break
        _compile_block_like(block, block_idx)

    return compiled
