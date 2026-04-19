"""FIRE target-selection excerpt."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, Set, cast

import torch
import torch.nn as nn


class _HasWeight(Protocol):
    weight: torch.Tensor


class _TransformerLike(Protocol):
    wte: nn.Module
    h: Iterable[nn.Module]


class _ModelWithTransformer(Protocol):
    transformer: _TransformerLike


class _AttnLike(Protocol):
    c_qkv: object


def _require_weight_module(module: object) -> _HasWeight:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected module with tensor weight, got {type(module).__name__}")
    return cast(_HasWeight, module)


def _require_parameter(tensor: torch.Tensor) -> nn.Parameter:
    if isinstance(tensor, nn.Parameter):
        return tensor
    raise TypeError(f"Expected nn.Parameter, got {type(tensor).__name__}")


def _require_transformer(model: nn.Module) -> _TransformerLike:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise TypeError("Expected model with transformer")
    return cast(_ModelWithTransformer, model).transformer


def _attn_module(block: nn.Module) -> object | None:
    return getattr(block, "attn", None)


def get_fire_targets(model: nn.Module, mode: str = "aggressive") -> Set[nn.Parameter]:
    """Topology-aware FIRE target selection using block.is_mamba."""
    targets: Set[nn.Parameter] = set()
    global_skips: Set[nn.Parameter] = set()

    if hasattr(model, "transformer"):
        transformer = _require_transformer(model)
        if hasattr(transformer, "wte"):
            global_skips.add(_require_parameter(_require_weight_module(transformer.wte).weight))
    if hasattr(model, "lm_head"):
        global_skips.add(_require_parameter(_require_weight_module(model.lm_head).weight))

    blocks = _require_transformer(model).h if hasattr(model, "transformer") else []

    for block in blocks:
        is_mamba = getattr(block, "is_mamba", False)

        if mode == "context_extension":
            if is_mamba:
                continue
            attn = _attn_module(block)
            if isinstance(attn, nn.Module):
                c_qkv = getattr(cast(_AttnLike, attn), "c_qkv", None)
                if c_qkv is not None:
                    c_qkv_weight = _require_weight_module(c_qkv).weight
                    if c_qkv_weight.dim() == 2:
                        targets.add(_require_parameter(c_qkv_weight))
                else:
                    for proj_name in ["c_q", "c_k"]:
                        proj = getattr(attn, proj_name, None)
                        if proj is not None:
                            proj_weight = _require_weight_module(proj).weight
                            if proj_weight.dim() == 2:
                                targets.add(_require_parameter(proj_weight))
        elif mode == "aggressive":
            for p in block.parameters():
                if p.dim() == 2 and p not in global_skips:
                    targets.add(p)

    return targets
