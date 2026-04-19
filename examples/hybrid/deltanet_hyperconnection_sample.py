"""DeltaNet and hyper-connection block selection sample.

What it is: a public-safe receipt for the real block choices that switch a
layer between attention, Mamba, DeltaNet, and multi-stream hyper-connections.

Why it exists: the hybrid stack is not one repeated attention block. Different
layers can change sequence mixer type and stream topology.

What problem it solves: it shows which config knobs move a layer onto the
DeltaNet or mHC path instead of hiding those decisions inside one opaque model
name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridLayerChoice:
    layer_kind: str
    uses_sparse_attention: bool
    uses_engram: bool
    uses_mhc: bool
    mhc_dynamic: bool
    mhc_fused_ops: bool


def choose_hybrid_layer(config: object, *, layer_symbol: str, use_engram: bool, use_mhc: bool) -> HybridLayerChoice:
    symbol = layer_symbol.upper()
    if symbol == "M":
        layer_kind = "mamba"
    elif symbol == "D":
        layer_kind = "deltanet"
    else:
        layer_kind = "attention"

    uses_sparse_attention = layer_kind == "attention" and bool(getattr(config, "dsa_enabled", False))

    return HybridLayerChoice(
        layer_kind=layer_kind,
        uses_sparse_attention=uses_sparse_attention,
        uses_engram=use_engram and layer_kind == "attention",
        uses_mhc=use_mhc and layer_kind == "attention",
        mhc_dynamic=bool(getattr(config, "mhc_dynamic", False)),
        mhc_fused_ops=bool(getattr(config, "mhc_fused_ops", False)),
    )


def describe_hyperconnection_surface(config: object) -> dict[str, object]:
    return {
        "n_streams": int(getattr(config, "mhc_n_streams", 4)),
        "sinkhorn_iters": int(getattr(config, "mhc_sinkhorn_iters", 20)),
        "dynamic": bool(getattr(config, "mhc_dynamic", False)),
        "dynamic_mode": str(getattr(config, "mhc_dynamic_mode", "maxtext")),
        "fused_ops": bool(getattr(config, "mhc_fused_ops", False)),
        "meaning": "mHC replaces one residual stream with several streams and then mixes them back with constrained weights",
    }
