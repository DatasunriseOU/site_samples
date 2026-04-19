"""Near-copy MegaCpp POC example: Megatron Hopper FLCE contract.

This sample keeps the public contract visible: Hopper-ready fused linear cross
entropy is not just a math choice. It is an output-layer and runtime-path
choice that must stay aligned with the model's loss path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HopperFlcePath:
    name: str
    fused_loss_path: bool
    output_layer_contract_ok: bool
    hopper_ready: bool
    notes: str


def plain_output_layer_path() -> HopperFlcePath:
    return HopperFlcePath(
        name="plain_column_parallel",
        fused_loss_path=False,
        output_layer_contract_ok=False,
        hopper_ready=False,
        notes="shape-compatible output layer, but not the fused CE contract Hopper path expects",
    )


def fused_output_layer_path() -> HopperFlcePath:
    return HopperFlcePath(
        name="fused_linear_cross_entropy",
        fused_loss_path=True,
        output_layer_contract_ok=True,
        hopper_ready=True,
        notes="output layer and loss path are aligned for fused Hopper execution",
    )


def compare_hopper_flce_paths() -> dict[str, HopperFlcePath]:
    return {
        "plain_path": plain_output_layer_path(),
        "fused_path": fused_output_layer_path(),
    }
