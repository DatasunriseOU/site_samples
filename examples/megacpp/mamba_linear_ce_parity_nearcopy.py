"""Near-copy MegaCpp POC example: Mamba linear-CE parity reproducer.

The real issue here is not a random loss mismatch. It is that one model path
can keep a plain column-parallel output layer while another uses the fused
linear-cross-entropy module that the language-model path expects.

This near-copy keeps the class and patch contract visible instead of collapsing
everything into one cross-entropy call.
"""

from __future__ import annotations

from dataclasses import dataclass


class ColumnParallelLinear:
    pass


class LinearCrossEntropyModule(ColumnParallelLinear):
    pass


@dataclass
class MockModel:
    name: str
    output_layer: ColumnParallelLinear
    fuse_linear_cross_entropy: bool = False


def build_gpt_model() -> MockModel:
    return MockModel(name="gpt", output_layer=LinearCrossEntropyModule())


def build_mamba_model() -> MockModel:
    return MockModel(name="mamba", output_layer=ColumnParallelLinear())


def apply_fix(mamba_model: MockModel) -> None:
    if not isinstance(mamba_model.output_layer, ColumnParallelLinear):
        raise TypeError("unexpected output_layer surface")
    mamba_model.output_layer.__class__ = LinearCrossEntropyModule
    mamba_model.fuse_linear_cross_entropy = True


def inspect_output_layer_classes() -> dict[str, str]:
    gpt = build_gpt_model()
    mamba = build_mamba_model()
    before = {
        "gpt": type(gpt.output_layer).__name__,
        "mamba_before": type(mamba.output_layer).__name__,
    }
    apply_fix(mamba)
    before["mamba_after"] = type(mamba.output_layer).__name__
    before["fuse_linear_cross_entropy"] = str(mamba.fuse_linear_cross_entropy)
    return before
