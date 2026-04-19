"""Opaque custom-op wrapper for a non-compile-friendly kernel.

What it is: a public-safe sketch of the MegaCpp POC pattern that hides a kernel
behind a custom op so `torch.compile` can still optimize the surrounding block.

Why it exists: some fused kernels expose autograd or backend details that fake
tensors and compile-time tracing cannot safely step through.

What problem it solves: it keeps the fragile kernel opaque while letting the
rest of the model block stay on the compiled fast path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpaqueCompileWrapperPlan:
    custom_op_name: str
    needs_fake_impl: bool
    supports_forward: bool
    supports_backward: bool
    expected_use: str


def build_opaque_compile_wrapper_plan() -> OpaqueCompileWrapperPlan:
    return OpaqueCompileWrapperPlan(
        custom_op_name="megacpp::opaque_kernel",
        needs_fake_impl=True,
        supports_forward=True,
        supports_backward=True,
        expected_use="keep one fragile fused kernel opaque so the surrounding compiled block can still lower cleanly",
    )
