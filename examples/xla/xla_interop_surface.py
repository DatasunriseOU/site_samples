"""Public-safe TPU/XLA interop surface.

This sample demonstrates how a publication excerpt can talk about torch_xla,
libtpu, JAX-side helpers, and Pallas-like kernel decisions without exposing any
private environment identifiers.
"""


def build_runtime_stack() -> dict:
    return {
        "frontend": "torch_xla",
        "runtime_api": "pjrt",
        "runtime_impl": "libtpu",
        "shape_policy": "static-first",
    }


def choose_kernel_path(*, needs_explicit_tiling: bool, dense_mask: bool, dynamic_shapes: bool) -> str:
    if dynamic_shapes:
        return "stay_in_xla"
    if needs_explicit_tiling or dense_mask:
        return "pallas_like_kernel"
    return "stay_in_xla"


def summarize_interop_contract() -> list[str]:
    stack = build_runtime_stack()
    return [
        f"{stack['frontend']} is the application frontend.",
        f"{stack['runtime_impl']} is recorded as runtime provenance behind {stack['runtime_api']}.",
        "JAX interactions stay at the helper or reference-taxonomy layer unless the model is authored for JAX.",
        "Kernel escape hatches should be justified by a concrete hot loop, not by backend fashion.",
    ]
