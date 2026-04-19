"""Public excerpt.

Source repo: MegaCpp public samples
Source material: https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_interop_surface.py
Purpose: show a public-safe TPU/XLA ownership split around torch_xla, libtpu, and JAX-side helpers
Edited for clarity.
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
