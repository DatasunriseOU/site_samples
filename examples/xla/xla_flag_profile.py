"""Canonical public-safe XLA sample derived from the internal TPU MegaCpp POCs.

The MegaCpp POC repo has two distinct XLA surfaces:
1. an XLA flag helper for pre-import runtime configuration.
2. an optimizer pattern that avoids graph-breaking
   scalar extraction.

This sample keeps a single public-safe module that documents those canonical
surfaces without carrying private launch policy, TPU fleet details, or forked
helper copies.
"""

from __future__ import annotations


XLA_PROFILE = {
    "frontend": "torch_xla",
    "runtime_api": "pjrt",
    "runtime_impl": "libtpu",
    "precision": "bf16",
    "spmd_mode": True,
    "shape_policy": "static-first",
    "compile_cache": "enabled",
}


XLA_RUNTIME_NOTES = {
    "torch_xla": "Apply runtime flags before importing torch_xla so libtpu sees the intended environment once.",
    "libtpu": "Treat runtime flags as pre-init configuration, not as post-import tuning knobs.",
    "optimizer": "Use tensor-backed step state for changing scalar values so the compiled graph stays stable.",
    "jax_interop": "Keep JAX helpers or Pallas-like kernels as narrow adjunct surfaces, not the primary execution frontend.",
}


def summarize_xla_profile(profile: dict) -> str:
    enabled = "on" if profile.get("spmd_mode") else "off"
    return (
        f"frontend={profile['frontend']} runtime={profile['runtime_api']} "
        f"precision={profile['precision']} spmd={enabled}"
    )


def build_public_xla_env() -> dict[str, str]:
    """Return a compact public-safe environment sketch.

    The names mirror the MegaCpp POC contract at a high level, but the values stay
    intentionally generic and publication-safe.
    """

    return {
        "PJRT_DEVICE": "TPU",
        "XLA_USE_BF16": "1",
        "XLA_NO_SPECIAL_SCALARS": "1",
        "LIBTPU_INIT_ARGS": "--xla_tpu_enable_latency_hiding_scheduler=true",
    }


def choose_kernel_path(*, needs_explicit_tiling: bool, dense_mask: bool, dynamic_shapes: bool) -> str:
    if dynamic_shapes:
        return "stay_in_xla"
    if needs_explicit_tiling or dense_mask:
        return "pallas_like_kernel"
    return "stay_in_xla"


def describe_framework_split() -> list[str]:
    return [
        "torch_xla owns PyTorch graph capture and SPMD-facing execution semantics.",
        "libtpu is the runtime implementation configured before torch_xla import.",
        "Optimizer step state should stay tensor-backed to avoid graph-breaking scalar extraction.",
        "Pallas-style or JAX-side helpers are justified only for concrete hot loops that default XLA lowering does not handle well.",
    ]
