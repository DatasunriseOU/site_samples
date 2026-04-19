"""Public TPU/XLA configuration sample used in publication references.

This file keeps only public-safe concepts: XLA flag bundles, framework/runtime
ownership boundaries, and compile-shape discipline.
"""

XLA_PROFILE = {
    "precision": "bf16",
    "compile_cache": "enabled",
    "spmd_mode": True,
    "shape_guard": "strict",
    "runtime": "pjrt",
    "frontend": "torch_xla",
}


XLA_RUNTIME_NOTES = {
    "torch_xla": "Import after runtime flags are fixed so tracing sees the intended environment.",
    "libtpu": "Treat as runtime provenance to record, not as an application-level tuning knob.",
    "jax_interop": "Keep JAX-side helpers isolated from PyTorch/XLA execution boundaries.",
    "pallas": "Reach for explicit tile control only when default XLA lowering leaves a clear hot path.",
}


def summarize_xla_profile(profile: dict) -> str:
    enabled = "on" if profile.get("spmd_mode") else "off"
    return (
        f"frontend={profile['frontend']} runtime={profile['runtime']} "
        f"precision={profile['precision']} spmd={enabled}"
    )


def build_public_xla_env() -> dict:
    """Return a public-safe environment sketch for TPU/XLA examples.

    The values are intentionally generic: they describe category choices rather
    than private fleet or host details.
    """

    return {
        "PJRT_DEVICE": "TPU",
        "XLA_USE_BF16": "1",
        "XLA_SPMD": "1",
        "LIBTPU_INIT_ARGS": "--xla_tpu_enable_latency_hiding_scheduler=true",
    }


def describe_framework_split() -> list[str]:
    return [
        "torch_xla owns PyTorch tracing, graph capture, and SPMD annotations.",
        "libtpu belongs to the TPU runtime provenance surface.",
        "JAX utilities can inform flag taxonomy without becoming the execution frontend.",
        "Pallas-style kernels are a narrow escape hatch for TPU hot loops, not the default path.",
    ]
