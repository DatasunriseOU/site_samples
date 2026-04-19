"""Compile warmup policy for CUDA regional-compile lanes.

What it is: a public-safe receipt of the MegaCpp POC rule that decides whether
an explicit compile-warmup pass should run before normal training.

Why it exists: some CUDA regional-compile plus MoE lanes looked healthy on
paper but stalled or crashed when explicit warmup was forced too early.

What problem it solves: it makes the skip policy explicit so readers can see
why warmup is disabled on certain runtime combinations instead of blaming all
compile issues on one framework version.
"""

from __future__ import annotations


def should_run_compile_warmup(*, device_type: str, regional_compile: bool, moe_enabled: bool, fsdp_cuda: bool, fsdp_no_compile: bool, no_compile_warmup: bool) -> tuple[bool, str]:
    if no_compile_warmup:
        return False, "user_disabled"
    if fsdp_cuda and fsdp_no_compile:
        return False, "fsdp_cuda_eager_mode"
    if device_type != "cuda":
        return False, "non_cuda_lane"
    if regional_compile and moe_enabled:
        return False, "cuda_regional_compile_plus_moe"
    return True, "explicit_warmup_allowed"


def summarize_compile_warmup_policy(**kwargs: object) -> dict[str, object]:
    enabled, reason = should_run_compile_warmup(
        device_type=str(kwargs.get("device_type", "cuda")),
        regional_compile=bool(kwargs.get("regional_compile", False)),
        moe_enabled=bool(kwargs.get("moe_enabled", False)),
        fsdp_cuda=bool(kwargs.get("fsdp_cuda", False)),
        fsdp_no_compile=bool(kwargs.get("fsdp_no_compile", False)),
        no_compile_warmup=bool(kwargs.get("no_compile_warmup", False)),
    )
    return {
        "warmup_enabled": enabled,
        "reason": reason,
        "notes": {
            "cuda_regional_compile_plus_moe": "skip explicit warmup and let lazy compile populate caches during real work",
            "fsdp_cuda_eager_mode": "compile was disabled upstream for this lane",
            "explicit_warmup_allowed": "safe to run the dedicated warmup pass",
        },
    }
