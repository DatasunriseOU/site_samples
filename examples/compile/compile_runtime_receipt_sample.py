"""Compile/runtime interaction receipt.

What it is: a compact public-safe summary of the main compile toggles that the
MegaCpp POC training runtime has to reconcile before a lane starts.

Why it exists: compile, CUDA graph capture, dynamic batch, and distributed
wrappers are not independent flags even though the CLI makes them look that way.

What problem it solves: it turns a pile of booleans into one readable runtime
receipt that explains which path will actually execute.
"""

from __future__ import annotations


def summarize_compile_runtime_lane(*, device_type: str, regional_compile: bool, compile_enabled: bool, dynamic_batch: bool, use_cuda_graphs: bool, moe_enabled: bool) -> dict[str, object]:
    warmup_enabled, warmup_reason = (False, "compile_disabled")
    if compile_enabled:
        if device_type == "cuda" and regional_compile and moe_enabled:
            warmup_enabled, warmup_reason = (False, "lazy_compile_after_skip")
        elif device_type == "cuda":
            warmup_enabled, warmup_reason = (True, "explicit_cuda_warmup")
        else:
            warmup_enabled, warmup_reason = (False, "non_cuda_compile")

    return {
        "device_type": device_type,
        "compile_enabled": compile_enabled,
        "regional_compile": regional_compile,
        "dynamic_batch": dynamic_batch,
        "use_cuda_graphs": use_cuda_graphs and device_type == "cuda",
        "moe_enabled": moe_enabled,
        "compile_warmup": {
            "enabled": warmup_enabled,
            "reason": warmup_reason,
        },
    }
