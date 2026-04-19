"""Compiled AdamW policy for shape-diverse training runs.

What it is: a public-safe receipt of the MegaCpp POC decision for when the
fused AdamW step is compiled dynamically and when it stays eager.

Why it exists: optimizer code sees a large mix of parameter shapes, and a
static compile plan can burn through recompile limits on the first step.

What problem it solves: it documents why the optimizer compile path is lazy,
dynamic, and disabled on TPU/XLA-style lanes.
"""

from __future__ import annotations


def compiled_adamw_policy(*, device_type: str, compile_adamw: bool, param_shape_families: int) -> dict[str, object]:
    should_compile = device_type == "cuda" and compile_adamw
    mode = "dynamic_compile" if should_compile else "eager"
    return {
        "device_type": device_type,
        "optimizer_mode": mode,
        "should_compile": should_compile,
        "dynamic": should_compile,
        "why": {
            "dynamic_compile": "one dynamic optimizer graph avoids per-shape recompiles across many parameter families",
            "eager": "TPU/XLA or explicitly disabled lanes let the native runtime handle optimization without inductor",
        }[mode],
        "shape_family_count": param_shape_families,
    }
