"""TPU compile/runtime control sample.

This shows the narrow TPU knobs that changed compile behavior in practice:
persistent cache location, barrier-split execution, startup retry limit, and
compiled versus eager optimizer stepping. The problem it solves is repeated
step-0 compile pain when the same cold graph is rebuilt without stable cache
and fallback controls.
"""

from __future__ import annotations

import os


def build_tpu_compile_runtime_controls(
    *,
    cache_dir: str = "/data/.xla_cache",
    disable_cache: bool = False,
    xla_barrier_every_n_layers: int = 0,
    no_compile_optimizer: bool = False,
    xla_compile_retry_limit: int = 2,
    xla_flag_profile: str = "none",
) -> dict[str, object]:
    env: dict[str, str] = {}
    if not disable_cache:
        env["XLA_COMPILATION_CACHE_DIR"] = cache_dir
    env["PJRT_DEVICE"] = "TPU"
    env["XLA_NO_SPECIAL_SCALARS"] = "1"

    execution_mode = "barrier_split" if xla_barrier_every_n_layers > 0 else "compiled_fwd_bwd"
    optimizer_mode = "eager_optimizer" if no_compile_optimizer else "compiled_optimizer"

    return {
        "env": env,
        "compile": {
            "execution_mode": execution_mode,
            "xla_barrier_every_n_layers": xla_barrier_every_n_layers,
            "optimizer_mode": optimizer_mode,
            "xla_compile_retry_limit": xla_compile_retry_limit,
            "xla_flag_profile": xla_flag_profile,
        },
        "notes": [
            "Startup retry is intentionally limited to step 0 and the first post-step0 compile window.",
            "Barrier-split mode inserts mark_step barriers instead of wrapping forward/backward in one compile region.",
            "Persistent cache reuse reduces cold-start recompiles when the graph contract stays stable.",
        ],
    }


def cache_enabled_from_env() -> bool:
    return not bool(os.environ.get("PUBLIC_XLA_CACHE_DISABLED"))
