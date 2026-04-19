"""Apply the MegaCpp POC's compile-safety environment switches for CUDA bringup.

This example shows the small runtime policy layer around compile. It exists to
reduce known failure modes during multi-GPU compile startup instead of changing
the whole training recipe.
"""

from __future__ import annotations

import os


def configure_compile_runtime_env(args) -> dict[str, str]:
    """Return the environment changes used by the MegaCpp POC compile path.

    The MegaCpp POC keeps this logic narrow: disable compile only for explicit eager
    mode or unstable lanes, and otherwise tune Inductor/NCCL settings so cold
    compile is less likely to fail before step 0.
    """

    updates: dict[str, str] = {}
    fsdp_disable_compile = bool(getattr(args, "fsdp_cuda", False) and getattr(args, "fsdp_no_compile", False))

    if getattr(args, "no_compile", False) or fsdp_disable_compile:
        updates["MEGACPP_NO_COMPILE"] = "1"
    elif getattr(args, "no_compile_optimizer", False):
        updates["MEGACPP_NO_OPTIMIZER_COMPILE"] = "1"

    if not getattr(args, "no_compile", False) and not fsdp_disable_compile:
        updates.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM", "1")
        updates.setdefault("TRITON_DEFAULT_NUM_STAGES", "2")
        updates.setdefault("TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE", "0")
        updates.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_SUBPROC_RESULT_TIMEOUT", "120")

    if int(getattr(args, "_cuda_compile_retry_attempt", 0) or 0) > 0:
        updates["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    for key, value in updates.items():
        os.environ[key] = value
    return updates
