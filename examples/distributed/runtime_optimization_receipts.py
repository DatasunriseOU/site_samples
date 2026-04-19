"""Runtime optimization receipts sample.

This example distills the donor's validated runtime fixes into a public-safe
table. It exists because optimization work is only useful when it is tied to a
real failure mode and an observed outcome. The problem it solves is separating
measured runtime wins from generic tuning advice.
"""

from __future__ import annotations


RUNTIME_RECEIPTS = (
    {
        "problem": "FSDP2 record_stream over-allocation caused a memory leak",
        "fix": "set TORCH_NCCL_AVOID_RECORD_STREAMS=1 and run gc.collect() every 50 steps",
        "impact": "OOM at about 60 steps before the fix; stable for 100+ steps after the fix",
    },
    {
        "problem": "mHC compile graph explosion drove memory to 105GB",
        "fix": "skip that surface from compile and use group recompute",
        "impact": "the receipt records this as the fix for the 105GB graph explosion",
    },
    {
        "problem": "CUDA compile retry launched an extra process and doubled memory pressure",
        "fix": "set --cuda_compile_retry_limit=0",
        "impact": "the donor receipt classifies this as the fix for double-process OOM",
    },
    {
        "problem": "FP8 activation hooks cost about 35% overhead",
        "fix": "apply a 16K size threshold, exclude selected layers, and use fused packing",
        "impact": "the donor receipt attributes the 35% overhead fix to those three changes",
    },
)


def explain_runtime_receipts() -> tuple[str, ...]:
    """Summarize why these receipts matter for GPU runtime work."""

    return (
        "these are grounded fixes from validated runtime notes, not generic advice",
        "each row links one concrete failure mode to the change that addressed it",
        "the list is useful when deciding whether a new slowdown is compile, memory, or hook related",
    )
