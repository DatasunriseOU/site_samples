"""Public-safe pipeline residency sketch."""

from __future__ import annotations


def activation_residency(
    *, stages: int, microbatches: int, checkpoint_every_stage: bool = False, overlap: bool = True
) -> dict[str, int | bool]:
    warmup_resident = min(stages - 1, microbatches) if stages > 1 else min(1, microbatches)
    steady_resident = min(stages, microbatches) if overlap else max(1, min(stages, microbatches) - 1)
    if checkpoint_every_stage:
        steady_resident = max(1, steady_resident // 2)
    return {
        "stages": stages,
        "microbatches": microbatches,
        "overlap": overlap,
        "checkpoint_every_stage": checkpoint_every_stage,
        "warmup_resident_microbatches": warmup_resident,
        "steady_resident_microbatches": steady_resident,
    }
