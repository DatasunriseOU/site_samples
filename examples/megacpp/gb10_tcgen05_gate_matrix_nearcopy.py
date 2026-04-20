"""Near-copy MegaCpp POC example: GB10 tcgen05 gate matrix.

This keeps the real contract closer to the internal GB10 receipts:
- the baseline arithmetic probe clears only the architecture gate;
- tcgen05-oriented probes hit additional metadata and signed-capability gates;
- reaching a deeper gate is not the same as proving tcgen05 runtime support.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateObservation:
    gate_id: int
    gate_name: str
    error_before_bypass: str
    bypass: str
    next_error: str | None


GATE_MATRIX = (
    GateObservation(
        gate_id=1,
        gate_name="ELF e_flags arch check",
        error_before_bypass="CUDA_ERROR_NO_BINARY_FOR_GPU",
        bypass="rewrite low 16 bits of e_flags",
        next_error="CUDA_ERROR_NOT_FOUND",
    ),
    GateObservation(
        gate_id=2,
        gate_name="reserved weak UND symbols",
        error_before_bypass="CUDA_ERROR_NOT_FOUND",
        bypass="rewrite UND symbols to ABS/GLOBAL",
        next_error="CUDA_ERROR_INVALID_IMAGE",
    ),
    GateObservation(
        gate_id=3,
        gate_name=".nv.info capability records",
        error_before_bypass="CUDA_ERROR_INVALID_IMAGE",
        bypass="zero the per-kernel capability records",
        next_error="CUDA_ERROR_INVALID_IMAGE",
    ),
    GateObservation(
        gate_id=4,
        gate_name=".nv.capmerc signed capability metadata",
        error_before_bypass="CUDA_ERROR_INVALID_IMAGE",
        bypass="no publication-safe bypass; integrity-protected capability data",
        next_error=None,
    ),
)


def first_blocking_gate(*, arch_rewritten: bool, symbols_patched: bool, nvinfo_zeroed: bool) -> GateObservation:
    if not arch_rewritten:
        return GATE_MATRIX[0]
    if not symbols_patched:
        return GATE_MATRIX[1]
    if not nvinfo_zeroed:
        return GATE_MATRIX[2]
    return GATE_MATRIX[3]


def publication_safe_verdict(*, arch_rewritten: bool, symbols_patched: bool, nvinfo_zeroed: bool) -> str:
    gate = first_blocking_gate(
        arch_rewritten=arch_rewritten,
        symbols_patched=symbols_patched,
        nvinfo_zeroed=nvinfo_zeroed,
    )
    if gate.gate_id < 4:
        return f"Still blocked at gate {gate.gate_id}: {gate.gate_name}."
    return (
        "The tcgen05 path reached the signed-capability boundary. That is stronger "
        "than a baseline arch patch, but it is still not a tcgen05 execute receipt."
    )


if __name__ == "__main__":
    print(
        publication_safe_verdict(
            arch_rewritten=True,
            symbols_patched=True,
            nvinfo_zeroed=True,
        )
    )
