"""MegaCpp POC example: baseline arch-patch probe for GB10.

What this solves in simple words:
- it keeps the narrow positive GB10 result separate from the stronger claims
  around tcgen05 or TMEM;
- it shows that rewriting the architecture field proves a loader/runtime fact,
  not full datacenter-path parity.
"""

from __future__ import annotations

from dataclasses import dataclass


SM_FLAGS = {
    "sm_100a": 0x6402,
    "sm_121a": 0x7902,
}


@dataclass(frozen=True)
class BaselineProbeReceipt:
    source_arch: str
    target_arch: str
    original_low16: int
    patched_low16: int
    load_status: str
    launch_status: str
    sync_status: str
    observed_output: tuple[int, ...]
    expected_output: tuple[int, ...]


def rewrite_arch_low16(e_flags: int, target_arch: str) -> int:
    """Rewrite only the low 16 bits that encode the cubin architecture."""

    if target_arch not in SM_FLAGS:
        raise KeyError(f"unknown target arch: {target_arch}")
    return (e_flags & 0xFFFF0000) | SM_FLAGS[target_arch]


def baseline_probe_receipt() -> BaselineProbeReceipt:
    """Publication-safe compact receipt for the GB10 baseline probe."""

    original = 0x09006402
    patched = rewrite_arch_low16(original, "sm_121a")
    return BaselineProbeReceipt(
        source_arch="sm_100a",
        target_arch="sm_121a",
        original_low16=original & 0xFFFF,
        patched_low16=patched & 0xFFFF,
        load_status="CUDA_SUCCESS",
        launch_status="CUDA_SUCCESS",
        sync_status="CUDA_SUCCESS",
        observed_output=(1, 3, 5, 7, 9, 11, 13, 15),
        expected_output=(1, 3, 5, 7, 9, 11, 13, 15),
    )


def is_narrow_positive(receipt: BaselineProbeReceipt) -> bool:
    """A success here proves baseline execution, not tcgen05 parity."""

    return (
        receipt.load_status == "CUDA_SUCCESS"
        and receipt.launch_status == "CUDA_SUCCESS"
        and receipt.sync_status == "CUDA_SUCCESS"
        and receipt.observed_output == receipt.expected_output
    )


def publication_note(receipt: BaselineProbeReceipt) -> str:
    if not is_narrow_positive(receipt):
        return "Baseline proof failed; do not generalize anything."
    return (
        "GB10 accepted and executed a baseline sm_100a cubin after an arch-field "
        "rewrite. That proves a loader/runtime fact only; it does not prove "
        "tcgen05.mma or TMEM execution."
    )


if __name__ == "__main__":
    receipt = baseline_probe_receipt()
    assert is_narrow_positive(receipt)
    print(publication_note(receipt))
