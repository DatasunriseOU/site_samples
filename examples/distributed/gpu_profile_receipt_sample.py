"""GPU profile receipt sample.

This is a donor-based public sample for matched GPU profiling receipts. It
exists to keep throughput comparisons honest: same shapes, same warmup, same
step count, then compare tok/sec, memory, and dispatch results. The problem it
solves is false speedup claims caused by changing more than one thing at once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileReceipt:
    label: str
    elapsed_s: float
    tok_per_sec: float
    peak_mem_gb: float
    steps: int
    dense_fa4_requested: bool = False
    dense_fa4_dispatch_observed: bool = False
    status: str = "ok"


def build_profile_receipt(
    *,
    label: str,
    elapsed_s: float,
    batch_size: int,
    seq_len: int,
    steps: int,
    peak_mem_gb: float,
    dense_fa4_requested: bool,
    dense_fa4_dispatch_observed: bool,
) -> ProfileReceipt:
    """Mirror the donor's central accounting: tokens divided by measured elapsed time.

    Grounded note: the donor profile runner resets state, performs warmup, then
    measures only the timed loop. That matters because the measured tok/sec is
    otherwise dominated by first-compile overhead.
    """

    total_tokens = batch_size * seq_len * steps
    tok_per_sec = total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    return ProfileReceipt(
        label=label,
        elapsed_s=elapsed_s,
        tok_per_sec=tok_per_sec,
        peak_mem_gb=peak_mem_gb,
        steps=steps,
        dense_fa4_requested=dense_fa4_requested,
        dense_fa4_dispatch_observed=dense_fa4_dispatch_observed,
    )


def compare_receipts(baseline: ProfileReceipt, candidate: ProfileReceipt) -> dict[str, float | str]:
    """Return the donor-style comparison summary for two matched profile lanes."""

    speedup = candidate.tok_per_sec / baseline.tok_per_sec if baseline.tok_per_sec > 0 else 0.0
    memory_delta_gb = candidate.peak_mem_gb - baseline.peak_mem_gb
    return {
        "baseline": baseline.label,
        "candidate": candidate.label,
        "speedup": round(speedup, 4),
        "memory_delta_gb": round(memory_delta_gb, 3),
        "winner": "candidate" if speedup >= 1.0 else "baseline",
    }


def fa4_profile_takeaways() -> tuple[str, ...]:
    """Measured-note strings grounded in donor receipt practice."""

    return (
        "The donor FA4-vs-Triton profiler measures only matched fwd+bwd loops after warmup.",
        "Dispatch confirmation matters: requested FA4 is not enough, the donor also records whether FA4 really dispatched.",
        "Peak memory belongs next to tok/sec because a faster lane that blows past the memory budget is not a usable promotion.",
    )
