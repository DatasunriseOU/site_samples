"""Regional-compile dynamic-batch contract.

This example captures the MegaCpp POC test shape used to prove that the same compiled
region can survive device-batch changes without recompiling. It exists because
regional compile only helps if warmup compilation is amortized across later
steps.

The problem it solves is silent recompilation. If a later batch shape triggers
new graphs, compile latency comes back and throughput collapses.
"""

from __future__ import annotations


def phase_summary(
    *,
    compilations_after_p1s1: int,
    compilations_after_p1s2: int,
    compilations_after_p2s1: int,
    compilations_after_p2s2: int,
    compilations_after_p3s1: int,
) -> dict[str, int | bool]:
    """Mirror the MegaCpp POC verdict logic for dbs 2 -> 8 -> 32."""

    new_compilations_p2 = compilations_after_p2s1 - compilations_after_p1s2
    new_compilations_p3 = compilations_after_p3s1 - compilations_after_p2s2
    return {
        "phase1_compilations": compilations_after_p1s1,
        "phase2_new_compilations": new_compilations_p2,
        "phase3_new_compilations": new_compilations_p3,
        "zero_recompiles": new_compilations_p2 == 0 and new_compilations_p3 == 0,
    }


def compile_frontier_note() -> tuple[str, ...]:
    return (
        "The MegaCpp POC tests treat zero recompiles across dbs 2, 8, and 32 as the success contract.",
        "The compile region must cover the stable math path, not per-microbatch Python unpacking.",
        "A recompilation on the larger batch size means the compile win was not preserved.",
    )
