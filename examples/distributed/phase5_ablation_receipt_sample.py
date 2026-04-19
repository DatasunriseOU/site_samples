"""Ablation receipt fragment for grounded optimization wins.

This example records one measured checkpoint from the MegaCpp POC ablation notes.
It exists so public examples can show a real optimization receipt without
inventing speed numbers or rewriting the larger planning document.

The problem it solves is unsupported performance summaries. Only measured
numbers with an in-repo source should appear in public-facing examples.
"""

from __future__ import annotations


def measured_phase5_receipt() -> dict[str, str | float | int]:
    """Return the documented MegaCpp POC checkpoint exactly as a structured note."""

    return {
        "scenario": "TP=8 with gradient checkpointing",
        "sequence_length": 128000,
        "per_device_memory_limit_gb": 31.25,
        "measured_throughput_tok_sec": 565000.0,
        "source_note": "docs/PHASE5_ABLATION_PLAN.md",
        "status": "WORKS",
    }


def receipt_interpretation() -> tuple[str, ...]:
    return (
        "This receipt is a single documented operating point, not a universal promise.",
        "The number comes from the MegaCpp POC ablation plan and should stay tied to that source note.",
        "If later runs disagree, the newer measured receipt should replace this one.",
    )
