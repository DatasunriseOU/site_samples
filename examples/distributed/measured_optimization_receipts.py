"""Measured optimization receipts sample.

This is a donor-based public helper for parsing measured optimization outcomes
from training result tables. It exists because optimization work needs receipt
discipline: what changed, under what condition, and what got faster or lighter.
The problem it solves is vague claims like "it was faster" without the matching
batch size, memory, or warm-cache condition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationReceipt:
    label: str
    tok_per_sec: float
    mfu_percent: float | None
    peak_mem_gb: float | None
    condition: str
    note: str


def parse_compact_tok_sec(value: str) -> float:
    """Convert donor-style compact throughput strings like ``304K`` into floats."""

    cleaned = value.strip().upper().replace(",", "")
    if cleaned.endswith("K"):
        return float(cleaned[:-1]) * 1_000.0
    if cleaned.endswith("M"):
        return float(cleaned[:-1]) * 1_000_000.0
    return float(cleaned)


def feature_delta(base: OptimizationReceipt, candidate: OptimizationReceipt) -> dict[str, float | str]:
    """Compute a small public-safe delta summary from two measured rows."""

    tok_delta = candidate.tok_per_sec - base.tok_per_sec
    pct_delta = (tok_delta / base.tok_per_sec * 100.0) if base.tok_per_sec else 0.0
    memory_delta = None
    if base.peak_mem_gb is not None and candidate.peak_mem_gb is not None:
        memory_delta = round(candidate.peak_mem_gb - base.peak_mem_gb, 2)
    return {
        "base": base.label,
        "candidate": candidate.label,
        "tok_per_sec_delta": round(tok_delta, 1),
        "tok_per_sec_pct": round(pct_delta, 2),
        "peak_mem_delta_gb": memory_delta,
        "condition": candidate.condition,
    }


def donor_measured_notes() -> tuple[OptimizationReceipt, ...]:
    """Receipt rows grounded in the donor hardware-results document.

    Grounded notes:
    - warm inductor/Triton cache was explicitly called out for these numbers
    - ``+enriched`` dropped throughput from 295K to 155K tok/sec at dbs=32 and
      was labeled bandwidth-bound
    - mHC group recompute kept 18.5K tok/sec at dbs=4 while reducing memory
      from 77GB to 23.7GB, so the measured win was memory relief without an
      observed throughput penalty in that lane
    """

    return (
        OptimizationReceipt(
            label="base_gqa_dsa_bf16_dbs32",
            tok_per_sec=parse_compact_tok_sec("295K"),
            mfu_percent=93.4,
            peak_mem_gb=117.0,
            condition="8xH200, dbs=32, warm inductor/Triton cache",
            note="Baseline feature-addition lane.",
        ),
        OptimizationReceipt(
            label="enriched_dbs32",
            tok_per_sec=parse_compact_tok_sec("155K"),
            mfu_percent=49.2,
            peak_mem_gb=120.0,
            condition="8xH200, dbs=32, warm inductor/Triton cache",
            note="Measured as bandwidth-bound with a 47 percent throughput drop.",
        ),
        OptimizationReceipt(
            label="mhc_old_eager_dbs4",
            tok_per_sec=parse_compact_tok_sec("18.5K"),
            mfu_percent=None,
            peak_mem_gb=77.0,
            condition="8xH200, dbs=4",
            note="Old eager lane before the group recompute fix.",
        ),
        OptimizationReceipt(
            label="mhc_group_ckpt_on_dbs4",
            tok_per_sec=parse_compact_tok_sec("18.5K"),
            mfu_percent=None,
            peak_mem_gb=23.7,
            condition="8xH200, dbs=4",
            note="Same measured throughput with 3.2x lower memory after group recompute.",
        ),
    )
