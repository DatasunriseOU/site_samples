"""Structured GPU receipt summary for FA4 smoke tests.

This example shows the MegaCpp POC-backed receipt shape used to summarize GPU runtime
smokes with kernel-truth, throughput, memory, and compile overhead. It exists
so optimization claims can point to one structured record instead of scattered
logs.

The problem it solves is unverifiable speed claims. A pass needs backend truth,
per-step evidence, and measured throughput in the same artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ModalFA4Receipt:
    preset: str
    backend_truth: str = "unknown"
    fa4_kernels_verified: bool = False
    throughput_tok_sec: float = 0.0
    throughput_min: float = 0.0
    throughput_max: float = 0.0
    mfu_mean: float = 0.0
    memory_peak_gb: float = 0.0
    memory_gpu_total_gb: float = 0.0
    loss_curve: list[float] = field(default_factory=list)
    loss_first: float | None = None
    loss_last: float | None = None
    compile_time_sec: float = 0.0
    wall_time_sec: float = 0.0
    gpu_type: str = ""
    gpu_count: int = 0
    num_steps: int = 0
    num_params: int = 0
    exit_code: int = -1
    status: str = "PENDING"
    failure_reason: str = ""

    def is_pass(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_raw_receipt(cls, raw: dict[str, Any], preset: str) -> "ModalFA4Receipt":
        summary = raw.get("summary", {})
        training = raw.get("training", {})
        execution = raw.get("execution_surface", {})
        checks = raw.get("checks", {})
        per_step = raw.get("per_step", [])
        loss_curve = [step.get("loss", 0.0) for step in per_step]
        compile_time_sec = per_step[0].get("dt_ms", 0.0) / 1000.0 if per_step else 0.0
        return cls(
            preset=preset,
            backend_truth=execution.get("actual_backend", "unknown"),
            fa4_kernels_verified=checks.get("fa4_backend_confirmed", False),
            throughput_tok_sec=summary.get("tok_sec_mean", 0.0),
            throughput_min=float(summary.get("tok_sec_min", 0.0)),
            throughput_max=float(summary.get("tok_sec_max", 0.0)),
            mfu_mean=summary.get("mfu_mean", 0.0),
            memory_peak_gb=execution.get("peak_memory_gb", 0.0),
            memory_gpu_total_gb=execution.get("gpu_mem_gb", 0.0),
            loss_curve=loss_curve,
            loss_first=summary.get("loss_first"),
            loss_last=summary.get("loss_last"),
            compile_time_sec=compile_time_sec,
            wall_time_sec=training.get("wall_time_sec", 0.0),
            gpu_type=execution.get("gpu_type", ""),
            gpu_count=execution.get("gpu_count", 0),
            num_steps=training.get("num_steps", 0),
            num_params=training.get("num_params", 0),
            exit_code=training.get("exit_code", -1),
            status=raw.get("status", "UNKNOWN"),
            failure_reason=raw.get("failure_reason", ""),
        )


def measured_receipt_fields() -> tuple[str, ...]:
    return (
        "throughput_tok_sec, throughput_min, and throughput_max are measured receipt fields, not estimates",
        "compile_time_sec is derived from the first per-step duration in the MegaCpp POC receipt",
        "kernel verification is kept separate from speed so a fast run without backend truth still fails the contract",
    )
