"""TPU XLA flag groups for memory and collective overlap.

This example shows the MegaCpp POC-backed flag group layer used before importing the
TPU runtime. It exists so one launch can describe VMEM limits, host-offload
overlap, and collective overlap in a stable way instead of hand-building long
flag strings.

The problem it solves is that TPU startup behavior is decided once at runtime
initialization. If the flags are inconsistent, the job can start with the wrong
memory or communication policy and fail before training settles.
"""

from __future__ import annotations

import os


_TPU_GENERATION_MAP = [
    ("v7", "v7"),
    ("ironwood", "v7"),
    ("tpu7", "v7"),
    ("v6e", "v6e"),
    ("v6", "v6e"),
    ("v5p", "v5p"),
    ("v5e", "v5e"),
    ("v5lite", "v5e"),
    ("v5", "v5e"),
    ("v4", "v4"),
]

DENSE_VMEM_LIMIT_KIB = 98304
MOE_VMEM_LIMIT_KIB = 81920

_HOST_OFFLOAD_FLAGS = [
    "--xla_tpu_enable_all_experimental_scheduler_features=true",
    "--xla_tpu_enable_scheduler_memory_pressure_tracking=true",
    "--xla_tpu_host_transfer_overlap_limit=24",
    "--xla_tpu_aggressive_opt_barrier_removal=ENABLED",
    "--xla_lhs_prioritize_async_depth_over_stall=ENABLED",
    "--xla_tpu_enable_ag_backward_pipelining=true",
    "--xla_should_allow_loop_variant_parameter_in_chain=ENABLED",
    "--xla_should_add_loop_invariant_op_in_chain=ENABLED",
    "--xla_max_concurrent_host_send_recv=100",
    "--xla_tpu_scheduler_percent_shared_memory_limit=100",
    "--xla_latency_hiding_scheduler_rerun=2",
]

_PIPELINING_FLAGS = ["--xla_tpu_iova_dma_chunk_size_bytes=16777216"]


def detect_tpu_generation(accelerator_type: str = "") -> str:
    """Return the MegaCpp POC's canonical TPU generation label."""

    if not accelerator_type:
        accelerator_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE", "")
            or os.environ.get("ACCELERATOR_TYPE", "")
        )
    lowered = accelerator_type.lower()
    for marker, generation in _TPU_GENERATION_MAP:
        if marker in lowered:
            return generation
    return "unknown"


def vmem_flags(*, is_moe: bool, vmem_limit_kib: int | None = None) -> list[str]:
    """Choose the MegaCpp POC VMEM cap used to trade scratch space against prefetch room."""

    if vmem_limit_kib is None:
        vmem_limit_kib = MOE_VMEM_LIMIT_KIB if is_moe else DENSE_VMEM_LIMIT_KIB
    return [f"--xla_tpu_scoped_vmem_limit_kib={vmem_limit_kib}"]


def continuation_fusion_flags(
    *,
    enable_all_gather: bool = False,
    enable_all_reduce: bool = False,
    enable_reduce_scatter: bool = False,
) -> list[str]:
    """Return the MegaCpp POC overlap flags for TPU collectives."""

    if not (enable_all_gather or enable_all_reduce or enable_reduce_scatter):
        return []
    flags = [
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
    ]
    if enable_all_gather:
        flags.extend([
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "--xla_enable_async_all_gather=true",
        ])
    if enable_all_reduce:
        flags.extend([
            "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true",
            "--xla_enable_async_all_reduce=true",
        ])
    if enable_reduce_scatter:
        flags.append("--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true")
    return flags


def host_offload_flags(*, enable: bool = False, enable_pipelining: bool = False) -> list[str]:
    """Return the MegaCpp POC flags that keep host transfers from stalling the step."""

    flags: list[str] = []
    if enable:
        flags.extend(_HOST_OFFLOAD_FLAGS)
    if enable_pipelining:
        flags.extend(_PIPELINING_FLAGS)
    return flags


def describe_flag_strategy() -> tuple[str, ...]:
    return (
        "VMEM caps decide how much TPU memory stays available for compiler prefetch and scratch space.",
        "Continuation fusion overlaps collectives with compute when SparseCore offload is not the chosen path.",
        "Host-offload flags reduce pipeline bubbles when parameters or optimizer state move through host memory.",
    )
