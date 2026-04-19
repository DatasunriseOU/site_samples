"""XLA flag-group example.

This shows how TPU runtime flags are grouped before torch_xla starts.
The problem is that libtpu reads important settings only once during startup,
so late changes do nothing. Grouping the flags early makes TPU runs repeatable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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


def detect_tpu_generation(accelerator_type: str = "") -> str:
    if not accelerator_type:
        accelerator_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE", "")
            or os.environ.get("ACCELERATOR_TYPE", "")
        )
    accel_lower = accelerator_type.lower()
    for substr, generation in _TPU_GENERATION_MAP:
        if substr in accel_lower:
            return generation
    return "unknown"


def vmem_flags(is_moe: bool, vmem_limit_kib: Optional[int] = None) -> list[str]:
    limit = vmem_limit_kib
    if limit is None:
        limit = MOE_VMEM_LIMIT_KIB if is_moe else DENSE_VMEM_LIMIT_KIB
    return [f"--xla_tpu_scoped_vmem_limit_kib={limit}"]


def continuation_fusion_flags(
    *,
    enable_all_gather: bool = False,
    enable_all_reduce: bool = False,
    enable_reduce_scatter: bool = False,
) -> list[str]:
    flags: list[str] = []
    if not (enable_all_gather or enable_all_reduce or enable_reduce_scatter):
        return flags
    flags.extend(
        [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
        ]
    )
    if enable_all_gather:
        flags.extend(
            [
                "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
                "--xla_enable_async_all_gather=true",
            ]
        )
    if enable_all_reduce:
        flags.extend(
            [
                "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true",
                "--xla_enable_async_all_reduce=true",
            ]
        )
    if enable_reduce_scatter:
        flags.append("--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true")
    return flags


@dataclass
class XlaFlagProfile:
    generation: str = "unknown"
    is_moe: bool = False
    enable_async_all_gather: bool = False
    enable_async_all_reduce: bool = False
    enable_async_reduce_scatter: bool = False


def build_flag_profile(profile: XlaFlagProfile) -> tuple[list[str], dict[str, str]]:
    flags: list[str] = []
    flags.extend(vmem_flags(profile.is_moe))
    flags.extend(
        continuation_fusion_flags(
            enable_all_gather=profile.enable_async_all_gather,
            enable_all_reduce=profile.enable_async_all_reduce,
            enable_reduce_scatter=profile.enable_async_reduce_scatter,
        )
    )
    env = {
        "PJRT_DEVICE": "TPU",
        "XLA_USE_BF16": "1",
        "XLA_NO_SPECIAL_SCALARS": "1",
    }
    return flags, env
