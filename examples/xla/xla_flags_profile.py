"""
XLA flags library for megacpp TPU training.

Adapted from MaxText's xla_flags_library.py for PyTorch/torch_xla.
These flags configure libtpu behaviour via LIBTPU_INIT_ARGS and other
environment variables.  They must be set BEFORE any torch_xla import
(libtpu reads LIBTPU_INIT_ARGS once at initialization time).

Usage in training script::

    from megacpp.xla_flags import apply_xla_flags
    apply_xla_flags(args)          # call before torch_xla import
    import torch_xla               # now libtpu picks up the flags

The library is organized into flag *groups*.  Each group targets a
specific optimization and is documented inline.  The main entry points
are:

- ``get_xla_flags(args)``   — returns ``(libtpu_flags: list[str], env_vars: dict[str, str])``
- ``apply_xla_flags(args)`` — calls ``get_xla_flags`` and writes to ``os.environ``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# TPU generation detection
# ---------------------------------------------------------------------------

# Map accelerator-type substrings to canonical generation names.
# Used to auto-enable generation-specific optimizations (e.g. SparseCore
# on Ironwood).  Order matters — first match wins.
_TPU_GENERATION_MAP = [
    # Ironwood / TPU v7 (has SparseCore for collective offloading)
    ("v7", "v7"),
    ("ironwood", "v7"),
    ("tpu7", "v7"),
    # Trillium / TPU v6e (megacpp's primary target)
    ("v6e", "v6e"),
    ("v6", "v6e"),
    # TPU v5
    ("v5p", "v5p"),
    ("v5e", "v5e"),
    ("v5lite", "v5e"),
    ("v5", "v5e"),
    # TPU v4
    ("v4", "v4"),
]


def detect_tpu_generation(accelerator_type: str = "") -> str:
    """Return canonical TPU generation from accelerator-type string.

    Args:
        accelerator_type: e.g. ``"v6e-4"``, ``"v5litepod-8"``.
            If empty, reads ``TPU_ACCELERATOR_TYPE`` / ``ACCELERATOR_TYPE``
            from the environment.

    Returns:
        One of ``"v7"``, ``"v6e"``, ``"v5p"``, ``"v5e"``, ``"v4"``, or
        ``"unknown"``.
    """
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


# ---------------------------------------------------------------------------
# VMEM limits
# ---------------------------------------------------------------------------
# xla_tpu_scoped_vmem_limit_kib controls how much VMEM the XLA compiler
# reserves for the current HLO operation.  The remaining VMEM is used for
# prefetching subsequent operations.  Lower values leave more room for
# prefetching but may spill compute intermediates to HBM.
#
# Recommended values (from MaxText, experimentally tuned for compute-bound
# models on Trillium / v6e):
#   Dense models:  98304 KiB  (96 MiB)
#   MoE models:    81920 KiB  (80 MiB) — MoE kernels use more scratch space

DENSE_VMEM_LIMIT_KIB = 98304
MOE_VMEM_LIMIT_KIB = 81920


def _vmem_flags(is_moe: bool, vmem_limit_kib: Optional[int] = None) -> list[str]:
    """Return VMEM scoped limit flags."""
    if vmem_limit_kib is None:
        vmem_limit_kib = MOE_VMEM_LIMIT_KIB if is_moe else DENSE_VMEM_LIMIT_KIB
    return [f"--xla_tpu_scoped_vmem_limit_kib={vmem_limit_kib}"]


# ---------------------------------------------------------------------------
# Continuation Fusion (CF)
# ---------------------------------------------------------------------------
# Continuation Fusion overlaps compute with collective communication.
# Instead of blocking on a collective (AllGather, AllReduce, ReduceScatter),
# the XLA compiler fuses the collective with subsequent compute ops so they
# execute concurrently on TensorCore and ICI respectively.
#
# Key interactions:
# - CF and SparseCore offloading are MUTUALLY EXCLUSIVE for the same
#   collective type.  If SparseCore offloading is enabled for AllGather,
#   CF for AllGather must be disabled (and vice versa).
# - CF is beneficial for TP (AllGather for weight gathering) and
#   FSDP (ReduceScatter for gradient reduction).


def _continuation_fusion_flags(
    enable_all_gather: bool = False,
    enable_all_reduce: bool = False,
    enable_reduce_scatter: bool = False,
) -> list[str]:
    """Return Continuation Fusion flags for the requested collective types."""
    flags: list[str] = []

    if not (enable_all_gather or enable_all_reduce or enable_reduce_scatter):
        return flags

    # Base CF flags (needed for any CF variant)
    flags.extend([
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
    ])

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
        flags.append(
            "--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true"
        )

    return flags


# ---------------------------------------------------------------------------
# SparseCore offloading (Ironwood / TPU v7+ only)
# ---------------------------------------------------------------------------
# On Ironwood (TPU v7), the SparseCore can execute collective operations
# (AllGather, ReduceScatter, AllReduce) independently from the TensorCore,
# achieving true overlap without needing Continuation Fusion.
#
# When SparseCore offloading is enabled for a collective type, the
# corresponding CF fusion must be DISABLED (they are mutually exclusive).
#
# On v6e and older TPUs, SparseCore offloading is not available — these
# flags are silently ignored by libtpu but we skip them to keep the
# flag string clean.

_SPARSECORE_BASE_FLAGS = [
    "--xla_tpu_use_tc_device_shape_on_sc=true",
    "--xla_sc_enable_instruction_fusion=false",
    "--xla_sc_disjoint_spmem=false",
    "--xla_sc_disable_megacore_partitioning=true",
]


def _sparsecore_flags(
    enable_all_gather: bool = False,
    enable_reduce_scatter: bool = False,
    enable_all_reduce: bool = False,
) -> list[str]:
    """Return SparseCore collective offloading flags."""
    if not (enable_all_gather or enable_reduce_scatter or enable_all_reduce):
        return []

    flags = list(_SPARSECORE_BASE_FLAGS)

    if enable_all_gather:
        flags.extend([
            # Disable CF for AllGather (mutually exclusive with SC offload)
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false",
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
            "--xla_tpu_enable_all_gather_offload_tracing=true",
        ])
    if enable_reduce_scatter:
        flags.extend([
            "--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false",
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
            "--xla_tpu_enable_reduce_scatter_offload_tracing=true",
        ])
    if enable_all_reduce:
        flags.extend([
            "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false",
            "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
            "--xla_tpu_enable_all_reduce_offload_tracing=true",
        ])

    return flags


# ---------------------------------------------------------------------------
# Host offloading
# ---------------------------------------------------------------------------
# When parameters or optimizer states are offloaded to host memory (CPU RAM),
# these flags tune the XLA scheduler to maximize overlap between host↔device
# transfers and on-device compute.  Without these flags, host offloading can
# create pipeline bubbles that negate the memory savings.
#
# These flags are also beneficial for pipeline parallelism over DCN where
# large tensors transit through host memory.

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

# Break DMA to/from host into 16 MiB chunks (helps pipeline parallelism
# over DCN with large host-offloaded tensors).
_PIPELINING_FLAGS = [
    "--xla_tpu_iova_dma_chunk_size_bytes=16777216",
]


def _host_offload_flags(enable: bool = False, enable_pipelining: bool = False) -> list[str]:
    """Return host offloading optimization flags."""
    flags: list[str] = []
    if enable:
        flags.extend(_HOST_OFFLOAD_FLAGS)
    if enable_pipelining:
        flags.extend(_PIPELINING_FLAGS)
    return flags


# ---------------------------------------------------------------------------
# Data parallel overlap
# ---------------------------------------------------------------------------
# When using data parallelism (DP > 1), gradient all-reduce operations can
# be pipelined across iterations and optimized for different-sized ops.
# This is particularly useful for DCN (inter-host) all-reduces in multi-host
# TPU pods.

_DATA_PARALLEL_OVERLAP_FLAGS = [
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
    "--xla_tpu_data_parallel_opt_different_sized_ops=true",
]


def _data_parallel_flags(dp_degree: int) -> list[str]:
    """Return data-parallel optimization flags when DP > 1."""
    if dp_degree <= 1:
        return []
    return list(_DATA_PARALLEL_OVERLAP_FLAGS)


# ---------------------------------------------------------------------------
# Layout optimization for ReduceScatter / AllReduce
# ---------------------------------------------------------------------------
# Improves memory layout for reduce-scatter and all-reduce collectives.
# Minor-sharding for major-trivial inputs reduces unnecessary data movement.

_LAYOUT_FLAGS = [
    "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
    "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1",
]

# Full layout optimization includes explicit layout assignment for AR+RS
_LAYOUT_FLAGS_FULL = _LAYOUT_FLAGS + [
    "--xla_tpu_assign_all_reduce_scatter_layout=true",
]


def _layout_flags(enable_fsdp: bool = False) -> list[str]:
    """Return layout optimization flags.

    When FSDP is enabled, uses the full set including explicit layout
    assignment for all-reduce-scatter.  Otherwise returns the base
    reduce-scatter fusion flags.
    """
    if enable_fsdp:
        return list(_LAYOUT_FLAGS_FULL)
    return list(_LAYOUT_FLAGS)


# ---------------------------------------------------------------------------
# Miscellaneous flags
# ---------------------------------------------------------------------------

def _misc_flags(
    disable_bundle_aware_cost_model: bool = True,
    enable_async_collective_permute: bool = True,
) -> list[str]:
    """Return miscellaneous optimization flags.

    - ``disable_bundle_aware_cost_model``: Disables the bundle-aware cost
      model which was causing 3x slower fusions in backward pass (b/357103386).
      Enabled by default as a safe optimization.
    - ``enable_async_collective_permute``: Enables async collective permute
      for better pipeline parallelism overlap.
    """
    flags: list[str] = []
    if disable_bundle_aware_cost_model:
        flags.append("--xla_tpu_use_bundle_aware_cost_model_for_fusions=false")
    if enable_async_collective_permute:
        flags.append("--xla_enable_async_collective_permute=true")
    return flags


# ---------------------------------------------------------------------------
# SDC (Silent Data Corruption) checker
# ---------------------------------------------------------------------------
# Detects chip / ICI / hardware corruption by re-running HLO/LLO and
# verifying deterministic results.  Performance cost: 1/(repeat_count+1)
# throughput.  Only enable for debugging suspected hardware issues.

# ---------------------------------------------------------------------------
# MoE MSA workaround (jellyfish memory_space_assignment crash)
# ---------------------------------------------------------------------------
# libtpu's internal MSA pass creates invalid same-memory-space async copy
# chains with MoE FusedExpertBank 3D weight tensors.  The fatal CHECK at
# memory_space_assignment.cc:1433 cannot be bypassed (CHECK_NE, no flag).
#
# Root cause: the Latency Hiding Scheduler (LHS) re-orders copies to overlap
# compute and communication.  With MoE's [N,H,D] bmm tensors, LHS creates
# copy chains that violate the "copies must go between different memory
# spaces" invariant.
#
# Workaround:
# 1. Disable LHS entirely — prevents the problematic copy chain creation
# 2. Set async copy bandwidth scaling very low — discourages async copies
# 3. Set MSA "inefficient use to copy" ratio high — avoids copy placement
# 4. Pre-set torch_xla's injected flags to prevent LHS re-enablement:
#    - xla_latency_hiding_scheduler_rerun=0 (torch_xla defaults to 1)
#    - xla_tpu_prefer_async_allgather_to_allreduce=false (torch_xla sets true)
#    - xla_tpu_use_enhanced_launch_barrier=false (torch_xla sets false, same)

_MOE_MSA_WORKAROUND_FLAGS = [
    # Core: disable Latency Hiding Scheduler — creates invalid async copy
    # chains with MoE FusedExpertBank 3D [N,H,D] bmm tensors
    "--xla_tpu_enable_latency_hiding_scheduler=false",
    # Prevent LHS rerun (torch_xla defaults to rerun=1)
    "--xla_latency_hiding_scheduler_rerun=0",
    # Disable copy fusion that merges small async copies into ones
    # that violate the "different memory space" invariant
    "--xla_tpu_enable_copy_fusion=false",
    # Prevent VMEM-to-VMEM DMA operations — the crash is specifically
    # about copies within memory space 1 (VMEM/alternate)
    "--xla_tpu_enable_vmem_to_vmem_dmas=false",
    # Disable async collective fusion — MoE all-to-all expert dispatch
    # creates complex VMEM patterns that trigger the MSA bug
    "--xla_tpu_enable_async_collective_fusion=false",
    # Conservative MSA parameters
    # NOTE: vmem limit is NOT set here — _vmem_flags() already handles
    # the MoE default (81920 KiB) and custom overrides.  Having it here
    # caused _deduplicate_flags() to overwrite user-specified custom vmem.
    "--xla_tpu_msa_inefficient_use_to_copy_ratio=1.0",
    "--xla_tpu_async_copy_bandwidth_scaling_factor=0.01",
    # Pre-set torch_xla's injected flags to prevent conflicting values
    "--xla_tpu_prefer_async_allgather_to_allreduce=false",
    "--xla_tpu_use_enhanced_launch_barrier=false",
    "--xla_tpu_enable_flash_attention=false",
]


def _moe_msa_workaround_flags() -> list[str]:
    """Return MoE MSA workaround flags for jellyfish crash prevention."""
    return list(_MOE_MSA_WORKAROUND_FLAGS)


def _moe_fusion_limit_flags(n_routed_experts: int = 8) -> list[str]:
    """Prevent LLO_CHECK overflow for large MoE without fully disabling fusion.

    XLA fuses ops into LLO regions. With 64+ experts, regions can exceed 4GB
    (int32 addressing limit in TPU LLO instruction encoding). Instead of
    disabling fusion entirely (--xla_tpu_rwb_fusion=false costs ~5% throughput),
    we limit fusion region sizes to keep each under 4GB.

    Strategy:
    - 8 experts: no limits needed (default fusion is fine)
    - 32+ experts: limit VMEM per fusion to prevent oversized regions
    - 64+ experts: aggressive limits + disable only rwb_fusion (keeps dot_dot)
    """
    if n_routed_experts < 32:
        return []

    flags = []
    if n_routed_experts >= 64:
        # Limit fusion region VMEM to prevent oversized LLO regions (>4GB).
        # This keeps fusion active but caps each region's size.
        # Disabling rwb_fusion entirely works but makes compilation 2x slower.
        flags.append("--xla_jf_fusion_max_vmem_mib=32")
        # Limit multi-output fusion (MoE dispatch creates multi-output fusions)
        flags.append("--xla_tpu_multi_output_fusion_limit=4")
    elif n_routed_experts >= 32:
        flags.append("--xla_jf_fusion_max_vmem_mib=48")

    return flags


def _sdc_flags(enable: bool = False, repeat_count: int = 5) -> list[str]:
    """Return SDC checker flags (debug only — significant perf cost)."""
    if not enable:
        return []
    return [
        "--xla_tpu_enable_sdc_checker=true",
        "--xla_tpu_sdc_check_halt_on_detection=true",
        "--xla_tpu_sdc_replicate_llo=true",
        f"--xla_tpu_sdc_check_repeat_count={repeat_count}",
    ]


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_DEBUG_ENV_VARS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TPU_VMODULE": "tpu_configuration_ops_impl=3",
}


# ---------------------------------------------------------------------------
# Core env vars (always set on TPU)
# ---------------------------------------------------------------------------
# XLA_NO_SPECIAL_SCALARS: Prevents scalars from silently moving to CPU.
# Without this, XLA may execute scalar operations on the host instead of
# the TPU, causing silent performance degradation and correctness issues.
#
# See AGENTS.md: "Set XLA_NO_SPECIAL_SCALARS=1 or scalars silently go to CPU"

_CORE_ENV_VARS = {
    "XLA_NO_SPECIAL_SCALARS": "1",
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

@dataclass
class XLAFlagsResult:
    """Result of XLA flag computation.

    Attributes:
        libtpu_flags: List of ``--xla_*`` flags for ``LIBTPU_INIT_ARGS``.
        env_vars: Dict of environment variables to set (e.g.
            ``XLA_NO_SPECIAL_SCALARS``).
        summary: Human-readable summary of enabled flag groups.
    """
    libtpu_flags: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)

    @property
    def libtpu_init_args(self) -> str:
        """Formatted LIBTPU_INIT_ARGS string (space-separated flags)."""
        return " ".join(self.libtpu_flags)


def get_xla_flags(
    args,
    *,
    tpu_generation: str = "",
    enable_sdc_checker: bool = False,
    enable_debug_logs: bool = False,
    enable_host_offload: bool = False,
    enable_pipelining: bool = False,
    vmem_limit_kib: Optional[int] = None,
) -> XLAFlagsResult:
    """Compute XLA flags based on training configuration.

    Reads the following from ``args`` (argparse namespace):
    - ``args.moe``:              bool — MoE model (affects VMEM limit)
    - ``args.tensor_parallel``:  int  — TP degree (affects CF AllGather)
    - ``args.fsdp``:             bool — FSDP enabled (affects CF ReduceScatter, layout)
    - ``args.expert_parallel``:  int  — EP degree (informational)

    Additional keyword arguments override auto-detection:
    - ``tpu_generation``: Override TPU gen detection (e.g. ``"v6e"``, ``"v7"``).
    - ``enable_sdc_checker``: Enable SDC checker (debug, ~6x perf cost).
    - ``enable_debug_logs``: Enable verbose TPU debug logging.
    - ``enable_host_offload``: Enable host offloading scheduler optimizations.
    - ``enable_pipelining``: Enable DMA chunking for pipeline parallelism.
    - ``vmem_limit_kib``: Override VMEM limit (default: auto based on dense/MoE).

    Returns:
        ``XLAFlagsResult`` with libtpu flags, env vars, and summary.
    """
    result = XLAFlagsResult()

    # --- Read config from args ---
    is_moe = getattr(args, "moe", False)
    tp_degree = getattr(args, "tensor_parallel", 1)
    fsdp_enabled = getattr(args, "fsdp", False)

    # Compute DP degree: num_devices / (tp * ep)
    # We don't know num_devices at flag-setting time (before torch_xla init),
    # but we can detect from FSDP or multi-chip setup.
    # For flag purposes, dp > 1 when fsdp is enabled or num_chips > tp.
    # --- Detect TPU generation ---
    if not tpu_generation:
        tpu_generation = detect_tpu_generation()
    is_ironwood = tpu_generation == "v7"

    # --- Core env vars (always set) ---
    result.env_vars.update(_CORE_ENV_VARS)
    result.summary.append("core: XLA_NO_SPECIAL_SCALARS=1")

    # --- 1. VMEM limits ---
    result.libtpu_flags.extend(_vmem_flags(is_moe, vmem_limit_kib))
    effective_vmem = vmem_limit_kib or (MOE_VMEM_LIMIT_KIB if is_moe else DENSE_VMEM_LIMIT_KIB)
    result.summary.append(
        f"vmem: {effective_vmem} KiB ({'MoE' if is_moe else 'dense'})"
    )

    # --- 2. Collective overlap (CF vs SparseCore) ---
    # On Ironwood: use SparseCore offloading (better overlap, dedicated hw).
    # On v6e/v5:   use Continuation Fusion (software overlap).
    if is_ironwood:
        # SparseCore offloading for all collective types
        sc_ag = tp_degree > 1
        sc_rs = fsdp_enabled
        sc_ar = True  # Always useful for gradient all-reduce
        result.libtpu_flags.extend(_sparsecore_flags(
            enable_all_gather=sc_ag,
            enable_reduce_scatter=sc_rs,
            enable_all_reduce=sc_ar,
        ))
        parts = []
        if sc_ag:
            parts.append("AG")
        if sc_rs:
            parts.append("RS")
        if sc_ar:
            parts.append("AR")
        result.summary.append(f"sparsecore: {'+'.join(parts)} offload (Ironwood)")
    else:
        # Continuation Fusion — overlap collectives with compute on TensorCore
        cf_ag = tp_degree > 1       # AllGather for TP weight gathering
        cf_ar = True                 # AllReduce for gradient reduction
        cf_rs = fsdp_enabled         # ReduceScatter for FSDP gradient sharding
        result.libtpu_flags.extend(_continuation_fusion_flags(
            enable_all_gather=cf_ag,
            enable_all_reduce=cf_ar,
            enable_reduce_scatter=cf_rs,
        ))
        parts = []
        if cf_ag:
            parts.append("AG")
        if cf_ar:
            parts.append("AR")
        if cf_rs:
            parts.append("RS")
        if parts:
            result.summary.append(f"continuation_fusion: {'+'.join(parts)}")

    # --- 3. Data parallel overlap ---
    # Always enable: DP > 1 whenever num_chips > TP * EP, and we don't
    # know num_chips before torch_xla init.  These flags are harmless
    # no-ops when DP == 1, so enabling unconditionally is safe.
    result.libtpu_flags.extend(_data_parallel_flags(dp_degree=2))  # conservative
    result.summary.append("data_parallel_overlap: enabled")

    # --- 4. Layout optimization ---
    result.libtpu_flags.extend(_layout_flags(enable_fsdp=fsdp_enabled))
    result.summary.append(
        f"layout: {'full (FSDP)' if fsdp_enabled else 'base (RS fusion)'}"
    )

    # --- 5. Host offloading ---
    if enable_host_offload:
        result.libtpu_flags.extend(
            _host_offload_flags(enable=True, enable_pipelining=enable_pipelining)
        )
        result.summary.append("host_offload: enabled")

    # --- 6. Miscellaneous ---
    result.libtpu_flags.extend(_misc_flags())
    result.summary.append("misc: disable_bundle_cost_model, async_collective_permute")

    # --- 7. MoE MSA workaround ---
    # libtpu's jellyfish MSA pass creates invalid VMEM-to-VMEM async copy
    # chains with MoE expert bank tensors (3D [N,H,D] bmm weights), causing
    # a fatal CHECK: from_memory_space != to_memory_space at
    # memory_space_assignment.cc:1433.  No flag disables the CHECK — instead
    # we must prevent the invalid chains from forming.
    #
    # The workaround: (a) disable the Latency Hiding Scheduler which creates
    # the problematic async copy chains, (b) set conservative VMEM placement,
    # (c) prevent torch_xla from re-enabling LHS via its injected flags.
    #
    # torch_xla's __init__.py uses _set_missing_flags() which only adds flags
    # NOT already present — so pre-setting these blocks injection.
    if is_moe:
        result.libtpu_flags.extend(_moe_msa_workaround_flags())
        result.summary.append("moe_msa_workaround: LHS disabled, conservative async copy")

        # Fusion size limits for large MoE — prevent LLO_CHECK 4GB overflow
        _n_routed = getattr(args, "moe_n_routed_experts", 8) if hasattr(args, "moe_n_routed_experts") else 8
        _fusion_flags = _moe_fusion_limit_flags(_n_routed)
        if _fusion_flags:
            result.libtpu_flags.extend(_fusion_flags)
            result.summary.append(f"moe_fusion_limit: {_n_routed} experts, VMEM capped")

    # --- 8. SDC checker (debug only) ---
    if enable_sdc_checker:
        result.libtpu_flags.extend(_sdc_flags(enable=True))
        result.summary.append("sdc_checker: ENABLED (debug, ~6x perf cost)")

    # --- 9. Debug logs ---
    if enable_debug_logs or enable_sdc_checker:
        result.env_vars.update(_DEBUG_ENV_VARS)
        result.summary.append("debug_logs: enabled")

    # Deduplicate flags (later flags win for conflicting keys)
    result.libtpu_flags = _deduplicate_flags(result.libtpu_flags)

    return result


def _deduplicate_flags(flags: list[str]) -> list[str]:
    """Deduplicate XLA flags, keeping the LAST occurrence of each flag name.

    Flags are ``--flag_name=value`` pairs.  If the same flag appears multiple
    times (e.g. CF enables a flag and SC disables it), the last one wins.
    """
    seen: dict[str, str] = {}
    for flag in flags:
        # Split on first '=' to get flag name
        if "=" in flag:
            name = flag.split("=", 1)[0]
        else:
            name = flag
        seen[name] = flag
    return list(seen.values())


def apply_xla_flags(
    args,
    *,
    tpu_generation: str = "",
    enable_sdc_checker: bool = False,
    enable_debug_logs: bool = False,
    enable_host_offload: bool = False,
    enable_pipelining: bool = False,
    vmem_limit_kib: Optional[int] = None,
    verbose: bool = True,
) -> XLAFlagsResult:
    """Compute and apply XLA flags to the process environment.

    MUST be called BEFORE any ``import torch_xla`` — libtpu reads
    ``LIBTPU_INIT_ARGS`` once at initialization.

    Sets:
    - ``os.environ["LIBTPU_INIT_ARGS"]``: Space-separated ``--xla_*`` flags.
      If ``LIBTPU_INIT_ARGS`` already has a value, the new flags are APPENDED
      (existing user flags take precedence via deduplication).
    - Additional env vars (e.g. ``XLA_NO_SPECIAL_SCALARS``).

    Args:
        args: argparse namespace from ``train_args.create_parser()``.
        verbose: Print applied flags summary.
        (other kwargs forwarded to ``get_xla_flags``).

    Returns:
        The computed ``XLAFlagsResult``.
    """
    result = get_xla_flags(
        args,
        tpu_generation=tpu_generation,
        enable_sdc_checker=enable_sdc_checker,
        enable_debug_logs=enable_debug_logs,
        enable_host_offload=enable_host_offload,
        enable_pipelining=enable_pipelining,
        vmem_limit_kib=vmem_limit_kib,
    )

    # Merge with existing LIBTPU_INIT_ARGS (user flags prepended, ours appended).
    # Deduplication keeps last occurrence, so our computed flags win for
    # conflicts. But if the user explicitly sets a flag in the env, prepending
    # it means our version (appended later) will override. To let user flags
    # win, we prepend ours and append theirs.
    existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
    if existing:
        # User's explicit flags come AFTER ours → user wins in dedup
        all_flags = result.libtpu_flags + existing.split()
        all_flags = _deduplicate_flags(all_flags)
        final_args = " ".join(all_flags)
    else:
        final_args = result.libtpu_init_args

    os.environ["LIBTPU_INIT_ARGS"] = final_args

    # Set additional env vars (don't overwrite user-set values)
    for key, value in result.env_vars.items():
        os.environ.setdefault(key, value)

    if verbose:
        print(f"XLA flags ({len(result.libtpu_flags)} libtpu flags):")
        for line in result.summary:
            print(f"  {line}")
        if existing:
            print("  (merged with existing LIBTPU_INIT_ARGS)")

    return result
