"""3D Parallelism (PP + FSDP2 + EP) integration for megacpp models.

Orchestrates three complementary parallelism strategies on a single cluster:

- **Pipeline Parallelism (PP)**: Partitions model layers across pipeline stages.
  Each stage lives on a separate group of devices and processes a different
  micro-batch, overlapping compute and communication.

- **Fully Sharded Data Parallelism (FSDP2)**: ZeRO-3 style parameter sharding
  within each pipeline stage.  Each data-parallel rank holds 1/dp of the
  parameters and optimizer states.

- **Expert Parallelism (EP)**: For MoE E-blocks, expert weights are partitioned
  across EP ranks.  Tokens are dispatched via AlltoAll between EP peers.

Optionally, Tensor Parallelism (TP) can be composed as a fourth axis, giving
a full 4D mesh.  When TP is active, the mesh has shape
``(pp, dp, tp, ep)``; otherwise ``(pp, dp, ep)`` (or ``(pp, dp)`` when ep=1).

Correct application order (following torchtitan conventions):
    TP -> PP -> FSDP2 -> EP
This ensures TP splits weight matrices first, PP partitions layers,
FSDP2 shards per-stage parameters, and EP dispatches expert tokens.

Usage:
    from megacpp.parallelism_3d import (
        ParallelismConfig,
        build_3d_mesh,
        apply_3d_parallelism,
        validate_3d_config,
    )

    config = ParallelismConfig(pp=4, dp=2, tp=1, ep=1)
    validate_3d_config(config, model_config)
    meshes = build_3d_mesh(config, device_type="cuda")
    result = apply_3d_parallelism(model, config, meshes, device=device)

Compatibility:
    - Works with both Nemotron-style (ABlock/MBlock/EBlock) and legacy Block layers.
    - CUDA: uses torch.distributed.device_mesh + FSDP2 (fully_shard).
    - XLA/TPU: uses SPMD mesh + mark_sharding via megacpp.fsdp.
    - Requires PyTorch 2.4+ for PP and FSDP2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sized
from typing import Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ParallelismConfig:
    """Configuration for 3D (PP x DP x EP) parallelism.

    Attributes:
        pp: Pipeline parallelism degree (number of pipeline stages).
            1 = no pipeline parallelism.
        dp: Data parallelism degree (FSDP2 sharding group size).
            1 = no data parallelism.
        tp: Tensor parallelism degree (Megatron-style weight sharding).
            1 = no tensor parallelism.
        ep: Expert parallelism degree (MoE expert partitioning).
            1 = no expert parallelism.
        pp_schedule: Pipeline schedule name ("1f1b", "interleaved_1f1b", "gpipe").
        pp_microbatches: Number of micro-batches for pipelining.  0 = auto
            (defaults to pp degree).
    """

    pp: int = 1
    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp_schedule: str = "1f1b"
    pp_microbatches: int = 0

    def __post_init__(self):
        if self.pp < 1:
            raise ValueError(f"pp must be >= 1, got {self.pp}")
        if self.dp < 1:
            raise ValueError(f"dp must be >= 1, got {self.dp}")
        if self.tp < 1:
            raise ValueError(f"tp must be >= 1, got {self.tp}")
        if self.ep < 1:
            raise ValueError(f"ep must be >= 1, got {self.ep}")
        if self.pp_microbatches < 0:
            raise ValueError(
                f"pp_microbatches must be >= 0, got {self.pp_microbatches}"
            )

    @property
    def world_size(self) -> int:
        """Total number of devices required: pp * dp * tp * ep."""
        return self.pp * self.dp * self.tp * self.ep

    @property
    def effective_dp(self) -> int:
        """Effective data-parallel degree (dp * ep for gradient reduction)."""
        return self.dp

    @property
    def has_pp(self) -> bool:
        return self.pp > 1

    @property
    def has_tp(self) -> bool:
        return self.tp > 1

    @property
    def has_ep(self) -> bool:
        return self.ep > 1

    @property
    def has_fsdp(self) -> bool:
        return self.dp > 1

    @property
    def num_axes(self) -> int:
        """Number of active parallelism axes (>1)."""
        return sum(1 for d in (self.pp, self.dp, self.tp, self.ep) if d > 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_3d_config(
    par: ParallelismConfig,
    model_config: Any,
    *,
    num_devices: Optional[int] = None,
) -> list[str]:
    """Validate a 3D parallelism config against a model config.

    Checks:
    1. World size matches available devices (if provided).
    2. PP degree divides number of model layers evenly (for balanced stages).
    3. TP degree divides num_heads and n_kv_head.
    4. EP degree divides moe_n_routed_experts (when MoE is active).
    5. NAM52-specific constraints: n_kv_head=8 compatibility, layer divisibility.

    Args:
        par: The parallelism configuration to validate.
        model_config: A GPTConfig (or similar) with model architecture fields.
        num_devices: Total available devices. If provided, checks that
            par.world_size == num_devices.

    Returns:
        List of warning strings (empty if all checks pass).

    Raises:
        ValueError: On hard constraint violations that would prevent training.
    """
    warnings: list[str] = []
    n_layer = getattr(model_config, "n_layer", None) or getattr(
        model_config, "depth", None
    )
    n_head = getattr(model_config, "n_head", 1)
    n_kv_head = getattr(model_config, "n_kv_head", 0)
    if n_kv_head == 0:
        n_kv_head = max(1, n_head // 4)
    moe_enabled = getattr(model_config, "moe_enabled", False)
    n_routed_experts = getattr(model_config, "moe_n_routed_experts", 8)
    nemotron_style = getattr(model_config, "nemotron_style", False)
    _standalone_mla = getattr(model_config, "use_mla", False) and not getattr(
        model_config, "dsa_enabled", False
    )

    # 1. World size vs available devices
    if num_devices is not None and par.world_size != num_devices:
        raise ValueError(
            f"ParallelismConfig requires {par.world_size} devices "
            f"(pp={par.pp} x dp={par.dp} x tp={par.tp} x ep={par.ep}), "
            f"but only {num_devices} devices are available."
        )

    # 2. PP divides layers
    if n_layer is not None and par.has_pp:
        if n_layer % par.pp != 0:
            warnings.append(
                f"n_layer={n_layer} is not evenly divisible by pp={par.pp}. "
                f"Stages will be imbalanced "
                f"(sizes: {_stage_sizes(n_layer, par.pp)})."
            )
        if par.pp > n_layer:
            raise ValueError(
                f"pp={par.pp} exceeds n_layer={n_layer}. "
                f"Cannot have more pipeline stages than layers."
            )

    # 3. TP divides heads
    if par.has_tp:
        if n_head % par.tp != 0:
            raise ValueError(
                f"n_head={n_head} not divisible by tp={par.tp}. "
                f"Tensor parallelism requires n_head % tp == 0."
            )
        # Standalone MLA (use_mla=True, dsa=False) does not use n_kv_head --
        # it has its own latent compression that only requires n_head
        # divisibility.  DSA+MLA still routes through MLAQKVProjector which
        # uses n_kv_head, so the check must remain when DSA is enabled.
        if not _standalone_mla and n_kv_head % par.tp != 0:
            raise ValueError(
                f"n_kv_head={n_kv_head} not divisible by tp={par.tp}. "
                f"Tensor parallelism requires n_kv_head % tp == 0."
            )

    # 4. EP divides experts
    if par.has_ep:
        if not moe_enabled:
            raise ValueError(
                f"ep={par.ep} > 1 requires moe_enabled=True in model config."
            )
        if n_routed_experts % par.ep != 0:
            raise ValueError(
                f"moe_n_routed_experts={n_routed_experts} not divisible by "
                f"ep={par.ep}. Expert parallelism requires even expert splits."
            )

    # 5. NAM52-specific checks
    if nemotron_style and n_layer is not None:
        # NAM52: 52 layers. PP=4 gives 13 per stage (perfect).
        # PP=2 gives 26 per stage (OK). PP=8 gives 6.5 (bad).
        if n_layer == 52 and par.has_pp:
            if 52 % par.pp != 0:
                warnings.append(
                    f"NAM52 (52 layers): pp={par.pp} does not divide 52 evenly. "
                    f"Recommended: pp=1, 2, 4, or 13."
                )
            # Check TP compatibility with NAM52's n_kv_head=8.
            # Standalone MLA (use_mla=True, dsa=False) does not use n_kv_head,
            # so skip this check in that case (same logic as the general check
            # at step 3 above).
            if par.has_tp and n_kv_head == 8 and not _standalone_mla:
                if 8 % par.tp != 0:
                    raise ValueError(
                        f"NAM52 has n_kv_head=8, but tp={par.tp} does not "
                        f"divide 8. Use tp=1, 2, 4, or 8."
                    )

    return warnings


def _stage_sizes(n_layers: int, pp: int) -> list[int]:
    """Compute per-stage layer counts for balanced partitioning."""
    base = n_layers // pp
    remainder = n_layers % pp
    return [base + (1 if i < remainder else 0) for i in range(pp)]


# ---------------------------------------------------------------------------
# Mesh Construction
# ---------------------------------------------------------------------------


def build_3d_mesh(
    par: ParallelismConfig,
    device_type: str = "cuda",
) -> dict[str, Any]:
    """Build device meshes for 3D parallelism.

    Creates a hierarchical mesh structure with named dimensions:
    - "pp": pipeline parallelism axis
    - "dp": data parallelism (FSDP2) axis
    - "tp": tensor parallelism axis (if tp > 1)
    - "ep": expert parallelism axis (if ep > 1)

    The full mesh tensor has shape ``(pp, dp, tp, ep)`` (with axes of size 1
    collapsed for backends that don't support degenerate dimensions).

    For CUDA, this uses ``torch.distributed.device_mesh.init_device_mesh``.
    For XLA/TPU, this returns mesh parameters suitable for ``xs.Mesh``.

    Args:
        par: Parallelism configuration specifying degrees.
        device_type: "cuda" or "xla".

    Returns:
        Dict with:
        - "mesh": The root DeviceMesh (or Mesh for XLA).
        - "pp": PP sub-mesh (or None if pp=1).
        - "dp": DP sub-mesh (always present, even if dp=1).
        - "tp": TP sub-mesh (or None if tp=1).
        - "ep": EP sub-mesh (or None if ep=1).
        - "shape": Tuple of (pp, dp, tp, ep) degrees.
        - "dim_names": Tuple of active dimension names.

    Raises:
        RuntimeError: If torch.distributed is not initialized (CUDA) or
            XLA devices are not available.
    """
    shape = []
    dim_names = []

    if par.has_pp:
        shape.append(par.pp)
        dim_names.append("pp")

    shape.append(par.dp)
    dim_names.append("dp")

    if par.has_tp:
        shape.append(par.tp)
        dim_names.append("tp")

    if par.has_ep:
        shape.append(par.ep)
        dim_names.append("ep")

    result: dict[str, Any] = {
        "mesh": None,
        "pp": None,
        "dp": None,
        "tp": None,
        "ep": None,
        "shape": tuple(shape),
        "dim_names": tuple(dim_names),
    }

    if device_type == "xla":
        result["mesh"] = _build_xla_mesh(par, shape, dim_names)
        # XLA: sub-meshes are extracted via axis names from the single SPMD mesh
        result["dp"] = result["mesh"]  # SPMD uses named axes, not sub-meshes
    elif device_type == "cuda":
        result["mesh"] = _build_cuda_mesh(shape, dim_names, device_type)
        # Extract sub-meshes for each parallelism dimension
        mesh = result["mesh"]
        if "pp" in dim_names:
            result["pp"] = mesh["pp"]
        if "dp" in dim_names:
            result["dp"] = mesh["dp"]
        if "tp" in dim_names:
            result["tp"] = mesh["tp"]
        if "ep" in dim_names:
            result["ep"] = mesh["ep"]
    else:
        # CPU fallback for testing — return a placeholder dict
        result["dim_names"] = tuple(dim_names)
        result["shape"] = tuple(shape)

    return result


def _build_cuda_mesh(shape, dim_names, device_type):
    """Build a CUDA DeviceMesh with the given shape and dimension names."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(
        device_type,
        tuple(shape),
        mesh_dim_names=tuple(dim_names),
    )


def _build_xla_mesh(par, shape, dim_names):
    """Build an XLA SPMD Mesh for TPU training."""
    import numpy as np
    import torch_xla.runtime as xr  # pyright: ignore[reportMissingImports]
    import torch_xla.distributed.spmd as xs  # pyright: ignore[reportMissingModuleSource]

    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices).reshape(tuple(shape))
    return xs.Mesh(device_ids, tuple(dim_names))


# ---------------------------------------------------------------------------
# Application Order
# ---------------------------------------------------------------------------


_APPLICATION_ORDER = ("tp", "pp", "fsdp2", "ep")
"""Canonical order for applying parallelism strategies.

1. TP first: splits weight matrices within each layer.
2. PP second: partitions layers into pipeline stages.
3. FSDP2 third: shards per-stage parameters across data-parallel ranks.
4. EP last: distributes expert weights across EP ranks.

This matches torchtitan conventions and ensures that:
- TP sharding happens on the original full model (before any layer splitting).
- PP operates on TP-sharded layers.
- FSDP2 wraps each pipeline stage's parameters.
- EP is applied last because it requires the MoE routing infrastructure
  to be in place after FSDP2 wrapping.
"""


@dataclass
class ParallelismResult:
    """Result of apply_3d_parallelism().

    Attributes:
        stages: List of pipeline stage modules (length = pp).
            If pp=1, this is a single-element list with the full model.
        application_order: Tuple of strategy names in the order they were applied.
        pp_partitions: Layer partition ranges [(start, end), ...] per stage.
        meshes: The mesh dict from build_3d_mesh().
        fsdp_active: Whether FSDP2 wrapping was applied.
    """

    stages: list[nn.Module] = field(default_factory=list)
    application_order: tuple[str, ...] = ()
    pp_partitions: list[tuple[int, int]] = field(default_factory=list)
    meshes: dict[str, Any] = field(default_factory=dict)
    fsdp_active: bool = False


def apply_3d_parallelism(
    model: nn.Module,
    par: ParallelismConfig,
    meshes: dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    device_type: str = "cuda",
) -> ParallelismResult:
    """Apply 3D parallelism to a model in the correct order.

    Orchestrates: TP -> PP -> FSDP2 -> EP.

    Args:
        model: The GPT model (already on device, already compiled if using
            regional compile).
        par: Parallelism configuration.
        meshes: Mesh dict from build_3d_mesh().
        device: Target device.  Defaults to model's current device.
        device_type: "cuda" or "xla".

    Returns:
        ParallelismResult with stages and metadata.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    result = ParallelismResult(meshes=meshes)
    applied: list[str] = []

    # --- Step 1: Tensor Parallelism ---
    if par.has_tp:
        _apply_tp(model, meshes, par, device_type)
        applied.append("tp")

    # --- Step 2: Pipeline Parallelism ---
    if par.has_pp:
        from megacpp.pipeline_parallel import partition_model, create_pipeline_stage

        partitions = partition_model(model, par.pp)
        result.pp_partitions = partitions

        stages: list[nn.Module] = []
        for stage_id in range(par.pp):
            stage = create_pipeline_stage(model, stage_id, par.pp, device)
            stages.append(stage)
        result.stages = stages
        applied.append("pp")
    else:
        result.stages = [model]
        n_layer = 0
        transformer = getattr(model, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if isinstance(blocks, Sized):
            n_layer = len(blocks)
        result.pp_partitions = [(0, max(0, n_layer - 1))] if n_layer > 0 else []

    # --- Step 3: FSDP2 ---
    if par.has_fsdp:
        dp_mesh = meshes.get("dp")
        if dp_mesh is not None:
            if device_type == "cuda":
                _apply_fsdp2_cuda(result.stages, dp_mesh, par)
            elif device_type == "xla":
                _apply_fsdp2_xla(model, dp_mesh, par)
            result.fsdp_active = True
            applied.append("fsdp2")

    # --- Step 4: Expert Parallelism ---
    if par.has_ep:
        ep_mesh = meshes.get("ep")
        if ep_mesh is not None:
            _apply_ep(model, result.stages, ep_mesh, par, device_type)
            applied.append("ep")

    result.application_order = tuple(applied)
    return result


def _apply_tp(model, meshes, par, device_type):
    """Apply tensor parallelism to the model.

    Verified CUDA contract from the H200 repro lane:
    - use the named TP sub-mesh (`meshes["tp"]`)
    - enable Sequence Parallel on the TP axis when TP > 1
    - keep FSDP2 on the separate DP sub-mesh (`meshes["dp"]`)
    """
    tp_mesh = meshes.get("tp")
    if tp_mesh is None:
        return

    if device_type == "cuda":
        # Use the CUDA TP path from base_train.py
        # Import deferred to avoid circular dependencies
        from scripts.base_train import _apply_cuda_tensor_parallel

        _apply_cuda_tensor_parallel(
            model,
            tp_mesh,
            enable_sp=bool(getattr(par, "has_tp", False) and getattr(par, "tp", 1) > 1),
        )
    # XLA TP: handled by apply_fsdp_sharding in the FSDP step (combined TP+DP)


def _apply_fsdp2_cuda(stages, dp_mesh, par):
    """Apply CUDA FSDP2 wrapping to each pipeline stage.

    The validated 2D repro uses a separate named DP mesh for FSDP2 and a
    separate TP mesh for TP/SP. This function should only receive the DP axis.
    """
    from megacpp.fsdp_cuda import apply_cuda_fsdp, apply_cuda_fsdp_to_pp_stages

    stage_list = [stage for stage in stages if isinstance(stage, nn.Module)]
    if not stage_list:
        return
    if par.has_pp:
        apply_cuda_fsdp_to_pp_stages(stage_list, dp_mesh)
        return
    for stage in stage_list:
        apply_cuda_fsdp(stage, dp_mesh)


def _apply_fsdp2_xla(model, mesh, par):
    """Apply XLA SPMD FSDP sharding."""
    from megacpp.fsdp import apply_fsdp_sharding

    apply_fsdp_sharding(
        model,
        mesh,
        tp_degree=par.tp,
        dp_degree=par.dp,
        ep_degree=par.ep,
    )


def _apply_ep(model, stages, ep_mesh, par, device_type):
    """Apply expert parallelism to MoE layers.

    On CUDA with PP: creates PP-scoped EP process groups so that AlltoAll
    dispatch only happens between ranks within the same pipeline stage.

    On CUDA without PP: EP process groups span the full world (handled
    by moe_dispatch.create_ep_process_groups in base_train.py).

    On XLA, EP is handled by SPMD mesh sharding (already applied in FSDP step).
    """
    if device_type == "cuda" and par.has_pp:
        # With PP active, EP groups must be scoped within each pipeline stage.
        # Ranks in different stages hold different layers and must NOT exchange
        # expert tokens across stage boundaries.
        import torch.distributed as dist

        from megacpp.moe_dispatch import create_pp_ep_process_groups

        ep_group, ep_dp_group = create_pp_ep_process_groups(
            world_size=dist.get_world_size(),
            pp_size=par.pp,
            ep_size=par.ep,
            tp_size=par.tp,
        )

        # Set EP groups on each stage's MoE layers.
        for stage in stages:
            for module in stage.modules():
                # TokenChoiceMoELayer stores ep_group/ep_size on self.
                if hasattr(module, "ep_group") and hasattr(module, "_ep_dispatcher"):
                    module.ep_group = ep_group
                    module.ep_size = par.ep
                    # Reinitialize the dispatcher with the new group.
                    if module._ep_dispatcher is not None:
                        from megacpp.moe_dispatch import AlltoAllTokenDispatcher

                        module._ep_dispatcher = AlltoAllTokenDispatcher(
                            ep_group=ep_group,
                            n_experts=module.n_routed,
                            ep_size=par.ep,
                            max_active_slots_per_token=module.k_max,
                            # Equal-split path for gradient_checkpointing
                            # determinism. See moe.py comment.
                            force_variable_split=False,
                            # DeepEP: hidden_size needed for Buffer sizing.
                            # use_deepep auto-detected from MEGACPP_MOE_DEEPEP
                            # env var (set by --deepep_dispatch CLI flag).
                            hidden_size=getattr(module, "n_embd", 0),
                            # Megatron-Core fused dispatch: auto-detected from
                            # config.use_megatron_dispatch.
                            use_megatron_dispatch=getattr(
                                getattr(module, "config", None),
                                "use_megatron_dispatch", False,
                            ) if hasattr(module, "config") else False,
                        )
    elif device_type == "cuda":
        # Non-PP CUDA: EP groups created by base_train.py via
        # create_ep_process_groups() before model construction.
        pass
    # XLA EP: already handled by mark_sharding in the FSDP step


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def get_stage_rank(par: ParallelismConfig, global_rank: int) -> dict[str, int]:
    """Compute per-axis ranks from a global rank.

    Given a global rank in a world of ``par.world_size`` devices, returns
    the rank along each parallelism axis.

    The device ordering follows row-major layout: (pp, dp, tp, ep).

    Args:
        par: Parallelism configuration.
        global_rank: Global rank (0-indexed).

    Returns:
        Dict with keys "pp", "dp", "tp", "ep" mapping to per-axis ranks.
    """
    if global_rank < 0 or global_rank >= par.world_size:
        raise ValueError(
            f"global_rank={global_rank} out of range [0, {par.world_size})"
        )

    remaining = global_rank
    ranks: dict[str, int] = {}

    # Row-major decomposition: pp is outermost, ep is innermost
    for name, degree in [("pp", par.pp), ("dp", par.dp), ("tp", par.tp), ("ep", par.ep)]:
        stride = par.world_size
        stride //= degree
        # Compute how many full strides fit
        # Actually, just do standard row-major decomposition
        pass

    # Correct row-major decomposition
    dims = [("pp", par.pp), ("dp", par.dp), ("tp", par.tp), ("ep", par.ep)]
    remaining = global_rank
    for name, degree in dims:
        # stride = product of all subsequent dimensions
        stride = 1
        found = False
        for n2, d2 in dims:
            if found:
                stride *= d2
            if n2 == name:
                found = True
        ranks[name] = remaining // stride
        remaining = remaining % stride

    return ranks


def estimate_memory_per_device(
    par: ParallelismConfig,
    total_params: int,
    bytes_per_param: float = 2.0,
    optimizer_multiplier: float = 10.0,
) -> dict[str, float]:
    """Rough memory estimate per device under 3D parallelism.

    This is a simplified model for capacity planning.  Actual memory depends
    on activation checkpointing, sequence length, batch size, etc.

    Args:
        par: Parallelism configuration.
        total_params: Total model parameters.
        bytes_per_param: Bytes per parameter (2.0 for bf16).
        optimizer_multiplier: Total bytes per param including optimizer states
            (10.0 for bf16 weight + fp32 Adam m + fp32 v).

    Returns:
        Dict with memory estimates in GB:
        - "params_gb": Parameter memory per device.
        - "optimizer_gb": Optimizer state memory per device.
        - "total_gb": Combined estimate.
    """
    # PP: each stage holds ~1/pp of the layers (params)
    params_per_stage = total_params / max(par.pp, 1)

    # TP: each TP rank holds ~1/tp of shardable params
    # (approximately 60-70% of params are shardable by TP)
    tp_shardable_frac = 0.65
    tp_reduction = 1.0 - tp_shardable_frac * (1.0 - 1.0 / max(par.tp, 1))
    params_per_device = params_per_stage * tp_reduction

    # FSDP: each DP rank holds ~1/dp of the per-stage params
    fsdp_reduction = 1.0 / max(par.dp, 1)

    param_bytes = params_per_device * bytes_per_param * fsdp_reduction
    opt_bytes = params_per_device * optimizer_multiplier * fsdp_reduction

    param_gb = param_bytes / (1024**3)
    opt_gb = opt_bytes / (1024**3)

    return {
        "params_gb": param_gb,
        "optimizer_gb": opt_gb,
        "total_gb": param_gb + opt_gb,
    }
