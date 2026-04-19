"""Pipeline Parallelism (PP) donor excerpt.

Public-facing example of how the donor stack partitions transformer layers into
pipeline stages with ``torch.distributed.pipelining``.

Usage:
    from examples.distributed.pipeline_parallel_sample import (
        partition_model, PipelineSchedule, create_pipeline_stage,
        apply_pipeline_parallel,
        partition_model_weighted,
    )

    # Get partition plan (layer-count balanced)
    partitions = partition_model(model, num_stages=4)
    # => [(0, 12), (13, 25), (26, 38), (39, 51)]

    # Weight-aware partition (balances by parameter count, important for MoE)
    partitions = partition_model_weighted(model, num_stages=4)
    # => E-block heavy stages get fewer layers to equalize memory

    # Create stages (requires distributed init)
    stage = create_pipeline_stage(model, stage_id=0, num_stages=4, device=device)

    # Or use the top-level entry point
    pipe = apply_pipeline_parallel(model, num_stages=4, schedule=PipelineSchedule.GPIPE)

Notes:
    - Works with both Nemotron-style (ABlock/MBlock/EBlock) and legacy Block layers.
    - Designed to be compatible with FSDP2 (apply PP first, then FSDP per-stage).
    - Weight-aware partitioning handles MoE E-blocks (64 experts = ~10x heavier).
    - Requires PyTorch 2.4+ with ``torch.distributed.pipelining``.
"""

from __future__ import annotations

import contextlib
import enum
import os
import threading
from typing import Any, Mapping, Optional, Protocol, Sequence, TypedDict, cast

import torch
import torch.nn as nn


class _TransformerWithLayers(Protocol):
    h: nn.ModuleList


class _TransformerWithEmbedding(Protocol):
    wte: nn.Module


class _ModelWithTransformerLayers(Protocol):
    transformer: _TransformerWithLayers


class _ModelWithTransformerEmbedding(Protocol):
    transformer: _TransformerWithEmbedding


class _ConfigWithEmbd(Protocol):
    n_embd: int


StructureMeta = Mapping[str, object]


class StageLayerInfo(TypedDict):
    stage_id: int
    start: int
    end: int
    n_layers: int
    n_params: int
    n_params_gb: float
    block_types: dict[str, int]


def _require_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if isinstance(model, nn.ModuleList):
        return model
    transformer = getattr(model, "transformer", None)
    layers = getattr(transformer, "h", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    raise ValueError(
        "Expected a GPT model with model.transformer.h (nn.ModuleList) "
        "or a bare nn.ModuleList."
    )


def _optional_module_attr(obj: object, name: str) -> nn.Module | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, nn.Module) else None


def _as_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    return value


# ---------------------------------------------------------------------------
# Aux loss injector
# ---------------------------------------------------------------------------


class _AuxLossInjector(torch.autograd.Function):
    """Inject auxiliary loss gradients into a hidden-state tensor.

    This follows the same basic pattern as Megatron-Core's auxiliary-loss
    scaler: attach an auxiliary loss to the hidden states' backward graph
    without changing forward values, then inject a scaled gradient during
    backward so router weights receive the intended signal.

    **Loss scaling** depends on the PP schedule:

    **torch.distributed.pipelining (1F1B, GPipe, VPP):**
    Scale = ``1.0 / grad_accum_steps``.  The schedule runs M microbatches
    and then calls ``stage.scale_grads(M)`` which divides ALL parameter
    gradients (including aux-injected router grads) by M.  So the
    injector fires M times at ``1/ga`` each, accumulates to ``M/ga``,
    then ``scale_grads`` divides by M => effective ``1/ga``, matching the
    main loss path.

    **DualPipe / DualPipeV:**
    Scale = ``1.0 / (grad_accum_steps * num_chunks)``.  DualPipe has NO
    ``scale_grads`` post-hoc mechanism.  Its loss function divides the
    main loss by ``ga * num_chunks``.  The injector must match this
    composite scale, otherwise router gradients are ``num_chunks`` times
    too large.

    Call :meth:`set_loss_scale` with the appropriate scale before the forward
    pass that produces the auxiliary losses.

    This is used for all non-last pipeline stages (both VPP chunks and
    standard 1f1b/gpipe PP) whose aux losses would otherwise be lost:
    - VPP: cleared by the next microbatch's forward before loss_fn drains
    - Standard PP: loss_fn only runs on the last stage, so non-last stage
      aux losses have no path to the backward pass without injection.
    """

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        if _AuxLossInjector.main_loss_backward_scale is not None:
            scaled_aux_loss_grad = torch.ones_like(aux_loss) * _AuxLossInjector.main_loss_backward_scale
        else:
            # Fallback: unit gradient (legacy behavior, but callers should
            # always set the scale via set_loss_scale).
            scaled_aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the backward scale for injected auxiliary losses.

        Must match the effective scale applied to the main loss before
        ``.backward()``.

        - torch.pipelining (1F1B/GPipe/VPP): ``1.0 / grad_accum_steps``
          (``scale_grads(M)`` handles microbatch factor).
        - DualPipe: ``1.0 / (grad_accum_steps * num_chunks)``
          (no ``scale_grads``; loss_fn divides by full product).

        Args:
            scale: A scalar tensor with the desired gradient scale.
        """
        if _AuxLossInjector.main_loss_backward_scale is None:
            _AuxLossInjector.main_loss_backward_scale = scale.clone().detach()
        else:
            _AuxLossInjector.main_loss_backward_scale.copy_(scale)


def inject_aux_loss(hidden_states: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """Attach *aux_loss* to *hidden_states*'s backward graph.

    Forward returns ``hidden_states`` unchanged.  Backward triggers
    ``aux_loss.backward()`` with a gradient scaled by
    :attr:`_AuxLossInjector.main_loss_backward_scale` (set via
    :meth:`_AuxLossInjector.set_loss_scale`) so the router that
    produced the loss receives correctly-scaled gradient signals.

    Args:
        hidden_states: The main activation tensor flowing through the
            pipeline (shape ``[B, T, D]``).
        aux_loss: A scalar auxiliary loss (e.g. MoE load-balance or
            MoD router loss).

    Returns:
        ``hidden_states`` — identical in forward, with aux_loss
        attached to the backward graph.
    """
    return cast(torch.Tensor, _AuxLossInjector.apply(hidden_states, aux_loss))


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_PIPELINING_AVAILABLE = False
try:
    from torch.distributed.pipelining import (  # noqa: F401
        PipelineStage,
        SplitPoint,
        pipeline,
    )

    _PIPELINING_AVAILABLE = True
except ImportError:
    pass


def _require_pipelining():
    """Raise if the pipelining backend is not available."""
    if not _PIPELINING_AVAILABLE:
        raise NotImplementedError(
            "Pipeline parallelism requires PyTorch 2.4+ with "
            "torch.distributed.pipelining support. "
            "Install a compatible PyTorch version or build from source."
        )


# ---------------------------------------------------------------------------
# Schedule enum
# ---------------------------------------------------------------------------


class PipelineSchedule(enum.Enum):
    """Supported pipeline schedules.

    SIMPLE_1F1B: One forward, one backward per micro-batch, sequential.
        Low memory (only 1 in-flight micro-batch per stage), simple to reason
        about.  Some pipeline bubble at the start and end.

    INTERLEAVED_1F1B: Interleaved 1F1B with virtual stages (Megatron-LM v3).
        Each rank holds multiple non-contiguous stage chunks.  Reduces pipeline
        bubble by a factor of ``v`` (number of virtual stages per rank) at the
        cost of more communication.

    GPIPE: All forwards then all backwards (Huang et al., 2019).
        Maximally simple, but keeps all micro-batch activations alive
        simultaneously — highest peak memory.
    """

    SIMPLE_1F1B = "1f1b"
    INTERLEAVED_1F1B = "interleaved_1f1b"
    GPIPE = "gpipe"
    VPP = "vpp"


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def partition_model(
    model: nn.Module,
    num_stages: int,
) -> list[tuple[int, int]]:
    """Partition model.transformer.h layers into ``num_stages`` groups.

    Returns a list of ``(start_layer_idx, end_layer_idx)`` tuples (inclusive on
    both ends) such that the union covers all layers and no stage has more than
    ``ceil(n / num_stages)`` layers (i.e., the maximum imbalance is at most 1
    layer).

    Args:
        model: A GPT model instance with ``model.transformer.h`` as an
            ``nn.ModuleList`` of transformer blocks.
        num_stages: Number of pipeline stages.  Must be >= 1 and <= the number
            of layers.

    Returns:
        List of (start, end) tuples of length ``num_stages``.

    Raises:
        ValueError: If ``num_stages`` is out of range or the model structure is
            unexpected.
    """
    # Accept both full GPT models and nn.ModuleList directly (for testing).
    layers = _require_transformer_layers(model)

    n_layers = len(layers)

    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if num_stages > n_layers:
        raise ValueError(
            f"num_stages ({num_stages}) > number of layers ({n_layers}). "
            f"Cannot have more stages than layers."
        )

    # Greedy balanced partition: distribute layers as evenly as possible.
    # First (n_layers % num_stages) stages get ceil(n_layers/num_stages)
    # layers, the rest get floor(n_layers/num_stages).
    base = n_layers // num_stages
    remainder = n_layers % num_stages

    partitions: list[tuple[int, int]] = []
    start = 0
    for i in range(num_stages):
        count = base + (1 if i < remainder else 0)
        end = start + count - 1
        partitions.append((start, end))
        start = end + 1

    assert len(partitions) == num_stages
    assert partitions[-1][1] == n_layers - 1
    return partitions


def partition_model_explicit(
    model: nn.Module,
    pp_boundaries: list[int],
) -> list[tuple[int, int]]:
    """Partition model using explicit PP stage boundaries.

    *pp_boundaries* is a list of layer indices where each new stage starts
    (excluding stage 0, which always starts at layer 0).  For example,
    boundaries [4] with 8 layers -> stages [(0,3), (4,7)].
    Boundaries [2, 6] with 8 layers -> [(0,1), (2,5), (6,7)].

    This is used when the user provides a pipe-delimited nem_pattern like
    "AEME|AEME" which explicitly marks where PP stages begin.

    Args:
        model: A GPT model with model.transformer.h or a bare
            nn.ModuleList.
        pp_boundaries: Sorted list of layer indices where new stages begin.
            Must not include 0.  All values must be < n_layers.

    Returns:
        List of (start, end) tuples of length len(pp_boundaries) + 1.
    """
    layers = _require_transformer_layers(model)

    n_layers = len(layers)

    if not pp_boundaries:
        raise ValueError("pp_boundaries must be non-empty")

    # Validate boundaries
    for i, b in enumerate(pp_boundaries):
        if b <= 0:
            raise ValueError(
                f"pp_boundaries[{i}]={b} must be > 0 (stage 0 always starts at 0)"
            )
        if b >= n_layers:
            raise ValueError(
                f"pp_boundaries[{i}]={b} >= n_layers={n_layers}"
            )
    # Check sorted and unique
    if pp_boundaries != sorted(set(pp_boundaries)):
        raise ValueError(
            f"pp_boundaries must be sorted and unique, got {pp_boundaries}"
        )

    # Build partitions from boundaries
    starts = [0] + list(pp_boundaries)
    ends = [b - 1 for b in pp_boundaries] + [n_layers - 1]
    partitions = list(zip(starts, ends))

    assert len(partitions) == len(pp_boundaries) + 1
    assert partitions[-1][1] == n_layers - 1
    # Verify contiguity
    for i in range(1, len(partitions)):
        assert partitions[i][0] == partitions[i - 1][1] + 1
    return partitions


def _layer_param_count(layer: nn.Module) -> int:
    """Count total parameters in a layer (including sub-modules)."""
    return sum(p.numel() for p in layer.parameters())


def partition_model_weighted(
    model: nn.Module,
    num_stages: int,
) -> list[tuple[int, int]]:
    """Partition model layers by parameter weight, not count.

    MoE E-blocks in NAM52 have 64 experts (each with W_fc, W_proj, W_gate),
    making them ~10x heavier than A-blocks or M-blocks in parameter count.
    Naive layer-count partitioning would put wildly unequal memory load on
    different pipeline stages.

    This function uses a greedy algorithm that assigns layers sequentially to
    the stage with the smallest accumulated weight so far, maintaining
    contiguous layer assignments (required for pipeline parallelism).

    Falls back to ``partition_model`` when all layers have equal weight
    (e.g., non-MoE models or uniform blocks).

    Args:
        model: A GPT model instance with ``model.transformer.h`` as an
            ``nn.ModuleList`` of transformer blocks.
        num_stages: Number of pipeline stages.

    Returns:
        List of (start, end) tuples of length ``num_stages``.

    Raises:
        ValueError: If ``num_stages`` is out of range or the model structure
            is unexpected.
    """
    layers = _require_transformer_layers(model)

    n_layers = len(layers)

    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if num_stages > n_layers:
        raise ValueError(
            f"num_stages ({num_stages}) > number of layers ({n_layers}). "
            f"Cannot have more stages than layers."
        )

    # Compute per-layer weights (parameter counts).
    weights = [_layer_param_count(layer) for layer in layers]
    total_weight = sum(weights)

    if total_weight == 0:
        # All layers empty (unlikely): fall back to count-based.
        return partition_model(model, num_stages)

    # Greedy contiguous partitioning: assign layers to stages such that
    # each stage's total weight is as close to (total_weight / num_stages)
    # as possible.  We must keep layers contiguous.
    target_per_stage = total_weight / num_stages
    partitions: list[tuple[int, int]] = []
    start = 0
    cumulative = 0.0

    for stage_idx in range(num_stages):
        if stage_idx == num_stages - 1:
            # Last stage gets all remaining layers.
            partitions.append((start, n_layers - 1))
            break

        # Greedily add layers until we exceed the target for this stage.
        # Always assign at least one layer per stage.
        stage_weight = 0.0
        end = start
        remaining_stages = num_stages - stage_idx

        # Ensure enough layers remain for subsequent stages (1 each).
        max_end = n_layers - remaining_stages  # exclusive upper bound for this stage

        while end <= max_end:
            stage_weight += weights[end]
            # Check if adding the next layer would overshoot more than not adding.
            if end + 1 <= max_end and stage_weight < target_per_stage:
                end += 1
            else:
                break

        partitions.append((start, end))
        cumulative += stage_weight
        # Recalculate target for remaining stages based on remaining weight.
        remaining_weight = total_weight - cumulative
        remaining_stages_after = num_stages - stage_idx - 1
        if remaining_stages_after > 0:
            target_per_stage = remaining_weight / remaining_stages_after
        start = end + 1

    assert len(partitions) == num_stages
    assert partitions[-1][1] == n_layers - 1
    # Verify contiguity and full coverage.
    for i in range(1, len(partitions)):
        assert partitions[i][0] == partitions[i - 1][1] + 1
    return partitions


def get_stage_layer_info(
    model: nn.Module,
    num_stages: int,
    *,
    weighted: bool = True,
) -> list[StageLayerInfo]:
    """Get detailed per-stage information for capacity planning.

    Returns per-stage dicts with layer counts, parameter counts, and
    block type breakdown (A/M/E blocks for Nemotron-style models).

    Args:
        model: GPT model with ``model.transformer.h``.
        num_stages: Number of pipeline stages.
        weighted: If True, use weight-aware partitioning.

    Returns:
        List of dicts, one per stage, with keys:
        - "stage_id": int
        - "start": int (first layer index)
        - "end": int (last layer index, inclusive)
        - "n_layers": int
        - "n_params": int (total parameters in this stage's layers)
        - "n_params_gb": float (params in GB at bf16)
        - "block_types": dict mapping block type name to count
    """
    layers = _require_transformer_layers(model)

    if weighted:
        partitions = partition_model_weighted(model, num_stages)
    else:
        partitions = partition_model(model, num_stages)

    result: list[StageLayerInfo] = []
    for stage_id, (start, end) in enumerate(partitions):
        stage_layers = [layers[i] for i in range(start, end + 1)]
        n_params = sum(_layer_param_count(layer) for layer in stage_layers)

        # Count block types.
        block_types: dict[str, int] = {}
        for layer in stage_layers:
            type_name = type(layer).__name__
            block_types[type_name] = block_types.get(type_name, 0) + 1

        result.append({
            "stage_id": stage_id,
            "start": start,
            "end": end,
            "n_layers": end - start + 1,
            "n_params": n_params,
            "n_params_gb": n_params * 2 / (1024 ** 3),  # bf16
            "block_types": block_types,
        })

    return result


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------


def _split_structure_meta(
    meta: StructureMeta | None,
    num_chunks: int,
) -> Sequence[StructureMeta | None]:
    """Split a batched structure_meta dict into per-microbatch chunks.

    When pipeline parallelism uses multiple microbatches (``pp_microbatches > 1``),
    the schedule splits ``x`` and ``target`` into microbatches internally, but
    ``structure_meta`` is set once before ``schedule.step()``.  This function
    pre-splits the metadata so each microbatch's forward sees the correct slice.

    **Splitting logic**:
    - ``None`` input returns ``[None] * num_chunks``.
    - Tensor values whose dim-0 matches the batch size (inferred from the first
      batched tensor found) are split with ``torch.tensor_split(t, num_chunks, dim=0)``.
    - Non-tensor values (ints, strings, None) and scalar/1-element tensors are
      replicated across all chunks.

    Args:
        meta: The full-batch structure_meta dict, or None.
        num_chunks: Number of microbatches to split into.

    Returns:
        A list of ``num_chunks`` dicts (or Nones), one per microbatch.
    """
    if meta is None or num_chunks <= 1:
        return [meta] * max(num_chunks, 1)

    # Infer batch size from the first batched tensor (dim-0).
    batch_size: Optional[int] = None
    for v in meta.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            batch_size = v.shape[0]
            break

    if batch_size is None:
        # No batched tensors — replicate the whole dict.
        return [meta] * num_chunks

    # Build per-chunk dicts.
    chunks: list[dict[str, object]] = [dict() for _ in range(num_chunks)]
    for key, val in meta.items():
        if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == batch_size:
            parts = torch.tensor_split(val, num_chunks, dim=0)
            for i, part in enumerate(parts):
                chunk = chunks[i]
                assert chunk is not None
                chunk[key] = part
        else:
            # Scalar tensor, non-tensor, or tensor with non-batch dim-0 — replicate.
            for i in range(num_chunks):
                chunk = chunks[i]
                assert chunk is not None
                chunk[key] = val

    return chunks


class _PipelineStageModule(nn.Module):
    """A thin wrapper around a contiguous subset of transformer layers.

    This module holds references (not copies) to the original layers and
    implements a forward pass that chains them sequentially.  It is designed
    to be passed to ``PipelineStage`` from ``torch.distributed.pipelining``.

    The forward signature matches what ABlock/MBlock/EBlock/Block expect:
    ``(x, cos_sin, window_size, kv_cache, doc_ids, **kwargs)``.  The first
    stage additionally handles embedding (idx -> x) and RoPE buffer slicing.

    **Output tensor deallocation** (Megatron's ``deallocate_output_tensor``
    pattern): Non-last stages free their output tensor's bulk storage after
    it has been sent to the next stage.  This is done lazily: at the start
    of the *next* micro-batch's forward, the previous output's ``.data`` is
    replaced with a scalar tensor, releasing the activation memory while
    keeping the autograd graph node alive for backward.  The last stage
    never deallocates because its output feeds directly into the loss.

    Enable with ``stage.enable_output_deallocation()`` (on by default for
    non-last stages).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        *,
        embed: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        config: object = None,
        global_layer_offset: int = 0,
        window_sizes: Optional[list] = None,
        nope_layers: Optional[set] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        ngram_hash: Optional[nn.Module] = None,
        structure_emb: Optional[nn.Module] = None,
        platform_emb: Optional[nn.Module] = None,
        dyt_embed: Optional[nn.Module] = None,
        relation_bias: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = layers
        self.embed = embed  # wte — only on first stage
        self.head = head  # lm_head — only on last stage
        self.config = config
        self.global_layer_offset = global_layer_offset
        self.window_sizes = window_sizes or [None] * len(layers)
        self.nope_layers = nope_layers or set()

        # Pre-block modules (first stage only).  These enrich the token
        # embeddings before they enter the transformer blocks, matching the
        # preprocessing that GPT.forward() performs.
        self.ngram_hash = ngram_hash
        self.structure_emb = structure_emb
        self.platform_emb = platform_emb
        self.dyt_embed = dyt_embed

        # Relation bias module reference — present on all stages so each
        # stage can compute the additive attention bias locally from the
        # per-step structure_meta.  This is a reference (not a copy) to
        # the model's RelationBias nn.Module.
        self.relation_bias = relation_bias

        # Per-step structure_meta dict, set by the training loop before
        # each schedule.step() call via set_structure_meta().  Read by
        # forward() to apply structure_emb, platform_emb, relation_bias,
        # and to pass structure_meta/attention_meta kwargs to blocks.
        self._structure_meta: StructureMeta | None = None

        # Microbatch-aware structure_meta splitting.  When
        # set_structure_meta() is called with num_microbatches > 1, the
        # full-batch meta is pre-split into chunks.  Each forward() call
        # pops the next chunk via _microbatch_counter.
        self._num_microbatches: int = 1
        self._structure_meta_chunks: Sequence[StructureMeta | None] = []
        self._microbatch_counter: int = 0

        # RoPE buffers — registered as persistent=False buffers so they move
        # with .to(device) and .half() but are not saved in state_dict.
        if cos is not None:
            self.register_buffer("cos", cos, persistent=False)
        else:
            self.cos = None
        if sin is not None:
            self.register_buffer("sin", sin, persistent=False)
        else:
            self.sin = None

        # Output tensor deallocation state.
        # _deallocate_output: whether to free output after send (non-last stages).
        # _saved_output: reference to the most recent forward output, deallocated
        #   at the start of the next forward call.
        # NOTE: Disabled by default — causes shape mismatch errors when the
        # pipeline schedule needs the cached output for backward AFTER the
        # next microbatch forward has started. Set PP_DEALLOC=1
        # to re-enable for memory savings (Megatron pattern).
        import os as _os
        self._deallocate_output: bool = (
            stage_id < num_stages - 1
            and _os.environ.get("PP_DEALLOC", "0") == "1"
        )
        self._saved_output: Optional[torch.Tensor] = None

        # ---- Auxiliary loss collection (MoE, MoD) ----
        # Per-microbatch lists populated during forward(), consumed by the PP
        # loss_fn closure.  Cleared at the start of each forward() call.
        self._moe_aux_losses: list[torch.Tensor] = []
        self._moe_z_losses: list[torch.Tensor] = []
        self._mod_aux_losses: list[torch.Tensor] = []
        # Pre-lm_head hidden states for MTP (last stage only).
        self._last_x_prenorm: Optional[torch.Tensor] = None
        # Stored cos_sin for MTP on last stage.
        self._last_cos_sin = None

        # Aux-loss injection flag.  When True, non-last stages inject their
        # accumulated aux losses into the hidden-state backward graph via
        # _AuxLossInjector at the end of forward() instead of relying on
        # drain-by-loss_fn.  This prevents aux losses from being lost:
        # - VPP: next microbatch's forward calls _clear_aux_losses()
        # - Standard PP: loss_fn only runs on the last stage
        # Set to True by enable_aux_loss_injection() for any non-last stage.
        self._inject_aux_into_graph: bool = False

    @staticmethod
    def deallocate_output_tensor(tensor: torch.Tensor) -> None:
        """Replace a tensor's storage with a scalar to free activation memory.

        This follows Megatron-LM's ``deallocate_output_tensor`` pattern.  The
        tensor object (and its autograd node) stays alive, but the underlying
        data storage is released.  Backward still works because gradient
        checkpointing recomputes activations as needed.

        Args:
            tensor: The output tensor whose bulk data should be freed.
                Must be a plain ``torch.Tensor`` (not a tuple/list).
        """
        if tensor is None:
            return
        assert isinstance(tensor, torch.Tensor), (
            f"Expected torch.Tensor, got {type(tensor).__name__}"
        )
        tensor.data = torch.empty(1, device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def overlapped_forward_backward(
        module0: "_PipelineStageModule",
        inputs0: Any,
        criterion0: Optional[Any],
        labels0: Any,
        module1: "_PipelineStageModule",
        loss1: Optional[torch.Tensor],
        outputs1: Optional[Any],
        output_grads1: Optional[Any],
    ) -> tuple[Any, Optional[torch.Tensor]]:
        """Sequential fallback matching the DualPipe overlap entry contract."""
        del module1

        if loss1 is not None:
            loss1.backward()
        elif outputs1 is not None and output_grads1 is not None:
            torch.autograd.backward(outputs1, output_grads1)

        if isinstance(inputs0, tuple):
            outputs0 = module0(*inputs0)
        elif isinstance(inputs0, list):
            outputs0 = module0(*tuple(inputs0))
        else:
            outputs0 = module0(inputs0)

        loss0 = criterion0(outputs0, labels0) if criterion0 is not None else None
        return outputs0, loss0

    def enable_aux_loss_injection(self) -> None:
        """Enable aux-loss injection into the hidden-state backward graph.

        For non-last pipeline stages (both VPP chunks and standard PP):
        instead of relying on the loss_fn closure to drain aux losses,
        inject them into the autograd graph at the end of forward().
        This follows Megatron-Core's ``MoEAuxLossAutoScaler`` pattern.
        """
        self._inject_aux_into_graph = True

    def enable_output_deallocation(self) -> None:
        """Enable output tensor deallocation (default for non-last stages)."""
        self._deallocate_output = True

    def disable_output_deallocation(self) -> None:
        """Disable output tensor deallocation."""
        self._deallocate_output = False
        self._saved_output = None

    def refresh_head(self, new_head: Optional[nn.Module]) -> None:
        """Re-bind the lm_head reference after TP or other module replacement.

        When Megatron TP replaces ``model.lm_head`` with a new
        ``ColumnParallelLinear``, the PP stage's ``self.head`` still points to
        the old ``nn.Linear``.  This method updates the reference so the last
        stage emits logits through the TP-sharded head, and the optimizer sees
        the correct (live) parameters.

        Only meaningful on the last pipeline stage (where ``self.head`` is not
        None).  Calling on non-last stages with ``new_head=None`` is a no-op.
        """
        self.head = new_head

    def set_structure_meta(
        self, meta: StructureMeta | None, *, num_microbatches: int = 1,
    ) -> None:
        """Set per-step structure_meta for this stage.

        Called by the training loop before each ``schedule.step()`` call.
        The forward() method reads ``self._structure_meta`` to:
        - Apply structure_emb / platform_emb enrichment (first stage).
        - Compute relation_struct_bias and pass it to blocks.
        - Build attention_meta from structure_meta fields.
        - Pass structure_meta to blocks via kwargs for structure-aware
          attention and TreeFFN.

        When ``num_microbatches > 1``, the metadata is pre-split into
        per-microbatch chunks via :func:`_split_structure_meta`.  Each
        subsequent ``forward()`` call selects the next chunk automatically.
        """
        self._num_microbatches = max(num_microbatches, 1)
        self._microbatch_counter = 0
        if self._num_microbatches > 1 and meta is not None:
            self._structure_meta_chunks = _split_structure_meta(
                meta, self._num_microbatches,
            )
            # Set _structure_meta to the first chunk so that callers who
            # read it directly (e.g. warmup code) get something valid.
            self._structure_meta = self._structure_meta_chunks[0]
        else:
            self._structure_meta = meta
            self._structure_meta_chunks = []

    @property
    def is_last_stage(self) -> bool:
        """Whether this is the last pipeline stage."""
        return self.stage_id == self.num_stages - 1

    def _clear_aux_losses(self) -> None:
        """Reset auxiliary loss accumulators at the start of each forward."""
        self._moe_aux_losses = []
        self._moe_z_losses = []
        self._mod_aux_losses = []
        self._last_x_prenorm = None
        self._last_cos_sin = None

    def pop_aux_losses(self) -> dict[str, list[torch.Tensor]]:
        """Return accumulated auxiliary losses and clear them.

        Called by the PP loss_fn closure after forward to add aux losses
        to the CE loss.  Returns a dict with keys:
        - ``moe_aux``: list of per-layer MoE load-balancing losses
        - ``moe_z``: list of per-layer MoE router z-losses
        - ``mod_aux``: list of per-layer MoD router auxiliary losses
        """
        result = {
            "moe_aux": self._moe_aux_losses,
            "moe_z": self._moe_z_losses,
            "mod_aux": self._mod_aux_losses,
        }
        self._moe_aux_losses = []
        self._moe_z_losses = []
        self._mod_aux_losses = []
        return result

    def _collect_block_aux_losses(self, result, layer=None) -> torch.Tensor:
        """Extract hidden states and aux losses from a block's return value.

        Block forward returns one of:
        - Plain tensor: just hidden states (no aux losses)
        - 2-tuple: (x, attn_summary) from attention blocks
        - 3-tuple (MoE): (x, aux_loss, z_loss) from MoE EBlocks
        - 3-tuple (MoD): (x, router_aux, zeros) from MoD wrapping a dense block
        - 5-tuple: (x, aux_loss, z_loss, router_aux, rank_metric) from MoD

        When *layer* is provided, MoD wrapper instances are detected by type
        so that 3-tuples from MoD (router_aux in slot 1) are routed to the
        ``_mod_aux_losses`` bucket instead of ``_moe_aux_losses``.

        Collects MoE and MoD losses into the stage's accumulators.
        Returns the hidden states tensor.
        """
        if not isinstance(result, tuple):
            return result

        if len(result) == 2:
            # Attention summary 2-tuple — no aux losses to collect
            return result[0]

        if len(result) == 5:
            # MoD 5-tuple: (x, aux_loss, z_loss, router_aux, rank_metric)
            x, aux_loss, z_loss, router_aux, _rank_metric = result
            # Router aux goes to MoD bucket
            if router_aux is not None and router_aux.numel() > 0:
                self._mod_aux_losses.append(router_aux)
            # Inner MoE aux/z losses (if the MoD wraps an MoE EBlock)
            if aux_loss.numel() > 0 and z_loss.numel() > 0:
                self._moe_aux_losses.append(aux_loss)
                self._moe_z_losses.append(z_loss)
            return x

        if len(result) == 3:
            x, aux_loss, z_loss = result

            # Detect MoD wrapper by type: MoD wrapping a dense (non-MoE) block
            # returns (x, router_aux, zeros) as a 3-tuple.  The router_aux must
            # go to _mod_aux_losses, not _moe_aux_losses.
            from runtime_model.mod import GammaMoDBlockWrapper, MoDBlockWrapper
            _is_mod = layer is not None and isinstance(
                layer, (MoDBlockWrapper, GammaMoDBlockWrapper)
            )
            if _is_mod:
                # MoD 3-tuple: slot 1 is router_aux, slot 2 is inner z_loss
                # (zero when wrapping a dense block, nonzero when wrapping MoE).
                inner_has_moe = getattr(layer, "_last_inner_aux_present", False)
                if aux_loss.numel() > 0:
                    if inner_has_moe:
                        self._moe_aux_losses.append(aux_loss)
                    else:
                        self._mod_aux_losses.append(aux_loss)
                if inner_has_moe and z_loss.numel() > 0:
                    self._moe_z_losses.append(z_loss)
            else:
                # True MoE 3-tuple: (x, aux_loss, z_loss)
                if aux_loss.numel() > 0:
                    self._moe_aux_losses.append(aux_loss)
                if z_loss.numel() > 0:
                    self._moe_z_losses.append(z_loss)
            return x

        # Unexpected tuple length — just return first element
        return result[0]

    def forward(
        self,
        x: torch.Tensor,
        cos_sin=None,
        window_size=None,
        kv_cache=None,
        doc_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """Sequential forward through the stage's layers.

        For the first stage, ``x`` is token IDs (Long) and gets embedded.
        For intermediate stages, ``x`` is hidden states (float).
        For the last stage, output is logits (float).

        All block-level args (cos_sin, window_size, kv_cache, doc_ids, kwargs)
        are passed through to each layer.  ``window_size`` is overridden
        per-layer from the stored ``self.window_sizes`` list.  Layers in
        ``self.nope_layers`` receive ``cos_sin=None``.

        **Auxiliary loss collection**: MoE and MoD auxiliary losses from each
        layer are accumulated in ``_moe_aux_losses``, ``_moe_z_losses``, and
        ``_mod_aux_losses``.  The PP loss_fn closure reads and clears these
        after forward via ``pop_aux_losses()``.

        **Output deallocation**: When ``_deallocate_output`` is True (non-last
        stages), the previous micro-batch's output tensor is deallocated at
        the start of this call, and the current output is saved for
        deallocation at the start of the next call.  This frees activation
        memory after the output has been sent to the next pipeline stage.
        """
        # Clear aux losses from the previous micro-batch.
        self._clear_aux_losses()

        # Deallocate previous micro-batch's output (Megatron pattern).
        # By the time the next forward runs, the pipeline schedule has already
        # sent the previous output to the next stage via P2P.
        if self._deallocate_output and self._saved_output is not None:
            _PipelineStageModule.deallocate_output_tensor(self._saved_output)
            self._saved_output = None

        # Select the correct per-microbatch structure_meta chunk.
        # When num_microbatches > 1, set_structure_meta() pre-splits the
        # full-batch meta.  Each forward() invocation (one per microbatch)
        # advances the counter.
        if self._structure_meta_chunks:
            idx_mb = self._microbatch_counter % len(self._structure_meta_chunks)
            self._structure_meta = self._structure_meta_chunks[idx_mb]
            self._microbatch_counter += 1

        if self.embed is not None:
            # First stage: x is token IDs (Long), embed them and apply all
            # pre-block enrichments matching GPT.forward()'s preprocessing.
            idx = x.detach().clone()  # detach+clone: DualPipe concurrent microbatch safety
            x = self.embed(idx)  # use cloned idx, not shared x
            if self.config is not None and getattr(self.config, "embed_scale", False):
                stage_config = cast(_ConfigWithEmbd, self.config)
                x = x * (stage_config.n_embd ** 0.5)

            # N-gram hash enrichment (additive, before norm).
            if self.ngram_hash is not None:
                x = x + self.ngram_hash(idx.clone())

            # Structure embeddings and platform embeddings — applied from
            # the per-step _structure_meta set via set_structure_meta().
            _smeta = self._structure_meta
            if self.structure_emb is not None and isinstance(_smeta, dict):
                import os as _os
                struct_out = self.structure_emb(
                    _smeta.get("structure_ids"),
                    _smeta.get("dep_levels"),
                    _smeta.get("ast_depth_ids"),
                    _smeta.get("sibling_index_ids"),
                    _smeta.get("node_type_ids"),
                    target_dtype=x.dtype,
                )
                # Match GPT.forward() CUDA detach guard for embedding backward.
                if (
                    x.device.type == "cuda"
                    and _os.environ.get("PP_STRUCTURE_EMB_DETACH", "1") != "0"
                ):
                    struct_out = struct_out.detach()
                _pre_dtype = x.dtype
                x = x + struct_out
                if x.dtype != _pre_dtype:
                    x = x.to(_pre_dtype)

            if self.platform_emb is not None and isinstance(_smeta, dict):
                platform_ids = _smeta.get("platform_ids")
                if platform_ids is not None:
                    x = x + self.platform_emb(platform_ids)

            # DyT / pre-norm (RMSNorm or DynamicTanh on the embeddings).
            if self.dyt_embed is not None:
                x = self.dyt_embed(x)

            # Auto-compute doc_ids from BOS tokens for document masking.
            # Prefer explicit per-step metadata first so PP matches
            # compute_loss()/GPT.forward() document-masking semantics.
            if doc_ids is None and isinstance(_smeta, dict):
                _meta_doc_ids = _smeta.get("doc_ids")
                if isinstance(_meta_doc_ids, torch.Tensor):
                    doc_ids = _meta_doc_ids
            if doc_ids is None and self.config is not None and kv_cache is None:
                bos_token_id = getattr(self.config, "bos_token_id", None)
                if isinstance(bos_token_id, int) and bos_token_id > 0:
                    doc_ids = (idx == bos_token_id).to(torch.int32).cumsum(dim=1)
            if doc_ids is not None:
                doc_ids = doc_ids.clone()

            # Compute cos_sin from stored RoPE buffers if not provided.
            if cos_sin is None and self.cos is not None and self.sin is not None:
                B, T = x.shape[:2]
                cos_sin = (self.cos[:, :T].clone(), self.sin[:, :T].clone())

        # ---- Build block-level kwargs from structure_meta ----
        # Mirrors GPT.forward()'s preprocessing of structure_meta into
        # relation_struct_bias, attention_meta, and block kwargs.  Each
        # stage reads self._structure_meta (set by the training loop via
        # set_structure_meta()) to provide these to its blocks.
        _block_kwargs = dict(kwargs)
        _smeta = self._structure_meta
        if isinstance(_smeta, dict):
            _block_kwargs["structure_meta"] = _smeta

            # Build attention_meta from structure_meta fields + doc_ids.
            _valid_token_counts = _smeta.get("row_valid_token_counts")
            _valid_block_counts = _smeta.get("row_valid_block_counts")
            _valid_slot_counts = _smeta.get("row_valid_slot_counts")
            _base_block_tokens = _smeta.get("base_block_tokens")
            _row_block_size_tokens = _smeta.get("row_block_size_tokens")
            if (
                doc_ids is not None
                or _valid_token_counts is not None
                or _valid_block_counts is not None
                or _valid_slot_counts is not None
            ):
                attention_meta = {
                    "doc_ids": doc_ids,
                    "row_valid_token_counts": _valid_token_counts,
                    "row_valid_block_counts": _valid_block_counts,
                    "row_valid_slot_counts": _valid_slot_counts,
                    "base_block_tokens": _base_block_tokens,
                    "row_block_size_tokens": _row_block_size_tokens,
                }
                from runtime_model.gpt import _build_attention_validity

                attention_validity = _build_attention_validity(
                    attention_meta,
                    doc_ids=doc_ids,
                )
                if attention_validity is not None:
                    attention_meta["attention_validity"] = attention_validity
                _block_kwargs["attention_meta"] = attention_meta
        elif doc_ids is not None:
            attention_meta = {"doc_ids": doc_ids}
            from runtime_model.gpt import _build_attention_validity

            attention_validity = _build_attention_validity(
                attention_meta,
                doc_ids=doc_ids,
            )
            if attention_validity is not None:
                attention_meta["attention_validity"] = attention_validity
            _block_kwargs["attention_meta"] = attention_meta

        # Compute relation_struct_bias from the relation_bias module and
        # structure_meta, matching GPT._build_relation_struct_bias().
        if self.relation_bias is not None and isinstance(_smeta, dict):
            _chunk_rel_mask = _smeta.get("chunk_relation_mask")
            if isinstance(_chunk_rel_mask, torch.Tensor) and _chunk_rel_mask.ndim == 4:
                relation_bias = cast(Any, self.relation_bias)
                _mask_float = _chunk_rel_mask.to(dtype=relation_bias.bias_table.dtype)
                _block_kwargs["relation_struct_bias"] = torch.einsum(
                    "brij,rh->bhij", _mask_float, relation_bias.bias_table
                )

        for local_idx, layer in enumerate(self.layers):
            global_idx = self.global_layer_offset + local_idx
            layer_window = self.window_sizes[local_idx]
            layer_cs = None if global_idx in self.nope_layers else cos_sin

            if isinstance(x, tuple):
                x = x[0]  # unwrap stale auxiliary returns from previous iteration

            x = layer(x, layer_cs, layer_window, kv_cache, doc_ids=doc_ids, **_block_kwargs)

            # Collect auxiliary losses (MoE, MoD) instead of discarding them.
            x = self._collect_block_aux_losses(x, layer=layer)

        if self.head is not None:
            # Save pre-lm_head hidden states for MTP (last stage only).
            self._last_x_prenorm = x
            self._last_cos_sin = cos_sin
            x = self.head(x)
        elif self._inject_aux_into_graph:
            # Non-last stage (VPP or standard PP): inject accumulated aux
            # losses into the hidden-state backward graph.  Without this,
            # non-last stage MoE/MoD aux losses are silently dropped —
            # cleared by _clear_aux_losses() before the loss_fn can drain
            # them (VPP: next microbatch's forward; standard PP: loss_fn
            # only runs on the last stage).
            #
            # Following Megatron-Core's MoEAuxLossAutoScaler pattern, we
            # attach the *weighted* combined aux loss to the output tensor's
            # autograd graph.  The forward values are unchanged; backward
            # triggers aux_loss.backward() with main_loss_backward_scale
            # (set to 1/grad_accum_steps by the training loop) so router
            # weights get correctly-sized gradients.
            #
            # Weights and averaging match the loss_fn's treatment of
            # drained losses: weight * mean(per_layer_losses).
            _cfg = self.config
            _moe_w = getattr(_cfg, "moe_aux_loss_weight", 0.01) if _cfg else 0.01
            _z_w = getattr(_cfg, "moe_router_z_loss_weight", 0.001) if _cfg else 0.001
            if self._moe_aux_losses:
                _combined = _moe_w * sum(self._moe_aux_losses) / len(self._moe_aux_losses)
                x = inject_aux_loss(x, _as_tensor(_combined))
            if self._moe_z_losses:
                _combined = _z_w * sum(self._moe_z_losses) / len(self._moe_z_losses)
                x = inject_aux_loss(x, _as_tensor(_combined))
            if self._mod_aux_losses:
                _combined = sum(self._mod_aux_losses) / len(self._mod_aux_losses)
                x = inject_aux_loss(x, _as_tensor(_combined))
            # Clear after injection — they are now in the autograd graph.
            self._moe_aux_losses = []
            self._moe_z_losses = []
            self._mod_aux_losses = []

        # Save output for deallocation at the start of the next forward.
        if self._deallocate_output:
            self._saved_output = x

        return x


def create_pipeline_stage(
    model: nn.Module,
    stage_id: int,
    num_stages: int,
    device: torch.device,
    *,
    weighted: bool = False,
    pp_boundaries: Optional[list[int]] = None,
) -> _PipelineStageModule:
    """Create a ``_PipelineStageModule`` wrapping the layers for ``stage_id``.

    This extracts the correct slice of ``model.transformer.h`` and optionally
    attaches the embedding (first stage) and lm_head (last stage).  It also
    propagates RoPE buffers, per-layer window sizes, and NoPE layer indices
    so that the stage's forward pass can supply them to each block.

    Args:
        model: A GPT model with ``model.transformer.h``.
        stage_id: Which stage this rank owns (0-indexed).
        num_stages: Total number of pipeline stages.
        device: Target device for the stage.
        weighted: Use weight-aware partitioning (for MoE models).
        pp_boundaries: Explicit PP boundaries from pipe-delimited nem_pattern.
            When provided, overrides auto/weighted partitioning.

    Returns:
        A ``_PipelineStageModule`` ready to be wrapped by
        ``PipelineStage`` or used directly.
    """
    if pp_boundaries:
        partitions = partition_model_explicit(model, pp_boundaries)
        if len(partitions) != num_stages:
            raise ValueError(
                f"pp_boundaries produces {len(partitions)} stages but "
                f"num_stages={num_stages} was requested. The pipe-delimited "
                f"nem_pattern must have exactly {num_stages - 1} pipe "
                f"delimiters to match the PP schedule's stage count."
            )
    elif weighted:
        partitions = partition_model_weighted(model, num_stages)
    else:
        partitions = partition_model(model, num_stages)
    start, end = partitions[stage_id]

    # Extract the layer slice as a new ModuleList (references, not copies).
    transformer_model = cast(_ModelWithTransformerLayers, model)
    stage_layers = nn.ModuleList(
        [transformer_model.transformer.h[i] for i in range(start, end + 1)]
    )

    # First stage gets embedding, last stage gets lm_head.
    embedding_model = cast(_ModelWithTransformerEmbedding, model)
    embed = embedding_model.transformer.wte if stage_id == 0 else None
    head = _optional_module_attr(model, "lm_head") if stage_id == num_stages - 1 else None

    config = getattr(model, "config", None)

    # Per-layer window sizes for this stage's layers.
    all_window_sizes = getattr(model, "window_sizes", None)
    stage_window_sizes = (
        all_window_sizes[start : end + 1]
        if all_window_sizes is not None
        else None
    )

    # NoPE layers (global indices that skip RoPE).
    nope_layers = getattr(model, "nope_layers", None) or set()

    # RoPE buffers — only needed on the first stage (which does embedding),
    # but provided to all stages so they can compute cos_sin if needed
    # (e.g., when resuming from a different pipeline split).
    cos = getattr(model, "cos", None)
    sin = getattr(model, "sin", None)

    # Pre-block modules (ngram_hash, structure_emb, platform_emb, dyt_embed)
    # are only relevant on the first stage where token IDs are embedded.
    ngram_hash = getattr(model, "ngram_hash", None) if stage_id == 0 else None
    structure_emb = getattr(model, "structure_emb", None) if stage_id == 0 else None
    platform_emb = getattr(model, "platform_emb", None) if stage_id == 0 else None
    dyt_embed = getattr(model, "dyt_embed", None) if stage_id == 0 else None

    # relation_bias: reference on ALL stages so each stage can compute
    # the additive chunk-relation attention bias from structure_meta.
    # This is a reference, not a copy — the parameter is shared.
    relation_bias = getattr(model, "relation_bias", None)

    stage_module = _PipelineStageModule(
        layers=stage_layers,
        stage_id=stage_id,
        num_stages=num_stages,
        embed=embed,
        head=head,
        config=config,
        global_layer_offset=start,
        window_sizes=stage_window_sizes,
        nope_layers=nope_layers,
        cos=cos,
        sin=sin,
        ngram_hash=ngram_hash,
        structure_emb=structure_emb,
        platform_emb=platform_emb,
        dyt_embed=dyt_embed,
        relation_bias=relation_bias,
    )

    # Propagate MTP module and wte reference to the last stage for aux loss
    # computation in the PP loss_fn.  MTP needs wte (shared embedding) and
    # lm_head.weight, plus the pre-lm_head hidden states saved during forward.
    if stage_id == num_stages - 1:
        mtp = getattr(model, "mtp", None)
        if mtp is not None:
            stage_module.mtp = mtp
            # wte reference for MTP teacher-forcing embeddings
            stage_module.wte = embedding_model.transformer.wte

    return stage_module


# ---------------------------------------------------------------------------
# Virtual Pipeline Parallelism (VPP) — Megatron-style interleaved stages
#
# Equivalence with Megatron-Core ``forward_backward_pipelining_with_interleaving``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our VPP runtime delegates to PyTorch's ``ScheduleInterleaved1F1B`` which is
# semantically equivalent to Megatron-Core's interleaved scheduler when using
# the default ``microbatch_group_size_per_vp_stage = pipeline_parallel_size``
# (Megatron-Core ``model_parallel_config.py`` line 453).  Specifically:
#
# 1. **Schedule table** — Megatron's ``get_schedule_table()`` and PyTorch's
#    implicit schedule (via ``forward_stage_index`` + ``microbatches_per_round``)
#    produce identical (microbatch_id, model_chunk_id) sequences.  PyTorch's
#    ``microbatches_per_round = n_microbatches // max(1, n_microbatches // pp)``
#    equals Megatron's default ``microbatch_group_size_per_vp_stage = pp_degree``
#    when ``n_microbatches`` is a multiple of ``pp_degree``.
#
# 2. **Warmup formula** — Megatron: ``(pp - rank - 1)*2 + (chunks-1)*group``.
#    PyTorch: ``(chunks-1)*mb_per_round + 2*((pp-1) - rank)``.  These are
#    identical when ``group == mb_per_round == pp_degree``.
#
# 3. **Backward chunk reversal** — Both reverse the chunk order for backward:
#    Megatron via ``num_model_chunks - chunk_id - 1``, PyTorch via reversed
#    ``backward_stage_index``.
#
# 4. **Gradient scaling** — PyTorch's ``scale_grads=True`` divides gradients by
#    ``n_microbatches`` (averaging over PP micro-batches within one step).  Our
#    loss function separately divides by ``grad_accum_steps`` (outer loop).
#    Together these correctly average over all micro-batches across all
#    accumulation steps.
#
# Differences from Megatron-Core (none affect correctness):
#
# - **Tunable group size**: Megatron exposes ``--microbatch-group-size-per-vp-stage``
#   to change how many contiguous microbatches run per VP chunk before switching.
#   PyTorch computes this automatically from ``n_microbatches // pp_group_size``.
#   Non-default group sizes (e.g., group=3 with pp=2, n_mb=6) are not configurable
#   in PyTorch's API.  This only matters for exotic configurations.
#
# - **P2P overlap**: Megatron has ``overlap_p2p_comm`` and
#   ``overlap_p2p_comm_warmup_flush`` for overlapping send/recv with compute.
#   PyTorch uses batched P2P (``_batch_p2p``) which is synchronous per time-step.
#   This may leave small performance gaps on very large pipeline degrees.
#
# - **Grad/param sync hooks**: Megatron integrates ``grad_sync_func`` and
#   ``param_sync_func`` directly into the schedule loop for overlapping gradient
#   all-reduce with pipeline execution.  In our stack, FSDP2's backward hooks
#   handle gradient sync outside the schedule.
#
# - **MoE expert-parallel comm overlap**: Megatron has a ``combined_1f1b`` path
#   that overlaps MoE expert-parallel communication with compute during 1F1B.
#   We do not use this (our MoE expert comm is synchronous).
#
# - **Partial activation checkpointing**: Megatron can checkpoint only some
#   microbatches (``num_microbatches_with_partial_activation_checkpoints``).
#   In our stack, activation checkpointing is per-block, not per-microbatch.
#
# The local helpers ``get_vpp_schedule_table()`` and ``get_vpp_warmup_count()``
# are NOT used at runtime — they exist for testing and schedule analysis.
# Runtime scheduling is handled entirely by ``ScheduleInterleaved1F1B``.
# ---------------------------------------------------------------------------


def compute_vpp_stage_ids(
    pp_rank: int,
    pp_degree: int,
    vpp_chunks: int,
) -> list[int]:
    """Compute the global stage IDs assigned to ``pp_rank`` under VPP.

    With VPP, the model is split into ``pp_degree * vpp_chunks`` virtual
    stages.  Each PP rank holds ``vpp_chunks`` non-contiguous stages,
    interleaved across the pipeline:

        total_virtual_stages = pp_degree * vpp_chunks
        rank r holds stages: [r, r + pp_degree, r + 2*pp_degree, ...]

    This mirrors Megatron-LM v3's virtual pipeline model parallelism,
    which reduces the pipeline bubble by a factor of ``vpp_chunks``
    compared to standard 1F1B.

    Example with pp_degree=4, vpp_chunks=2 (8 virtual stages total):
        rank 0 → stages [0, 4]
        rank 1 → stages [1, 5]
        rank 2 → stages [2, 6]
        rank 3 → stages [3, 7]

    Args:
        pp_rank: This rank's position in the pipeline (0-indexed).
        pp_degree: Number of physical pipeline ranks.
        vpp_chunks: Number of virtual chunks per rank.

    Returns:
        List of ``vpp_chunks`` global stage IDs (ascending order).
    """
    total_stages = pp_degree * vpp_chunks
    stage_ids = [pp_rank + chunk * pp_degree for chunk in range(vpp_chunks)]
    assert all(0 <= s < total_stages for s in stage_ids)
    return stage_ids


def create_vpp_stages(
    model: nn.Module,
    pp_rank: int,
    pp_degree: int,
    vpp_chunks: int,
    device: torch.device,
    *,
    weighted: bool = False,
    pp_boundaries: Optional[list[int]] = None,
) -> list[_PipelineStageModule]:
    """Create the ``vpp_chunks`` stage modules for a VPP rank.

    Each PP rank holds ``vpp_chunks`` non-contiguous slices of the model.
    The total number of virtual stages is ``pp_degree * vpp_chunks``, and
    the layers are partitioned into that many contiguous groups.  This rank
    gets the groups at positions ``[pp_rank, pp_rank + pp_degree, ...]``.

    The first virtual stage of the first rank gets the embedding; the last
    virtual stage of the last rank gets the lm_head.  Intermediate chunks
    are pure transformer blocks.

    This follows the Megatron-LM v3 virtual pipeline parallelism design:
    - Narayanan et al., "Efficient Large-Scale Language Model Training on
      GPU Clusters Using Megatron-LM" (2021), Section 3.2.
    - Reduces pipeline bubble from ``(pp-1)/(pp-1+m)`` to
      ``(pp-1)/(pp*v-1+m)`` where ``v`` = vpp_chunks, ``m`` = microbatches.

    Args:
        model: Full GPT model (before any PP partitioning).
        pp_rank: This rank's position in the pipeline (0-indexed).
        pp_degree: Number of physical pipeline ranks.
        vpp_chunks: Number of virtual chunks per rank (default: 2).
        device: Target device for the stage modules.
        weighted: Use weight-aware partitioning for MoE models.
        pp_boundaries: Explicit PP boundaries from pipe-delimited nem_pattern.

    Returns:
        List of ``vpp_chunks`` ``_PipelineStageModule`` instances, ordered
        by their global virtual stage ID (ascending).

    Raises:
        ValueError: If the model cannot be evenly split into
            ``pp_degree * vpp_chunks`` stages, or if arguments are invalid.
    """
    total_stages = pp_degree * vpp_chunks
    stage_ids = compute_vpp_stage_ids(pp_rank, pp_degree, vpp_chunks)

    stages: list[_PipelineStageModule] = []
    for virtual_stage_id in stage_ids:
        stage = create_pipeline_stage(
            model,
            stage_id=virtual_stage_id,
            num_stages=total_stages,
            device=device,
            weighted=weighted,
            pp_boundaries=pp_boundaries,
        )
        # Disable output deallocation — the interleaved schedule switches
        # between model chunks, so a chunk's output may still be needed
        # for backward when another chunk is running forward.  The
        # ScheduleInterleaved1F1B from torch.distributed.pipelining manages
        # tensor lifetimes internally.
        stage.disable_output_deallocation()
        # Enable aux-loss injection for non-last-stage chunks.  In VPP,
        # the interleaved schedule runs multiple microbatches through
        # different chunks concurrently.  Non-last chunks' aux losses
        # would be cleared by the next microbatch's forward before the
        # loss_fn fires.  Injection attaches them to the autograd graph
        # so they survive (Megatron MoEAuxLossAutoScaler pattern).
        if virtual_stage_id != total_stages - 1:
            stage.enable_aux_loss_injection()
        stage.to(device)
        stages.append(stage)

    return stages


def get_vpp_schedule_table(
    num_microbatches: int,
    num_model_chunks: int,
    microbatch_group_size: Optional[int] = None,
    *,
    pp_degree: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Build the VPP schedule lookup table (Megatron-style).

    The schedule table maps a ``virtual_microbatch_id`` (0-indexed across
    all chunks and microbatches) to a ``(microbatch_id, model_chunk_id)``
    pair.  This determines which model chunk processes which microbatch at
    each step of the interleaved schedule.

    Ported from Megatron-Core ``get_schedule_table``
    (``megatron/core/pipeline_parallel/schedules.py``).

    Note: This helper is NOT used at runtime.  PyTorch's
    ``ScheduleInterleaved1F1B`` builds and executes its own schedule
    table internally with equivalent semantics in the currently proven
    operating regime: default depth-first grouping
    (``microbatch_group_size = pp_degree``) with ``n_microbatches`` a
    multiple of ``pp_degree``.  This function exists for analysis and
    testing of that proven regime.

    Args:
        num_microbatches: Number of microbatches per pipeline stage.
        num_model_chunks: Number of virtual chunks per rank (vpp_chunks).
        microbatch_group_size: Number of contiguous microbatches per
            virtual stage before switching to the next chunk.  Defaults
            to ``pp_degree`` if provided, otherwise ``num_microbatches``
            (fallback for backward compatibility).  Megatron-Core defaults
            to ``pipeline_parallel_size`` (depth-first schedule), and
            PyTorch ``ScheduleInterleaved1F1B`` uses the equivalent
            ``microbatches_per_round = n_microbatches // max(1, n_mb // pp)``.
        pp_degree: Pipeline parallel degree.  Used to compute the default
            ``microbatch_group_size`` when not explicitly provided.

    Returns:
        List of ``(microbatch_id, model_chunk_id)`` tuples of length
        ``num_microbatches * num_model_chunks``.

    Example with num_microbatches=8, num_model_chunks=2, pp_degree=4
    (microbatch_group_size defaults to 4):
        [(0,0),(1,0),(2,0),(3,0), (0,1),(1,1),(2,1),(3,1),
         (4,0),(5,0),(6,0),(7,0), (4,1),(5,1),(6,1),(7,1)]
    """
    if microbatch_group_size is None:
        # Match Megatron-Core default: microbatch_group_size = pp_degree.
        # PyTorch ScheduleInterleaved1F1B uses the equivalent formula:
        #   number_of_rounds = max(1, n_microbatches // pp_group_size)
        #   microbatches_per_round = n_microbatches // number_of_rounds
        # which equals pp_degree when n_microbatches >= pp_degree.
        microbatch_group_size = pp_degree if pp_degree is not None else num_microbatches

    schedule_table: list[tuple[int, int]] = []
    for min_mb_in_group in range(0, num_microbatches, microbatch_group_size):
        if min_mb_in_group + microbatch_group_size >= num_microbatches:
            # Last (or only) group — include remaining microbatches
            schedule_table.extend(
                (mb_id, chunk_id)
                for chunk_id in range(num_model_chunks)
                for mb_id in range(min_mb_in_group, num_microbatches)
            )
        else:
            schedule_table.extend(
                (mb_id, chunk_id)
                for chunk_id in range(num_model_chunks)
                for mb_id in range(
                    min_mb_in_group, min_mb_in_group + microbatch_group_size
                )
            )
    return schedule_table


def get_vpp_warmup_count(
    pp_rank: int,
    pp_degree: int,
    num_microbatches: int,
    num_model_chunks: int,
    microbatch_group_size: Optional[int] = None,
    *,
    forward_only: bool = False,
) -> tuple[int, int, bool]:
    """Compute warmup / steady-state counts for VPP interleaved schedule.

    Ported from Megatron-Core ``get_pp_rank_microbatches``
    (``megatron/core/pipeline_parallel/schedules.py``).

    Note: This helper is NOT used at runtime.  PyTorch's
    ``ScheduleInterleaved1F1B`` computes warmup/steady-state internally
    with equivalent formulas in the currently proven operating regime:
    ``n_microbatches`` is a multiple of ``pp_degree`` and
    ``microbatch_group_size`` is left at the default ``pp_degree``.
    This function exists for analysis, testing, and validation of that
    proven regime.

    Args:
        pp_rank: This rank's position in the pipeline.
        pp_degree: Number of physical pipeline ranks.
        num_microbatches: Microbatches per stage.
        num_model_chunks: Virtual chunks per rank (vpp_chunks).
        microbatch_group_size: Contiguous microbatches per VP stage.
            Defaults to ``pp_degree`` to match both Megatron-Core's default
            (``pipeline_model_parallel_size``) and PyTorch's equivalent
            ``microbatches_per_round``.
        forward_only: If True, no backward passes (inference).

    Returns:
        (num_warmup, num_remaining, all_warmup):
        - num_warmup: Number of forward-only steps before 1F1B starts.
        - num_remaining: Number of 1F1B steady-state steps.
        - all_warmup: True if all microbatches are processed in warmup
          (no steady-state phase).
    """
    if microbatch_group_size is None:
        # Match Megatron-Core default: pipeline_model_parallel_size.
        # This also matches PyTorch's ScheduleInterleaved1F1B warmup formula:
        #   warmup = (n_local_stages - 1) * microbatches_per_round
        #          + 2 * (pp_group_size - 1 - rank)
        # where microbatches_per_round = pp_degree (when n_mb >= pp_degree).
        microbatch_group_size = pp_degree

    total = num_microbatches * num_model_chunks

    if forward_only:
        return total, 0, True

    # VPP warmup: (pp_degree - pp_rank - 1) * 2 accounts for the stagger
    # across ranks in both forward and backward directions.
    # (num_model_chunks - 1) * microbatch_group_size fills the pipeline
    # across all virtual stages before 1F1B steady-state can begin.
    num_warmup = (pp_degree - pp_rank - 1) * 2
    num_warmup += (num_model_chunks - 1) * microbatch_group_size

    all_warmup = num_warmup >= total
    if all_warmup:
        num_warmup = total

    num_remaining = total - num_warmup
    return num_warmup, num_remaining, all_warmup


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _get_schedule_class(schedule: PipelineSchedule):
    """Map our enum to the torch.distributed.pipelining schedule class."""
    _require_pipelining()
    from torch.distributed.pipelining import (
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
    )

    _map = {
        PipelineSchedule.SIMPLE_1F1B: Schedule1F1B,
        PipelineSchedule.INTERLEAVED_1F1B: ScheduleInterleaved1F1B,
        PipelineSchedule.GPIPE: ScheduleGPipe,
        PipelineSchedule.VPP: ScheduleInterleaved1F1B,
    }
    return _map[schedule]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def apply_pipeline_parallel(
    model: nn.Module,
    num_stages: int,
    schedule: PipelineSchedule = PipelineSchedule.SIMPLE_1F1B,
    *,
    device: Optional[torch.device] = None,
    group: object = None,
) -> dict:
    """Top-level entry point for pipeline parallelism.

    Partitions the model into pipeline stages and returns a dict with:
    - ``"partitions"``: list of (start, end) tuples
    - ``"schedule_cls"``: the torch schedule class to use
    - ``"stages"``: list of ``_PipelineStageModule`` instances (one per stage)
    - ``"num_stages"``: number of stages

    In a real distributed setting, each rank would only instantiate its own
    stage.  This prototype creates all stages for inspection/testing.

    Args:
        model: GPT model.
        num_stages: Number of pipeline stages.
        schedule: Which pipeline schedule to use.
        device: Target device.  Defaults to the model's current device.
        group: Optional process group for distributed communication.

    Returns:
        Dict with partition info and stage modules.
    """
    _require_pipelining()

    if device is None:
        # Try to infer device from model parameters.
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    partitions = partition_model(model, num_stages)
    schedule_cls = _get_schedule_class(schedule)

    stages = []
    for stage_id in range(num_stages):
        stage_module = create_pipeline_stage(model, stage_id, num_stages, device)
        stages.append(stage_module)

    return {
        "partitions": partitions,
        "schedule_cls": schedule_cls,
        "stages": stages,
        "num_stages": num_stages,
    }


# ---------------------------------------------------------------------------
# Comm-stream separation (Megatron pattern, donor research item #1)
# ---------------------------------------------------------------------------
#
# torch.distributed.pipelining schedules issue NCCL P2P via the module-level
# helpers ``_batch_p2p`` and ``_wait_batch_p2p`` in
# ``torch.distributed.pipelining.schedules``.  Both serialize on the default
# CUDA compute stream:
#
#     for op in p2p_ops: dist.batch_isend_irecv(p2p_ops)
#     for w in works: w.wait()  # host-side blocking wait
#
# torch.profiler on NAM56R PP=2 vpp dbs=4 shows ``ncclDevKernel_SendRecv``
# burning **23.6 s out of 66 s wall (35%)** of the entire step, almost
# entirely serialized with compute.  Megatron's ``combined_1f1b`` solves this
# by routing P2P onto a dedicated comm stream with explicit
# ``cuda.Event``-based cross-stream synchronization, letting the compute
# stream advance through the next microbatch's GEMM/activation kernels while
# NCCL is still pumping the previous microbatch's tensors over NVLink.
#
# This module exposes a thin context manager that achieves the same effect
# without forking ``Schedule1F1B``: it monkey-patches the module-level
# ``_batch_p2p`` / ``_wait_batch_p2p`` helpers to (a) enqueue NCCL kernels on
# a dedicated high-priority CUDA stream and (b) replace the host-side
# ``work.wait()`` with a cross-stream ``compute_stream.wait_event(comm_event)``
# barrier.  Because all P2P ops within one schedule.step() share the same
# comm stream and CUDA enforces in-stream ordering, waiting on the latest
# event implicitly waits on every prior P2P op queued before it — making the
# patch correct without per-tensor tracking.
#
# Use it from base_train.py around the schedule's ``.step()`` call:
#
#     from examples.distributed.pipeline_parallel_sample import pp_comm_stream_context
#     with pp_comm_stream_context():
#         pp_schedule_obj.step(input, target=target, losses=losses)
#
# When CUDA is not available the context manager is a no-op so the same
# code path works on CPU/XLA test runs.

_comm_stream_singleton: dict[int, "torch.cuda.Stream"] = {}
_comm_stream_lock = threading.Lock()
_pp_p2p_trace_counter = 0
_pp_p2p_trace_lock = threading.Lock()


def _pp_trace_batch_p2p_enabled() -> bool:
    return os.environ.get("PP_TRACE_BATCH_P2P", "0") == "1"


def _describe_pp_p2p_op(op: object) -> str:
    op_name = type(op).__name__
    peer = getattr(op, "peer", None)
    tensor = getattr(op, "tensor", None)
    if tensor is None:
        return f"{op_name}(peer={peer}, tensor=None)"
    shape = tuple(tensor.shape)
    return (
        f"{op_name}(peer={peer}, shape={shape}, dtype={tensor.dtype}, "
        f"device={tensor.device}, requires_grad={bool(tensor.requires_grad)})"
    )


def _get_pp_comm_stream() -> "Optional[torch.cuda.Stream]":
    """Lazily allocate one high-priority CUDA stream per device for PP P2P.

    The comm stream is shared across all schedules on the same device.  We
    pick the highest priority CUDA exposes (-1 on H100/H200) so NCCL kernel
    launches preempt low-priority compute kernels and minimize end-to-end
    latency, leaving the bulk of the compute stream free for matmul/SiLU/
    SwiGLU work.
    """
    if not torch.cuda.is_available():
        return None
    device_idx = torch.cuda.current_device()
    with _comm_stream_lock:
        stream = _comm_stream_singleton.get(device_idx)
        if stream is None:
            try:
                # CUDA stream priorities: lower number = higher priority.
                # H100/H200 expose [-5, 0]; we take the highest available.
                lo, _hi = torch.cuda.Stream.priority_range()  # type: ignore[attr-defined]
            except Exception:
                lo = -1
            stream = torch.cuda.Stream(device=device_idx, priority=lo)
            _comm_stream_singleton[device_idx] = stream
        return stream


@contextlib.contextmanager
def pp_comm_stream_context(enabled: bool = True):
    """Patch ``torch.distributed.pipelining.schedules`` so PP P2P runs on a
    dedicated comm stream with cross-stream event sync.

    Use as a context manager around ``schedule.step(...)``.  Outside the
    context the schedule reverts to the upstream serialized behavior.

    Args:
        enabled: When False, this context manager is a strict no-op.  This
            lets callers gate the optimization behind a CLI flag without
            wrapping every call site in a conditional.

    Notes:
        - Single-threaded only.  The patch mutates the
          ``torch.distributed.pipelining.schedules`` module namespace; nested
          context entries are reference-counted by the wrapper but
          concurrent uses from different threads will race.
        - The patch is a no-op on CPU/XLA paths (no CUDA stream).  On those
          devices the upstream module helpers run unchanged.
        - **The wrapper is now a strict NO-OP** because the patch design
          is fundamentally incompatible with PyTorch's NCCL backend.  See
          ``is_pp_comm_stream_enabled`` for the full architectural
          writeup.  TL;DR: ``ProcessGroupNCCL`` keeps a private per-rank-pair
          NCCL stream pool (``ncclStreams_``) and ignores any
          ``with torch.cuda.stream(...)`` context applied at the Python
          level.  Wrapping ``dist.batch_isend_irecv`` in a stream context
          does NOT redirect NCCL kernels; it only changes which stream
          the *event-based sync* records on, which is harmless but also
          useless.  Worse, replacing the host-side ``work.wait()`` with a
          no-op REMOVES the cross-stream ``cudaStreamWaitEvent`` that
          PyTorch normally inserts inside ``Work.wait()``, causing
          downstream compute to read recv buffers before the NCCL kernel
          fills them — verified inf-grad failure on bench3 H200:8 d20
          PP=2 1f1b/gpipe/vpp dbs=16.

    The function bodies are kept as scaffolding for future iteration on
    a real solution (separate ``ProcessGroup`` for PP P2P, custom NCCL
    C++ extension, or upstream pytorch#138074 / pytorch#147729 landing).
    They are guarded behind ``enabled=True`` and only run if the user
    explicitly opts in via ``PP_COMM_STREAM=1``.
    """
    # The patch is broken (see is_pp_comm_stream_enabled docstring).
    # Default behavior: strict no-op.  The wrapper code below is kept
    # for experimentation but is unreachable on the default path.
    if not enabled or not torch.cuda.is_available():
        yield
        return

    try:
        import torch.distributed.pipelining.schedules as _sched
    except ImportError:
        # PyTorch without the pipelining module; nothing to patch.
        yield
        return

    comm_stream = _get_pp_comm_stream()
    if comm_stream is None:
        yield
        return

    orig_batch_p2p = _sched._batch_p2p
    orig_wait_batch_p2p = _sched._wait_batch_p2p
    trace_enabled = _pp_trace_batch_p2p_enabled()

    def _patched_batch_p2p(p2p_ops, desc=None):
        """Issue NCCL P2P on the comm stream and queue a per-call cross-stream sync.

        Each ``_batch_p2p`` call records its OWN dedicated CUDA event on the
        comm stream and immediately enqueues
        ``compute_stream.wait_event(comm_event)`` so the compute stream
        resumes only after THIS specific P2P op completes.  Critically we
        do NOT share state between calls — that's what broke the first
        iteration of this patch on PP=2 1F1B (separate fwd_send / fwd_recv
        / bwd_send / bwd_recv calls each need their own event; a single
        shared ``latest_event`` made downstream waiters consume recv
        buffers before the matching NCCL kernel finished, producing Inf
        gradients on the very first step at d20 PP=2 1f1b dbs=16).

        The compute stream's wait_event is non-blocking on the host: the
        host loop returns immediately and is free to enqueue subsequent
        compute kernels onto compute_stream.  Those kernels are queued
        AFTER the wait_event so they run only after the P2P completes —
        correctness is preserved.  Throughput wins come from (a)
        eliminating the host-side ``work.wait()`` block in
        ``_wait_batch_p2p`` (the host loop pipelines ahead) and (b) NCCL
        kernels running on a separate, high-priority stream where they
        don't compete with compute kernel launches.
        """
        if len(p2p_ops) == 0:
            if _pp_trace_batch_p2p_enabled():
                print(
                    "[trace][pp_batch_p2p] call=empty desc="
                    f"{desc!r} ops=[]",
                    flush=True,
                )
            return []
        call_idx = -1
        if trace_enabled:
            global _pp_p2p_trace_counter
            with _pp_p2p_trace_lock:
                call_idx = _pp_p2p_trace_counter
                _pp_p2p_trace_counter += 1
            op_descs = ", ".join(_describe_pp_p2p_op(op) for op in p2p_ops)
            print(
                "[trace][pp_batch_p2p] "
                f"call={call_idx} phase=enter desc={desc!r} "
                f"op_count={len(p2p_ops)} ops=[{op_descs}]",
                flush=True,
            )
        compute_stream = torch.cuda.current_stream()
        # Send buffers may have been written by an in-flight compute kernel
        # on the compute stream — wait for it before NCCL reads.
        compute_done = torch.cuda.Event()
        compute_stream.record_event(compute_done)  # type: ignore[arg-type]
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(compute_done)  # type: ignore[arg-type]
            works = orig_batch_p2p(p2p_ops, desc=desc)
            comm_done = torch.cuda.Event()
            comm_stream.record_event(comm_done)  # type: ignore[arg-type]
        # IMMEDIATELY queue the compute stream's wait on THIS call's
        # event.  No shared state — each call's wait is correct for its
        # own data.  This is non-blocking on the host: the wait_event is
        # GPU-side, so the host loop returns and continues queuing work.
        compute_stream.wait_event(comm_done)  # type: ignore[arg-type]
        if trace_enabled:
            work_desc = ", ".join(type(work).__name__ for work in works)
            print(
                "[trace][pp_batch_p2p] "
                f"call={call_idx} phase=issued desc={desc!r} "
                f"work_count={len(works)} works=[{work_desc}]",
                flush=True,
            )
        return works

    def _patched_wait_batch_p2p(work):
        """No-op host-side wait.

        The patched ``_batch_p2p`` already enqueued
        ``compute_stream.wait_event(comm_done)`` for each call's own
        event, so the compute stream is GPU-side serialized after each
        P2P completes.  Replacing the host-side ``work.wait()`` with a
        no-op lets the schedule's host loop pipeline ahead and queue
        more compute kernels without blocking on NCCL completion.

        ``work`` references are still held by the schedule's local
        variables until they fall out of scope, which is when the NCCL
        Work objects can be safely garbage collected.
        """
        if _pp_trace_batch_p2p_enabled():
            work_desc = (
                "[" + ", ".join(type(w).__name__ for w in work) + "]"
                if isinstance(work, (list, tuple))
                else type(work).__name__
            )
            print(
                "[trace][pp_batch_p2p] "
                f"phase=wait_bypassed work={work_desc}",
                flush=True,
            )
        # Intentionally do nothing.  Compute-stream cross-stream sync is
        # handled inside ``_patched_batch_p2p`` above.
        del work

    _sched._batch_p2p = _patched_batch_p2p
    _sched._wait_batch_p2p = _patched_wait_batch_p2p
    if trace_enabled:
        print(
            "[trace][pp_batch_p2p] "
            f"phase=context_apply sched_module={_sched.__name__} "
            f"orig_batch_id={id(orig_batch_p2p)} patched_batch_id={id(_sched._batch_p2p)} "
            f"orig_wait_id={id(orig_wait_batch_p2p)} patched_wait_id={id(_sched._wait_batch_p2p)}",
            flush=True,
        )
    try:
        yield
    finally:
        if trace_enabled:
            print(
                "[trace][pp_batch_p2p] "
                f"phase=context_restore sched_module={_sched.__name__} "
                f"current_batch_id={id(_sched._batch_p2p)} current_wait_id={id(_sched._wait_batch_p2p)}",
                flush=True,
            )
        _sched._batch_p2p = orig_batch_p2p
        _sched._wait_batch_p2p = orig_wait_batch_p2p


def is_pp_comm_stream_enabled() -> bool:
    """Return True if pp comm-stream separation is currently active.

    Reads the ``PP_COMM_STREAM`` env var.  **Default OFF — the
    patch is architecturally broken and cannot deliver on its promise.**

    Architectural reality (verified by reading PyTorch source +
    multi-source web research, 2026-04-08):

    1. ``ProcessGroupNCCL`` maintains a PRIVATE per-rank-pair internal
       NCCL stream pool (``ncclStreams_`` keyed by ``"{src_rank}:{dst_rank}"``).
       NCCL kernels run on those internal streams regardless of the
       Python-side ``with torch.cuda.stream(s):`` context.
       (Reference: ``ProcessGroupNCCL.hpp:1365`` and pytorch#132852.)

    2. Wrapping ``dist.batch_isend_irecv`` in
       ``with torch.cuda.stream(comm_stream):`` does NOT redirect NCCL
       kernels.  It only causes PyTorch to insert a
       ``cudaStreamWaitEvent`` on the user's stream (data-ready
       checkpoint), then NCCL kernels still run on the internal stream.
       (See pytorch#67158 / #138074 / #147729 — all open RFEs to expose
       a public API for stream override; none have landed.)

    3. ``Work.wait()`` for the NCCL backend is **non-blocking on the
       host** — it inserts a ``cudaStreamWaitEvent`` on the current
       CUDA stream pointing at the NCCL kernel's completion event.
       This is the cross-stream sync that prevents downstream compute
       from reading recv buffers before NCCL fills them.

    4. Replacing ``_wait_batch_p2p(work)`` with a no-op (which the
       initial patch did) **removes that critical cross-stream sync**,
       causing compute to consume garbage recv data and producing Inf
       gradients on the very first training step.  Verified on bench3
       H200:8 d20 PP=2 dbs=16 across all 3 schedules (1f1b/gpipe/vpp).

    5. Megatron-LM does NOT patch streams either.  Its
       ``p2p_communication.py`` calls ``dist.batch_isend_irecv``
       directly and relies on:
         (a) ``overlap_p2p_comm=True`` deferring ``req.wait()`` past
             non-dependent compute (schedule-level optimization);
         (b) Interleaved 1F1B / VPP exposing more in-flight microbatches;
         (c) PyTorch's automatic per-(src,dst)-pair NCCL stream pool
             giving naturally concurrent compute and P2P streams.
       Megatron's ``_COMM_STREAM`` is internal bookkeeping, NOT an
       NCCL stream override.

    6. A pipeline runtime can already create a separate ``ProcessGroup`` for PP
       via a dedicated PP communicator.
       That group has its own NCCL communicator, hence its own internal
       NCCL stream pool, hence already runs P2P concurrently with FSDP
       all-reduce / TP all-gather on a separate stream.  The "Megatron
       comm-stream separation" is **already happening for free**.

    The 35% NCCL P2P time observed in NAM56R's torch.profiler is data
    transfer time (or cross-stream sync wait time) — NOT serialization
    on the wrong stream.  Reducing it requires either:
      - schedule-level optimizations (overlap_p2p_comm pattern: defer
        the wait until just before consuming recv data, exposing more
        compute work in between)
      - reducing P2P data volume (TP+SP scatter, FP8 activations)
      - DualPipe-style F+B kernel interleaving on the same stream

    None of these are solved by the wrapper in this module.

    The wrapper code is left as scaffolding for future work, but is
    DEFAULT OFF and unreachable unless the user explicitly opts in via
    ``PP_COMM_STREAM=1`` — and even then it's broken.
    """
    import os
    return os.environ.get("PP_COMM_STREAM", "0") == "1"


# ---------------------------------------------------------------------------
# Shared-parameter gradient sync for Pipeline Parallelism
# ---------------------------------------------------------------------------
#
# Megatron-Core solves this with ``finalize_model_grads`` which calls
# ``_allreduce_word_embedding_grads`` and ``_allreduce_conditional_embedding_grads``
# after backward.  We follow the same pattern.
#
# **Why this is needed:**
# In multi-process PP, each rank creates the full model then slices out its
# stage.  Parameters that are logically shared across stages (relation_bias,
# wte when MTP is active) become independent copies — one per process.  Each
# copy only receives gradients from its local stage's layers, so the copies
# diverge after the first optimizer step.
#
# **What we sync:**
# 1. ``relation_bias`` — present on ALL stages, receives gradients only from
#    attention layers on that stage.  All-reduce (SUM) across the PP group.
# 2. ``wte`` (embedding) — on the first stage for token embedding, and on the
#    last stage for MTP auxiliary loss.  All-reduce (SUM) between first and
#    last PP ranks only.


_cached_embd_group: dict[int, object] = {}


def _get_or_create_pp_first_last_group(
    pp_group: "torch.distributed.ProcessGroup",
    pp_degree: int,
):
    """Return the cached first/last-stage subgroup for MTP embedding grad sync.

    PyTorch requires *all* ranks in the main group to enter ``dist.new_group()``
    and to do so in the same global order. Creating the first/last PP subgroup
    lazily only on the ranks that happen to own ``wte`` gradients violates that
    contract and can deadlock NCCL.

    Our PP layout is stage-major contiguous:

        stage 0: [0 .. R-1]
        stage 1: [R .. 2R-1]
        ...

    where ``R = world_size // pp_degree`` is the number of parallel pipelines
    (same intra-stage position across PP stages). We therefore create *all*
    first/last PP subgroups in deterministic lane order and cache the one
    matching the current rank's ``pp_group``.
    """
    import torch.distributed as dist

    if pp_degree <= 2:
        return pp_group

    _cache_key = id(pp_group)
    cached_group = _cached_embd_group.get(_cache_key)
    if cached_group is not None:
        return cached_group

    pp_ranks = dist.get_process_group_ranks(pp_group)
    if len(pp_ranks) != pp_degree:
        raise RuntimeError(
            "PP first/last subgroup setup expected pp_group size to match "
            f"pp_degree ({len(pp_ranks)} != {pp_degree})"
        )

    world_size = dist.get_world_size()
    if world_size % pp_degree != 0:
        raise RuntimeError(
            "PP first/last subgroup setup requires world_size divisible by "
            f"pp_degree ({world_size} % {pp_degree} != 0)"
        )

    ranks_per_pp_stage = world_size // pp_degree
    target_first_last = (pp_ranks[0], pp_ranks[-1])
    embd_group = None
    for lane_idx in range(ranks_per_pp_stage):
        first_last = [
            lane_idx,
            (pp_degree - 1) * ranks_per_pp_stage + lane_idx,
        ]
        group = dist.new_group(first_last)
        if tuple(first_last) == target_first_last:
            embd_group = group

    if embd_group is None:
        raise RuntimeError(
            "Failed to create first/last PP subgroup for ranks "
            f"{target_first_last}"
        )

    _cached_embd_group[_cache_key] = embd_group
    return embd_group


def sync_pp_shared_param_grads(
    stage_modules: list[_PipelineStageModule],
    pp_group: "torch.distributed.ProcessGroup",
    pp_rank: int,
    pp_degree: int,
    *,
    has_mtp: bool = False,
) -> None:
    """All-reduce gradients of parameters shared across PP stages.

    Must be called AFTER backward and BEFORE optimizer.step() on every PP
    rank.  Follows the Megatron-Core ``finalize_model_grads`` pattern.

    Args:
        stage_modules: This rank's ``_PipelineStageModule``(s). For standard
            PP this is a single-element list; for VPP it contains all chunks.
        pp_group: The pipeline-parallel process group (ranks forming one
            pipeline, i.e. same intra-stage position across all PP stages).
        pp_rank: This rank's position in the pipeline (0-indexed).
        pp_degree: Total number of pipeline stages.
        has_mtp: Whether MTP is active (last stage uses wte for MTP loss).
    """
    import torch.distributed as dist

    if pp_degree <= 1:
        return  # No cross-stage sync needed.

    # ---- DTensor safety check ----
    # FSDP2 wraps parameters as DTensors.  Our all-reduce operates on .grad
    # tensors directly, which is NOT DTensor-aware: it would all-reduce local
    # shards instead of full tensors, producing incorrect gradients.  Detect
    # this and raise an error early rather than silently training with wrong
    # gradients.
    _DTensor: type[torch.Tensor] | None
    try:
        from torch.distributed._tensor import DTensor as _ImportedDTensor

        _DTensor = cast(type[torch.Tensor], _ImportedDTensor)
    except ImportError:
        _DTensor = None

    def _check_not_dtensor(param: torch.Tensor, name: str) -> None:
        if _DTensor is not None and isinstance(param, _DTensor):
            raise RuntimeError(
                f"sync_pp_shared_param_grads: {name} is a DTensor (FSDP2-managed). "
                f"Plain all-reduce on DTensor .grad operates on local shards, not "
                f"full tensors, producing incorrect gradient sync. This combination "
                f"(PP + FSDP2 + shared params as DTensors) is not yet supported. "
                f"Workaround: ensure shared params (relation_bias, wte) are excluded "
                f"from FSDP2 wrapping, or use PP without FSDP2."
            )
        if (
            _DTensor is not None
            and param.grad is not None
            and isinstance(param.grad, _DTensor)
        ):
            raise RuntimeError(
                f"sync_pp_shared_param_grads: {name}.grad is a DTensor. "
                f"Plain all-reduce on DTensor gradients operates on local shards, "
                f"not full tensors. This combination (PP + FSDP2 + shared params) "
                f"is not yet supported."
            )

    # ---- 1. relation_bias: all-reduce across entire PP group ----
    # relation_bias is on every stage.  Each stage's attention layers
    # produce independent gradient contributions.  SUM them so the
    # optimizer sees the combined gradient from all layers.
    _relation_bias_grads = []
    for stage_mod in stage_modules:
        rb = getattr(stage_mod, "relation_bias", None)
        if rb is not None:
            for p in rb.parameters():
                _check_not_dtensor(p, "relation_bias")
                if p.requires_grad and p.grad is not None:
                    _relation_bias_grads.append(p.grad)

    if _relation_bias_grads:
        # Coalesce into a single flat tensor for one all-reduce.
        flat = torch.cat([g.reshape(-1) for g in _relation_bias_grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=pp_group)
        # Scatter back.
        offset = 0
        for g in _relation_bias_grads:
            numel = g.numel()
            g.copy_(flat[offset : offset + numel].reshape(g.shape))
            offset += numel

    # ---- 2. wte (embedding) for MTP: all-reduce between first & last ----
    # When MTP is active, the last stage references wte for MTP loss.
    # The first stage also has wte for token embedding.  In multi-process
    # PP, these are separate copies.  We need to sum their gradients
    # so both copies see the combined embedding + MTP gradient.
    #
    # Only ranks 0 and (pp_degree-1) participate.  For pp_degree==2 this
    # is the full PP group, so we use it directly.  For pp_degree>2 we
    # create a sub-group of just the first and last ranks.
    if has_mtp and pp_degree >= 2:
        embd_group = _get_or_create_pp_first_last_group(pp_group, pp_degree)
        _wte_grads = []
        for stage_mod in stage_modules:
            # wte is set on the last stage by create_pipeline_stage
            wte = getattr(stage_mod, "wte", None)
            if wte is not None:
                for p in wte.parameters():
                    _check_not_dtensor(p, "wte")
                    if p.requires_grad and p.grad is not None:
                        _wte_grads.append(p.grad)
            # embed is set on the first stage
            embed = getattr(stage_mod, "embed", None)
            if embed is not None and embed is not wte:
                for p in embed.parameters():
                    _check_not_dtensor(p, "embed/wte")
                    if p.requires_grad and p.grad is not None:
                        _wte_grads.append(p.grad)

        if _wte_grads:
            # Only first and last ranks participate.
            if pp_rank == 0 or pp_rank == pp_degree - 1:
                flat = torch.cat([g.reshape(-1) for g in _wte_grads])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=embd_group)
                offset = 0
                for g in _wte_grads:
                    numel = g.numel()
                    g.copy_(flat[offset : offset + numel].reshape(g.shape))
                    offset += numel


def validate_pp_shared_params(
    model: nn.Module,
    pp_degree: int,
    has_mtp: bool = False,
) -> list[str]:
    """Validate shared parameter configuration for PP and return warnings.

    Call at PP setup time to detect potential shared-param divergence issues.
    Returns a list of warning strings (empty if everything is clean).

    Args:
        model: The full GPT model (before stage extraction).
        pp_degree: Number of pipeline stages.
        has_mtp: Whether MTP is active.

    Returns:
        List of warning/info messages about shared parameter handling.
    """
    warnings: list[str] = []

    if pp_degree <= 1:
        return warnings

    relation_bias = getattr(model, "relation_bias", None)
    if relation_bias is not None:
        n_params = sum(p.numel() for p in relation_bias.parameters())
        warnings.append(
            f"PP shared param: relation_bias ({n_params} params) is replicated "
            f"on all {pp_degree} PP stages. Gradients will be all-reduced across "
            f"the PP group after backward (sync_pp_shared_param_grads)."
        )

    if has_mtp:
        wte = getattr(model, "transformer", None)
        wte = getattr(wte, "wte", None) if wte is not None else None
        if wte is not None:
            n_params = sum(p.numel() for p in wte.parameters())
            warnings.append(
                f"PP shared param: wte ({n_params} params) is used on first "
                f"stage (embedding) and last stage (MTP). Gradients will be "
                f"all-reduced between PP ranks 0 and {pp_degree - 1} "
                f"(sync_pp_shared_param_grads)."
            )

    return warnings
