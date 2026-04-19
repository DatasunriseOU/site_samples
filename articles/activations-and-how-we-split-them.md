---
title: "Activations and how we split them"
description: "What activations actually are in a hybrid Mamba 3, Transformer, and MoE stack, why they dominate memory at long context, and the levers we have: selective recompute per layer or op, sequence parallel, context parallel, and the trade-offs we live with."
date: "2026-04-18"
tags: ["activations", "gradient-checkpointing", "sequence-parallel", "context-parallel", "mamba", "moe", "h200"]
---

Activations are the memory term that grows with your ambition. Parameters are fixed once the architecture is chosen; optimizer state scales with parameters and the optimizer's per-weight footprint; gradients scale with parameters after sharding. Activations scale with batch, sequence, depth, and the specific set of intermediate tensors the backward pass needs. Doubling context doubles them. Turning on a new head doubles a slice of them. They are the single term that responds to how we execute the forward pass, which means they are the term we have the most levers on — and the one most likely to turn a "small" 4B-8B hybrid model into an OOM on 141 GB of HBM.

This post is the mechanics of activations in our hybrid Mamba 3 + Transformer + MoE stack, and the specific knobs we use to fit them.

## Why this matters

Our target is a long-context specialist. The model is small by design; the context is not. At 16K or 32K we cannot afford a naive autograd save-every-intermediate regime, and we cannot afford a blanket global gradient checkpointing regime either — the first OOMs, the second drops a measurable fraction of throughput and breaks specific composed lanes (packed-doc Mamba under FP8, regional-compile's inductor remat). We need per-layer, per-op decisions, and we need parallelism axes that shard activations rather than recompute them.

## What activations actually are

An activation is a tensor produced in the forward pass that is kept alive so the backward pass can differentiate. Any intermediate that torch's autograd stores on the saved-tensors list is an activation; anything recomputed in the backward is not. The memory cost is `shape * dtype`, summed over every saved intermediate, summed over every layer.

For a dense attention block using flash attention (FA2/FA4), the Megatron accounting gives roughly 34 half-precision elements per token-channel per layer: the norm input, the norm output, Q, K, V, flash attention's softmax logsumexp stats, the projected output, the MLP input, the two (or three, for SwiGLU) MLP hiddens, dropout masks when present, and residuals. Flash attention never materializes the full `QK^T` matrix; that is the reason a 4K-context dense block is tractable at all.

For a Mamba block, the dominant activations are the input projection, the conv1d output, the BC projections, and the selective-scan state. The scan kernel is itself expensive to recompute, which changes the recompute policy math.

For an M2RNN (RBlock), the recurrent state itself is small; the large linears around it are where the activation bytes live.

For a MoE layer (EBlock), activations include the router input, the router logits (fp32, replicated), the dispatch buffer, the per-expert gathered token tensor, each expert's MLP intermediates, the combine buffer, and the output. The dispatch and combine buffers are the terms that scale with the capacity factor and the top-k, not linearly with experts.

For features layered on top: MTP saves K depth steps of full-block activations; Engram's gated path saves five D-sized intermediates per layer; cross-layer mHC carries persistent and temporary streams; the TOP head materializes a fp32 `(B*T, V)` logits tensor that at V=65536 is routinely 8+ GB per step.

The memory estimator carries these as separate fields on a structured estimate because they respond to different levers.

## The levers we have

### Per-layer, per-op recompute

The core mechanism in the main model runtime module is selective recompute. A top-level flag exists but is deliberately secondary; the real decisions are on per-block, per-op booleans parsed in `_parse_recompute_modules` and applied by `_maybe_recompute` / `_maybe_recompute_attention` wrappers. The specific flags the production code understands:

- `recompute_moe_experts` (default True): recompute the expert GEMM in backward. On our depth-56 ~4.7B MoE preset, this single decision saves roughly 44 GB across 22 E-layers. The savings come from not storing the per-expert gathered token tensor or the expert MLP hiddens; we pay one extra grouped-GEMM in backward, which is cheap relative to the communication and norm work that dominate the E-layer.
- `recompute_m2rnn` (default True): checkpoint the M2RNN recurrence. Saves roughly 9 GB across four RBlocks.
- `recompute_mamba_conv` (default True): recompute the conv1d, SiLU, qknorm, and bias inside the Mamba backward. Saves roughly 6 GB across 13 MBlocks at negligible throughput cost. This is narrower than wrapping the entire MBlock, which we deliberately do not do — wrapping collides with the FP8 packed-doc Mamba pack hook's `[-448, 448]` activation bound.
- `recompute_norms`: discard the RMSNorm/LayerNorm output and recompute in backward. Adds roughly 7 GB of savings on the depth-56 preset.
- `mla_recompute_q_upproj`: recompute the MLA Q up-projection in backward. Saves `H * qk_head_dim` bytes per token on MLA-equipped blocks.
- `recompute_modules = "core_attn,mlp"`: a finer-grained list the forward walks to decide whether to wrap the core attention call, the MLP call, or the MoE call in a non-reentrant `torch.utils.checkpoint`. Orthogonal to the per-kind recompute booleans above.
- `skip_eblock_checkpointing`: the coarse-grained "don't checkpoint EBlocks at all" flag. Known brittle under specific MoE dispatch configs; kept off by default.
- `mtp_recompute` (default True): activation checkpointing inside the MTP depth loop. Saves K-1 blocks of activations.
- `mhc_recompute_group_size`: for the cross-layer hypercycle path, H_mixed and h_post intermediates within each group are recomputed; the group size auto-picks, `-1` disables.

The non-reentrant `torch.utils.checkpoint` is what the wrappers use. Under our token-choice MoE path we pass `determinism_check='none'` because CUDA atomic-add non-determinism in the dispatch scatter produces tiny numerical differences between forward and recompute, which flip the routing of a few tokens and fail the strict shape/dtype check. The values are correct; the metadata is not, and the check is misleading here.

### Last-layer exception

`_should_checkpoint` in the main model runtime module always returns False for the last layer. Its activations are consumed immediately by backward, so checkpointing them is pure recompute cost with no memory benefit.

### CPU offload

`cpu_offload.py` provides a `cpu_offload_checkpoint` path that, instead of recomputing, copies the saved tensor to pinned host memory and restores it for backward. It trades PCIe bandwidth for FLOPs. On an 8x H200 single-host ~4B-MoE ablation we measured a low single-digit percent throughput lift with peak HBM roughly unchanged in the high-80s GB — a good trade when memory is the binding constraint and the PCIe link has headroom, not a general-purpose default.

### Random-projection compression

`compact_activation.py` implements the "compressed activations via random projection" idea (arXiv:2410.15352). A Linear forward compresses its input via a seeded random projection before saving, and backward reconstructs an approximation for the weight gradient. Seeded projections cost zero storage. The paper's ~17% memory saving replicates in this setup with a small but non-zero accuracy cost on the eval mix, so we keep it as an opt-in lever rather than a default.

### Regional-compile inductor remat

When `apply_regional_compile` is active and `activation_memory_budget < 1.0` is passed, the inductor compiler inspects each compiled block's graph and inserts recompute nodes to hit the budget. This is a different mechanism than the manual wrappers above; manual checkpointing inside a regional-compile block is explicitly disabled by the helper because the two would double-count. The preset surface translates the top-level `gradient_checkpointing` flag into a sub-1.0 `activation_memory_budget` for this path, which is why a silent "force off" regression broke receipted throughput two months ago.

### TE checkpoint

Blocks running under TE `fp8_autocast` need `te_checkpoint`, not stock checkpoint, because stock PyTorch checkpoint loses the FP8 amax history and the recompute draws different scales from the forward. The forward and recompute then disagree on scales and the backward produces garbage gradients silently. The rule is: TE owns any block under `fp8_autocast`.

### Sequence parallel

Sequence parallel (SP) shards activations along the sequence axis inside the TP region. Rather than replicating the full-sequence LayerNorm or RMSNorm output on every TP rank, SP keeps each TP rank responsible for `T / tp_degree` tokens during the norm and dropout regions, with explicit all-gathers at the TP boundary. In practice, the forward hooks live in the TP wrappers and the block pre-norm path gathers `x` back at the boundary. For a block whose activation bill is dominated by the norm/residual regions, this is effectively a `tp_degree`-way reduction in activation memory for those regions. On a TP=2 run the reduction is 2x; on TP=4 it is 4x.

SP composes with selective recompute. We do both on the production preset.

### Context parallel

Context parallel (CP) shards the sequence across a dedicated process group, so different ranks own different chunks of the sequence for the same sample. It is the scaling axis for very long contexts; at training time, it is how a 32K or 64K context fits on a device whose activation term would otherwise overflow. In practice, the launcher constructs the CP process group and wires it through as the model's CP group.

CP is currently CUDA-only in this implementation and is explicitly not implemented for MBlocks (Mamba layers). The Mamba selective scan is a left-to-right recurrence whose parallelisation across the sequence axis does not match the ring-attention pattern CP uses for attention; the correct answer is probably chunked-state handoff with a communication primitive we have not written yet. For now, if a preset turns on CP, its Mamba blocks must run at `context_parallel=1`, which in practice means CP is most useful on long-context dense-attention-heavy recipes and omit it on the pure hybrid long-context recipes.

### The trade-off curves

The curves that matter in practice: per-op recompute cost varies widely (attention core is cheap because FA4 already recomputes in backward, MLP is a full SwiGLU re-forward, MoE expert-GEMM is cheap relative to the routing comms); SP beats recompute for the norm/residual region because it shards rather than recomputes and the all-gather runs at NVLink line rate, so we always turn on SP before we turn on recompute; CP and FSDP are orthogonal (CP shards sequence, FSDP shards parameters) and we compose them; and global-versus-per-block is not close — global saves a flat fraction at a flat cost, while per-block moves along a Pareto that is strictly better on every hybrid preset we have measured.

## How it lands in a production stack

In production, activation sizing becomes a typed policy rather than a collection of flags. Lifted as-is: per-op recompute wrappers, the last-layer exception, the TE checkpoint rule for FP8 blocks, `determinism_check='none'` on the non-reentrant path, SP forward hooks, CP process-group construction, and the MBlock CP-not-supported guard. Rewritten: the flag surface. The overlapping set (top-level `gradient_checkpointing`, six `recompute_*` flags, `checkpoint_spacing`, `activation_memory_budget`, `cpu_offload_checkpointing`, and an autocast-selected TE path) collapses into a `CheckpointingPolicy` dataclass with per-block-kind entries and a single regional-compile translation path. Moved to kernels: fused RMSNorm+QKV, fused MLA Q up-projection, fused residual/bias/dropout, and TE grouped-GEMM with `moe_permute_with_probs`. Feature-flagged: `compact_activation`, MTP depth, Engram gates, TOP head, CP degree. Dropped: the coarse `skip_eblock_checkpointing` default, global-only checkpointing, and any retry that silently flips `gradient_checkpointing` without regression coverage.

## Ablations and what we kept

- Per-block always beats global on the depth-56 preset; selective expert-GEMM recompute matches full-EBlock savings with a simpler failure surface; narrow Mamba conv+BC recompute avoids the FP8 packed-doc full-MBlock crash.
- SP on whenever TP>1; activation saving is real and the NVLink all-gather is cheap.
- CP on for dense long-context recipes, off for hybrid recipes until the Mamba CP handoff exists.
- `compact_activation` opt-in when memory is binding; the ~17% saving is real, the accuracy cost is small but measurable.
- CPU offload checkpointing on-demand, not a baseline; good trade when PCIe is idle.
- `determinism_check='none'` on non-reentrant MoE checkpoint; the alternative is a metadata false-positive on a known-benign shape/dtype mismatch.

## Public checklist

- SP on whenever TP>1.
- Selective recompute is the first lever; global checkpointing is not a supported default.
- FP8 blocks use `te_checkpoint`, not stock checkpoint.
- Regional-compile's `activation_memory_budget` is the single authority when regional-compile is on; manual checkpoints inside compiled blocks are disabled.
- CP off on presets that include Mamba layers until the chunked-state handoff lands.
- Last layer is never checkpointed.
- Non-reentrant MoE checkpoint always passes `determinism_check='none'`.
- Auto-fit retry preserves `gradient_checkpointing`; covered by a regression test.
- `compact_activation`, MTP depth, Engram, TOP, and CP degree are declared in the preset, not via env.
- The estimator's activation fields attach to the launch record.

## Snapshot

| Lever | Scope | Cost | When to reach for it |
|-------|-------|------|----------------------|
| SP (sequence parallel) | norm/residual region | NVLink all-gather | always on when TP>1 |
| Selective recompute | per-op, per-block-kind | bounded by op cost | first memory lever |
| `te_checkpoint` | FP8-autocast blocks | preserves amax history | required under FP8 |
| CP (context parallel) | attention sequence axis | ring comms | dense long-context only |
| `compact_activation` | opt-in micro-batch path | ~17% memory, small accuracy cost | binding memory pressure |
| CPU offload checkpoint | whole-block copy-out | PCIe bandwidth | PCIe idle, memory binding |

```python
# sketch: per-block-kind policy the production code collapses to
policy = CheckpointingPolicy(
    ablock="selective",
    eblock="expert_gemm_only",
    mblock="conv_bc_only",
    rblock="full_plus_norms",
    last_layer="never",
    fp8_wrapper="te_checkpoint",
)
```

## References

- the main model runtime module (per-block `_should_checkpoint`, `_maybe_recompute`, `_parse_recompute_modules`, SP forward hooks, CP guards)
- compact-activation, CPU-offload, residual-fusion, and bias-dropout-add helpers
- tensor-parallel, block-construction, and Mamba runtime components
- analytical memory estimation and runtime memory-debug tooling
- launcher wiring for CP-group construction and SP flag routing
- change notes for activation-shaping regressions and fixes
- [Reducing Activation Recomputation in Large Transformer Models — Korthikanti et al., NVIDIA]
- [Ring Attention with Blockwise Transformers — Liu et al.]
- [Sequence Parallelism in Megatron — Korthikanti et al., NVIDIA]
- [CompAct: Compressed Activations via Random Projection — arXiv:2410.15352]
- [Mamba 3 — Gu et al.]
