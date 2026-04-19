---
title: "DualPipe and 3D Parallelism on H200 and GB10"
description: "How MegaCpp lays out the TP × PP × DP × EP cube on H200 multi-node systems and GB10, integrates DualPipe / DualPipeV with our hybrid layer pattern, accounts for pipeline bubbles, and launches the deployment training job."
date: "2026-04-18"
tags: ["pipeline-parallelism", "tensor-parallelism", "dualpipe", "h200", "gb10", "nvidia", "3d-parallelism"]
---

Pipeline parallelism is the axis everyone wants to skip and nobody can. With a depth-52 hybrid model and an 8-way specialist ensemble we cannot fit the largest preset under FSDP-only sharding on H200:8. The 3D cube — TP × PP × DP × EP — is the answer, and DualPipe / DualPipeV is the schedule that buys back the bubble pure 1F1B leaves on the floor. This post is the layout we run, the DualPipe integration that survived our hybrid layer pattern, and the launch story.

## Why MegaCpp cares about this

The constraint is shape, not size. A depth-52 hybrid preset interleaves A-blocks (dense attention), M-blocks (Mamba), E-blocks (MoE), and DSA-attention layers in patterns like AEMEAEMEAEMR. The blocks have wildly different parameter counts and activation costs: E-blocks dominate parameters (64 experts each), M-blocks dominate activation memory (selective-scan state), DSA layers carry the indexer and absorbed-MLA caches. Naive layer-count partitioning produces stages that differ by 5-10x and pushes us off the device on the heaviest stage.

So the job is twofold: lay out a mesh that fits — TP for dense, PP for depth, DP/FSDP2 for replication, EP for MoE — then schedule the pipeline so the bubble stops being the dominant cost. DualPipe and DualPipeV give us bidirectional F/B overlap plus a W (weight-grad) decomposition that drives the bubble below 1F1B at the same stage count. The codebase built the cube and the DualPipe wrapper end-to-end; MegaCpp lifts the layout planning while delegating the schedule to a focused DualPipeV integration on top of upstream Megatron.

## What we built in the public MegaCpp parallelism path

Five modules carry the parallelism story.

the distributed parallelism module is the cube. `ParallelismConfig(pp, dp, tp, ep)` plus `pp_schedule` and `pp_microbatches` is the entire surface; `validate_3d_config` checks the layout against the model config. The validator is load-bearing: PP must divide layer count, TP must divide `n_head` and `n_kv_head`, EP must divide `moe_n_routed_experts`, and the depth-52 preset gets its own checks (`pp ∈ {1, 2, 4, 13}`, `tp ∈ {1, 2, 4, 8}` to keep `n_kv_head=8` divisible). `build_3d_mesh` calls `init_device_mesh` with named axes `("pp", "dp", "tp", "ep")`. `apply_3d_parallelism` applies the strategies in the only order that composes: TP, PP, FSDP2, EP. This is the torchtitan-canonical order and what Megatron's `parallel_state` follows; deviating leaves FSDP trying to shard parameters PP has already moved off-rank.

The pipeline-partitioning module owns the layer partition. `partition_model` does layer-count balanced splits; `partition_model_weighted` does parameter-count balanced splits, which is what we use for any preset with E-blocks (a 64-expert E-block is roughly 10× the weight of an A-block). `create_pipeline_stage` materialises a `PipelineStageModule` that owns the embedding on stage 0 and the final norm + lm_head on the last stage. The schedule abstraction routes through `torch.distributed.pipelining`. The aux-loss injector (`_AuxLossInjector`, mirroring Megatron's `MoEAuxLossAutoScaler`) attaches MoE / MoD aux losses to the hidden-state backward graph on non-terminal stages; without it those losses get cleared at the next forward and never see a gradient.

The Megatron TP adapter is the TP layer swap. After model construction and before FSDP, it walks the module tree and replaces `nn.Linear` with Megatron's `ColumnParallelLinear` / `RowParallelLinear`. The choice is communication direction: column-parallel splits output dim and reduces in backward (QKV), row-parallel splits input dim and reduces in forward (output projection, second MLP linear). The wrapper unwraps Megatron's `(output, bias)` return so callers see a plain tensor. Faster than DTensor TP and the path we use for TP > 1 on Hopper.

The Megatron block wrapper is the alternative: wrap a Megatron `TransformerLayer` (TE spec) and call it as a research-project-shaped block. It does not own our hybrid features (Engram, MHC, DSA, Mamba, MoE), so it is only useful for A-blocks where the architectural features live elsewhere. The Megatron bridge is the lazy import firewall — same pattern as the TE bridge, TPU path unaffected.

the public STP module sample is the Semantic Tube Prediction auxiliary loss. It is in the brief because layer placement matters: Variant C samples hidden states from layers `[0, 4, 8, 12]`, and that aggregation must live on a stage that owns those layers. Under PP > 1 we restrict STP to single-layer mode because the cross-stage gather forces an extra P2P that defeats the bubble reduction.

The research DualPipe schedule layer is the schedule. Standard DualPipe pairs ranks: `N` ranks hold `N` stages, each rank stores 2 stages (its own + its mirror's), mirror ranks must sync grads, memory per rank is `2/N` — meaningful only at `N ≥ 4`. DualPipeV is the V-shape variant: `N` ranks hold `2N` stages, each rank holds 2 unique stages (no mirror), only rank 0 provides input. Memory per rank is `1/N` — half DualPipe's at the same rank count. The bubble formula `(PP/2 - 1) × (F + B - 3W)` collapses to zero at `PP=2` (no real overlap, only two stages); at `PP ≥ 4` the bidirectional schedule eats most of the bubble and the W decomposition fills the rest. The stage builder constructs the stage modules, the P2P configurator sets the inter-stage shape, the DualPipe loss wrapper owns the criterion, and the gradient sync closes the loop on reduction.

Non-obvious bookkeeping inside the integration: non-terminal stages must enable the aux-loss injector (otherwise MoE / MoD aux losses are dropped at the next forward), output deallocation must be disabled (DualPipe overlaps F and B across micro-batches; a stage's output can be needed after the next micro-batch's forward starts, and Megatron's deallocation pattern would corrupt the autograd graph), and loss must be scaled by `grad_accum_steps × num_chunks` so accumulated gradients match the non-PP path. These are the bugs we hit in that order.

## How it lands in MegaCpp

The deployment substrate is Megatron-LM with a 2-rank DualPipeV carved out of the world.

The deployment DualPipeV wrapper is the bridge. The decision that shaped everything else: Megatron runs with `--pipeline-model-parallel-size 1`. Each Megatron rank holds the full 52-layer model; we do not let Megatron split the pipeline. We carve a dedicated 2-rank DualPipe process group out of the world — on H200:8 the pipe groups are `(0,1), (2,3), (4,5), (6,7)`, giving DP=4 replicas. DualPipeV splits the 52 layers into 4 virtual stages of 13 layers, assigning `(stage0, stage3)` to `pipe_rank=0` and `(stage1, stage2)` to `pipe_rank=1` — the canonical V-shape. DP gradient sync runs across same-`pipe_rank` peers via standard `new_group` collectives.

Why PP=1 inside Megatron and DualPipeV outside? Megatron's PP schedule and DualPipeV both want to own the forward/backward dance; layering them produces an intractable cross-product schedule. Carving DualPipeV out of the world keeps TP and DP entirely under Megatron's `parallel_state`. The cost is that each rank holds the full model, so DualPipeV's `1/N` per-rank memory relief is what buys us the depth.

The DualPipeV activation patch wires `setup_dualpipev_from_args` into Megatron's `setup_model_and_optimizer`. Activation is gated by `CPPMEGA_DUALPIPEV=1`; failures crash loudly because silent fallback on a 2-rank pipeline is worse than a hard error.

The hybrid schedule plan patch handles our layer mix under Megatron's combined-1F1B path. The stock plan assumes a uniform transformer stack and stalls the comm stream every time we hit a Mamba or DSA layer. The patch wraps Mamba and DSA layers as opaque schedule nodes on the compute stream, leaves MoE layers decomposed into dispatch / MLP / combine nodes on the comm stream, and keeps both streams busy across the hybrid pattern.

The family-aware spec builder and TE spec builder assemble the `MambaStackSubmodules` spec consumed by Megatron's `MambaStack`. They override the Mamba mixer (Mamba3 vs M²RNN per layer), keep upstream TE submodules for everything else, and route attention through the DSA-absorbed MLA spec on DSA layers.

The Megatron launch-argument builder emits the runtime flags. The 8-GPU H200 single-node fragment is `--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --sequence-parallel`. Multi-node H200 moves TP up to 4 and EP up to 8. DP is implicit (`world_size / (TP × PP × EP)`). GB10 collapses the cube to TP=1, PP=1, EP=1, DP=1 — on that path the parallelism story is "all the work is in the kernel, not the cube".

MegaCpp pieces that did not survive the lift: our early pipeline partitioner, the Megatron block wrapper, and the in-process aux-loss injector — Megatron's `TransformerLayer` and `parallel_state` already do those jobs. The stage modules in the deployment DualPipeV wrapper are a focused re-implementation that knows how to slice a Megatron `MambaModel` into the 4 virtual stages DualPipeV needs.

## Bubble accounting

The honest version. DualPipeV's bubble formula `(PP/2 - 1) × (F + B - 3W)` is a lower bound, not a measurement. On our depth-52 preset with `pp=2` (4 stages V-shape, `pipe_rank=0` holds stages `(0, 3)` and `pipe_rank=1` holds stages `(1, 2)`), the bubble term collapses to zero by the formula but the real bubble is the warmup + cooldown of the schedule plus the P2P round-trip latency, which is in the 3-5% range on a single 8-GPU H200 node. At `pp=4` (8 stages V-shape) on a multi-node H200 cluster, the formula bubble is small and the measured bubble is dominated by the cross-host P2P, which we keep in the 5-8% range with bf16 P2P tensors. The W decomposition matters because it lets the schedule fill the would-be bubble with weight-gradient compute on micro-batches whose backward-input has already finished.

The hybrid pattern adds a confound the formula does not see. Mamba layers have a different `F + B - W` ratio than dense attention, and a stage that contains a high Mamba count is slower per micro-batch than a stage of pure attention. We balance stages by parameter count (`partition_model_weighted`) and verify the actual per-stage step time matches within 5% via instrumentation. When it does not, we adjust the explicit boundary list (`pp_boundaries`) until it does. There is no algorithm here; it is a feedback loop on instrumentation.

## Ablations and what we kept

The wins: DualPipeV over standard DualPipe (`1/N` vs `2/N` memory at the same rank count), Megatron `ColumnParallelLinear` / `RowParallelLinear` over DTensor TP (faster custom NCCL overlap, async allreduce), the TP-then-PP-then-FSDP2-then-EP application order, parameter-weighted layer partitioning for any MoE-bearing preset, the aux-loss injector for non-terminal PP stages, output-deallocation off for DualPipe stages, the explicit-boundary path through `pp_boundaries` for hybrid presets where the equal-split is too uneven, and bf16 P2P tensors (fp32 P2P is bandwidth-bound on cross-host).

The losses: DualPipe (non-V) at `pp=2` (no memory relief, no real overlap), DualPipeV at `pp=1` on GB10 (no second rank, the schedule degenerates), `pipe`-delimited `pp_boundaries` patterns that did not produce the right total stage count for DualPipeV (4 segments for `PP=4` is wrong, we need 8), and Megatron PP > 1 layered under DualPipeV (the cross-product schedule is not tractable).

The neutrals: TP=2 vs TP=4 on a single 8-GPU H200 node is roughly a wash for the dense layers; we run TP=2 because it leaves more room for EP=4 and DP=2 inside the same world. Sequence parallelism is on whenever TP > 1; the activation savings on MLA projections justify the allgather/reduce-scatter. STP Variant C is disabled under PP > 1.

The boring engineering: the validator in `validate_3d_config` runs on every job submission, not just at startup. Every smoke run also executes the `dualpipev_forward_backward` path on a depth-13 toy model so the schedule code itself has continuous coverage. The mirror-grad and DP-grad sync paths are tested on a 2-rank and 4-rank smoke, respectively, on every PR.

## Production checklist

- Apply parallelism in TP → PP → FSDP2 → EP order. Out-of-order produces silent sharding corruption.
- The depth-52 hybrid preset uses Megatron `--pipeline-model-parallel-size 1` and a 2-rank DualPipeV carved out of the world.
- Each Megatron rank holds the full model; DualPipeV splits 52 layers into 4 virtual stages of 13 layers, assigned in V-shape to `pipe_rank` 0 and 1.
- DP gradient sync runs across same-`pipe_rank` peers via standard `torch.distributed.new_group` collectives.
- Aux-loss injection is enabled on non-terminal stages.
- Output deallocation is disabled on DualPipeV stages (otherwise the autograd graph corrupts under bidirectional overlap).
- Loss is scaled by `grad_accum_steps × num_chunks` to match the non-PP gradient magnitude.
- The hybrid schedule plan patch is applied; opaque nodes wrap Mamba and DSA layers so the comm stream stays busy through the AEMEM... pattern.
- Layer partition is parameter-weighted on any MoE-bearing preset; explicit `pp_boundaries` overrides apply when the auto-balanced split exceeds 5% per-stage step-time imbalance.
- TP layers go through Megatron `ColumnParallelLinear` / `RowParallelLinear`. Sequence parallelism is on whenever TP > 1.
- P2P tensors are bf16; the inter-stage shape is `(B, T, D)` set via `configure_p2p_tensors` before the first step.
- The 3D validator runs on every job submission; failures are hard errors, warnings are logged.
- GB10 paths ignore DualPipeV entirely. The cube collapses to TP=1, PP=1, EP=1, DP=1; throughput comes from the kernels.
- STP is restricted to single-layer mode under PP > 1.

## Application order and what it costs

| Axis | What it splits | Comm | Notes |
|---|---|---|---|
| TP | weight/activation | all-reduce per layer | Megatron-Core preferred over DTensor on CUDA |
| PP (DualPipeV) | layers (V-shape) | P2P across `pipe_rank` | bidirectional overlap |
| FSDP2 | params + grads + optimizer | all-gather + reduce-scatter | applied after TP+PP |
| EP | experts | all-to-all | applied last |

```python
# Apply order: TP -> PP -> FSDP2 -> EP. Out-of-order silently corrupts shards.
apply_megatron_tp(model, tp_size=tp)
apply_dualpipe_v(model, num_chunks=4, virtual_stages=4, pipe_rank=rank)
apply_fully_shard(model, mesh=fsdp_mesh)
apply_expert_parallel(model, ep_size=ep)
```

## References

- [DualPipe — DeepSeek-AI, GitHub](https://github.com/deepseek-ai/DualPipe)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Megatron-LM — NVIDIA, GitHub](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch `torch.distributed.pipelining`](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [Hybrid layout docs](https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md)
- [STP sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/stp/stp_sample.py)
