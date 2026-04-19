---
title: "Why a 4B-8B model fills an H200 and still OOMs"
description: "A detailed accounting of where 141 GB of HBM goes when you train a 4B-8B hybrid Mamba 3, Transformer, and MoE specialist: parameters, gradients, optimizer state, activations, KV cache, MoE routing buffers, and allocator fragmentation."
date: "2026-04-18"
tags: ["memory", "h200", "moe", "mamba", "activations", "fsdp2", "training"]
---

The first time an engineer used to LLaMA-class training looks at a hybrid training stack, they say the same thing: "It is a 4B model, you have 141 GB per GPU, why are we OOMing." The short answer is that the parameter count is almost the smallest term in the memory budget once every contributor is written down. The long answer is this post. The same arithmetic that makes a dense 7B look comfortable on an 80 GB card makes a hybrid Mamba 3 + Transformer + MoE specialist at 4K-16K context push the H200's 141 GB ceiling, and turns the 80 GB H100 into "literally does not fit at our target microbatch".

## Why this matters

We train specialists, not generalists. A 4B-8B model for code-like domains is the right size for quality, latency and cost. That choice is non-negotiable; what we negotiate is every other knob that lets the model actually fit on the device. The pre-flight estimator is not a nice-to-have, it is the gate on whether a preset is allowed to run at all, and it is the tool we reach for before buying more compute. An operator who cannot articulate where the last 10 GB went is one step from "oh, it worked on Monday".

## Where the bytes go

Three components do this job: an analytical pre-flight memory estimator, runtime allocator inspection via `torch.cuda.memory_stats`, and a persisted calibration record that the auto-fit retry ladder reads. The estimator carries per-component fields on a structured memory estimate, and the rest of the system reasons over those fields rather than over a single peak number.

| Component | Typical share at TP=2 EP=4 | Lever |
|---|---|---|
| Parameters (bf16) | 2-4 GB / device | TP, EP shard |
| Gradients | ~params (eager); /dp (FSDP2) | FSDP2 |
| Optimizer state | 8-32 GB (Muon vs pure AdamW) | Muon, FSDP2, DP degree |
| Activations (4K ctx, depth 52) | 10-20 GB | Gradient checkpointing, recompute |
| MoE routing/dispatch | several GB | Capacity factor, EP, dispatcher |
| KV cache (training: none; eval: hot) | 0 / variable | Paged KV at eval only |
| Allocator overhead | 5-10% of peak | `expandable_segments:True` |

### Parameters

The parameter footprint is the term operators intuit correctly. At bf16, a 4B-8B model is 8-16 GB of weights before sharding. A serious estimator breaks this down by shard class such as replicated, TP-sharded, EP-sharded, or CPU-only, and by optimizer class such as Muon or AdamW. That matters because TP shards Q/K/V/O and MLP weights across the tensor-parallel axis while EP shards the routed-expert bank across the expert-parallel axis. Replicated weights (embeddings, norm affines, router projections, some feature banks) stay full-size on every rank.

### Gradients

For the eager CUDA path, the estimator counts materialised gradients at the same byte-for-byte size as the sharded parameter set. FSDP2 keeps gradients sharded along the DP axis after the reduce-scatter, so the gradient term scales down with `dp_degree` for anything that is not replicated. For replicated tensors, gradients are not sharded; that is a small tax but not negligible once the feature stack gets dense. On XLA SPMD we deliberately do not count a full extra gradient copy, because the compiled step folds update scratch into a fused region; counting it separately would overstate TPU HBM. On CUDA, we count materialised, because it is materialised.

### Optimizer state

This is where the arithmetic surprises the newcomer. AdamW keeps `m` and `v` moments in fp32, which is 8 bytes per parameter. Muon keeps a single bf16 momentum buffer at ~2 bytes per parameter. A 4B model under pure AdamW needs 32 GB of optimizer state before sharding; the same model under Muon needs 8 GB. On a well-sharded FSDP2 run the optimizer state divides across the DP group, but it is the single largest memory component on any configuration with a small DP degree. Our production route is Muon on most matrix parameters and selective AdamW on specific output projections; the 4x saving over pure AdamW is the single reason the 4B-8B size is feasible on this topology at all.

### Activations

Activation memory scales with `batch * seq * hidden * layers * dtype * K` where `K` is the implementation-specific constant of how many intermediates per layer are saved for backward. For a dense attention block with flash attention (no materialised QK^T), the Megatron formula gives roughly 34 half-precision elements per token-channel per layer. In bytes, `B * T * D * 34 * n_layer * bpe / 2`, At `B=1, T=4096, D=1536, n_layer=52, bpe=2` that is on the order of 10-20 GB before any hybrid-specific extras. With gradient checkpointing the estimator switches to a much smaller formula, which is the single biggest memory lever available.

The hybrid stack adds real terms. Mamba layers in fp32 are roughly 2x the per-layer activation of a bf16 attention layer (we scale by the M-layer fraction). MoE routing adds per-layer router logits in fp32 plus gathered input and expert output buffers sized by capacity factor, summed over the MoE layer count. DSA indexer and Engram add their own intermediates. The accumulated total dominates when context grows.

## MoE routing, KV cache, and allocator overhead

MoE routing buffers are the term newcomers forget. With capacity factor 1.25, top-6 routing on 64 experts, and a 4 K context, the per-layer dispatch buffers are several hundred MB; summed over the MoE layer count of the deep preset, that is several GB. The estimator computes per-layer buffers as `B * T * N * 4` (router logits, fp32, replicated) plus `local_experts * C_e * D * bpe` twice (gathered input and expert output), with `C_e = ceil(1.25 * B * T * top_k / N)`. This is also why the MoE preset cares about the dispatcher: the alltoall path holds a smaller working set than the padded path, but the padded path is faster on small EP.

KV cache during training is zero because we recompute attention via flash. KV cache during eval is variable, capped by paged KV, and is what makes the inference memory budget look completely different from the training memory budget. The estimator does not count it on the training path; the eval path has its own calibration.

Allocator overhead is the last term and the one that breaks late. Without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, MoE activation churn fragments the caching allocator into a state where `reserved - allocated` drifts upward and a step that had headroom at step 50 OOMs at step 500. With it on, fragmentation is bounded; we still budget 5-10% of peak as allocator overhead because the allocator does not give back what it does not have to.

```python
from memory_estimator import estimate_memory

est = estimate_memory(
    model_config=cfg, world_size=8, tp_degree=2, ep_degree=4,
    dp_degree=1, batch_size=1, seq_len=4096,
    grad_checkpoint=True, optimizer="muon+adamw",
)
for name, gb in est.as_dict().items():
    print(f"{name:24s} {gb:6.2f} GiB")
```

## How the estimator drives auto-fit

The startup calibration pass writes a small JSON record at the end of every successful step 0 with the per-component sizes the estimator predicted versus what `torch.cuda.memory_stats` actually reported. The auto-fit retry ladder reads that record before the next launch: if predicted activations were within 5% of measured, the next launch trusts the estimator; if they were off by more, the next launch widens the safety margin. The same record is what lets `dbs` step down from 8 to 4 to 2 without re-discovering the memory cliff every time. Without the calibration loop, the estimator is just a calculator; with it, the estimator is the contract that decides whether a launch is allowed.

The runtime forensics path pairs `torch.cuda.memory_stats` with memory snapshots. When a step OOMs, we capture a memory snapshot, diff it against the previous step's snapshot, and look for the largest growing bucket. Most late-stage OOMs are not "model too big"; they are "MoE dispatch buffer grew because a new expert got hot" or "Mamba conv1d state grew because the chunk size changed". The forensics has to be granular enough to tell those apart.

## What proved worth keeping

The analytical estimator stays as the launch gate, the calibration loop stays as the auto-fit input, Muon plus selective AdamW remains the optimizer baseline, gradient checkpointing remains standard above shallow presets, `expandable_segments:True` stays the allocator default, and any new feature should ship with an estimator entry before it ships with a launcher entry.

Pure AdamW at the 4B-8B size is not attractive because a 32 GB optimizer state for a 4 GB parameter set is hard to justify. Per-step memory snapshots in the hot path are also too expensive. It is also a mistake to assume attention dominates activation memory on every deep preset, or to plan capacity while ignoring MoE dispatch buffers and allocator overhead. That discipline is not glamorous, but it is the reason a 4B model can fit on an H200 without relying on guesswork.

## The 16K context wall

At 4K context the deep preset fits on an H200 with room to breathe; at 16K the same preset is on the edge of OOM, and the contributors that grow are predictable. Activation memory scales linearly with sequence length, so the bf16 attention term roughly quadruples between 4K and 16K. Mamba's per-layer fp32 activation also grows linearly. MoE dispatch buffers grow linearly because `C_e` is proportional to `B * T`. KV cache during training is still zero (we recompute), so the long-context tax is entirely in activations and dispatch. Gradient checkpointing absorbs most of the activation growth; what it does not absorb is the dispatch buffer growth, which is why the 16K configuration shifts the dominant memory term from "Mamba activations" to "MoE dispatch".

The implication for capacity planning is that the 16K configuration cannot run at the same EP and TP shapes as the 4K configuration. We documented two operating points: 4K at TP=2 EP=4 dp=1 with full FSDP2, and 16K at TP=2 EP=4 dp=1 with FSDP2 plus selective recompute on the MoE blocks. The 16K configuration is not faster per token; it produces longer sequences for the data mixes that need them, and the memory budget is the binding constraint.

## What CUDA reports versus what is true

`torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` are different numbers and they diverge during a run. Allocated is what tensors currently hold; reserved is what the caching allocator has held onto. The gap is allocator overhead, and it grows under MoE workloads because the dispatch buffer cycles through different shapes per expert popularity wave. With `expandable_segments:True` the gap is bounded; without it, the gap drifts upward indefinitely.

Two diagnostics matter. The first is `torch.cuda.memory_stats()`, which exposes per-segment counters that let us tell "we held a 2 GB chunk for 30 seconds and freed it" from "we held it for the whole run". The second is the snapshot diff: capture a snapshot at step N and step N+100, sort buckets by growth, and the largest growing bucket is almost always the diagnosis. We snapshot every 1000 steps in production, not every step, because the snapshot itself takes a few hundred milliseconds and we do not want to perturb the steady-state.

The measurement record should capture peak `memory_reserved` per rank and the largest growing bucket between the first and last steady-state snapshot. When a regression appears, that pair of numbers is usually enough to point at the cause.

## What this implies for the next architecture

The memory accounting above is what disqualifies a number of attractive architecture moves. Doubling the expert count from 64 to 128 doubles the dispatch buffer at fixed `C_e`, and the deep preset cannot absorb that on H200 at 16K context. Moving to a denser MoE (top-k = 8 instead of 6) increases `C_e` proportionally, with the same effect. Adding a second Mamba branch per layer doubles the per-layer fp32 activation, which the gradient checkpointing absorbs but only at the cost of an extra recompute pass.

Each of these is a real proposal someone has made; each was rejected on the strength of the analytical estimator, not on a benchmark. The estimator is the gate. The benchmark exists to confirm the estimator and to catch the cases where the estimator is wrong about a specific shape. If we lose the estimator we lose the gate, and architecture proposals start landing on hope rather than arithmetic.

## References

- Analytical memory estimation for launch gating
- Runtime allocator statistics and memory snapshots for forensics
- Startup calibration that compares predicted versus measured memory
- FSDP2, MoE, and Muon components that shape real training memory
