---
title: "Benchmarking the MegaCpp stack on Modal: multi-GPU lessons from rented boxes"
description: "What we learned running the training stack on rented H100, H200, and B200 boxes through Modal: three benchmark lanes, an 8-GPU FSDP2 hang, and the bookkeeping that lets the numbers survive a week."
date: "2026-04-18"
tags: ["modal", "benchmarks", "multi-gpu", "fsdp", "h200", "b200", "reproducibility"]
---

Modal is, for us, a benchmarking surface and an overflow capacity pool — not the production training cluster. We use it to answer questions of the form "is the current MegaCpp stack still the best variant on H200?" and "how much faster is B200 than H100 on a real training step, not a synthetic kernel?". The numbers we get back have a much shorter half-life than people assume; the bookkeeping around them is what keeps them honest after a week.

## Why this matters

Rented Blackwell- and Hopper-class boxes are how we sanity-check what our long-lived training systems tell us. They expose the things that a warm, well-loved system hides: cold inductor caches, ABI mismatches between the wheel set we built locally and the image we rent, network and storage paths that do not look like our own. A throughput number from a clean container is harder won and more honest than the same number off a system that has been compiling kernels for a month. This post is the operating manual we wish we had had before the first 8-GPU run on Modal — the three benchmark lanes we treat as distinct, the multi-GPU failure mode that ate the most time, and the bookkeeping that lets a number outlive the wave that produced it.

## 1. Three lanes, not one

There is a recurring temptation to wave at "the Modal benchmark" as if it were a single number. It is not. The repo carries three distinct Modal surfaces, and treating them as interchangeable is how stale claims sneak back into reports.

### Whole-model training benchmarks

Where we measure throughput in real `tok/sec` from the global batch (not per-GPU), compare structural variants under a controlled launch regime, and answer questions like "does removing MTP hurt steady-state throughput?" or "is FSDP2 still worth it versus DDP on this geometry?". Steady-state means post-warmup steps; step 0 is discarded by hygiene. Eval is disabled for the duration of the benchmark window, so the number reported is throughput, not throughput plus eval overhead.

### Exact-token sparse detached benchmark

Benchmarks the sparse-attention path in isolated eval/no-grad form so we can record exact runtime telemetry and the backend identity that actually ran. The supported launcher uses an explicit `app.run(detach=True)` lifecycle plus a collector script; the local `modal run ... --detach` shortcut is intentionally not the accepted contract here, because we want lifecycle objects we can audit later. The artifacts on this lane are not throughput numbers; they are `bench_result`, `bench_telemetry`, `backend_identity`, the remote runtime provenance blob, and the saved `app_id` plus `function_call_id`.

### Sparse validation and FA4 promotion

Bounded acceptance: parity checks, promotion-readiness, summary manifests for a wave. The success criterion is not "highest tok/sec". It is promotion status, promotion readiness, and pass/fail summaries. A benchmark record here showing Triton promotion success does not imply that FA4 runtime import and bootstrap are healthy on the same image; that is a different field on the manifest, and you have to read it.

The reason we keep these separate is purely operational: collapsing them produces wrong claims. A green sparse validation does not justify a number on the training lane. A throughput number from the training lane does not prove the sparse acceptance lane works. The cost of conflation is that someone quotes one as the other in a chat thread two weeks later, and we re-run the box to disprove the misquote.

## 2. What works today

The single-GPU story is straightforward. Full training of our current depth-52 dense preset runs end-to-end on H100, H200, and B200 instances. The full pytest suite passes on a Modal H100 image without skips.

Multi-host throughput is where Modal earned its keep. Same model, same recipe, same image, three GPU classes:

| GPU  | Best `tok/sec` (single device) | Relative to H100 |
|------|--------------------------------|------------------|
| H100 | ~2.8k                          | 1.00x            |
| H200 | intermediate                   | varies by recipe |
| B200 | ~4.3k                          | ~1.55x           |

The B200 number is the one that actually matters for capacity planning. It is roughly 1.55x H100 on the same recipe — not 2x, not "Blackwell magic", just 1.55x — and at current spot prices it does not pay for itself versus H200 unless we can keep it saturated. We mention that explicitly because an earlier draft of this table once carried a "B200 is the future" caption that was charitable to silicon and unfair to procurement.

Pricing context for capacity planning, per GPU-hour at the time of writing:

| GPU  | $/hr  | 8-GPU $/hr |
|------|-------|------------|
| B200 | 6.25  | 50.00      |
| H200 | 4.54  | 36.32      |
| H100 | 3.95  | 31.60      |

These move; the ratios move slower. Decision rule: B200 only when we are bottlenecked on memory bandwidth or HBM capacity for the specific recipe, H200 by default, H100 only for cheap regression sweeps where a low-double-digit gap is irrelevant.

## 3. The 8-GPU hang

The honest part of any benchmarking post is the failure mode that cost the most time. For us, on Modal, that was the 8-GPU FSDP2 path with `regional_compile` enabled. The symptom is the worst kind: the run launches, ranks initialize, the first forward pass enters — and then nothing. No traceback, no NCCL timeout for a long while, no useful log slice.

The root cause is mundane once you see it. With a cold inductor cache, Triton JITs each kernel on first use. JIT time is not deterministic across ranks. Eight ranks therefore enter the first NCCL collective at eight different moments, and the collective deadlocks because some ranks are still inside the compiler.

The same code does not hang on our long-lived H200 training systems because those systems have warm inductor caches from prior runs; the JIT path is effectively a cache lookup, the rank skew collapses, and the collective proceeds. Modal containers are clean by default, so they hit the slow path every time.

### Options we evaluated

There is no clever single-line fix.

1. Pre-bake the inductor cache into the Docker image, sourced from a warm H200 VM. Cleanest fix and the one we are converging on; it moves the variance off the hot path and into image build time.
2. Mount a Modal Volume with a pre-populated cache from a prior 8-GPU run on the same image. Works, but the cache must come from an 8-GPU run on the matching image; an 8-GPU cache from a different image, or a 1-GPU cache from the right image, does not cover the kernel set.
3. Sequential compile warmup. Tempting, but FSDP2 changes the graph in ways that make a "compile once on rank 0 and fan out" strategy unsafe. Discarded.
4. Reduce model complexity for the Modal lane — fewer MoE experts, no MoD — to shrink the kernel set the first compile has to JIT. What we actually do for quick acceptance runs while the cache-baked image is being prepared.

The relevant lesson is that "works on the long-lived training host" was hiding a real determinism gap. Modal forced it into the open by giving us a fresh container every time.

## 4. Data plumbing

Training data lives in private cloud storage. For multi-GPU training, FUSE-style parallel reads from inside Modal's container did not survive eight concurrent readers. We saw read stalls and partial-shard reads, not corruption, but the training step took the latency hit. The fix is dull and effective: pre-copy the relevant shards into a Modal Volume once, then mount the Volume and read from local disk. Throughput becomes deterministic and the egress bill drops.

Fused kernel wheels live in the same bucket area. We pin them by image so the "which kernels does this run actually use" question always has a single answer in the run manifest.

## 5. Bookkeeping is the deliverable

A throughput number with no provenance is a rumor. A throughput number with the right metadata is a benchmark record that survives the next stack upgrade. For each lane we record a different set of fields, deliberately.

For the **whole-model training lane**: `app_id`, `function_call_id`, the exact launch flags (verbatim, not paraphrased), the parsed steady-state step metrics, and the exact distributed mode. Without the distributed mode, "180k tok/sec on H200" is unreproducible — it could be DDP, FSDP2, FSDP2 with compile, or Megatron-style.

For the **exact-token sparse lane**: launcher args, case metadata, the exact sparse env selectors, the runtime telemetry payload, the `backend_identity` the run actually used, the remote runtime provenance, and the detached collector's state transitions. The last one is what lets us reconstruct "did this finish on its own or did the collector reattach?".

For the **sparse validation / FA4 lane**: the validation/promotion mode, `promotion_status`, `promotion_ready`, the saved summary manifests, and whether the run was detached or blocking.

If a Modal output does not carry these, we treat it as anecdote. If it does, it stays useful for months.

### A benchmark record sketch

```yaml
# whole-model training lane
app_id: app_xxxx
function_call_id: fc_xxxx
image_digest: sha256:...
gpu_class: H200:8
distributed_mode: FSDP2+regional_compile
launch_flags: |
  --backend fsdp2 --regional-compile --batch 4 --grad-accum 8 ...
steady_state:
  tok_per_sec_global: ...
  step_p50_ms: ...
  warmup_steps_discarded: 8
kernel_wheel_pins:
  flash_attn: 2.8.3
  mamba_ssm: 2.3.1
  causal_conv1d: 1.6.1
```

## 6. Practical routing

For anyone using the same surfaces, the routing is:

- the benchmark matrix for whole-model benchmark intent.
- the detached sparse benchmark launcher, paired with the collector, for exact-token sparse acceptance.
- the sparse-validation launcher for sparse and FA4 promotion waves.

The convenience harness is fine for one-off curiosity runs. It is not the source of truth for distributed H200 claims, and we do not let it become one.

## What we kept and what we threw away

Kept: three explicit benchmark lanes with separate launchers and separate benchmark records, single-GPU end-to-end runs on H100, H200, and B200 as the smoke-test surface, the cache-baked image as the long-term answer to the cold-cache hang, pre-copied dataset shards in a Modal Volume, image-pinned kernel wheels, and per-lane schemas that make the run reproducible.

Threw away: the convenience harness as a source of truth for distributed claims, "Modal benchmark" as a single number, FUSE reads from eight concurrent ranks, ad-hoc compile-and-fan-out warmup under FSDP2, the assumption that an 8-GPU cache from a different image would cover this image's kernel set, and any Modal output that does not carry the lane-appropriate metadata.

The Modal numbers that survive a stack upgrade are the ones whose benchmark record names the lane, the image, the distributed mode, and the steady-state slice. Everything else is a rumor that costs a re-run.

## What we will not claim

We will not claim that older checked-in training outputs prove the current training lane is universally healthy. They are dated evidence from the wave they came from. The multi-GPU FSDP2 plus compile lane is alive on warm-cache H200 systems and is being made reliable on Modal via the cache-baked image; it is not yet a one-command experience for anyone starting from scratch. When it is, we will say so in the benchmark record, not in a tweet.

## References

- MegaCpp benchmark planning notes
- MegaCpp multi-GPU status notes
- Modal documentation on containers, volumes, and GPUs
- PyTorch documentation for FSDP2 and `torch.compile`
