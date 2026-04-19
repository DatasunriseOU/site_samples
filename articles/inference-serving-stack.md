---
title: "Serving the Eight: Router, Per-Specialist Scheduler, and the KV Layout That Keeps Them Honest"
description: "How we actually serve an eight-specialist C++ ensemble: top-level router, per-specialist continuous-batch schedulers, paged KV per model, admission control, and the SLOs we publish."
date: "2026-04-18"
tags: ["inference", "serving", "ensemble", "kv-cache", "scheduler", "slo"]
---

The MegaCpp ensemble is eight specialists, not one model. That single architectural fact reshapes every serving decision we make. A monolithic generalist can live inside one vLLM-style engine with one paged KV pool and one scheduler. An ensemble cannot: the models have different sizes, sparsity patterns, hybrid interleavings, and quantization targets. This post is about the stack above the kernels — router, per-specialist scheduler, KV layout, admission control, and published SLOs. Kernel choices and NVFP4 layout are covered elsewhere.

## Why this matters

An ensemble exposes the same product surface as a single model and is much harder to serve well. The cost of a bad serving design is not a slow demo; it is silent specialist starvation, KV evictions that cascade across replicas, tail latency that wanders by intent class, and a debugging story where nobody can answer "which model actually answered this request" without grepping logs. We learned the hard way that the only way to make eight specialists feel like one product is to keep their boundaries crisp inside the serving stack: one router on top, one scheduler per specialist, one paged-KV pool per specialist, one published SLO per specialist. This post is the design that survived two redesigns.

## 1. The shape of the problem

An incoming request is a chat-like blob of C++ context: a prompt, optional repo snippets, optional tool outputs, and a caller-declared intent (`codegen`, `debug`, `build-fix`, `review`, or `unspecified`). It arrives with a priority integer, an optional deadline, and a preferred decoding policy (greedy, typical, temperature-sampled). The serving stack has to decide three things before any token comes out:

1. Which specialist (or specialists) answer this request.
2. Which instance of that specialist takes it, on which GPU, in which batch.
3. How its KV footprint is paid for, and what to preempt if it will not fit.

Each decision has its own timescale. Routing is per-request and happens once. Scheduling is per-token and happens thousands of times per second. KV allocation sits underneath scheduling and decides whether the next decode step can even run. We kept the three decisions in three different components on purpose, because conflating them is how serving systems end up with router logic leaking into kernel dispatch.

## 2. The top-level router

The router is a small model plus a rules layer. The small model is a distilled classifier in the tens of millions of parameters, trained on labeled dogfood traffic and curated examples from our enriched-corpus dataset family; it produces a distribution over the eight specialists plus a `reject` class. The rules layer overrides it in two cases: when the caller declared an intent that maps directly to a specialist (debug traces always go to Debug-SLM, build files to Build-SLM), and when the prompt tripwires a structural detector (SFINAE-heavy headers route to Template-SLM regardless of the classifier).

### Shadow dispatch and what the router does not do

The router outputs a primary specialist and, for high-stakes requests, a shadow specialist. The shadow is only dispatched when the primary's top-1 probability is below a threshold tuned per intent. For pure `codegen` traffic the threshold is low and we almost never shadow. For `review` traffic, where the cost of a wrong specialist is a confidently wrong review, the threshold is high and we shadow more often.

A few things the router deliberately does not do. No token-level reassignment — once routed, a request stays on its specialist for the whole generation. No cross-specialist output stitching; the quality penalty is visible. No online learning from real-time feedback; the classifier is retrained offline on labeled traffic.

## 3. One scheduler per specialist, not one across all of them

The instinct is to run one global scheduler that picks the best GPU for each request across all specialists. We built that first and threw it away. Specialists have different KV-per-token footprints (hybrid ratios and head counts vary), different maximum contexts, and different ideal decode batch sizes. A global scheduler has to carry all of that simultaneously; admission becomes a constraint solver and tail latency gets worse, not better, because requests sit behind decisions they have no dependency on.

Each specialist instance now runs its own continuous-batching scheduler. The in-repo primitive is a continuous-batch scheduler that sits between incoming requests and a paged-KV block manager for that specialist. Its contract is small and explicit:

- Hold a waiting queue ordered by `(priority DESC, arrival ASC)`.
- Try prefix-cache reuse before allocating fresh blocks.
- Admit a request only if enough free blocks exist to cover its prompt plus at least one decode step.
- Preempt the lowest-priority running sequence when a strictly-higher-priority request is waiting.
- Group within the scheduler by adapter identity, because adapter swaps are the second most expensive thing we do after KV eviction.

The router sits above the eight schedulers as a thin fan-out. It picks a specialist, picks one of its replicas (least-loaded queue depth, with a small penalty for replicas currently preempting), and hands off. The scheduler does not know about routing; the router does not know about blocks. This boundary is the single most important structural decision in the stack.

## 4. KV cache layout across specialists

Paged KV is non-negotiable for continuous batching; we inherited the design from vLLM and FlashAttention's `block_table` and stayed close to canonical. What is specialist-specific is the block size, the pool size, and the prefix-cache key.

| Specialist | Block size (tokens) | Why |
|---|---|---|
| Template-SLM | 8 | Long, repetitive header sequences; deep cross-request reuse |
| STL-SLM | 8 | Same prefix-heavy pattern over `<ranges>` / `<algorithm>` |
| Algo-, Memory-, Concurrency-, Systems-, Build-, Debug-SLM | 16 | Default; balances reuse vs `block_table` indexing cost |

Pool size is set from the GPU's free HBM after weights and a reserved activation budget; we deliberately avoid dynamic pool growth because every growth event is a serving stall waiting to happen. The prefix-cache key is `(specialist_id, adapter_id, hashed_token_prefix)` — including the adapter is what stops cross-adapter cache poisoning that bit us once on a Debug-SLM A/B.

### Hybrid layers change the math

Several specialists interleave attention with Mamba3 SSM blocks. SSM layers do not have KV cache in the usual sense; they have per-step conv state and SSM recurrence state that are bytes-per-layer, not bytes-per-token. So a hybrid specialist's KV footprint per token is lower than a pure-attention model of the same parameter count, and its preempt-and-resume path has to snapshot SSM state separately. The scheduler treats the SSM snapshot as part of the sequence's serialized state; the block manager only owns the attention KV.

The practical consequence is that a hybrid specialist with a heavy Mamba3 ratio can hold more concurrent sequences in the same HBM than a pure-attention model of the same parameter budget — but the per-sequence preempt cost is higher, because we have to copy the SSM state to host-pinned memory on eviction and copy it back on resume. The scheduler accounts for that asymmetry when it picks a preemption victim: at equal priority it prefers to preempt a pure-attention sequence over a hybrid one.

## 5. Admission control and SLOs

Admission is where the serving stack acquires its honesty. A request is admitted when (a) the chosen specialist's scheduler has room for prompt + one decode step, (b) the caller's deadline is achievable given current queue depth, and (c) admitting does not push another in-flight request below its own deadline. If any of those fail, we either preempt a strictly-lower-priority request, return a typed `429`-equivalent with a retry-after, or shadow-route to a less-loaded specialist when the router said the second-choice was viable.

### What we publish

Per-specialist SLOs, surfaced on the serving dashboard and in the response headers:

- Time-to-first-token (TTFT) p50/p95/p99.
- Inter-token latency (ITL) p50/p95/p99.
- Admission-to-first-token p95 (admission queueing visible to the caller).
- Per-specialist correctness floor: rolling 24h C++ compilation-pass rate.
- Preempted-once fraction; this is a transparency knob, not an SLO, and we publish it because it matters to callers building latency-sensitive pipelines.

We do not publish a single ensemble latency number. Asked once, given on demand, never on the dashboard, because nobody can act on it.

### Backpressure and circuit breakers

Two failure modes have to be caught before they propagate. The first is a specialist whose correctness SLO is drifting — a model refresh ships a regression and the rolling compilation-pass rate drops below the floor. The router carries a circuit breaker per specialist that, on sustained breach, demotes that specialist to shadow-only and lets a configured fallback specialist take primary traffic for the affected intent. The breaker resets only when the rolling rate climbs back above the floor for a sustained window; we do not flap.

The second is a load spike against a single specialist. Admission already returns a typed retry-after when the chosen scheduler is full, but if the spike is sustained the router escalates: it lowers the shadow threshold for that specialist (so the second-choice catches more low-confidence traffic), and if that is not enough it raises the priority floor for the affected intent. The result is that low-priority traffic gets throttled before high-priority traffic ever sees a deadline miss. Both knobs are operator-visible on the serving dashboard, and both are reversible without a deploy.

## 6. Adapters, quantization, and the dispatch matrix

Each specialist replica carries its base NVFP4 weights plus a small number of adapter rotations. Adapters are LoRA-shaped delta tensors held in BF16 because the rank is low and the bandwidth cost is dominated by the base weights anyway. The scheduler groups by adapter identity inside its batch so the dispatch kernel can fold the LoRA delta in a single pass. Cross-adapter batching is technically supported and operationally avoided; it doubles the activation traffic for marginal batch fill.

The dispatch surface looks like:

```python
# pseudocode for the per-specialist dispatch step
batch = scheduler.assemble_step()              # group by adapter, decode/prefill split
kv_view = block_manager.view(batch.sequences)  # paged-KV table for FA-Blackwell / SDPA
adapter = adapter_pool.get(batch.adapter_id)   # LoRA delta in BF16

with serving_telemetry(batch):
    logits = engine.forward(
        tokens=batch.tokens,
        kv=kv_view,
        adapter=adapter,
        positions=batch.positions,
        mode=batch.mode,                        # "prefill" | "decode" | "spec_verify"
    )
    sampled = sampler.sample(logits, batch.policies)

scheduler.commit(batch, sampled)
```

`spec_verify` is the path that speculative decoding uses; it shares everything with `decode` except the K-token block shape. Keeping it in the same dispatch surface is what lets the spec-decode path inherit all of the admission, preemption, and KV-rollback bookkeeping for free.

## What we kept and what we threw away

Kept: a small classifier plus rules at the top, one continuous-batching scheduler per specialist, one paged-KV pool per specialist, prefix-cache keyed by `(specialist, adapter, prefix-hash)`, per-specialist SLOs published as the only latency contract, adapter-grouped batches, a single dispatch surface for prefill/decode/spec-verify, and SSM state owned by the sequence rather than the block manager.

Threw away: a global cross-specialist scheduler, token-level rerouting mid-generation, cross-specialist output stitching, online drafter learning, dynamic KV-pool growth, cross-adapter batching at scale, and an aggregate ensemble latency number on the dashboard.

The boundary between routing, scheduling, and KV management is the only thing keeping the eight specialists feeling like one product. Every redesign we made on either side of that boundary stuck; every redesign that tried to dissolve the boundary got rolled back inside a week.

## Public references

- [MegaCpp public repository](https://github.com/DatasunriseOU/cppmega)
- [vLLM documentation](https://docs.vllm.ai/)
- [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
