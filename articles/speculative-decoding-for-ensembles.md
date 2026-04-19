---
title: "Speculative Decoding Inside an Eight-Specialist Ensemble"
description: "Drafter choice, acceptance rates on real C++ workloads, and the failure modes we hit adapting speculative decoding to an ensemble of specialists."
date: "2026-04-18"
tags: ["speculative-decoding", "inference", "ensemble", "mtp", "eagle", "serving"]
---

Speculative decoding reads as trivially correct in a paper and lands as a pile of contract work in a real serving stack: drafter choice, KV rollback, acceptance sampling, tree verification, tool-call interaction. For our eight-specialist C++ ensemble the question is sharper: where does the drafter live, how much does it actually save, and what does it break on the way. This post is what we learned running the experiment and shipping one bounded version.

## Why this matters

Per-token decode latency is the price callers actually feel, and a credible 1.3x-1.6x compounded across millions of tokens a day is real money and real product latency. Most published numbers come from single generalist models with no tool calls, no hybrid SSM/attention layers, no per-specialist adapters, and no router. We have all four. So the interesting question is not "does speculative decoding work" — it does — but which variant survives an ensemble's coordination cost without erasing its own gain. This post is the version that did, and the variants that did not.

## 1. Where the drafter can live

There are four plausible places to put the drafter in an ensemble this shape. We evaluated all of them.

1. A small shared drafter outside the specialists. One tiny model drafts for all eight.
2. A per-specialist external drafter. Each specialist has its own small model trained against it.
3. A per-specialist self-drafter. Each specialist drafts itself via an MTP head or EAGLE-style feature-space head.
4. Cross-specialist drafting. A fast specialist drafts for a slow one when the router is unsure.

Option 1 is the cheapest and the one we wanted to work. It does not. A shared drafter has to cover eight training distributions — template-heavy headers, GDB transcripts, CMake files, SFINAE-heavy concepts, lock-free queues — and per-specialist acceptance collapses toward the worst of them. A drafter in the low hundreds of millions of parameters gave acceptance well below the speedup threshold once you pay for the extra forward pass.

Option 2 works and is operationally miserable. Eight extra models to train, checkpoint, quantize, deploy, and keep version-aligned, with a re-train every time a main specialist refreshes weekly. We shipped it for two weeks on one specialist and retired it.

Option 3 is what we run. The Multi-Token-Prediction (MTP) module trains jointly with the main model and is otherwise unused at inference. Repurposing it as a self-drafter costs one thin wrapper that runs the MTP block autoregressively, and rides on weights already in the checkpoint.

Option 4, cross-specialist drafting, is structurally appealing and structurally broken. Acceptance math requires draft and verify to share a sampling distribution. Specialists trained on different data mixes with different adapters do not. Measured acceptance across pairs we expected to be close was worse than generating from scratch.

### The choice we shipped

| Drafter design | Verdict | Why |
|---|---|---|
| Shared external | Rejected | Eight distributions collapse acceptance |
| Per-specialist external | Rejected | Eight extra training/deploy lanes; weekly drift |
| Per-specialist self (MTP) | Shipped | Reuses existing weights, one wrapper, native distribution alignment |
| Cross-specialist | Rejected | Distributions mismatched by design |

## 2. The MTP self-drafter in practice

Every specialist is trained with MTP. `MTPModule` is a single transformer block plus a fusion projection that runs K autoregressive steps per position (K=3 is the default `mtp_depth`) with a cross-entropy loss against teacher-forced ground truth. At inference it is normally ignored.

To turn it into a drafter we wrap it in a small `MTPDraftModel` that runs the same K-step loop but samples instead of computing a loss. The decode loop becomes draft-then-verify: run the MTP block K times against the last accepted hidden state, produce K candidates, feed them as a K-token block into the main model's next forward, and apply typical acceptance sampling — accept a prefix, resample one bonus token at the rejection boundary, discard the rest.

The decode loop, in code:

```python
# Simplified serving-side draft-then-verify loop with the MTP self-drafter
def speculative_step(state, K=3, tau=0.1):
    drafts = mtp_draft.sample_k(state.last_hidden, K=K)        # K candidate tokens
    logits = main_model.forward_block(state, drafts)           # one fwd, K positions
    accepted = typical_accept(logits, drafts, tau=tau)         # prefix + 1 bonus
    state.commit(accepted)
    return accepted
```

Attractive properties, all structural:

- No new model to train. Repurposing MTP changes only what is called.
- Shares token embedding and LM head weights with the main model; token distributions are natively aligned.
- Drafter parameters are roughly one percent of the main specialist.
- Fits inside the contiguous-KV contract without needing paged-KV tree-verification plumbing.

The unattractive property: a single-block draft on a multi-billion-parameter specialist is weak. Paper-class numbers do not transport. On C++ workloads we see acceptance in the 0.4-0.7 band depending on specialist and task shape, and end-to-end speedup in the 1.3-1.6x band. Worth shipping; not a headline.

## 3. Acceptance rates we actually see

Measured on our evaluation harness at 2K prompt / 512 decode, typical acceptance at tau=0.1 unless noted, on current NVIDIA serving hardware. Numbers are rounded to the nearest five hundredths to discourage over-reading the second digit.

| Specialist | Task shape | Acceptance | End-to-end speedup |
|---|---|---|---|
| Algo-SLM | algorithm-heavy `codegen` | ~0.60 | ~1.5x |
| Template-SLM | deep template `codegen` | ~0.50 | ~1.3x |
| Memory-SLM | RAII / allocator `codegen` | ~0.55 | ~1.4x |
| Concurrency-SLM | lock-free / coroutine `codegen` | ~0.50 | ~1.3x |
| Systems-SLM | POSIX / syscall `codegen` | ~0.55 | ~1.4x |
| Build-SLM | CMake / Bazel `codegen` | ~0.65 | ~1.5x |
| Debug-SLM | live GDB transcripts | ~0.40 | ~1.25x |
| STL-SLM | `<ranges>` / `<algorithm>` rewrites | ~0.65 | ~1.5x |

The greedy-vs-typical gap is smaller than expected. Typical acceptance at tau=0.1 is already tight on C++ tokens — the next-token distribution is narrower than on natural language, so the gap between "most probable" and "typical set" closes. Moving tau changes outputs before it changes acceptance interestingly. Debug-SLM has the lowest acceptance because its inputs are unstructured trace text where the drafter cannot exploit the syntactic regularities that help the codegen specialists.

## 4. The failure modes we hit

### Tool-call triggers

Our decode loop reacts to tokens like `<QUERY_TOOL>` and `<SCRIPT_START>` to pause and dispatch external calls. The first draft-then-verify loop happily drafted past a trigger, accepted the trigger plus a few speculative tokens after it, and the main model then replayed a tool call that had never been issued. Fix: check each accepted token against the trigger set; on a hit, truncate the prefix at the trigger, fall back to sequential decode for the tool interaction, then resume speculation.

### Mamba3 hybrid layers

Several specialists interleave attention with Mamba3 SSM blocks. Mamba has per-step state in the conv and the SSM recurrence that cannot be trivially rolled back on rejection. The first rejection would leave SSM state advanced past the accepted position, and subsequent decode generated from a future that had not happened. We now snapshot conv and SSM state before verify and restore on rejection — a few kilobytes per layer, wired through every SSM variant we ship.

### Paged KV rollback

Contiguous KV rollback is trivial — overwrite the slot. Paged KV under continuous batching has to walk the `block_table`, reset the write pointer in the last partial block, and free blocks allocated only for the rejected tail. Getting this wrong is silent: decode keeps working, but a later prefix-cache hit from an unrelated request can find stale KV. We found this under load when two callers started getting each other's draft tokens roughly once in a few hundred. Fix: zero the partial-block tail on rollback and assert the write pointer matches accepted length every step.

### Tree verification

We implemented EAGLE-2-style tree drafting once and backed it out. Kernels are fine — `GPT.forward()` takes a custom attention mask and position offsets — but tree-shaped `block_table` plumbing under paged KV is its own project, and in our numbers it moved acceptance from the mid-0.5s into the low-0.6s for a roughly 1.2x end-to-end win. Not enough to justify a second cache-rollback implementation. Code stays in `experiments/`; it comes back if we ever sit at the memory-bound end of longer decodes.

### Warmup

Draft and verify have different shapes (K-long candidate block versus 1-long step), which triggers a new compile the first time spec decode runs against a given batch size. Under `torch.compile` with FA-Blackwell and MoBA paths, warmup is several seconds per shape, and the first request on a newly scaled replica ate the whole SLO budget. Fix: pre-warm both shapes during health check; reject traffic until warmup completes. Less glamorous than the algorithmic work and mattered more in practice.

### XLA / TPU

TPU does not support speculative decoding today. `xla_generate_greedy()` has no KV cache, and PJRT fixed-shape graphs cannot accept variable accepted-token counts per step. A padded version (pad to `max_draft_tokens` with masking) would work; we have not prioritised it because TPU inference is not in our product path today.

## 5. What is gated and why

`ServingConfig` carries a `SpeculativeDecodeConfig` slot per specialist. On by default only where we have validated acceptance and validated interaction with the hybrid-layer pattern and adapter stack. Currently on for Algo-, Memory-, Systems-, STL-, and Build-SLM; off for Template-SLM (acceptance too low to justify the verify-pass cost), Concurrency-SLM (SSM rollback path still flakes one test in ten on the heavier hybrid ratio), and Debug-SLM (tool-segment acceptance floor still being measured). Gating surfaces in a `speculation-status` response header (`on`/`off`/`fallback`) so callers can correlate latency telemetry with policy.

## 6. The numbers that matter

End-to-end, across production traffic weighted by request volume, with self-speculative decoding enabled on the five eligible specialists (qualitative bands rather than precise percentages, because the underlying mix shifts week to week):

- Median tokens-per-second on enabled specialists: up by a high-double-digit percent versus the non-speculative baseline.
- p50 inter-token latency on enabled specialists: down by a mid-double-digit percent.
- p95 inter-token latency on enabled specialists: down by a low-double-digit percent.
- p95 time-to-first-token: essentially unchanged, as expected — speculation is a decode-only win.
- Throughput ceiling under sustained load: up by a low-double-digit percent, limited by the increased memory bandwidth pressure of the K-block verify pass.
- Rejection-driven rollback overhead: a low single-digit fraction of decode time, depending on acceptance rate.

These numbers are softer than "2-3x on a single model" from the literature. The softness is honest: eight models with different acceptance curves, tool interactions, hybrid-layer rollback, and NVFP4 quantization tightening the acceptance distribution. The speedup is real and worth the complexity, and it is not the headline number — claiming otherwise would mislead anyone thinking of copying the approach.

## What we kept and what we threw away

Kept: MTP-as-drafter per specialist, sequential (non-tree) draft with K=3, typical acceptance at tau=0.1, explicit per-specialist gating, tool-trigger truncation, Mamba SSM state snapshot/restore, paged-KV tail zeroing on rollback, warmup pre-compile of both shapes in the health check.

Threw away: shared global drafter, per-specialist external drafters, cross-specialist drafting, EAGLE-2 tree verification (left in `experiments/` as a future option), K=5 for non-repetitive workloads, draft-verify across mismatched quantization levels, and online learning of the drafter against production traffic.

The through-line, as elsewhere in the stack, is that ensembles resist shared components. The drafter that works is the one each specialist already has inside itself. Everything we tried to share either regressed acceptance or introduced coordination cost that erased the speedup. Once each specialist drafted itself, the numbers stabilised and the complexity stayed bounded.

## Public references

- [MegaCpp source repository](https://github.com/DatasunriseOU/MegaCpp source repository)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [vLLM documentation](https://docs.vllm.ai/)
