---
title: "SOTA Ablation and Comparison: How MegaCpp Decides What to Keep"
description: "The ablation plan, the comparison methodology, and the honest numbers behind the MegaCpp SLM stack — what stacked, what didn't, and what we threw out even though the paper said it would help."
date: "2026-04-18"
tags: ["ablation", "sota", "moe", "dsa", "ifim", "evaluation", "c++"]
---

The default state of any new module — sparse-attention variant, dynamic-depth scheme, training-objective auxiliary, fancy positional encoding — is off. It earns its way into the production NAM-class run by clearing a bounded ablation, on the same data, at the same depth, against a baseline we trust. If it does not, we drop it, even when the paper looks great. This post is the methodology behind that decision: how we structure the plan, how we keep comparisons honest across phases, the bugs we caught in our own runs that invalidated entire experiment groups, and the scorecard of what actually stacks for C++.

## Why this matters

SOTA-shopping is the failure mode of every codebase that has too much GPU. A team adds Mamba because the curve looked good in a paper, then DSA because a different curve looked good in a different paper, then a Phase-4 training objective on top, and suddenly nothing reproduces because the substrate changed three times along the way. We have done that and paid for it. The substrate-pinning rule, the invalidation-is-loud rule, and the best-checkpoint-with-step rule all exist to keep the comparison meaningful when twenty things are moving.

The other reason this matters is honesty about our own bugs. Phase 1 had a real one — applying an attention-side trick to Mamba layers — and the right response was to mark the affected runs INVALID and re-run them in Phase 2 on a corrected configuration. Hiding it would have made an entire family of objectives look better than it was.

## 1. The fixed substrate

Every comparison number in this post is on the same substrate. That part is not negotiable.

```yaml
# Fixed substrate for every Phase 1-5 cell in this post
hardware:    spot v6e-4 TPU slice (one regional pool, six in parallel per wave)
data:        cpp_enriched_16k (compiler-pretokenized C++ parquet shards)
geometry:
  depth:           16
  head_dim:        64       # model dim 1024, 16 heads
  context:         4096
  total_batch:     131072   # tokens
  steps:           10000
topology:    TP=2, dp=2 over the 4 chips of a v6e-4
metric:      val_bpb on held-out C++ split @ step 10K
```

If a number in this post is on a different substrate, it is labeled. If it is not labeled, it is on this one. We refuse to print a number without naming the substrate it came from after spending too much time chasing apparent regressions that turned out to be a head-dim or context-length change.

## 2. The catalog

Modules we are willing to consider live in three tiers because the cost of evaluating them is not uniform.

| Tier | Examples | Cost class |
| --- | --- | --- |
| 1 — wired into the core training entrypoint | Mamba-3 (AAM hybrid), DSA, Engram, mHC, MTP, NCP, MoD | flag-flip + run |
| 2 — believed promising, integrated | TOP, SRI, IFIM, GateSkip, MoD variants, FlexiDepth, continual backprop, Jacobi forcing, YaRN RoPE | integration plus run |
| 3 — inference-only, separate lane | ADEPT early-exit, EAGLE-2 speculative decoding, ring attention | excluded from training ablation |

## 3. Phase 1: structure first

Phase 1 was the architectural foundation question: holding everything else fixed, which structural change moves `val_bpb` the most? Six experiments, all at `depth=16`, all on `cpp_enriched_16k`, all 10K steps.

| Exp  | Config                          | Val BPB @ 10K | Delta vs baseline |
| ---- | ------------------------------- | ------------- | ----------------- |
| EXP1 | Baseline (attention only)       | ~1.866        | —                 |
| EXP2 | + Mamba-3 AAM                   | ~1.80         | low single-digit % |
| EXP3 | + DSA (sparse attention)        | **1.562**     | mid-double-digit % (winner) |
| EXP4 | Engram + mHC + MTP              | INVALID       | Engram-on-Mamba-layer bug |
| EXP5 | Full stack                      | INVALID       | same bug + NaN loop |
| EXP6 | Full stack + NCP                | ~1.7          | NCP marginal      |

The headline is that DSA was the largest single-feature improvement we have seen in any phase. From Phase 2 onward, DSA is the baseline.

### The asterisk that matters

EXP4 and EXP5 used `--engram_layers=0,5,10`. With our AAM Mamba pattern at `depth=16`, layers 2, 5, 8, 11, 14 are Mamba layers. Engram is an embedding-side trick designed for attention layers; applying it on top of a Mamba layer is not a supported configuration and produces `val_bpb` north of 3.5 at init. Both experiments were invalid for evaluating Engram, mHC, and MTP. The correct layer set on `depth=16` AAM is `0,1,3,4,6,7,9,10,12,13,15` — every attention layer, no Mamba layers — and a model-init guard now refuses launches that violate it.

We publish the bug because pretending it did not happen is how a number like "1.578 with Engram + mHC + MTP" survives into a marketing slide a year later. The honest version: those experiments did not happen on a configuration that means what we thought it meant, and we re-ran the relevant cells in Phase 2.

## 4. Phase 2: stacking with corrected layers

With DSA fixed as the base, Phase 2 added one secondary feature at a time, with the corrected Engram layer set:

| Exp     | Config                | Val BPB              | Status          |
| ------- | --------------------- | -------------------- | --------------- |
| p2_e01  | DSA only              | ~1.68 @ 3.75K        | reference       |
| p2_e03  | DSA + Engram (TP=2)   | **~1.60 @ 1.25K**    | best stable     |
| p2_e04  | DSA + mHC (TP=2)      | ~1.58 @ 10K          | complete        |
| p2_e05  | DSA + MTP             | ~1.93 @ 2.75K        | converged early |

Two takeaways. First, Engram and mHC each cleanly improve on DSA with the corrected layer set. Second, MTP is a regression on this substrate at this scale; we park it rather than promote it. The decision rule lives in the methodology, not the prior.

## 5. Phase 4: secondary objectives on the MoE base

Phase 4 stacked secondary training-time objectives on top of a MoE base (DSA + 2 shared / 16 routed top-2). All 4K context, 10K steps, on the substrate above unless TP is noted.

| Exp      | Config                       | TP | Val BPB           | Note                          |
| -------- | ---------------------------- | -- | ----------------- | ----------------------------- |
| p2_e00   | DSA baseline                 | 4  | ~1.96             | reference                     |
| p4_e06   | DSA + IFIM                   | 4  | **~1.57**         | winner — large gain           |
| p4_e03   | DSA + GateSkip               | 2  | ~1.58             | TP=2 required                 |
| p4_e01   | DSA + TOP                    | 4  | ~1.73             | T_top=2048 aligned            |
| p4_e05   | DSA + SRI                    | 4  | ~1.75             |                               |
| p4_e04b  | DSA + Mamba-3                | 4  | ~2.32             | hurts when stacked with DSA   |

IFIM — instruction-aware fill-in-the-middle — was the clean winner: a training-time data transformation that prepends docstrings and comments as instruction prefixes to FIM examples. It costs nothing at inference time and yielded a ~20% improvement over the dense DSA baseline. GateSkip was second-best, with the TP<=2 caveat. SRI and TOP both helped over baseline but less. DSA + Mamba-3 stacked badly on this geometry — a result that contradicts what the AAM hybrid did under MoE in `moe_e06`, which is exactly why we run things in groups instead of trusting any one result.

The asterisk on Phase 4 is the TOP number. The original TOP run was at ~101 seconds per step (a 12-day ETA) because of a Python loop in the auxiliary loss. The batched matmul fix replaces eight sequential `torch.mm` calls with one `torch.mm(h_2d, w.T)` and brings step time to ~12 seconds. The ~1.73 number is from the slow run; we are re-running TOP under the batched implementation in Phase 5 before letting it into any final comparison.

## 6. Phase 5: do they stack?

The Phase 5 question is the only one that matters for the production candidate: which Phase 4 objectives stack additively on top of the MoE base, and is there a single configuration we can commit to for the production run?

| ID     | Config                            | TP | Expected BPB |
| ------ | --------------------------------- | -- | ------------ |
| p5_e01 | MoE base re-validation (moe_e06)  | 2  | ~1.20        |
| p5_e02 | MoE + IFIM                        | 2  | 1.05-1.20    |
| p5_e03 | MoE + GateSkip + IFIM             | 2  | 1.00-1.15    |
| p5_e04 | MoE + SRI + IFIM                  | 2  | 1.05-1.20    |
| p5_e05 | Dense TOP re-validation (batched) | 4  | ~1.72        |
| p5_e06 | MoE + TOP                         | 2  | 1.05-1.20    |

The hypothesis behind `p5_e02` is that IFIM and MoE operate at orthogonal levels — IFIM rewrites the data, MoE routes the activations — so they should compose without interference. IFIM may even improve MoE routing by giving the router a cleaner semantic signal in the prefix. If that holds, `p5_e02` is the production candidate.

The expected-BPB column is honest: it is a range, not a target. We do not pretend to predict the exact stacking gain ahead of the run. The decision rule is in the comparison methodology, not the prediction.

## 7. Comparison methodology

Three rules, no exceptions.

### Same substrate or labeled differently

Two numbers from different substrates do not appear in the same table. If they have to be discussed together, the substrate is part of the row.

### Best-checkpoint reporting, with the step recorded

We report the best `val_bpb` and the step it was achieved at, never the final-step number alone. A model that hits 1.20 at step 3,750 and drifts to 1.25 by 10K is a different signal than one that hits 1.20 at 10K — the first is a stability problem, the second is a converged result.

### Invalidation is loud

If a configuration was run on a buggy layer set, a corrupted preset, or during a known loss spike, its numbers are removed from the comparison and replaced with the word INVALID and the reason. The d24 hybrid checkpoint at step 25K is the canonical example: a transient loss spike (loss jumped from ~0.8 to ~3.4 around step 24,850, recovered fully by ~25,700) coincided with the periodic save, and the saved weights were degraded. The 25K eval (3.1% compile rate vs 11.0% at step 20K) was not a model regression but a snapshot during recovery. We say so. The next save is the comparable one.

## What we kept and what we threw away

The shape we plan to commit to: DSA (sparse attention, start-layer 8) as the base; Mamba-3 AAM hybrid with `qknorm + bias + trapezoidal` defaults (complex RoPE opt-in) for the SSM portion; MoE with 2 shared + 16 routed experts top-2 and `z_loss_weight=0.01`; Engram on attention layers only with the explicit layer set; mHC enabled; IFIM as the training-time objective.

What we threw out: MTP at 4K context (slight regression on this substrate); dense DSA + Mamba-3 stacking (bad without MoE); NCP (marginal gain at the cost of training complexity, parked); MoD variants beyond the baseline (parked pending head-to-head against GateSkip on the MoE base); and the Phase 1 Engram/mHC/MTP cells (void due to the layer-set bug, replaced by Phase 2 on a corrected setup).

That stack is what survives the methodology, not what survives a single best-of-N run. If Phase 5 contradicts the prediction, we will rebuild the candidate around the actual numbers, not the prior. That is the point of running ablations instead of adopting papers.

## References

- internal ablation plans
- internal comparison notes
- internal phase-5 planning notes
- internal training/eval reports
- internal evaluation methodology notes
- internal checkpoint-evaluation notes
- internal training review notes
