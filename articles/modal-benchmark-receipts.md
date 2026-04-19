---
title: "Modal Benchmark Receipts: What Counted as Evidence and What Did Not"
date: 2026-04-18
author: MegaCpp Engineering
tags: [modal, benchmarks, receipts, throughput, evidence]
summary: >
  The useful benchmark story was not the headline number. It was the receipt:
  exact config, exact backend, exact compile posture, and a clear separation of stable lanes from unstable ones.
description: >
  A grounded guide to benchmark receipts using compile posture, backend
  identity, and narrow evidence records rather than headline throughput
  claims.
---

# Modal Benchmark Receipts: What Counted as Evidence and What Did Not

A benchmark number is only trustworthy when it comes with a receipt: the exact model lane, exact operator family, compile posture, and known exclusions. The same nominal model can produce very different throughput depending on whether it used padded MoE or jagged MoE, stable dense compile or graph breaks, sparse recovery or a still-regressing backend.

Benchmark arguments get sloppy fast when infrastructure is moving. One run is quoted after a backend fix, another after a config change, another after a compile improvement, and suddenly the comparison sounds cleaner than it is. The project avoided that trap in its better artifacts by treating benchmark claims as evidence records rather than wins to advertise.

That is what a receipt is here. A receipt is not just a log file. It is a bounded claim about one lane: what model mix it used, what backend family was active, what compile status was in play, and what nearby caveats still existed.

## Why Headline Numbers Are Dangerous

The most direct evidence is in the run history itself. It records several moments where the apparent faster path was not actually the best benchmark lane once compile behavior was included. The MoE examples are especially important. A jagged grouped path could look better in local operator terms, but if it forced graph breaks or required `@torch.compiler.disable`, then the end-to-end training lane could lose to a padded alternative that preserved a more coherent compiled graph.

That is not a small footnote. It changes what a throughput number means. If one lane is measuring arithmetic efficiency and another is measuring a compiled end-to-end system with fewer breaks, they are not substitutes. The receipt has to say which one it is.

| Receipt ingredient | Why it matters |
| --- | --- |
| Exact lane or recipe | Throughput only makes sense for one concrete topology and feature mix |
| Compile posture | Eager, partial compile, and whole-model compile are different systems |
| Operator family | Dense, sparse, padded MoE, and jagged MoE can behave very differently |
| Known caveats | Graph breaks, fallbacks, and disabled paths explain why one number moved |

The point is not to make reporting harder for its own sake. The point is to stop lying accidentally.

That may sound severe, but the repo history earns the severity. Performance changed for multiple legitimate reasons at once: compile behavior, sparse backend recovery, MoE path selection, and validation narrowing. A benchmark note that hides those causes may still be numerically accurate for one run and yet be misleading for every comparison that follows it.

## A Good Receipt Names the Structural Tradeoff

The better benchmark notes do this explicitly. For example, the MoE compile writeups explain that the padded path could be faster overall because it remained compilable while the jagged fused path introduced graph breaks. That is not merely a "backend detail." It is the central interpretation key for the benchmark.

A good benchmark receipt therefore has to identify the structural tradeoff in plain language. If a run used padded MoE, say it. If the fused path was disabled because compile could not tolerate it, say that too. If a sparse attention recovery restored correctness but did not fully recover the old throughput envelope, the receipt should separate those statements instead of compressing them into a single success narrative.

```text
Receipt pattern:
model family + feature mix + compile posture + backend family + known caveats
```

That structure may look verbose, but it is still cheaper than debugging a misleading benchmark comparison later.

## The Best Receipts in the Repo Narrow the Frontier First

an H200 bring-up receipt is not a Modal-only document, but it demonstrates the right habit. It distinguishes passing dense compile lanes from failing real MoE lanes. It records when `PP + TP + SP + compile` is alive at one depth and when deeper or different operator seams remain the next frontier. It does not report one blended number as if all nearby variants were equally solved.

That style is exactly what benchmark reporting needs on hosted accelerators as well. The benchmark should inherit the same frontier language:

1. this exact lane is alive,
2. this adjacent lane is still unstable,
3. this number belongs only to the first lane.

Without that separation, teams start quoting the best number from the easiest slice and mentally attaching it to the hardest configuration. The repo's better reports resist that temptation.

That resistance is what makes them reusable. A later engineer can come back, see which lane the number belonged to, and decide whether a new run is actually comparable or only superficially adjacent.

## Modal Receipts Needed More Than Throughput

Hosted benchmark environments add their own source of confusion. Startup overhead, artifact staging, compile cache state, and exact environment drift all influence observed speed. That means a hosted benchmark receipt needs more than model arguments.

Even when the repo does not encode every hosted-environment detail in one file, it repeatedly encodes the right philosophy: preserve runtime notes, preserve validation context, and do not blur time-sensitive environment details with stable code-backed facts. The TPU runtime bundles make this distinction explicit by separating repo-local runtime evidence from cloud-product claims that may drift. The same discipline applies to hosted GPU benchmarking. Environment facts that drift should be treated as unstable metadata, not folded into the model-performance claim itself.

That is also why receipts are better than screenshots or one-line summaries. A screenshot can show a number. A receipt explains whether the number came from a stable compiled lane, a fallback backend, or a warm-cache run that no adjacent lane can reproduce.

This is where hosted benchmarking usually becomes misleading. The same nominal recipe can move because of cache warmth, compile posture, fallback behavior, or environment drift. If those are not logged alongside the number, later comparisons start looking more stable than the code actually was.

## Hybrid Patterns Make Benchmark Labels More Important

In a pure dense stack, benchmark labeling is still easy to get wrong, but at least the model family is simple. In a hybrid stack with `A`, `M`, `E`, and `R` blocks, labels like NAM52 and NAM56R carry real architectural meaning. They imply different operator mixes and therefore different benchmark expectations.

An `AEMEAEMEAEMR`-style pattern should never be benchmarked or discussed as if it were just "another transformer run." The `E` blocks introduce expert routing and dispatch behavior. The `M` blocks change the runtime profile. The `R` suffix or recurrent family changes state handling again. If a receipt does not mention that structure, it is too weak to compare against anything serious.

| Family cue | Benchmark implication |
| --- | --- |
| `A` heavy dense lane | Throughput mostly reflects dense attention and projection economics |
| `E` heavy lane | Routing, capacity, dispatch, and compile behavior become central |
| Mixed `A/M/E/R` pattern | End-to-end number reflects family interactions, not one kernel story |
| NAM52 vs NAM56R | Different receipts, different claims, different performance envelopes |

This is one reason the repo's use of notation is valuable. It forces performance claims back onto real structure.

It also improves review quality. Once a benchmark is tied to a named family mix rather than a generic model label, reviewers can challenge the comparison on architecture grounds instead of only arguing about logging hygiene.

It also prevents benchmark laundering across adjacent lanes. Once a pattern name carries real architectural meaning, a result from the easiest runnable subset cannot honestly be promoted into a claim about the hardest intended target.

## Sparse and Recovery Work Changed What Benchmarks Meant

The DSA SDPA recovery bundle is another reminder that correctness, backend choice, and throughput should be logged separately. One March 8 fix materially recovered throughput without pretending to erase all later regression analysis. The accompanying validation note also states the remaining gap plainly instead of overfitting the story to one recovered run.

That is the kind of evidence discipline benchmark reporting needs. A benchmark after a recovery should say whether it reflects restored correctness, restored backend selection, partial throughput recovery, or all three. Otherwise every later comparison inherits ambiguity from the earlier one.

The validation note in that recovery bundle is a good example of restraint. It leaves the remaining gap open instead of pretending one recovered run restored the entire older envelope. That makes the benchmark more useful, not less useful, because later readers know exactly what still needs explanation.

## What a Strong Modal Benchmark Receipt Looks Like

A strong receipt in this stack should include:

| Field | Example meaning |
| --- | --- |
| Model lane | NAM52 dense compile lane, or NAM56R hybrid lane with MoE enabled |
| Pattern or family | Dense `A`-heavy, or hybrid `AEMEAEMEAEMR` |
| Compile status | eager, partially compiled, or whole-model compiled |
| Backend detail | padded MoE, jagged grouped MoE, sparse SDPA, dense attention |
| Stability note | stable frontier or adjacent known blocker |
| Artifact pointer | report, receipt, or preserved validation note |

That may feel like more ceremony than most benchmark dashboards want. But the alternative is worse: optimistic numbers nobody can align to code.

A receipt-heavy culture may feel slower in the short term, but it saves time later because fewer benchmark disputes have to be re-opened from scratch.

And code alignment is the only standard that survives fast-moving infrastructure. A benchmark with exact report anchors and caveats can still be interpreted after several later patches. A benchmark without that context ages into trivia almost immediately.

## The Main Rule: Report the Lane You Actually Measured

The strongest through-line across the codebase is this: report the exact lane you measured, not the family you wish you had measured. If compile was only stable on the padded MoE path, the receipt belongs to the padded MoE path. If the dense TP+SP+FSDP compile lane passed but the real MoE layer remained the next blocker, the benchmark belongs to the dense lane. If a sparse recovery restored one backend path but left a broader performance gap unexplained, say that too.

Hosted benchmark arguments usually go wrong when people merge nearby truths into one sentence. The receipts in this project are useful because they refuse to do that. They tell you what worked, what did not, and why one number should not be generalized across the whole stack.

That is what made a benchmark believable here. Not the highest tokens-per-second line, but the narrowest honest claim around it.

That standard is stricter than typical benchmark culture, but it is the right one for a hybrid system where compile posture, sparse routing, and hosted-runtime state can all change independently. The narrower the receipt, the longer it remains useful.

## References

- MegaCpp benchmark notes
- an H200 bring-up receipt
- a DSA/SDPA recovery note
- a DSA/SDPA validation note
- the main training entrypoint
