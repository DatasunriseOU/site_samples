---
title: "What changed after 10K steps: the ablations that stayed honest"
description: "A grounded reading of the prototype's post-10K training changes: STP activation, auxiliary-head timing, plasticity scheduling, and why later ablations are more trustworthy than warmup receipts."
date: "2026-04-18"
tags: ["ablation", "training", "stp", "fire", "mtp", "nam52", "nam56r"]
---

**TL;DR:** The most useful ablations in the research repo begin after the first 10K steps because that is where the training stack stops hiding behind warmup noise. a delayed STP activation gate postpones the geodesic regularizer by design, multi-token prediction is explicitly documented as misleading in very short receipts, and the plasticity toolkit is scheduled around real phase changes rather than tiny startup windows. If you compare feature sets only at step 20, step 100, or even step 1000, you mostly learn how initialization behaves. If you compare after 10K, you start learning how the model family actually trains.

The training code already tells you this if you read it literally. The public STP loss sample defines STP as a curvature penalty on hidden-state trajectories; it is not supposed to matter before it is enabled. Public training configuration and runtime notes describe STP as an optional auxiliary term with a delayed start and separate weight. The short ladder receipts around NAM52 also state that MTP hurts in very short runs and that `no_mtp` is the correct baseline for early acceptance. The lesson is not that auxiliary features are bad. The lesson is that honest ablations must be aligned to the activation schedule of the feature being studied.

That becomes even more important in hybrid families such as NAM52 and NAM56R. In the local notation, `A` is an attention block, `M` is a Mamba block, `E` is an expert or MoE block, and `R` is a recurrent block. The same taxonomy also appears in block-level naming such as `ablock`, `mblock`, `eblock`, `rblock`, and `cblock` in design notes and helper code. A feature may interact with one part of that stack much earlier than another. So the headline question is not "does the feature help?" but "when is the feature actually live enough to measure?"

## Warmup receipts mostly measure the wrong thing

Very short receipts look rigorous because they are easy to compare, but they often collapse three different effects into one number: startup transients, immature optimizer state, and the actual feature under test. The prototype has multiple examples of this failure mode.

The cleanest one is MTP. In a NAM52 training ladder receipt, the 20-step ladder reports `mtp1` as +2.4% worse than the dense baseline and `mtp3` as +15.3% worse, with an explicit note that MTP hurts at fewer than 20 steps and that `no_mtp` is the correct starting point for acceptance. That is a strong statement because it is not a generic paper claim. It is a local receipt tied to a concrete NAM52 preset. The report is effectively saying that short-window comparisons overstate the early cost of the auxiliary objective.

The same logic applies to STP. The public STP loss sample makes the feature conceptually cheap, but the training stack still delays it so that the base model can establish representations first. Once you know that a delayed STP start exists, a pre-start ablation becomes almost meaningless. You are comparing a dormant option against another dormant option and attributing the result to a feature that is not yet contributing gradient signal.

Plasticity tools follow the same pattern from another angle. sanitized public integration notes and sanitized public review notes are explicit that FIRE is a phase-boundary tool, DASH is periodic, and ReDo is useful only when the activation family can produce dormant neurons in the first place. Those mechanisms are not intended to show their value in the first few hundred steps of a fresh run. A short receipt can capture overhead, but it cannot tell you whether the intervention improves long-run plasticity.

| Surface | What a short receipt sees | What a post-10K receipt sees |
| --- | --- | --- |
| STP | Mostly disabled or weakly coupled | Real geodesic regularization on live hidden trajectories |
| MTP | Extra loss competing with unstable warmup | Auxiliary head after the base loss has settled |
| FIRE | Usually irrelevant unless a curriculum shift happens | True phase-transition reset behavior |
| DASH / ReDo | Local perturbation without stable baseline | Whether plasticity maintenance helps late training |

The practical rule is simple: if the code delays a feature, the ablation must delay its conclusion.

## What STP actually changes after the cutoff

The STP implementation is unusually transparent. The loss samples ordered triples `(s, r, t)` from a hidden-state trajectory and penalizes curvature with `1 - cos(h[t] - h[r], h[r] - h[s])`. That matters because it tells you what STP is and what it is not. It is not a second language-model head. It is not another token-level classification target. It is a geometric prior over hidden-state evolution.

That geometry-based design is exactly why early measurements are easy to misread. At the start of training, hidden-state trajectories are still being organized by the main objective. A curvature regularizer can either appear inert or look deceptively expensive, depending on how noisy those first trajectories are. After 10K, the same regularizer is applied to a representation space that has actual shape. That is the first point where an STP-on versus STP-off comparison starts to answer a real question.

The base training code also preserves STP as a separable knob. The argument surface keeps the STP weight distinct from the primary loss and logs it as its own auxiliary component. That separation is important for receipts. If an ablation changes the total loss after 10K, you want to know whether the difference came from the base objective, the auxiliary term itself, or a throughput tradeoff that changed effective optimization rate.

A minimal honest receipt therefore needs at least three channels: base loss, STP loss, and throughput. The repo does not require a giant dashboard to make this point; the contract can stay small.

```yaml
ablation_window:
  compare_from_step: 10000
  report:
    - train/loss
    - train/stp_loss
    - tok_per_sec
    - active_pattern
```

The `active_pattern` field matters in hybrids. A run with a mostly `A`-heavy schedule can expose STP differently from a schedule with more `M`, `E`, or `R` pressure, even if the top-line model name is still NAM52 or NAM56R.

## Hybrid patterns make timing more important, not less

One reason the local notation is useful is that it forces you to think in blocks instead of slogans. NAM52 and NAM56R are not generic dense transformers. They are patterned hybrids, and the pattern notation explains why two runs with the same parameter count can react differently to the same ablation.

In the research repo's design notes, `A`, `M`, `E`, and `R` are not decorative. They are the training topology. A hybrid pattern string encodes where attention, state-space, MoE, and recurrent pressure are actually placed. The cost of an auxiliary head or a plasticity intervention may concentrate in only one of those categories. That means an honest post-10K ablation should preserve the pattern string, not reduce everything to "feature on" and "feature off."

This is also where MegaCpp becomes relevant. The ported Megatron-side code in the public hybrid block spec sample and the public MTP loss integration sample keeps the same idea alive: hybrid structure and MTP configuration are first-class runtime contracts. The port is not merely copying names. It is preserving the fact that a block mix and an auxiliary path interact structurally.

For post-10K analysis, that yields a better comparison matrix.

| Family | Pattern lens | Ablation question that survives warmup |
| --- | --- | --- |
| NAM52 | Mostly hybrid `A`/`M` with targeted extras | Does STP or MTP improve the settled optimization path? |
| NAM56R | Larger mixed `A`/`M`/`E`/`R` family | Which auxiliary terms are still worth paying for after the stack is fully live? |
| MegaCpp port lanes | Megatron-native hybrid specs | Which prototype ablations transfer as real runtime knobs rather than research-only hacks? |

Without the pattern lens, a short receipt invites the wrong conclusion: "feature X is slower." With the pattern lens, the better conclusion is: "feature X is slow or useful under this block topology and after this activation schedule."

## Why post-10K ablations are the first ones worth operationalizing

The repo contains several examples where a feature's apparent cost changes once neighboring issues are fixed. That is why the 10K cutoff is methodological, not mystical. It gives the training system enough time to move from setup behavior to operating behavior.

The MTP evidence is the clearest operational example. The ladder receipt explicitly distinguishes between single-GPU short receipts and more stable comparisons, and it warns that early MTP behavior is not the right acceptance criterion. That means the operational baseline for a new backend, new launcher, or new substrate should usually be `no_mtp` first, then a later MTP ablation once the lane is known-good.

STP follows the same principle via delayed activation. Plasticity tools follow it via event-based scheduling. Once you line those up, the correct ablation order becomes obvious:

1. Establish a stable base lane.
2. Let delayed auxiliaries actually turn on.
3. Compare only in the interval where the feature is live.
4. Keep the hybrid pattern fixed while comparing.

That sequence also matches what the Megatron-side port is trying to preserve. The port docs around NAM56R repeatedly treat selective recompute, MTP, and memory-saving knobs as runtime contracts that must be evaluated in a realistic active lane, not in a toy receipt. This is the same intellectual move: do not judge a feature before the surrounding system has entered the state where that feature is supposed to matter.

## What to carry forward into future receipts

The best part of the prototype is not any single feature. It is the discipline of separating dormant, warming, and active regimes. That discipline should survive the move into MegaCpp and any future training reports.

For practical reporting, the carry-forward checklist is short:

| Requirement | Why it matters |
| --- | --- |
| Preserve model family and pattern string | Avoid mixing topology changes with feature changes |
| State the feature's activation schedule | Prevent dormant-window receipts from being over-interpreted |
| Report throughput with the loss | Distinguish algorithmic gain from rate distortion |
| Use a post-10K comparison window when possible | Measure active behavior rather than warmup quirks |
| Keep references file-level and code-grounded | Make the claim reproducible by rereading the repo |

The core lesson is narrow but durable. A good ablation is not just a pair of numbers. It is a timing claim. The repo already encodes the timing: STP starts late, MTP is misleading early, FIRE is for boundaries, and hybrid patterns shape every comparison. Once you accept that, the post-10K window stops looking arbitrary. It becomes the first interval where the experiment is telling the truth.

## References

- public STP loss sample
- public training runtime notes
- public training configuration notes
- public NAM52 ladder receipt excerpt
- sanitized public integration notes
- sanitized public review notes
- public hybrid block spec sample
- public MTP integration sample
