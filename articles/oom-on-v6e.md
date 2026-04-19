---
title: "OOM on v6e: Why Memory Pressure Looked Different on TPU"
description: "What TPU v6e out-of-memory failures taught us, why the obvious fixes were often wrong, and how the lane eventually measured memory honestly."
date: "2026-04-18"
tags: ["tpu", "v6e", "oom", "memory", "xla"]
---

# OOM on v6e: Why Memory Pressure Looked Different on TPU

OOM on TPU v6e was not a slightly different version of GPU OOM. The lane needed chip-level visibility, a deterministic retry ladder, and much less trust in GPU-origin intuitions. The winning move was not “reduce batch until it fits.” It was “measure memory on the right unit, preserve the exact model shape in the report, and shrink the right dimension in the right order.”

TPU memory failures are frustrating because they often look generic at first. The process dies, the trace is noisy, and the first instinct is to do what worked on GPU: shrink batch size, toggle checkpointing, and hope the next run lands. The local TPU docs and runtime notes argue for a more disciplined view. On the TPU lane, compiler decisions, SPMD layout, and per-chip pressure can dominate the story. If you only watch aggregate memory, you can easily debug the wrong problem.

## Why TPU OOM felt different from GPU OOM

The first conceptual shift was to stop asking “did the TPU device run out of memory?” and start asking “which physical chip hit the limit, under what shape, and after which compile/layout decision?” The TPU setup notes emphasize exactly that split between high-level runtime state and the lower-level reality of per-chip pressure.

That distinction also helps keep the TPU story separate from the CUDA story. The validated H200 notes focus on filesystem placement, fused-package discipline, and multi-GPU runtime invariants. The TPU lane has a different first layer of truth: PJRT, SPMD lowering, and chip-local pressure. Treating them as the same class of memory problem is one of the easiest ways to waste time.

| Question | Wrong default answer | Better answer |
| --- | --- | --- |
| How much memory is in use? | One aggregate SPMD number | Per-physical-chip `bytes_used` versus `bytes_limit` |
| Why did step zero fail? | “The model is too big” | One chip or one layout crossed the limit under this exact shape |
| What should shrink next? | Whatever dimension is easiest to type | The first dimension in the retry ladder that reduces the active pressure surface |

That difference sounds administrative until you actually hit repeated OOMs. Once the team started treating the physical chip as the right unit of observation, two things changed immediately. First, the auto-fit logic became more trustworthy because it was no longer guessing from a blended number. Second, humans stopped overreacting to failures that looked global but were actually localized.

The TPU documents are useful here because they separate TPU truths clearly. There is a documented preferred wheel lineage and a live validated TPU lane. That separation matters for memory work too. If the runtime stack differs, the memory profile can differ. Treating all v6e failures as one bucket is a good way to lose days.

## The retry ladder mattered more than the first fix

Once compile windows are expensive, random retries become a tax. That is why the TPU lane needed an ordered response rather than ad hoc operator guesses. The purpose of a retry ladder is not elegance. It is to avoid paying the compile cost for a sequence of bad instincts.

The local docs and scripts imply a practical hierarchy: preserve the core experimental question if possible, but shrink the dimension that most directly reduces the pressure you are seeing. Sometimes that means device batch size. Sometimes it means sequence length. Sometimes it means abandoning a topology that is not honest for the hardware budget.

That choice is easier when the report preserves exact model shape. A dense NAM52 lane, a hybrid NAM56R lane, and a pattern such as `AEMEAEMEAEMR` do not put pressure on memory in the same way. If the report only says “large run on v6e,” the retry ladder becomes generic faster than it should.

```yaml
oom_retry_ladder:
  1: reduce_device_batch_size
  2: reduce_sequence_length
  3: disable_nonessential_feature_tax
  4: lower_parallelism_or_change_topology
  5: stop_and_record_report
```

That block is inferred policy, not a literal checked-in config. It captures the behavior the TPU notes are arguing for: OOM handling should be ordered, scripted, and explainable.

The most important line in that ladder is the last one. There is a point where repeated shrinking stops being recovery and starts being denial about the lane’s viability. A good OOM workflow records the frontier instead of pretending there is always one more harmless knob turn.

## Why GPU heuristics were often wrong on TPU

GPU OOM diagnosis is shaped by a different set of instincts: allocator fragmentation, extension workspace spikes, flash-attention workspace, compiler cache behavior, or delayed activation rematerialization changes. Some of those intuitions still matter on TPU, but they are not the safest first guess.

On TPU, graph partitioning and SPMD layout decisions can change where memory pressure lives. That means the same model shape can behave differently depending on how the runtime lowered and partitioned it. This is why the TPU lane needed its own measurement logic rather than a copy of GPU dashboards.

It also explains why exact model notation must survive in OOM reports. If the model is `NAM56R` or uses a pattern like `AEMEAEMEAEMR`, that is not decorative context. It tells the reader whether the lane includes attention-heavy regions, Mamba regions, MoE regions, and recurrent tail behavior. Those block families do not stress memory in the same way. Saying “the big hybrid model OOMed” throws away the shape information that could help explain the pressure.

There is a second benefit to that naming discipline: it protects cross-run comparison. TPU OOM investigations are often spread over days because compile-heavy retries are expensive. If one report uses the exact lane label and the next report uses softened prose, engineers can end up comparing non-equivalent runs without noticing.

The block glossary remains useful on TPU for exactly this reason:

| Token | Meaning | Typical memory concern |
| --- | --- | --- |
| `A` | attention block | attention activations and long-context surfaces |
| `M` | Mamba block | state tensors and scan-related residency |
| `E` | expert block | routed-token hot spots and expert-side intermediates |
| `R` | recurrent block | recurrent state and tail-structure changes |

A precise OOM report should preserve both the runtime shape and the architectural shape. Without both, remediation becomes guesswork.

## The right measurement unit was the chip

The decisive change in the TPU memory story was moving from a vague “device memory” concept to per-chip accounting. The local TPU notes describe a need for chip-level `bytes_used` and `bytes_limit`, and that is the right abstraction. Once the team could see which chip was actually near the edge, broad fixes gave way to narrower reasoning.

That kind of observability changes behavior more than people expect. Without it, operators reach for the same levers in the same order every time. With it, they can ask better questions:

- Is one chip peaking far earlier than its peers?
- Did a topology change move the pressure instead of reducing it?
- Is the problem tied to long sequence length or to a particular block family?
- Did a compile or layout change alter residency without any visible model-code change?

That is a far healthier loop than “reduce something and rerun.” It turns OOM work into diagnosis instead of ritual.

It also makes escalation more honest. Once pressure is localized to one chip and one shape, the team can decide whether the next move belongs in launcher geometry, model shape, or runtime tooling. Before that, every discussion stays broad and unproductive.

## Why exact receipts changed operator behavior

The measurement culture around throughput and run records carries over directly to memory work. If the team is already disciplined about preserving model names, feature sets, and lane topology in a report, then OOM reports become much more actionable. They stop being stories and start being frontiers.

This is especially important because a step-zero OOM is easy to summarize lazily. “Did not fit” is not enough. What did not fit, under which wheel lineage, on which validated lane, with which exact model shape, after which reduction attempts? The local TPU docs push toward keeping those details together, and that is the right habit.

That matters because TPU OOM is often expensive to reproduce. A long compile window followed by a step-zero failure is not something you want to rediscover just because the earlier report forgot whether the lane used the NAM52 dense shape or the NAM56R hybrid shape.

The same logic also keeps TPU and GPU lanes from bleeding into each other conceptually. The H200 notes are very explicit about CUDA runtime invariants, root-volume hygiene, and validated launcher environments. The TPU notes are explicit about PJRT, wheel lineage, and chip-level truth. Keeping those receipts separate is how the team avoids applying the wrong fix to the wrong platform.

## What survived from the v6e OOM work

Several lessons survived and seem durable.

First, chip-level memory reporting is mandatory. Anything less is too lossy for serious OOM diagnosis on TPU.

Second, OOM recovery needs an ordered ladder. Compile-heavy experimentation punishes random retries.

Third, GPU heuristics should be treated as hypotheses on TPU, not as defaults. They may still help, but they do not get priority automatically.

Fourth, keep the distinction between the documented preferred lineage and the currently validated host state in the memory report. A drifted environment can produce an OOM story that looks fundamental when it is really environmental.

Fourth, exact naming should stay inside the report. `NAM52`, `NAM56R`, and pattern strings are operational details because they preserve the architectural shape behind the failure.

Finally, an honest frontier is better than a heroic myth. If a given topology does not fit on v6e under the real runtime and real stack, the useful output is a recorded limit, not a vague promise that one more small tweak will surely fix it.

## What A Cross-Platform Training Stack Should Inherit

A cross-platform training stack should inherit the TPU lesson even when running on different hardware: measure the bottleneck on the unit that actually fails, keep the lane description exact, and script the retry behavior instead of relying on operator memory. Those are not TPU-only ideas. TPU just made the cost of ignoring them impossible to hide.

There is a deeper mixed-platform lesson here too. When one project spans TPU research and GPU production lanes, the louder platform tends to dictate debugging style. The v6e memory work is a reminder that shared discipline matters more than shared folklore: exact measurement, exact receipts, and platform-specific units of truth.

## References

- `TPU_SETUP.md`
- `00-vision.md`
- the TPU training launcher
- the public training-args sample
- the distributed parallelism module
- the public project README
- `production_status.md`
