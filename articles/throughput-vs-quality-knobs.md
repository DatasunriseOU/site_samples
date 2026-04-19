---
title: "Throughput vs quality knobs: which trade-offs are real"
description: "A grounded map of the knobs that actually move the throughput-quality frontier in hybrid NAM52 and NAM56R training, based on public MegaCpp code, articles, and upstream references."
date: "2026-04-18"
tags: ["throughput", "quality", "nam52", "nam56r", "training", "moe", "mamba"]
---

The real throughput-versus-quality knobs are not cosmetic flags. The big levers are block pattern, expert routing behavior, auxiliary-head policy, precision scope, and checkpoint or recompute policy. Those knobs matter because they change which work dominates a step. A pattern like `AEMEAEMEAEMR` pays for attention, expert routing, and recurrent-state handling differently than a mostly dense stack, so the same optimization can be a major gain in one family and almost irrelevant in another.

## Public code and notes

- [DSA indexer upstream example set](https://github.com/DatasunriseOU/cppmega/tree/main/upstream_prs/examples/12_dsa_indexer_memory)
- [Mamba linear CE upstream example set](https://github.com/DatasunriseOU/cppmega/tree/main/upstream_prs/examples/11_mamba_linear_ce)
- [MegaCpp DSA patch surface](https://github.com/DatasunriseOU/cppmega/blob/main/cppmega/megatron/dsa_indexer_fused_patch.py)
- [MegaCpp Megatron recipe surface](https://github.com/DatasunriseOU/cppmega/blob/main/cppmega/recipes/nam56r_megatron.py)

The easiest mistake in model tuning is to treat throughput and quality as separate checklists. Public MegaCpp materials point the other way. The published architecture description combines attention, Mamba, expert, and recurrent-style pieces, and the public implementation surfaces make it clear that runtime changes can also affect parity and training behavior. If you want a useful knob map, you have to start from the pattern notation and from the exact block mix that a run uses.

## Start from the block pattern, not from folklore

The hybrid notation used in MegaCpp is the right place to start. `A` means attention, `M` means Mamba, `E` means expert or MoE, and `R` means recurrent-style work. In public code and docs that same split often appears as `ablock`, `mblock`, `eblock`, and `rblock`. The published notes also describe the stack as combining Mamba-3 hybrid layers, sparse attention work, MTP, and other extensions rather than a single uniform transformer shape. That is why pattern strings such as `AEMEAEMEAEMR` are more than labels: they are cost models in shorthand.

Once you read the training stack this way, the throughput-quality frontier becomes much less mysterious. An `A`-heavy model responds strongly to attention-kernel changes, KV handling, normalization, and sequence-length policy. An `E`-heavy model responds much more to router behavior, token distribution, expert overlap, and capacity handling. An `M`-heavy model shifts the story again, because recurrent state and Mamba-specific projections can dominate the non-attention part of the step.

| Pattern element | Main compute pressure | Typical quality lever | Why it changes the frontier |
| --- | --- | --- | --- |
| `A` / `ablock` | attention kernels, QKV projection, sequence length | context handling, mask semantics, RoPE behavior | attention cost scales differently from expert or recurrent work |
| `M` / `mblock` | state update, mixer kernels, specialized projections | long-context recurrence behavior | recurrent compute can replace or complement attention cost |
| `E` / `eblock` | router, expert GEMMs, token shuffles | specialization, capacity, token coverage | routing policy changes both speed and learning behavior |
| `R` / `rblock` | recurrent memory update and scheduling | persistence and temporal bias | different state path than pure attention or MoE |

That table is the reason generic advice fails. "Enable the faster kernel" is not a universal tuning strategy when half the time is not in the kernel you are staring at.

## Architectural knobs are the first-order ones

The biggest real knobs are architectural. In practice that means block pattern, MoE routing policy, shared-versus-routed expert balance, MTP depth, and whether a feature is truly on or merely parsed. Public issue discussions and code examples show why this matters. For example, if a routing mode is exposed in configuration but execution still follows a different policy, a user can think they changed a quality or throughput knob when they actually did not. That is the worst kind of knob: visible in config, absent in execution.

MoE settings are especially high leverage because they change both math and traffic. Public MegaCpp materials provide several useful reality checks: router dtype is not an independent truth when a run is already under bf16 autocast, shared expert overlap is not free if there is no concurrent stream path, and FP8-related claims need to be tied to the exact expert path rather than treated as a blanket model speedup. In other words, expert knobs are real, but only when they are wired into the path that actually executes.

The same logic applies to MTP. Public MegaCpp examples separate MTP as a training-only feature and make `mtp=False` the real off switch, with explicit `mtp_depth=1` and `mtp_depth=3` lanes. That matters because auxiliary heads are not just a quality idea; they add projection, loss, memory, and optimizer work. If you compare two runs and forget to name MTP depth, you are not comparing like with like.

```yaml
model_family: NAM56R
pattern: AEMEAEMEAEMR
major_knobs:
  moe_enabled: true
  mtp_depth: 3
  regional_compile: true
  grad_reduce_in_fp32: true
  recompute: selective
report_rule: always publish pattern plus active architectural knobs
```

That kind of structured report is much more useful than a one-line claim that one run was "faster" or "better."

## Precision and communication knobs are real, but conditional

The second tier of knobs sits around precision, gradient movement, and overlap. These are still real, but they only pay off if the bottleneck actually matches the knob. Public MegaCpp examples are clear on this point. One visible setting is `grad_reduce_in_fp32` through the optimizer path, explicitly keeping gradient buffers in float32 for the communication and writeback route. Another group of public notes covers Megatron-style bucket handling and overlap policy so the fast path is not paying unnecessary synchronization cost.

Those changes matter because communication policy can alter both stability and throughput. But they are not interchangeable with architecture. If attention is only a small fraction of the step, then an attention-kernel win moves the total less than you expect. In public MegaCpp analysis, attention can be a small share of step time under some regional-compile configurations. That single detail is enough to kill a lot of misleading performance narratives.

A similar story shows up in the Mamba path. Public MegaCpp notes describe targeted work on Mamba in-proj fusion and also explain why a full replacement is higher risk: state-dict migration, extension parity, and FP8 integration all complicate the move. That is a good example of a knob that looks local but is actually architectural. Fusing an in-proj can help throughput, but if the rest of the Mamba path is still excluded from the broader precision stack, the total impact is narrower than a dashboard might suggest.

| Knob | Throughput upside | Quality or stability risk | Grounded reading |
| --- | --- | --- | --- |
| `grad_reduce_in_fp32` | better reduction robustness, sometimes steadier scaling | higher comm cost than bf16-only path | useful when reduction quality is part of the bottleneck |
| FP8-scoped expert path | expert GEMM speedups | recipe mismatch, coverage gaps, parity debt | only count wins on the path actually using FP8 |
| TE fusion in Mamba ingress | lower projection overhead | migration and extension compatibility risk | useful, but not a blanket Mamba rewrite |
| bucket and overlap tuning | better comm-compute overlap | easy to mis-measure if the step is compute-bound elsewhere | worth naming in structured reports, not overgeneralizing |

The key idea is that these are path-sensitive knobs. You have to know where the time is before you can rank them.

## Checkpointing and recompute policy are often the most honest trade-off

Engineers often describe checkpointing as a pure throughput loss taken only to fit memory. The current codebase gives a more nuanced picture. Recompute policy changes the shape of the step, the peak-memory envelope, and sometimes which model size is even runnable on a given lane. That means it is one of the cleanest real throughput-quality knobs because it determines whether you can afford a larger or richer model at all.

Public MegaCpp notes repeatedly separate memory-saving moves from architectural ones. There are explicit discussions of activation checkpointing, deferred comparisons on recompute policy, and the importance of reporting exact runtime lanes instead of collapsing everything into one speed number. In the Mamba path this is even clearer: the implementation includes a dedicated recompute surface, which is a reminder that recurrent-style paths have their own memory behavior and their own honest trade-off surface.

For MoE, the memory story is even sharper. Public MegaCpp examples show how changing an implementation surface can shrink a large intermediate into a much smaller fused buffer while preserving the intended math. The point is not only that one kernel is better; it is that memory shape changes what model and batch configurations are feasible. On a real training lane, that feasibility boundary feeds back into quality because the affordable model, context, and batch policy change.

This is why recompute and memory-shape decisions deserve to sit next to routing and MTP in any serious tuning discussion. They are not housekeeping.

## Quality knobs must be described with activation windows

Some features do not impose steady per-step cost. They activate later, activate conditionally, or matter only after a schedule boundary. If you measure them without stating the active window, you can easily mark a costly feature as free or a useful feature as irrelevant.

Public MegaCpp discussions already show concrete examples of this measurement problem. When quality-facing behavior is silently different from what a flag suggests, the benchmark itself becomes hard to trust. Those are not just correctness bugs; they are benchmarking traps. If a feature activates differently than you think, your throughput-quality chart is fiction.

That is also why family labels such as NAM52 and NAM56R should never be dropped from performance reports. They encode scale, intended recipe, and usually a different mix of active paths. A sentence like "better on a modern accelerator" is vague. A sentence like "better on NAM52, `AEME`-leaning pattern, no MTP, selective recompute" is usable.

Here the MegaCpp articles and public examples complement each other well. The article side provides the model taxonomy and training-path framing, while the public examples show how a local kernel or class-parity fix can move memory or throughput on named configurations. For example, the Mamba linear cross-entropy example shows how restoring output-layer parity removes an unnecessary logits allocation and stabilizes an otherwise OOM-prone lane. That is a throughput gain with a direct architectural interpretation: it comes from changing which output-layer implementation is in play, not from generic optimization folklore.

## What to standardize in every tuning report

If you want throughput-versus-quality discussions to become cumulative instead of repetitive, standardize the report format. The strongest version is simple.

Include the model family, the exact pattern string, the active architectural knobs, the precision or communication knobs, the memory policy, and the activation window for any delayed feature. Then publish both the observed speed metric and the quality metric in the same report. That does not solve every interpretation problem, but it removes most of the avoidable ones.

One practical template is:

```text
family=NAME
pattern=AEMEAEMEAEMR
lane=dense accelerator | dense TPU | MoE eval
arch_knobs=moe_enabled,mtp_depth=3,router_policy=topk
runtime_knobs=regional_compile,grad_reduce_in_fp32,recompute=selective
quality_window=steps[1000:2000]
throughput_metric=tokens_per_second
quality_metric=task_loss_or_eval
```

That kind of structure turns performance discussion into engineering evidence. It also makes it easier to compare notes across MegaCpp articles and public examples, because both speak in terms of concrete lanes, concrete modules, and reproducible reports.

The short conclusion is straightforward. Real knobs are the ones that change executed structure, memory shape, routing behavior, or communication semantics. Fake knobs are the ones that exist only in parsed args, in partial audits, or in reports that omit the active path. Once you enforce that distinction, throughput-versus-quality trade-offs stop looking mystical and start looking like what they are: model- and lane-specific engineering choices.

## References

- https://github.com/DatasunriseOU/cppmega/tree/main/upstream_prs/examples/11_mamba_linear_ce
- https://github.com/DatasunriseOU/cppmega/tree/main/upstream_prs/examples/12_dsa_indexer_memory
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md
