---
title: "Torch XLA and PJRT reality: what actually matters"
description: "A grounded look at the current TPU stack: custom torch_xla, PJRT contracts, SPMD setup order, and the real failure modes that still shape evaluation and training."
date: "2026-04-18"
tags: ["torch-xla", "pjrt", "xla", "tpu", "training", "evaluation"]
---

The current TPU lane is not a generic "install XLA and go" story. It depends on a modern PJRT contract, early SPMD initialization, and careful reduction behavior in evaluation code. The high-level idea is simple: use the TPU stack that matches the runtime contract, set flags before imports, enable SPMD before tensors exist, and never assume a TPU metric is globally reduced unless the code path proves it.

There is a lot of stale advice around Torch XLA. Some of it was valid for older XRT-era setups, some of it assumes stock wheels are enough, and some of it ignores how hybrid training code mixes evaluation, optimizer state, and mesh construction. The current repo materials are useful precisely because they document the boring but decisive details: exact runtime versions, import order, cache initialization, and the places where TPU reductions can still fail silently.

## The first reality: the stack is a contract, not just a package list

The current public README states the validated TPU lane very plainly. The preferred install path remains the documented installer flow, and the validated stack uses custom `torch` and custom `torch_xla` builds together with a modern `libtpu`. It also states the reason: the custom `torch_xla` line is required for the modern libtpu and PJRT contract, while the stock `torch_xla==2.9.0` wheel still targets an older lane.

That one paragraph explains most TPU confusion. When people say "XLA is unstable," they often mean they mixed a current TPU runtime with an older software contract. When they say "PJRT changed everything," they are partly right, but the practical lesson is not philosophical. It is operational: use the build line that matches the runtime interface you are actually on.

The product and tech-stack docs reinforce the same picture from another angle. They describe TPU tensor parallelism and XLA SPMD 2D mesh support as a first-class design target, not an experiment glued on later. That matters because a project built around SPMD mesh semantics will naturally place more weight on early mesh initialization, mark-sharding correctness, and collective behavior than a codebase that only occasionally touches TPU.

| Question | Repo-grounded answer | Why it matters |
| --- | --- | --- |
| Is stock `torch_xla` enough? | Not for the current validated PJRT lane | version mismatch can turn setup issues into fake model issues |
| Is TPU support bolt-on? | No, TPU SPMD is part of the intended runtime design | initialization order and mesh semantics are central |
| Can I trust a local TPU metric by default? | No, only after the reduction path proves it | silent fallback can misreport eval quality or throughput |

If you take only one thing from this, make it this: TPU stability starts with a correct contract between `torch`, `torch_xla`, and `libtpu`.

## The second reality: import order and SPMD timing are part of correctness

The training startup path in the main training script is explicit about setup order. It notes that `xr.use_spmd()` must be called before any Torch XLA tensor exists. Later, it also emphasizes that XLA and libtpu flags must be applied before any `torch_xla` import because initialization code reads the environment and fills in missing flags. This is not cosmetic startup sequencing. It is part of the runtime contract.

That means TPU setup has two early gates.

1. Set the environment and flag policy before importing the runtime.
2. Enable SPMD before constructing tensors or letting helper paths import XLA indirectly.

The code also initializes an XLA compilation cache after flag setup and before the main model flow. This is another small but important detail. Caching is not just a convenience; it changes how repeated runs behave and helps separate compilation artifacts from actual execution regressions.

```text
example TPU startup contract:
  set PJRT_DEVICE = TPU
  apply TPU runtime flags before importing the runtime
  enable SPMD before tensors exist
  start training with tensor parallelism and an explicit XLA flag profile
```

The exact launch wrapper varies, but the principle does not. If a helper imports XLA too early, the rest of the run can become hard to reason about. That is why the comments in the training script matter so much more than generic TPU blog advice.

## The third reality: evaluation can lie if reduction semantics are weak

One of the most useful code surfaces for understanding TPU behavior is the evaluation path used for loss aggregation. It contains helper functions that try to all-reduce totals through the XLA runtime and branch on `PJRT_DEVICE=TPU`. At first glance that sounds fine. The live bug report shows why it is not enough.

The report calls out a specific high-severity issue: BPB reduction on TPU can silently return per-chip numbers. The root cause is that the reduction path swallows exceptions too broadly and falls back without making enough noise. That means a run can look healthy while reporting local rather than global totals.

This is not a niche bookkeeping error. In practice it affects how you interpret evaluation curves, cost-per-token calculations, and any claim about TPU scaling efficiency. If your global metric silently degraded into a local metric, then your dashboard is not merely noisy; it is wrong.

The same report also documents TPU-side hazards around optimizer grad materialization and autodetection. One issue ties XLA optimizer gradient materialization to clipping state, so disabling clipping can unexpectedly starve the compiled path of the parameter list it needs. Another notes that TPU autodetect can still force XLA on a broken or non-runtime host. These are different bugs, but they point to the same engineering lesson: TPU correctness in a hybrid training system is often determined by the edges where runtime assumptions meet convenience logic.

| Failure mode | Surface | Practical consequence |
| --- | --- | --- |
| reduction fallback too quiet | the evaluation reduction path | per-chip metrics can masquerade as global ones |
| grad materialization tied to clipping | optimizer path | compiled TPU training can break when clipping is disabled |
| autodetect forces XLA in the wrong place | startup logic | host selection bugs look like model or compiler failures |

For operators, the rule should be simple: if the metric depends on a collective, confirm the collective path explicitly.

## Mesh construction is the real TPU mental model

The current the distributed parallelism module module makes the intended abstraction explicit. CUDA uses the device-mesh stack and FSDP2 style sharding, while XLA or TPU uses SPMD mesh semantics and `mark_sharding` through the local FSDP support. The point is not that TPU imitates CUDA. The point is that both runtimes map into the same conceptual model of data, tensor, pipeline, and expert dimensions, but with different implementation backends.

That is the right way to think about PJRT in this repo. PJRT is not a magical optimizer; it is the runtime contract underneath the TPU execution model. The engineering task is to build the right mesh, shard the right tensors, and preserve those assumptions through the training and eval stack.

This matters especially in a project that mixes dense, sparse, and hybrid paths. The README already warns readers to keep exact-token sparse, blockized sparse CUDA, and dense/full FA4 receipts separate. The same discipline applies on TPU. Do not collapse all TPU results into one bucket. A dense TPU lane, a sparse eval lane, and a hybrid Mamba-plus-expert lane may all use XLA, but they are not testing the same execution surface.

The repo's TPU design target is also specific: scaling up to eight Trillium chips with SPMD mesh semantics. That is a stronger statement than "supports TPU." It tells you what the code is trying to optimize for and why mesh logic is so central.

That target also changes how you should read model notation on TPU. These architecture labels are kept intentionally here as public glossary terms, not as ad-hoc experiment strings. The associated block vocabulary matters on TPU too. `ablock` usually implies attention-side sharding and reduction pressure. `eblock` implies routed-token motion and expert-local work. `mblock` and `rblock` imply state-carrying or selective paths whose shard boundaries may behave very differently from dense attention. If a TPU result regresses, those names help isolate whether the problem lives in collective shape, state handling, or expert routing rather than in “XLA” as a monolith.

This is one reason PJRT discussions become unproductive when they stay too generic. The runtime contract may be global, but the failure surfaces are still local. A wrong import order can poison the entire job. A weak reduction path can corrupt evaluation only. A bad sharding annotation can destabilize one block family while leaving another apparently healthy. The practical value of the glossary is that it keeps those surfaces separate while the TPU stack is still evolving.

## The setup story is mature enough to be useful, but not simple enough to ignore

A good sign is that the docs are no longer aspirational here. The README names a validated TPU stack. The training script has detailed comments about flag application, cache initialization, and early SPMD setup. The evaluation code already contains TPU-specific reduction handling. In other words, the TPU lane is real.

But the live bug report is equally important because it explains what is still fragile. TPU backward for clustered sparse paths can still ignore attention softcap. Some optimizer cleanup on compiled XLA paths can touch `None` grads. And the reduction path can still turn global metrics into per-process ones if exceptions are hidden. None of these invalidate the TPU lane. They simply define the current boundary of trust.

The mature posture is therefore neither optimism nor panic. It is operational discipline. Re-read the setup comments before large runs. Re-check reduction code before trusting TPU dashboards. Keep mesh and sharding assumptions close to the block family they serve. And avoid the lazy habit of treating one successful dense training run as proof that every hybrid or routed path is equally safe. TPU support is strongest when the project treats setup, sharding, and measurement as first-class code contracts rather than one-time bringup chores.

This is where many public posts go wrong. They either declare TPU support "done" because one run succeeded, or they declare it unusable because one reduction path is imperfect. The repo gives a more useful answer. TPU is a real lane with a concrete setup contract and a concrete list of live caveats. That is exactly the kind of maturity you want from active research infrastructure.

## What an honest PJRT checklist should include

If you are bringing up or evaluating a TPU lane in a similar project, the minimal honest checklist is short.

1. Confirm the software contract: matching `torch`, `torch_xla`, and `libtpu` for the intended PJRT lane.
2. Apply runtime flags before any XLA import.
3. Enable SPMD before any XLA tensor exists.
4. Initialize compilation cache intentionally.
5. Verify that evaluation reductions really reduce globally.
6. Treat autodetect and convenience fallbacks as suspicious until verified.

One compact configuration sketch looks like this:

```text
runtime=TPU
PJRT_DEVICE=TPU
mesh=SPMD
tp=4
cache=enabled
metric_reductions=verified
autodetect=disabled_or_confirmed
```

That is not glamorous, but it is the difference between a trustworthy TPU lane and an anecdote.

The deeper point is that PJRT reality is mostly about disciplined initialization and disciplined measurement. The prototype already contains both the positive evidence and the warning signs. Use the validated stack, respect setup order, and distrust any TPU metric path that has not been re-read recently. That is a much more productive posture than arguing in the abstract about whether Torch XLA is good or bad.

## Code and notes

- [PyTorch/XLA docs](https://docs.pytorch.org/xla/master/)
- [PyTorch/XLA SPMD docs](https://docs.pytorch.org/xla/master/spmd.html)
- [PyTorch DTensor docs](https://pytorch.org/docs/stable/distributed.tensor.html)

## Further reading

- [PyTorch compile docs](https://pytorch.org/docs/stable/generated/torch.compile.html)
