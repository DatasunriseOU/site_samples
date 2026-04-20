---
title: "Clustered sparse on TPU: the planner stages"
description: "How MegaCpp decomposes clustered sparse TPU attention into planner stages, legality checks, and backend dispatch rather than treating sparse attention as one giant kernel."
date: "2026-04-19"
tags: ["tpu", "pallas", "sparse-attention", "kernels", "xla"]
---

The most misleading way to describe sparse TPU attention is as one kernel.
MegaCpp's public examples show something more useful: a planner pipeline.
Coarse routing, union selection, legality checks, causal windowing, and block
expansion all happen before the final sparse kernel is even the interesting
part.

That is the right public story because it explains why sparse TPU work is hard.
The challenge is not only writing a kernel. It is keeping the planner stages
shape-safe and backend-safe enough that the sparse lane cannot silently drift
into an invalid or misleading execution path.

## The planner is where the sparse contract becomes real

The kernel examples in this repo already expose the main stages:

- three-phase clustered sparse flow
- union selection and query-mask legality
- causal windowing predicates
- hierarchical block expansion before final top-k decisions
- exact mask-contract cache handling
- backend dispatch between Splash, Pallas, and sparse variants

That is enough to support a strong architectural claim. Sparse TPU attention is
not one feature flag. It is a planner plus execution stack.

## Why MegaCpp separates the stages publicly

The separation solves two problems.

First, it makes correctness legible. If legality and windowing are buried inside
one opaque sparse kernel, it becomes hard to prove which boundary failed.
Breaking the pipeline into planner stages makes it possible to reason about each
contract individually.

Second, it makes backend ownership honest. A request for clustered sparse
attention does not mean one exact backend will always execute it. The backend
dispatch and fallback examples keep that visible. Sometimes the sparse request
stays on the intended Pallas path. Sometimes a different backend or fallback is
more honest for the current shape and mask contract.

## The planner stages that matter most

The local examples suggest a practical decomposition.

Union selection is where top-k block choices are converted into a compact work
set the later sparse kernel can actually consume. Causal windowing is where
future-illegal tiles are removed before they pollute the sparse plan.
Hierarchical block expansion is where a coarse routing choice is refined into a
more precise sparse workset. Exact mask-contract helpers ensure that cache keys
and legality decisions remain tied to the actual runtime mask semantics.

This is a better engineering story than saying "we implemented clustered sparse
attention on TPU." It shows where the planner can fail and why the sparse lane
needs more than one test surface.

## Why Pallas and Splash both appear in the story

The point is not that Splash and Pallas are interchangeable. The point is that
they own different parts of the TPU attention story. Splash is the stable path
for more standard dense or local attention surfaces. Pallas matters when the
mask, sparse plan, or softcap behavior needs lower-level control. MegaCpp's
public examples keep that distinction visible instead of flattening everything
into "the TPU backend."

That is also why clustered sparse examples belong beside backend-dispatch and
bridge examples. The planner stages are only meaningful if the runtime can make
a defensible backend choice after the planner has spoken.

## Prior art and context

The broad ideas are not unique. Pallas TPU docs and sparse TPU docs explain the
kernel-language side. Splash kernel sources show the stable TPU attention lane.
MoBA and related sparse-attention work provide the broader block-sparse routing
context. MegaCpp's public contribution is the narrower planner view: examples
that show how legality, routing, block expansion, and backend dispatch are kept
as explicit stages around clustered sparse TPU attention.

## References

- [Clustered sparse three-phase sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/kernels/clustered_sparse_three_phase_sample.py)
- [Union selection query mask sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/kernels/union_selection_query_mask_sample.py)
- [Causal windowing predicate sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/kernels/causal_windowing_predicate_sample.py)
- [Hierarchical block expansion sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/kernels/hierarchical_block_expansion_sample.py)
- [Exact mask contract cache sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/kernels/exact_mask_contract_cache_sample.py)
- [XLA backend dispatch sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_backend_dispatch.py)
- [Pallas docs](https://jax.readthedocs.io/en/latest/pallas/index.html)
- [TPU Pallas docs](https://jax.readthedocs.io/en/latest/pallas/tpu/index.html)
- [Sparse TPU Pallas docs](https://jax.readthedocs.io/en/latest/pallas/tpu/sparse.html)
- [Splash attention kernel source](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
- [MoBA](https://arxiv.org/abs/2502.13189)
- [MoBA repository](https://github.com/MoonshotAI/MoBA)
