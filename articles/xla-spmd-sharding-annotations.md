---
title: "XLA SPMD Sharding Annotations We Actually Ship"
description: "The mark_sharding call sites that survived real training on TPU v6e, the propagation surprises that taught us to pin everything, and the annotation audit that cut our recompile rate."
date: "2026-04-18"
tags: ["xla", "spmd", "tpu", "sharding", "the POC"]
---

This is the companion to the tensor-parallel post. That one covers the parallelism design. This one is strictly about the annotation surface in our research repo: where we call `mark_sharding`, where we do not, which specs we settled on, and what XLA's sharding propagation will do to you if you leave a parameter unannotated. The short version: on v6e, any parameter you do not annotate is a parameter XLA gets to interpret. That interpretation is sometimes catastrophically wrong. The fix is boring: annotate everything, including the tensors you want replicated.

## Why propagation is not your friend

SPMD propagation turns "you only annotated half the parameters" into "your residual stream is silently corrupted." Unlike a recompile or an OOM, propagation bugs do not crash. They produce loss curves that look slightly off, validation BPB that drifts, and a debugging trail that goes through a dozen activation tensors before you find the unannotated weight. We hit this on attention bottlenecks, on indexer projections, on adapter ranks that coincidentally divided into `n_embd / tp`, and on MoE expert weights when the hidden dim happened to match a mesh axis size.

The other reason it matters: a sharding spec is part of the compile cache key. An unannotated parameter risks correctness and performance, because the inferred spec on step 0 is not guaranteed to match the inferred spec on step 1 if anything in the surrounding graph changes. Pinning every parameter to an explicit spec, real or replicated, removes both classes of failure from the cache key.

## Where the annotations live

Two call sites matter in the training startup. Both are reached from the TPU training launcher via the public FSDP sharding sample:

- `_apply_tensor_parallel_sharding(model, mesh, ep_degree)` walks the parameter list and applies TP/EP specs.
- `apply_fsdp_sharding(model, mesh, tp_degree, dp_degree, ep_degree)` does ZeRO-3 on top, folding a `"data"` shard into the existing TP/EP spec and calling `xs.mark_sharding` with the combined tuple.

They run in that order. By the time the model is on the XLA device, every parameter has had `mark_sharding` called on it exactly once with its final spec. Activations get separate annotations via `set_spmd_mesh(...)` on the model object, which caches `xs.mark_sharding` and calls it on hidden states at layer boundaries in the forward.

The mesh shape depends on active parallelism: `("data",)` when TP=1 and EP=1; `("data", "model")` when TP>1; `("data", "expert")` when EP>1 and TP=1; `("data", "expert", "model")` when EP>1 and TP>1. The annotation code branches on `mesh.axis_names`, not on the args, which has saved us from at least two mismatched-mesh bugs.

## The specs we settled on

| Module | Parameter | Spec on `("data", "model")` |
|---|---|---|
| Dense attention | `c_qkv.weight`, `c_q/c_k/c_v.weight` | column-parallel `("model", None)` |
| Dense attention | `attn.c_proj.weight` | row-parallel `(None, "model")` |
| Dense MLP | `mlp.c_gate_fc.weight` (fused SwiGLU) | column-parallel |
| Dense MLP | `mlp.c_proj.weight` | row-parallel |
| MLA | `w_uq.weight`, `w_ukv.weight` | column-parallel on dim 0 |
| MLA | `w_dq/w_dkv/w_dqkv.weight` | replicated (explicit) |
| MLA | `attn.rope_cos`, `attn.rope_sin` | replicated (explicit) |
| MoE EP>1 | `experts.W_fc/W_proj/W_gate` | `("expert", None, None)` |
| MoE | router, shared experts, null expert | replicated (explicit) |
| DSA | `indexer.q_proj/w_proj.weight` | column-parallel |
| DSA | `indexer.k_proj.weight` | replicated (explicit; shared across heads) |
| Auxiliary | Engram, mHC, N-gram hash, MTP, structure embeddings | replicated (explicit, loud) |

On ZeRO-3, the FSDP pass picks a shard dim from whatever is still `None` after TP/EP, prefers the largest one that is divisible by `dp_degree`, and writes the combined spec back. For 2D TP-sharded weights it builds 2D SPMD sharding like `("data", "model")` or `("model", "data")`; for replicated-after-TP weights it shards on dim 0 unless the shape is too small (the threshold is 4). It falls back to replication when nothing divides cleanly.

## The propagation surprises

XLA's sharding propagation is useful when it works, and it works most of the time. The problem is the cases where it infers a wrong shard because a dimension coincidentally matches a mesh axis.

The archetypal example is Engram at TP=4. `engram_bottleneck_dim` defaults to `n_embd // 4`. At TP=4 on a 1024-wide model, that is 256. `n_embd / tp` is also 256. Propagation sees activations feeding the Engram path that carry a `"model"` shard on the last dim (they come out of column-parallel Q/K/V intermediates) and concludes the Engram bottleneck is also head-parallel. The Engram `out_proj` then produces partial sums over the `n_embd` dim instead of the full vector. The residual stream is quietly corrupted. Validation BPB roughly doubles. No crash. No warning.

The fix is one line per parameter: `xs.mark_sharding(param, mesh, _replicated(param.ndim))` on every Engram weight. After that, XLA has no excuse to infer anything. The same surprise has bitten us on indexer K projections (shared across heads, shape divides cleanly under TP), on LoRA matrices whose ranks coincidentally divided into `n_embd / tp`, and on N-gram hash output projections.

```python
import torch_xla.distributed.spmd as xs

def _replicated(ndim: int):
    return tuple(None for _ in range(ndim))

for name, param in model.named_parameters():
    if name in REPLICATED_NAMES:
        xs.mark_sharding(param, mesh, _replicated(param.ndim))
```

## LoRA and adapter annotations

LoRA adds its own `mark_sharding` surface. `apply_lora_spmd_sharding()` in the public LoRA sample annotates LoRA A/B matrices with specs aligned to the base layer's parallelism: `("model", None)` for column-parallel LoRA-A, `(None, "model")` for row-parallel LoRA-B, and `(None, None)` for the replicated case. The replicated LoRA matrices are still annotated explicitly, for the same reason every other replicated tensor is: an unannotated tensor is an interpreted tensor. Adapter merge does not require new annotations because the merge happens on the base parameter, which is already annotated.

## The annotation audit

After enough propagation surprises, we wrote a one-shot audit: walk every named parameter at the end of `_apply_tensor_parallel_sharding` and `apply_fsdp_sharding`, query its current sharding spec, and assert it is non-default. Anything left at the default sharding is logged at error level and the launch refuses to start. The audit caught three real bugs the first time we ran it: a forgotten N-gram hash projection, an Engram bias that had been added in a recent feature commit without an annotation update, and an MLA RoPE buffer that is not a parameter but is loaded into the device and was previously left to propagation.

The audit also produces a fingerprint: a hash over `(name, dtype, shape, spec)` for every parameter. The fingerprint goes into the receipt and is checked against the cache key. When a launch's fingerprint disagrees with the previous launch's, the persistent compilation cache is invalidated for that run; this is much cheaper than letting XLA discover a stale cache mid-step and SIGABRT.

## What we kept and threw away

We kept the rule that every parameter has an explicit spec, the audit at the end of the sharding pass, the fingerprint in the receipt, the LoRA explicit-replicated annotations, and the practice of annotating activations at layer boundaries via `set_spmd_mesh`. We kept the explicit replicated annotation on every "small" tensor whose shape might coincidentally match a mesh axis (RoPE buffers, MLA down-projections, indexer K projections, all router projections).

We threw away reliance on XLA propagation for any parameter we own (we still rely on it for activations between annotated boundaries, but never for parameters), the assumption that "tiny replicated tensor doesn't matter", per-launch ad-hoc spec edits that did not go through the audit, and the convenience of skipping annotations on a feature that "is just a Linear". Every unannotated tensor is a future propagation bug; we have proven that at our own expense more than once.

## How the audit and the cache key compose

The annotation audit runs at the end of the sharding pass; the cache key is computed at the start of the first compile. Between the two, the only thing that changes is the parameter tensor placement on the device, and that is deterministic given the spec set. The implication is that the audit's fingerprint is a strict prefix of what XLA hashes into the cache key. When the fingerprint changes, the cache key changes; when the cache key changes, every cached compilation is invalidated for the affected sub-graph.

We use this property deliberately. When a feature commit changes a sharding spec (for example, moving Engram from inferred-sharded to explicitly-replicated after the propagation bug), the fingerprint changes, the cache key changes, and the next launch does a cold compile of the affected sub-graphs. That is more expensive than we would like, but it is correct: a cached compilation against the old spec would silently produce different results, and we would much rather pay a cold compile than ship a precision regression.

The flip side is that we do not change sharding specs casually. Every spec change is recorded in the public engineering changelog against the receipt that motivated it, the fingerprint diff is captured, and the next launch's cold-compile cost is budgeted into the receipt. That discipline keeps the cache useful and makes spec changes auditable after the fact.

## What activation annotations buy

`set_spmd_mesh(...)` on the model object caches `xs.mark_sharding` for activations and applies it at layer boundaries in the forward. The default activation sharding mirrors the parameter sharding: an activation entering a column-parallel `Linear` is annotated `("data", "model")`, an activation leaving a row-parallel `Linear` is annotated `("data", None)`. We pin the boundary annotations rather than letting propagation infer because, again, the cache key includes the inferred annotation and we do not want it to drift.

The activation annotations are also where the SP/non-SP split is most visible. With sequence parallelism on, the activation entering the column-parallel input is `("data", "model")` on the sequence dim; without SP, it is `("data", None)` and the gather happens inside the linear. We do not let the same code path serve both worlds; the SP build is a different annotation pass entirely, selected at startup, with its own fingerprint and its own cache key.

## When propagation is allowed

There is one place we still let propagation work: the activations between two annotated boundaries inside a compiled region. The compiler's reasoning over those is reliable enough that pinning every intermediate would be overkill and would defeat the fusion windows that make XLA fast on v6e in the first place. The rule is that boundaries are pinned, intermediates are not. When an intermediate produces an unexpected shard (the kind of thing the Engram bug produced for parameters), it is almost always because the producing op was reading an unannotated tensor; the fix is to annotate the producer, not the intermediate.

The audit gates the producer side. If every parameter and every layer-boundary activation has an explicit spec, the intermediates fall into place. The bugs we have hit have all been at the producer side, never at the intermediate side. That is the operating evidence we have for keeping the rule.

## References

- the TPU training launcher
- fsdp.py
- lora.py
- engram.py
- m2rnn.py
- dsa_training.py
- the public engineering changelog
- TENSOR_PARALLELISM.md
