---
title: "Kernel Catalog and Impact: Why the Runtime Needed a Real Map"
date: 2026-04-18
author: MegaCpp Engineering
tags: [kernels, h200, moe, attention, triton, systems]
summary: >
  The POC and MegaCpp codebases do not use one magical fast kernel. They use a
  growing catalog of kernels and backend seams, and performance only made sense
  once that catalog was treated as an explicit design surface.
description: >
  A grounded tour of the kernel catalog across attention, sparse MLA, MoE, MTP,
  and dispatch/combine paths, with emphasis on why naming the kernel family and
  backend contract changed system-level decisions.
---

# Kernel Catalog and Impact: Why the Runtime Needed a Real Map

**TL;DR:** The important performance lesson in the POC was not “switch to a faster kernel.” It was that the project needed an explicit kernel catalog: which family handled attention, which family handled expert dispatch and combine, which family handled sparse MLA, and which family was only a donor or deferred substrate. Once those boundaries were named and documented, system-level decisions about memory, launchers, and model variants became much more trustworthy.

When teams talk about kernel work, they often compress everything into a single heroic path: one fused kernel, one extension, one benchmark chart. The repo evidence here points in the opposite direction. The working system is a catalog, not a monolith. Different block families use different kernel backends, and the backend choice affects not only raw speed but also memory materialization, autograd structure, optimizer assumptions, and the comparability of benchmark receipts.

That is why the design note on the MoEBlaze substrate matters so much. It states the target explicitly: not merely fewer saved tensors, but lower end-to-end runtime materialization and HBM traffic. That is the right lens for understanding kernel impact in a real training stack.

## The catalog starts with model structure, not with CUDA code

Before listing any backend, it helps to remember why a catalog exists at all. NAM56R is already a mixed architecture. In the recipe layer it is declared as `AEMEAEMEAEMR`, which means the runtime must support at least attention-style layers, expert layers, Mamba-family layers, and recurrent-style or selective layers. A single kernel family cannot carry all of that.

The POC runtime reflects this directly.

- the main model runtime module owns attention-side integration, precision plans, and block composition.
- the main MoE runtime module and the MoE dispatch runtime module own router-adjacent, dispatch, and combine behavior.
- MegaCpp contributes sparse MLA and native Hopper-facing components under dedicated modules.

This is already a kernel catalog in embryo. The performance problem is not “find the best kernel,” but “assign the right kernel family to the right structural surface.”

| Model surface | Typical backend family | Why the distinction matters |
| --- | --- | --- |
| Dense attention | Flash-attention family and related attention kernels | Throughput, causal masking, layout assumptions |
| Sparse MLA | TileLang sparse MLA kernels | Different tensor layout, different scale handling |
| MoE dispatch/combine | Triton-first substrate plus donor-inspired structure | Materialization and routing metadata dominate |
| MTP / CE path | Native Hopper-oriented kernels where available | Avoid logits materialization and extra reshapes |

The table is useful because it turns “kernel optimization” into a routing problem across real subsystems.

It also pairs naturally with the MegaCpp block glossary. `ablock` questions usually land in attention kernels, projection kernels, positional handling, and their surrounding layout adapters. `eblock` questions land in routing, dispatch, combine, grouped GEMM, and metadata motion. `mblock` questions land in state-space mixers, selective scan style kernels, and their runtime boundary code. `rblock` questions land in recurrent persistence logic and state-carrying update paths. `cblock` is useful when discussing a composite runtime region that bundles multiple sub-operations under one scheduling or checkpointing policy. A real kernel catalog should make those distinctions visible because different block families fail, scale, and optimize for different reasons.

## Attention kernels were a family, not a single switch

The top-level model file makes this obvious. the main model runtime module imports the project’s flash-attention entry points and also references decode and cache-aware helpers. That means “attention kernel” already means more than one thing: training-time full attention, decode-time single-token paths, and backend-specific variants.

The repo comments also reflect a practical split between what is patched in from outside and what is kept as the project’s own high-level contract. That separation matters because a training stack cannot tolerate every attention experiment rewriting the whole module surface.

The impact of a clean attention catalog is mostly defensive.

1. It prevents backend-parity claims from being made too early.
2. It makes it clear whether a benchmark changed the entire module path or only a low-level kernel.
3. It keeps decode, cache, and full-attention cases from being collapsed into one benchmark story.

That same discipline shows up again in sparse MLA.

## Sparse MLA needed its own kernel family and layout contract

The MegaCpp sparse MLA modules are unusually explicit. `sparse_mla.py` states that it wraps TileLang fused sparse MLA forward and backward kernels into a single interface. It also documents layout assumptions such as permuting query from `[sq, b, np, hn]` to `[b, sq, np, hn]`, narrowing KV for MLA, and reshaping the output back after the kernel call.

The FP8 variant goes further. `tilelang_sparse_mla_fwd_fp8.py` maintains an LRU-style kernel cache, builds a forward kernel keyed by launch parameters, and returns BF16 output even though the compute path is FP8. Those details are not trivia. They tell you exactly why sparse MLA belongs in its own catalog row.

It is not just a different implementation of dense attention. It has a different data contract, different scale metadata, and different caching behavior.

```python
q = query.permute(1, 0, 2, 3).contiguous()
kv = key.permute(1, 0, 2, 3)[:, :, 0:1, :].contiguous()
out, lse = kernel(q, kv, q_scale, kv_scale, indices)
output = out.permute(1, 0, 2, 3).contiguous().reshape(sq, b, np_ * hnv)
```

That schematic captures the key point: a sparse MLA kernel family imposes its own ingress and egress layout rules. Once that is true, the choice of kernel affects not only speed but the surrounding adapter code, memory traffic, and error surface.

## The MoE catalog was about substrate, not just compute

The design note `25-moeblaze-kernel-substrate-decision.md` is the clearest catalog document in the repo. It explicitly compares Triton, vLLM modular fused MoE, SGLang fused MoE Triton, Megatron-Core MoE, MegaBlocks, fusedswiglu, TensorRT-LLM, and CUTLASS or CuTe. More importantly, it labels each one by role: chosen first, donor, deferred donor, or deferred substrate.

That is exactly what a kernel catalog should do. It should not only say what exists. It should say what each option is for.

| Option | Role in the catalog | Reason |
| --- | --- | --- |
| Triton | Chosen first | Best fit with current repo and lowest integration cost |
| vLLM modular fused MoE | Donor | Good decomposition for permute, unpermute, and weighted finalize |
| SGLang fused MoE Triton | Donor | Useful Triton organization and top-k handling |
| Megatron-Core MoE | Donor | Strong training-system boundaries for router, dispatch, experts, combine |
| MegaBlocks | Donor | Metadata and topology ideas |
| fusedswiglu | Narrow donor | Direct fused gate-up activation shape |
| CUTLASS / CuTe | Deferred substrate | High ceiling, higher integration burden |

The impact of this document is bigger than a backend choice. It stops the project from pretending every kernel source is equally ready for direct adoption. It also keeps the main goal focused on end-to-end traffic and intermediate materialization rather than isolated arithmetic throughput.

That distinction is why the runtime catalog matters. A fast compute kernel can still lose if it requires the wrong staging buffers.

The donor labeling also prevented a lot of bad engineering behavior. Without it, every attractive upstream kernel starts to look like a near-term integration candidate. With it, reviewers can say something sharper: this source is useful as a decomposition donor, this one is useful for metadata ideas, this one is useful only as a ceiling reference, and this other one is not worth pulling into the training path until the surrounding ABI and materialization story are under control. That is a systems decision, not a benchmark vanity choice.

## Native Hopper-oriented kernels changed some ceilings

The MegaCpp side also contains targeted Hopper-facing work outside the main MoE substrate conversation. `mtp_native_hopper_ce.py` exists specifically to route a multitoken prediction cross-entropy path through a native Hopper kernel so that logits do not need to be materialized in the usual way. The comments in that file are clear that masked positions are handled inside the kernel and that the path is meant to exploit a Hopper-native capability rather than a generic fallback.

This is a good example of why a catalog can improve system design even before it improves every benchmark. Once a path is recognized as a separate kernel family with its own capability profile, the launcher and model stack can decide when to use it and when not to. Without that, a specialized kernel either gets overused or forgotten.

The broader lesson is that some kernels are system-level enablers more than raw-throughput stars. Avoiding logits materialization can matter as much as a few percentage points of arithmetic speed, especially in large sequence or multitask training regimes.

That same lesson appears in how the repo treats launch seams. Several modules are not “the kernel” in the narrow sense, but they matter just as much because they determine whether a good kernel is fed cleanly or surrounded by expensive copies, reshapes, and compatibility buffers. In practice, many performance wins come from moving a boundary so that a kernel family receives the layout it actually wants, rather than from rewriting the arithmetic body itself. A kernel catalog that ignores adapters and launch objects is incomplete.

## Kernel impact was mostly about reducing ambiguity

The most practical impact of the catalog was that it made benchmark interpretation less sloppy.

If a sparse MLA result improved, reviewers could ask whether TileLang FP8 caching, scale handling, or layout adapters changed. If an MoE lane improved, reviewers could ask whether the gain came from better Triton substrate staging, less routing materialization, or a donor-inspired dispatch shape. If an attention configuration changed, it could be traced to the flash-attention family rather than being conflated with sparse MLA or decode behavior.

That kind of separation is what keeps optimization work cumulative instead of anecdotal.

| Question | Without a catalog | With a catalog |
| --- | --- | --- |
| “Why did this benchmark improve?” | Hard to know which backend moved | Usually attributable to one kernel family or adapter seam |
| “Can we compare these runs?” | Easy to compare unlike with unlike | Easier to see when backend families differ |
| “What should we optimize next?” | Chases symptoms | Targets a named substrate or launch seam |

The impact, in other words, was operational clarity.

## Why the catalog matters for future work too

A project like this will keep gaining new kernels. Some will be direct implementations, some will remain donor references, and some will stay deferred because they require a new extension or ABI lane. The right response is not to hide that diversity. It is to keep the catalog explicit.

The repo already shows what that looks like.

- Pattern notation identifies which model surfaces exist.
- Runtime files keep family-specific adapters near their real call sites.
- Design notes declare which backend is chosen, deferred, or donor-only.
- Specialized kernel modules document their layout and scale contracts.

The next benefit is educational for maintainers. When someone says a benchmark moved after a kernel change, the catalog gives reviewers a checklist. Was it the dense attention family, the sparse MLA family, the MoE dispatch family, or a launch adapter around one of them? Did the change affect only an `ablock` region, or did it alter an `eblock` dispatch path that would never show up in a dense-only benchmark? Those questions sound simple, but they are exactly what keeps future optimization work grounded instead of turning every speedup into folklore.

That is the foundation for honest systems work. It lets the project say not just “we have a faster kernel,” but “this model surface is now served by this backend family, under this contract, with these tradeoffs.”

For a mixed architecture like NAM56R, that is the only definition of a useful kernel improvement.

## References

- `25-moeblaze-kernel-substrate-decision.md`
- the main model runtime module
- the main MoE runtime module
- the MoE dispatch runtime module
- the public sparse MLA sample
- the public TileLang FP8 sparse MLA sample
- the public Hopper-native CE sample
- the public NAM56R recipe sample
