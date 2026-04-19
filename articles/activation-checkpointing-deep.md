---
title: "Activation checkpointing deep dive: why per-block policies beat one global switch"
description: "Full, selective, and narrow recompute across attention, MoE, Mamba-style, and recurrent blocks: what saves memory, what costs too much compute, and why a per-block policy usually wins."
date: "2026-04-18"
tags: ["activation-checkpointing", "selective-recompute", "mamba", "moe", "mla", "h200", "ablation"]
---

This post covers the ablation history behind a practical checkpointing policy. Full checkpointing everywhere was too expensive. Per-operator selective activation checkpointing helped in a few places but became hard to reason about at system level. What held up was a per-block policy: attention blocks use full-block or framework-level selective recompute, MoE blocks recompute expert GEMMs only, Mamba-style blocks avoid full checkpointing and keep a narrow conv-plus-projection recompute, and recurrent blocks use full checkpointing plus a small in-module recompute.

## Why this matters

Hybrid models do not have one dominant activation bottleneck. Attention, MoE, Mamba-style sequence layers, and recurrent blocks all concentrate memory in different operators, and the cost of recomputing those operators is very different. A Mamba selective scan is expensive to rerun. Core attention is moderately expensive. A standard MLP is usually much cheaper. One global flag throws away most of that structure. Per-block policy keeps most of the memory benefit without turning the runtime into a maze of special cases.

FP8 adds another constraint. Standard PyTorch checkpointing does not preserve FP8 amax history across recompute, while Transformer Engine checkpointing does. If the checkpoint boundary and autocast scope are mismatched, the training curve can drift long before the failure is obvious.

## The mechanisms that matter

MegaCpp needed four mechanisms that are often all called "checkpointing," even though they behave very differently.

Manual block checkpointing wraps each block forward with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. In practice, the main model forward decides layer by layer using block type, layer index, and a spacing policy. This is the eager-mode path.

Inductor automatic rematerialization is the compiled regional path. A compile helper translates `gradient_checkpointing=True` into an activation-memory budget, then inductor inserts recompute nodes to satisfy that budget. Compiled rematerialization and manual checkpointing do not compose well. If both are active in the same region, they can double-count work, slow training sharply, and still save less memory than expected.

CPU offload checkpointing trades recompute for host-link traffic. Instead of rerunning the whole block, it copies large saved inputs to pinned CPU memory and brings them back during backward. A finer-grained variant can use saved-tensor hooks above a size threshold. This is CUDA-only and only useful when recompute is expensive enough to justify the transfer.

Transformer Engine checkpointing is the FP8-safe path. It preserves amax history across the recompute boundary and is the right choice whenever an FP8-autocast block must be checkpointed.

Operator-local recompute is the last piece. Recurrent blocks can rerun only the recurrence instead of the whole block. MLA can save compact latent KV state and regenerate full K and V during backward. Mamba-style layers can rerun only the convolution and projection pieces. Those narrow cuts matter because they often recover most of the memory win at a fraction of the wall-clock cost of full-block checkpointing.

## The per-block ablation history

### Attention blocks

The useful comparison was between no checkpointing, full-block checkpointing, framework-level selective core-attention recompute, and a custom per-operator policy at the SDPA boundary. The landed policy was simple: full-block checkpointing in eager mode and framework-level `selective` recompute in the standard compiled configuration. That boundary worked because MLA up-projection was already being recomputed from compact latent state, so recomputing core attention captured the expensive part without duplicating the rest.

### MoE blocks

MoE made the tradeoff much clearer. Full-block checkpointing was too expensive because backward had to replay dispatch, permutation, collective traffic, expert compute, and combine. The winning policy was selective expert-GEMM recompute only. That left dispatch metadata alone and reran just the cheapest part of the MoE chain. It delivered the largest memory win in the stack while keeping throughput cost small.

### Mamba-style blocks

Mamba-style layers were the opposite of MoE. Full-block checkpointing was a bad trade because selective scan is expensive to rerun and tends to dominate the block cost. It also interacted badly with FP8 packed-token paths when recompute re-entered packing logic with already quantized inputs. The narrow conv-plus-projection recompute path was much better. It recovered meaningful memory and cost little in throughput because convolution backward already performed part of that work.

### Recurrent blocks

Recurrent blocks benefited from a combination of coarse and narrow recompute. The recurrence chain alone can pin several gigabytes, so rerunning just that chain is highly effective. In practice, full-block checkpointing plus the narrow recurrence recompute produced the cleanest result: memory close to the narrow path with a simpler block-level rule.

## Why custom per-op SAC usually lost

Custom per-operator SAC looked attractive because it promised exact control over what to save and what to recompute. In practice it lost on complexity.

One policy saved only expensive operators. That worked on attention-heavy stacks but missed important buffer-heavy work in MoE paths. Another policy used a raw tensor-size threshold. That was easy to explain but unstable because compact latent states could fall just below the threshold and get recomputed even when they were the wrong tensors to rerun. A block-aware operator policy worked better, but by then it was effectively rebuilding the block-level policy in a more fragile form. The result was clear: if your intended rule already depends on block identity and runtime context, it is usually cleaner to encode that policy at the block boundary.

## The platform-specific lessons

On compiled CUDA paths with an activation-memory budget below full retention, let the compiler own rematerialization. That path depends on gradient checkpointing being exposed to the compile layer, because the compile layer turns that signal into the inductor budget. If a higher-level configuration silently disables the flag, the compiled rematerialization lane disappears.

On TPU-class systems, autotuned rematerialization already does much of the work. Manual checkpointing still helps for attention-heavy regions, but CPU offload is not available, and MoE tradeoffs differ because dispatch buffers scale differently than they do on CUDA systems.

CPU offload is reserved for the narrow cases where recompute is expensive and host-link bandwidth is mostly idle. It is not a default training strategy. It is a situational escape hatch.

## The policy that survived

The practical policy is short:

- Attention: full-block checkpointing in eager mode, compiler or framework-level selective recompute in compiled mode.
- MoE: recompute expert GEMMs, not dispatch.
- Mamba-style blocks: avoid full-block checkpointing; use narrow conv-plus-projection recompute.
- Recurrent blocks: combine full-block checkpointing with narrow recurrence recompute.
- FP8: use an FP8-safe checkpoint path whenever recompute crosses an FP8-autocast boundary.

That policy is less elegant than a single global flag, but it matches where memory is actually spent and where recompute is actually cheap.
