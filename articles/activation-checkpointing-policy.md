---
title: "Activation Checkpointing Policy: The Per-Block Pareto That Held Up"
description: "Selective versus full activation checkpointing across attention, MoE, Mamba-style, and recurrent blocks, and why the best policy depends on where each block actually spends memory and compute."
date: "2026-04-18"
tags: ["activation-checkpointing", "training", "H200", "tpu", "moe", "mamba"]
---

# Activation Checkpointing Policy

Activation checkpointing is easy to enable and hard to tune. Turn it on everywhere and memory drops, but throughput often falls too much. Turn it off and the model may not fit at all. For hybrid architectures, neither extreme is right. The practical answer is a per-block policy.

## What "checkpointing" actually means

Several different mechanisms are often grouped under the same label.

Manual block checkpointing wraps a forward callable with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. In eager mode, this is the straightforward path.

Compiled rematerialization is different. When the compile pipeline is given an activation-memory budget below full retention, the compiler inserts recompute into the graph directly. That path should own the recompute decision inside compiled regions. Stacking manual checkpointing on top of compiler rematerialization tends to duplicate work.

CPU offload checkpointing replaces recompute with copies to pinned CPU memory and restores tensors during backward. It trades compute for host-link bandwidth and is only useful in narrow cases.

FP8-safe checkpointing is its own category. If recompute crosses an FP8-autocast boundary, the checkpoint path must preserve amax history correctly. Otherwise forward and recompute can quantize differently.

## Per-block policy

### Attention blocks

Checkpoint by default. Attention is usually the largest activation consumer on the dense path, and its recompute cost is manageable. In eager mode, full-block checkpointing is acceptable. In compiled mode, framework-level selective recompute around core attention is often the better fit. Under FP8, use an FP8-safe checkpoint path.

### MoE blocks

Use selective expert-GEMM recompute, not full-block checkpointing. Full-block MoE checkpointing reruns dispatch, permutation, communication, expert compute, and combine on backward, which is too expensive. Selective expert recompute captures most of the memory win while avoiding the expensive dispatch side of the block.

### Mamba-style blocks

Do not checkpoint the whole block by default. Selective scan is expensive to rerun, so full-block checkpointing gives a poor throughput tradeoff. A narrow conv-plus-projection recompute path is much better and avoids FP8 issues that can appear when recompute re-enters packed-token logic.

### Recurrent blocks

Checkpoint the block, but keep a narrow recurrence recompute enabled as well. The recurrence chain can hold a surprising amount of memory, and rerunning it is much cheaper than rerunning the entire block payload.

### Last layer

Do not checkpoint the last layer. Its activations are consumed immediately by backward, so the recompute adds cost without relieving peak memory.

## Mechanism selection by runtime

On compiled CUDA paths with an activation-memory budget below full retention, let the compiler own rematerialization.

On eager CUDA paths, use manual block checkpointing plus the per-block rules above.

On TPU-class systems, autotuned rematerialization does much of the work already. Manual checkpointing still helps on attention-heavy regions, but CPU offload is unavailable and MoE tradeoffs differ because dispatch buffers dominate differently.

CPU offload should stay narrow. It can help when recompute is expensive and host-link bandwidth is available, but it is not a general-purpose default.

## The Pareto that mattered

The key lesson from the measurements was not that one mode won everywhere. It was that each block family had a different efficient frontier.

- Attention favored either full-block checkpointing or framework-level selective recompute, depending on whether the runtime was eager or compiled.
- MoE favored selective expert recompute.
- Mamba-style layers favored narrow recompute only.
- Recurrent blocks favored full-block checkpointing plus a small in-module recompute.

That mix moved the stack from out-of-memory regimes into usable training regimes without paying the full throughput cost of blanket checkpointing.

## Failure modes worth remembering

Non-reentrant checkpointing can require relaxed determinism checks on token-routed MoE paths because tiny numerical differences in scatter-style updates can change routing metadata even when the backward signal is still acceptable.

Backend-specific runtime hooks matter. TPU and CUDA do not share the same rematerialization vocabulary, so policies that look similar at a high level still need backend-specific handling.

Automatic retry logic should preserve the intended recompute mode when it searches for a smaller batch geometry. If retries silently disable checkpointing, they can invalidate the very lane they are supposed to rescue.

## What we threw away

- One global `gradient_checkpointing` story as the source of truth.
- Full-block checkpointing for MoE by default.
- Full-block checkpointing for Mamba-style layers.
- Manual checkpointing inside compiled regions that already use compiler-driven rematerialization.

## Policy snapshot

| Block kind | Mechanism | Notes |
|------------|-----------|-------|
| Attention | manual checkpoint or framework-level `selective` | FP8-safe checkpointing under FP8 |
| MoE | selective expert-GEMM recompute | avoid full-block replay of dispatch |
| Mamba-style | never full-block by default | narrow conv-plus-projection recompute only |
| Recurrent | full block plus narrow recurrence recompute | good memory win at modest cost |
| Last layer | never | little or no peak-memory benefit |

```python
# Eager-mode sketch
def should_checkpoint(layer_idx, n_layers, block_kind, spacing):
    if layer_idx == n_layers - 1:
        return False
    if block_kind == "MambaStyle":
        return False
    if spacing > 0 and layer_idx % spacing:
        return False
    return True
```

## References

- PyTorch checkpointing and CUDA memory-management documentation
- Transformer Engine documentation for FP8-safe checkpointing
- public literature on activation recomputation in large transformer training
