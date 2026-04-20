---
title: "OOM Debugging Playbook for H200 Training Runs"
description: "A practical playbook for triaging H200 out-of-memory failures: distinguish fragmentation from true exhaustion, isolate the largest activation surfaces, and apply the cheapest fix first."
date: "2026-04-18"
tags: ["oom", "H200", "memory", "debugging", "training"]
---

Out-of-memory failures on modern accelerators are often diagnosed too loosely. "Needs a smaller batch" is only one of several possibilities. In practice, most H200 OOMs fall into one of four buckets:

1. True activation pressure: the model really does not fit at the current geometry.
2. Fragmentation: the allocator has enough total free memory but cannot serve the next large request cleanly.
3. Workspace spikes: a fused kernel, attention backend, or MoE path asks for a temporary buffer much larger than usual.
4. Optimizer or cache bursts: the step fits most of the time, then a later phase produces a short-lived peak.

The playbook is to identify which bucket you are in before changing everything at once.

## Step 1: separate fragmentation from real exhaustion

Look at allocator retry counts, inactive split bytes, and the largest failed allocation request. If retries are climbing and inactive split memory is large, the run is fragmentation-bound. If retries are low but the requested allocation itself is too large relative to free space, the run is truly out of memory.

That distinction matters because the fixes are different. Fragmentation problems usually respond to allocator configuration or a slightly smaller peak burst. True exhaustion requires reducing retained state or batch geometry.

## Step 2: locate the largest activation surface

The next question is where the peak comes from.

- If the largest temporary is attention workspace, reduce the number of layers using the most memory-hungry attention backend or lower the sequence-related pressure that drives workspace size.
- If the largest temporary is inside MoE, check whether dispatch scratch and expert activations are being reused efficiently.
- If the largest burst appears during the optimizer step, focus on optimizer partitioning rather than activation checkpointing.
- If the peak comes from evaluation or serving mixed into the same process, check KV-cache growth before touching training settings.

The fastest way to waste time is to tune recompute when the real culprit is a cache or optimizer burst.

## Step 3: apply the cheapest structural fix first

Once the peak surface is known, fix it in order of cost.

### Allocator-level fixes

If fragmentation is the issue, use allocator settings that favor expandable segments and avoid unnecessarily aggressive segment splitting. These changes are low-risk and should be confirmed before reshaping the model.

### Activation-memory fixes

If attention dominates, use selective recompute or reduce the set of layers using the heaviest attention backend.

If MoE dominates, prefer selective expert recompute over full-block checkpointing. Replaying dispatch and collective-heavy paths is usually too expensive.

If recurrent or Mamba-style blocks dominate, look for narrow in-module recompute before wrapping the whole block.

### Optimizer-state fixes

If the burst appears on the optimizer step, verify that optimizer state is actually partitioned as intended. Silent regressions here can multiply the optimizer footprint by the data-parallel degree.

### Cache and serving fixes

If a long-context evaluation or serving path shares the node, cap KV cache explicitly. Unbounded cache growth is one of the most avoidable OOM sources in mixed workloads.

## Step 4: verify buffer reuse and temporary sizing

Large scratch buffers are often supposed to be reused. When reuse breaks because shapes drift between calls, every block allocates a fresh buffer and the run drowns in temporaries.

That is especially important for MoE dispatch scratch and attention workspace. Check whether metadata stays shape-stable across layers and whether temporary buffers are sized to actual per-rank token counts rather than pessimistic worst-case maxima.

## Step 5: keep the debug loop minimal

A good OOM debug loop is short.

1. Run one step with allocator and memory debugging enabled.
2. Read retry counts, inactive split bytes, and top allocation sites.
3. Decide whether the failure is fragmentation, activations, workspace, optimizer, or cache.
4. Change one class of fix at a time.
5. Re-run and compare the same signals.

Per-step memory tracing is usually too expensive to leave on. A targeted early-step snapshot is usually enough to identify the dominant pressure source.

## What we generally keep

- Selective activation recompute as the default, rather than full checkpointing everywhere.
- Partitioned optimizer state for larger training jobs.
- Allocator settings that reduce fragmentation under bursty demand.
- Explicit KV-cache limits on mixed training and evaluation workloads.
- Shape-stable temporary-buffer reuse on MoE and attention-heavy paths.

## What we avoid by default

- Full checkpointing everywhere.
- Over-provisioned temporary buffers sized to worst-case token counts when actual traffic is much lower.
- Long-running, per-step memory tracing during normal training.
- Mixing serving-style KV growth into a training node without an explicit cap.

## Fast triage checklist

| Signal | Likely issue | First move |
| --- | --- | --- |
| High allocator retries and large inactive split memory | Fragmentation | Favor expandable segments; reduce peak burst slightly |
| Largest allocation inside attention workspace | Attention backend pressure | Lower heavy-backend usage or sequence-related pressure |
| Largest allocation inside MoE scratch | Dispatch or expert temporary growth | Verify scratch reuse and actual token-based sizing |
| Peak arrives on optimizer step | Optimizer-state burst | Verify optimizer partitioning |
| Peak arrives during co-located eval or serving | KV-cache growth | Apply an explicit KV cap |

The point of the checklist is not to be exhaustive. It is to keep you from treating every OOM as the same bug.
