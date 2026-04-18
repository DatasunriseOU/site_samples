---
title: "Training on 8x H200 SXM: The Operator Playbook"
description: "End-to-end operator notes for driving an 8x H200 SXM node: topology, NCCL tuning, storage layout, and the invariants that keep a run from silently drifting."
date: "2026-04-18"
tags: ["h200", "nccl", "nvlink", "fsdp2", "torchrun", "training", "operations"]
---

An 8x H200 SXM node is a practical unit for training a mid-sized specialist model from scratch. On paper it looks like a larger-memory successor to H100. In practice the gap between a fresh machine and steady-state high-throughput training with reliable checkpoints is a sequence of small operational choices that, taken in the wrong order, cost days. This post focuses on that operator surface: how to drive the node, what the topology forces on the launch flow, which NCCL settings survived ablation, where state should live, and what is worth monitoring.

## Why the operator surface is the contract

MegaCPP training is not about one heroic run. It is about repeating comparable launches across the same hardware class. That is a discipline problem, not a performance problem. The launch surface is the contract. If two operators can produce different steady-state throughput on the same configuration because one forgot to pin a compile cache or left a stale NCCL setting behind, the comparison stops meaning anything within a week.

The H200's 141 GB of HBM also changes the shape of the decision surface. A 4B-8B specialist fits comfortably at 4K context, but the same model at 16K or with FP8 experts plus TE fused norms pushes us back into activation-memory pressure. We want one launcher that lands cleanly in both regimes and escalates predictably when it does not.

## Topology, launcher, and what the wrapper owns

The training stack should expose a single entry point, launched with `torchrun --standalone --nproc_per_node=8`. Around it, operators usually keep a thin shell layer that pins provenance, environment, and arguments for a given configuration. That split is deliberate: the training code owns correctness, while the shell layer owns the host contract.

Each rank owns one H200. The eight GPUs are fully connected by an NVLink/NVSwitch fabric that advertises 900 GB/s per-GPU NVLink bandwidth. For our training mix the fabric is fast enough that NCCL collectives are rarely the wall-clock bottleneck; the bottleneck is either compute (GEMM) or elementwise overhead (bf16 add/fill kernels dominate our traces). What the operator has to get right is that NCCL actually uses the fabric in the way we expect, and that nothing in the host image is silently routing a collective through the PCIe root complex.

A fresh node requires exactly one preflight: `nvidia-smi topo -m` must show NV-switch links between all pairs, not PHB. If you see PHB, stop, you are on a misconfigured host and every NCCL number you produce will be noise.

| Concern | Knob | What it does |
|---|---|---|
| Stream serialisation | `CUDA_DEVICE_MAX_CONNECTIONS=1` | Deterministic NCCL/compute interleave; required for PP P2P determinism |
| Async free | `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | Removes per-op `record_stream` syncs |
| NVLink-SHARP | `NCCL_NVLS_ENABLE=0` | Disables SHARP on single-host fabric where it stalls init |
| PP chunking | `NCCL_P2P_NET_CHUNKSIZE=524288` | Megatron-Bridge default once PP is on |
| Stream priority | `TORCH_NCCL_HIGH_PRIORITY=1` | Comm streams cannot be starved by compute |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Prevents MoE activation churn from fragmenting cache |
| Heartbeat | `TORCH_NCCL_ENABLE_MONITORING=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` | Survives 15-20 min Triton JIT compile windows |
| Inductor autotune GEMM | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0` | Stops Triton workspace OOMs from poisoning cache |

These survived ablation on our 8xH200 configuration and are now applied from Python at launch in the training entrypoint, so an operator who forgets to export them still gets the right floor. The CUDA-graph trees pair (`TORCHINDUCTOR_TRITON_CUDAGRAPHS=1`, `TORCH_COMPILE_CUDAGRAPH_TREES=1`) is also on by default; on a depth-20 1.2 B FSDP `dbs=16` bench we measured a small-single-digit-percent steady-state throughput lift with no correctness regression.

### What the wrapper owns

Provenance, environment, the `torchrun` line, and log extraction. A good launcher records the source revision, a short working-tree status, and the machine class into the log before the first Python import. That single practice catches many "mystery regressions" that later turn out to be uncommitted edits or host drift. The launch line should stay intentionally small:

```bash
exec torchrun --standalone --nproc_per_node=8 \
  -m <training-entrypoint> \
  --config "$CONFIG" --run_name "$RUN_NAME" \
  --device_batch_size "$DBS" --max_seq_len "$SEQ" \
  --compile_cache_dir "$CACHE" 2>&1 | tee "$LOG"
```

## State on disk and live monitoring

There are three categories of state and they should go to three places. Persistent training state such as checkpoints and evaluation outputs belongs on a durable high-capacity data volume. Per-process artefacts such as compile caches, autotune caches, and bootstrap logs belong in a per-run scratch directory so stale state cannot cross-contaminate a different experiment. Operator artefacts such as the merged log, launch summary, and environment dump belong with the launch materials for that run.

The invariant that should not drift is simple: the training data root must point at the dedicated data volume, not at a home directory on the boot disk. Teams eventually hit the same failure mode when this is left loose: the boot disk fills, checkpoint writes fail, and the next restart comes from a stale checkpoint. The fix is to assert the data-root location at launch.

Live monitoring can stay simple: one poller for GPU power, utilization, and memory; one parser for canonical training-step lines; and one rule. If someone inspects the machine while training is live, they should use tooling that does not create a CUDA context on a busy GPU. Query-only `nvidia-smi` calls are fine. Attaching a process that touches device state is not.

## What we keep verifying

Every serious benchmark run starts with a stack line: torch, fused-attention library, recurrent-block library, Triton, optional TPU runtime, and CUDA driver. Every serious benchmark run ends with a post-run summary: peak HBM per rank, sustained tok/sec over the steady-state window, a short communication slice from profiling, and the final loss. If any of those are missing, the run should not be used for comparison, regardless of what the wall clock said. This rule sounds bureaucratic; it has saved teams from "hero numbers" that later turned out to come from a stale stack.

The other invariants are negative ones. No CUDA Graphs around expert-parallel paths until the all-to-all dispatch fully replays. No `torch.compile(model)` whole-model on the hybrid; only per-block compile around attention and MoE blocks. No mixing of incompatible fused-normalization implementations in the same model tree. And no live `pip install` on a training host; if the stack changes, the image changes, the environment changes, and the benchmark record changes.

## What we kept and threw away

We kept the wrapper-plus-entrypoint split, the eight environment defaults above, the dedicated-data-volume convention, the per-run scratch isolation, the benchmark-summary rule, and the `nvidia-smi topo -m` preflight as a hard gate. We threw away whole-model `torch.compile`, `record_stream` reliance, NVLink-SHARP on single-host fabric, NCCL heartbeat at default, and any monitoring tool that opens a CUDA context against the live GPUs. None of this is novel. It is the operating contract that lets the 8xH200 node behave like a known quantity instead of a haunted house.

## Failure modes the playbook is calibrated to catch

Three classes of failure dominate on this node and each one has a documented detection path.

The first is silent allocator drift. With `expandable_segments:True`, the allocator no longer fragments catastrophically, but on long MoE runs the gap between `reserved` and `allocated` still grows slowly because the dispatch buffer cycles through different shapes per expert popularity wave. The detection is a per-step `torch.cuda.memory_stats` poll the bench launcher writes to a JSONL file; the dashboards alarm when reserved drift exceeds a configurable bound. The action is to step `dbs` down at the next checkpoint boundary, not mid-run, because mid-run resizes invalidate the Inductor cache for the affected blocks.

The second is NCCL heartbeat near a long compile. We removed the canonical 10-minute kill by raising the heartbeat to 7200 seconds, but a `torch.compile` phase that exceeds even that budget still fails. The detection is the launcher's own elapsed-time print: any compile that takes longer than half the heartbeat triggers a warning, the launcher saves a partial benchmark summary, and the next launch is instructed to use a smaller per-block compile budget via equivalent CUDA-side controls. We have seen this on deep hybrid configurations with FP8 enabled across many blocks; keeping the first and last layers in BF16 shortened compile time enough to fit the budget.

The third is collective stalls under expert-parallel. NCCL all-to-all on the MoE dispatcher occasionally stalls when an expert wave is so imbalanced that one rank holds most of the dispatched tokens. The detection is a kernel trace collected automatically every N steps in benchmark mode; the alarm is when the all-to-all slice exceeds a configurable share of step time. The action is to widen the load-balancing loss weight or, in extreme cases, swap the dispatcher variant for the affected configuration. These interventions belong in public-facing change notes, because they explain why a configuration changed and what signal forced the change.

## What the run-summary collector actually consumes

A run summary collected at the end of every training launch should consume four streams: the merged training log, the structured stats stream, the per-step memory snapshots, and the kernel trace if one was collected. It should produce one machine-readable report with the software stack, hardware class, topology, launch environment, steady-state throughput window, peak HBM per rank, communication slice from the profile, and final loss. That report is what dashboards consume, and it is also what teams diff when a regression appears.

The collector refuses to produce a summary if any of those streams is missing. The launcher refuses to start a real training run if the bench preflight, the environment dump, or the stack line is incomplete. Together these two refusals are the entire reason the comparison matrix means anything: a run either produces a complete summary or it does not exist for comparison purposes.

## References

- the MegaCPP training entrypoint and launch summaries
- topology and throughput notes from the H200 training lane
- public training-state notes
