---
title: "CPU Offload and Startup Memory Calibration on H200 and GB10"
description: "How MegaCpp picks microbatch and offload knobs at boot, the zero-copy pinned offload paths, the AdamW-only optimizer offload trade-offs, and what shipped versus what stayed experimental."
date: "2026-04-18"
tags: ["CPU Offload", "Memory", "H200", "GB10", "Calibration"]
---

The two memory questions before every CUDA training run are "what microbatch fits" and "what stays on the GPU". The answer is not a one-time benchmark; it is a calibration loop that combines a feature-aware estimator, a persistent records catalog, and two distinct offload paths (activation and optimizer state). On H200 we mostly use the loop to confirm headroom; on GB10 it is the only thing standing between us and a 47-minute compile-then-OOM cycle. This post walks through how the loop is wired, what the offload mechanisms actually cost, and which knobs remained useful in sustained training.

## Why MegaCpp cares about this

H200 has 141 GB of HBM3e per device and a fast NVLink fabric; GB10 (DGX Spark, sm_121a) has 128 GB of unified memory and consumer-tier interconnect. The same model definition with the same flags runs differently on the two. A representative configuration may also enable a long list of memory-hungry features at the same time: hybrid Mamba-3 plus Transformer blocks, multi-token prediction (MTP), engram memory, structure embeddings, n-gram hash tables, FIM masking, and the eight-specialist routing. A naive "set device_batch_size and pray" approach used to lose us a full GB10 day per misconfigured launch.

The calibration loop replaces that with three artifacts: a pre-flight estimate that accounts for every enabled feature, a persistent JSONL catalog of past launches and their OOM outcomes, and a small set of escape hatches (automatic fit search, AdamW offload, activation offload) the runtime can pull when the estimate is tight.

## What we built in the POC

The pre-flight estimator carries a hardware table with usable HBM per chip for the SKUs we care about — H200: 141.0 GB, GB10: 128.0 GB unified, plus B200, GB200, H100, A100, L40S, and T4 — and a dtype byte table that knows about bf16, fp8, and nvfp4 (0.5 bytes per element). The estimate is feature-aware: it walks the configuration, adds parameter bytes per layer including the right Mamba state-space contribution and MoE expert count, optimizer-state bytes for AdamW plus Muon (Muon's two momentum buffers vs AdamW's two moments), gradient bytes, and activation bytes per microbatch, then subtracts a configurable headroom reserve. The Mamba contribution is the trickiest because the SSM state caches scale with `seq_len * d_state * n_groups` rather than with `n_embd` alone; that audit caught a 6 GB blowup on the depth-52 hybrid preset before any GPU touched it.

The automatic fit search explores the joint space of tensor parallelism, expert parallelism, data parallelism, device batch size, and checkpointing for valid combinations that fit. On CUDA we pin checkpointing off in this search because Nemotron Nano 3 trains 30B without recompute and our shape behaves the same way. The candidate scoring is not "minimize peak HBM"; it is shape-aware. Large multi-GPU CUDA MoE runs prefer sharded-expert meshes (target around 16 local experts per device on 8x H200, dropping toward 8 on larger hosts) before they chase extra DP, because EP=1 maximizes DP on paper but recreates the routing/gather OOMs have already been observed. The selection reason is recorded with each candidate so you can inspect why one combination beat another.

The startup calibration loop makes the process persistent. Each boot writes a JSONL record with the schema version, hardware shape, requested config, estimator output, and runtime outcome. We also maintain a small set of retryable memory failures that mean "the estimator was too optimistic, retry with safer knobs" rather than "there is a logic bug"; that list covers compile-time HBM OOMs, runtime CUDA OOMs, XLA resource exhaustion, and Triton shared-memory or resource limits. Historical records are then used to demote shapes that have already OOMed on the same hardware. CUDA and XLA history stay separated so one backend does not pollute the other's fit decisions.

The runtime introspection layer dumps `torch.cuda.memory_stats()` (current and peak allocated/reserved/active, allocation retries, OOM count, inactive split bytes) and the top-N allocations from `torch.cuda.memory_snapshot()` when the snapshot recorder is active. It is gated behind a debug-only environment flag and prints once per phase boundary. The fragmentation signal (allocation retries plus inactive split bytes) is what catches the slow-creep OOMs that look like leaks but are really allocator fragmentation under FSDP2 reshard churn.

Meta-device initialization removes the double-allocation tax. We build the model on `torch.device("meta")`, then move it to empty CUDA storage, then run weight initialization in place. We never allocate the full model on CPU first. For the depth-52 hybrid preset this is the difference between "boot fits in the calibrator's headroom budget" and "OOM during construction before the first forward". The same idiom makes resume-from-checkpoint cheap because we materialize empty and then load weights in shards.

The system has two offload mechanisms.

The activation-offload path has two modes. The coarse mode is an Unsloth-style autograd `Function` that replaces gradient checkpointing for a whole transformer block: instead of discarding the saved hidden state and recomputing, it copies it to pinned CPU memory and pulls it back during backward. Recomputation is gone; PCIe traffic takes its place. The fine-grained mode uses `torch.autograd.graph.saved_tensors_hooks` to selectively offload saved tensors above a `min_numel` threshold from registered submodules. Both use async CUDA streams to overlap D2H/H2D with compute. This is CUDA-only; XLA/TPU keeps standard gradient checkpointing.

The optimizer-offload path wraps an existing AdamW optimizer in a CPU-backed stepper. The contract is deliberately narrow: Muon parameters always stay on the GPU because orthogonalization needs fast matmul; AdamW parameters are candidates for offload based on `offload_fraction` and the largest tensors go first to maximize per-byte savings. Steps copy gradients D2H async, run the CPU AdamW step, copy params back H2D async. The wrapper preserves the `torch.optim.Optimizer` interface so the training loop does not branch. A guard returns the original optimizer unchanged when `offload_fraction <= 0` or CUDA is unavailable.

## How it lands in MegaCpp

The estimator and the calibration catalog graduate almost as-is. The estimator becomes the boot-time gate: MegaCpp refuses to launch a CUDA training job whose estimated peak exceeds 95% of the device's usable HBM, period. A debug-only override exists for kernel development, but it is not used in standard launch flows. Automatic fit search becomes the default microbatch picker; the operator only specifies a global token budget and the search picks a tensor/expert/data-parallel layout plus device batch size deterministically.

The startup calibration store lifts as the cross-platform record layer; the CUDA wrapper merges into the production calibrator as a lightweight namespace. Records persist under the MegaCpp data root and are never indexed by identifying metadata, so they replicate cleanly across hosts.

Meta-device initialization is the lifted idiom. Every model construction in MegaCpp goes through the meta-device path; we removed the legacy "build on CPU then `.to(cuda)`" code paths.

The activation-offload split survives. Whole-block offload becomes the supported coarse primitive. Fine-grained saved-tensor offload ships as an opt-in that is wired only on GB10, where unified memory makes the D2H/H2D path effectively free for tensors above the size threshold; on H200 it is available but rarely beats plain checkpointing. The runtime flag for this path enumerates its target regions explicitly rather than trying to infer them from the model.

The CPU AdamW offload path remains available but is used selectively. It ships behind a feature flag, defaults off on H200 (where AdamW state for the eight specialists fits comfortably), and defaults to `offload_fraction=0.5` on GB10 for the largest specialists. We do not offload Muon, ever. The CPU step's float64 reduction tail can be a few percent of step time when offload_fraction is high, so the practical rule is: only offload enough AdamW state to keep peak headroom above 8 GB; do not chase the maximum.

What stayed experimental: the multi-tier offload scheme that paged optimizer state across pinned host memory, NVMe, and a remote shared store. The complexity-versus-savings ratio was bad and the failure modes were operationally awful. The MegaCpp production codebase has two tiers — GPU and pinned host — and that is the contract.

## Ablations and what we kept

The estimator audit history is the most useful artifact in this whole stack. Every time the estimator under-counted, we either added the missing feature term or tightened the headroom reserve. The current model accounts for 26+ features explicitly: Engram, MoE (Token Choice plus shared plus null experts), Mamba-3, DSA, mHC (intra plus cross-layer HC), MTP, MoD, n-gram hash, structure embeddings, Tree FFN, relation bias, NCP, TOP, GateSkip, FIM/IFIM, FIRE/ReDo/DASH, gradient checkpointing, FSDP, TP, EP. The "feature-aware" tag is not marketing; missing any of them and the estimate is wrong by gigabytes.

Calibration on H200 mostly confirms what `auto_fit` already picked. The catalog still earns its keep because the same shape that fits at compile time can OOM during a particular curriculum phase when sequence length grows or when MoE token routing concentrates. Recording those events lets the next launch start from a safer microbatch automatically.

Calibration on GB10 is the load-bearing story. The unified-memory ceiling is hard, the compile takes long enough that retry costs money, and the consumer-tier silicon (sm_121a, no `tcgen05`, no TMEM, ~128 KiB physical SMEM/SM versus 228 KiB on H100/H200) means the kernel mix differs and the activation footprint is not transferable from H200. The catalog has caught:

- A Mamba-3 SSM-state blowup at the 8K-context curriculum step, fixed by demoting microbatch by 1.
- A first-compile temporary that fits in 141 GB on H200 and does not in 128 GB on GB10. The retry rule (compile-time OOM bumps the headroom floor by 8 GB on the next attempt) prevents the second 47-minute compile from happening.
- A pinned-buffer collision between the CPU-offload optimizer step and the FSDP2 bucket buffers; the rule is now "offload_fraction must leave at least 8 GB of pinned-host headroom for NCCL".

The activation-offload ablation set: on H200, the coarse whole-block offload path is roughly compute-equivalent to standard checkpointing because PCIe is the bottleneck and we have plenty of headroom anyway; it remains useful for the rare configuration where recompute breaks compile. On GB10, the unified-memory path makes activation offload genuinely cheap; it is useful for the attention block in long-context training and saw the long-context compile fit where it would not have otherwise.

The optimizer-offload ablation set: `offload_fraction=0.0` is the default everywhere. At 0.5 on GB10 we recover roughly enough HBM to enable the next-larger microbatch, but only after the AdamW CPU step is parallelized over OpenMP threads (single-threaded was a 12% step regression). At 1.0 we lose more in step time than we gain in throughput. The largest-first selection rule matters; an even split across param groups loses the largest-tensor benefit and ends up paying overhead per group with no proportional savings.

What we threw out:

- Predictive offload schedulers that promised to swap optimizer state in and out per layer. Too clever, too fragile, never beat the simple "split AdamW once at boot" rule.
- A learned estimator trained on past records. The hand-written feature accounting is more accurate and more debuggable.
- Estimator runtime calibration ("compile, measure, redo"). The compile cost makes it useless on GB10; on H200 the headroom is wide enough that it never fired.

## Public checklist

- Always boot through the automatic fit search; do not hand-set `device_batch_size` for routine launches.
- Refuse to launch when estimated peak exceeds 95% of usable HBM.
- Build models via `create_model_on_device` (meta-device + `to_empty` + `init_weights`); never allocate the full model on CPU first.
- Persist every launch into the calibration catalog with outcome and OOM type; rerank future candidates from the history.
- Treat the listed `RETRYABLE_MEMORY_FAILURE_TYPES` as estimator misses, not bugs; bump the headroom floor on retry.
- Keep Muon parameters on the GPU; never route them through the CPU optimizer wrapper.
- Cap AdamW offload at the fraction that keeps at least 8 GB of pinned-host headroom for NCCL.
- Enable fine-grained activation offload only on GB10 and only for explicitly enumerated submodule targets.
- Gate memory-debug dumps behind a debug-only environment flag; record allocation retries and inactive split bytes per phase to catch fragmentation creep.
- Two offload tiers only: GPU and pinned host. Do not ship NVMe or remote tiers.

## Calibration snapshot

| Hardware | Offload path | Primary use |
|----------|--------------|-------------|
| H200 | activation offload (pinned) | claw back activation memory when PCIe idle |
| H200 | AdamW optimizer offload | fit larger `dbs` on memory-binding presets |
| GB10 | activation offload | required on tight-memory presets |
| GB10 | optimizer offload | bypasses the compile-then-OOM cycle |

```python
# boot-time calibration: pick microbatch from the estimator plus probe
from cppmega.memory import estimator, probe
plan = estimator.plan(model, preset)
dbs = probe.fit(model, plan, headroom_gb=8)
```

## References

- public memory-estimation, calibration, debug, meta-init, and CPU-offload modules in the MegaCpp repo
- [PyTorch CUDA caching allocator and `torch.cuda.memory_stats` — pytorch.org]
- [Unsloth gradient checkpointing offload — Unsloth blog]
- [Megatron-LM HybridDeviceOptimizer (NVIDIA + Alibaba PAI) — Megatron-LM]
- [Nemotron Nano 3 training memory analysis — NVIDIA technical report]
