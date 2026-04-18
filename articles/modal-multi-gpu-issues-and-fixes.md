---
title: "Modal Multi-GPU Pain and the Fixes That Actually Landed"
description: "NCCL topology, GPU isolation, eviction and OOM-kill behavior, observability gaps, and the guide we follow when a Modal multi-GPU job hangs on the first forward pass."
date: "2026-04-18"
tags: ["modal", "multi-gpu", "nccl", "fsdp2", "h200", "b200", "runbook"]
---

Single-GPU Modal is boring: pick an H100, H200, or B200, warm the inductor cache, and launch. Multi-GPU Modal is where the interesting failure modes live. Our experience is captured plainly in a Modal multi-GPU status note: the full pytest suite passes on Modal H100, single-GPU training on H100, H200, and B200 runs our depth-52 hybrid preset end-to-end, and B200 is roughly 1.55x faster than H100 on the same preset. But 8-GPU FSDP2 with regional `torch.compile` hangs during the first forward pass. This post is about why it hangs, what we changed in the main training entrypoint to stop it hanging in most cases, what we still cannot fix from userspace, and the guide we follow when a multi-GPU job goes silent.

## Why MegaCpp cares about this

Modal is where we do ad-hoc benchmarking and multi-GPU smokes between long production runs on reserved H200 systems and TPU slices. "Does this new block compile cleanly under FSDP2 on 8 H200s" and "does this kernel change actually scale past 2 GPUs" are the questions we want Modal to answer quickly. A multi-GPU training surface that hangs on the first forward pass 30% of the time is not useful for that; it is actively misleading, because a hung job burns the same per-GPU-hour budget as a successful one. At roughly $36/hour for 8x H200 and $50/hour for 8x B200, a single wasted 2-hour hang costs real money, and two in a row turns the tool into a trust problem.

## What we built in the POC

The multi-GPU surface is the same training function as the single-GPU one. GPU count is part of the Modal GPU spec (`H200:8`, `H100:4`, `B200:8`); the function body branches on `torch.cuda.device_count()` to set distributed-mode environment variables before importing torch. The distributed mode is computed from GPU count and parallelism settings and returns one of `single`, `ddp`, `tensor_parallel`, `ddp+tensor_parallel`, or `fsdp2`, and the choice is recorded in the benchmark record so we can later correlate failures by mode.

The environment hardening for multi-GPU lives in a single block at the top of `_train_impl`. When `n_gpus > 1`, we set:

- `TRITON_DEFAULT_NUM_STAGES=2`: caps Triton's shared-memory budget so autotune does not request more than the SM90 hardware limit. Without this, the largest GEMM config (`triton_mm` at block 128, num_stages=4) asks for 262144 bytes of shared memory against a 232448-byte hardware limit on H200, and autotune reports "No valid triton configs".
- `TORCHELASTIC_ERROR_FILE set to an ephemeral runtime error-report file`: forces torchelastic to dump structured rank failure data to a known location. Without this, a single rank crashing shows up as a generic NCCL timeout on the other ranks and nothing else.
- `TORCH_DISTRIBUTED_DEBUG=INFO` (never `DETAIL`): DETAIL wraps every NCCL process group in a `ProcessGroupWrapper` that creates a secondary Gloo PG for validation collectives. That extra Gloo PG deadlocks FSDP2's `fully_shard()` init on Modal containers because (a) validation collectives can block forever if any rank diverges on an init code path, (b) the production training entrypoint downgrades DETAIL to INFO, but that happens after `dist.init_process_group()` so the wrapper is already installed, and (c) Modal's container IPC restrictions interfere with the extra Gloo PG.
- `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, `FSDP_USE_ORIG_PARAMS=true`: standard FSDP2 hardening.
- `TORCH_NCCL_ENABLE_MONITORING=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`: the NCCL heartbeat monitor otherwise fires after 600 seconds of no collectives and kills the job. On a cold inductor cache on a 4B hybrid preset, compile warmup can easily run 15 minutes without a collective. We disable the heartbeat for the compile phase and rely on an extended timeout as a safety net.
- Under FSDP, autotune is forced off (`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0`). The Triton autotune subprocess opens its own CUDA context and competes for HBM with the 4B model; on DDP we observed the subprocess OOM while benchmarking large matmuls (32768x65536 bf16 = 8.6 GB), all configs return `inf`, and training crashes with "No valid triton configs".

On the data side, multi-GPU runs pre-copy shards from the cloud-bucket FUSE mount into a local Modal volume before starting training. The cloud-bucket FUSE layer is fine for single-GPU sequential reads; under parallel reads from 8 worker processes it produces sporadic stalls and occasional corrupt reads, and the easiest fix is to not read from FUSE during training.

The sparse-attention and FA4-adjacent multi-GPU lanes are separate. There is a dedicated FA4-under-FSDP2 scaling path with its own benchmark schema, so a failed FSDP2 hang does not contaminate whole-model throughput claims.

## How it lands in MegaCpp

The production version keeps the environment hardening as-is, but lifts it out of the training prologue into a small shared module so the nightly runner, the FA4 validation lane, and the sparse-validation lane get the same settings without copy-paste drift.

The FSDP2 plus regional-compile hang on 8x H200 is the single blocker that is not fixed from userspace. The documented root cause in the multi-GPU status notes is cold-inductor-cache divergence: Triton JIT compilation takes different wall times on different ranks, one or more ranks enters a NCCL collective before the laggards, and the collective deadlocks. The same code works on reserved H200 systems because the inductor cache is already warm from previous runs. The production fix has three layers, only the first of which is userspace:

1. Force a warm inductor cache on the Modal volume before any 8-GPU launch. `_seed_inductor_cache` is already layered (checkpoint-volume tarball, cloud-bucket tarball, Docker-baked seed, per-blob API download). Production adds a gate: multi-GPU launches refuse to start if the cache is under a size threshold, and pre-seed automatically from a tarball captured on a successful 8-GPU run.
2. Bake the last-known-good inductor cache into the Docker image itself (`/opt/inductor_cache_seed`). This turns "cold volume on a fresh deployment" from a guaranteed hang into a slow-but-successful first launch, and the volume gets populated from there.
3. Offer a reduced-complexity fallback preset for Modal multi-GPU work: fewer MoE experts and no MoD. The point is not to ship that preset; the point is to have a diagnostic preset that isolates the compile-divergence problem from the correctness problem when we debug a new hang.

What is dropped: we do not attempt sequential rank-by-rank compile warmup. It sounds tempting until you remember that FSDP2 changes the compile graph when `fully_shard` wraps modules, so warming rank 0 and then running FSDP init on all ranks warms the wrong graph.

Observability becomes a real contract. Production runs emit a structured benchmark record with `distributed_mode`, `gpu_type`, `gpu_count`, NCCL env snapshot, inductor-cache size at start and end, and per-rank first-forward-pass wall time. The benchmark record is written to the checkpoint volume on every shutdown path, not just success, so the watchdog can pick it up even on crashes. Modal's built-in logs and app state transitions remain the ground truth for "did the container die", but we do not rely on them for "why".

## Ablations and what we kept

Most of the multi-GPU pain showed up in the March audit. Of the 20 audited H200 training records, 5 were multi-GPU failures: 4 DDP 2-GPU crashes and 1 FSDP2 4-GPU crash. All 5 cascaded from problems on individual ranks that presented as NCCL timeouts on the survivors. None of those were "NCCL itself is broken"; all of them were "one rank died, NCCL correctly reports a collective timeout, and we have no benchmark record to tell us which rank or why".

What survived contact with 8-GPU H200 work in practice:

- `TRITON_DEFAULT_NUM_STAGES=2` under multi-GPU: non-negotiable; without it autotune cannot produce a valid config.
- Autotune off under FSDP: the subprocess OOM is reproducible, and disabling autotune entirely is a ~30% throughput hit that we accept in exchange for a stable compile path.
- NCCL heartbeat monitoring off during compile, extended heartbeat timeout as safety net: the 600-second default was the single most common cause of "job dies at minute 10 with no useful error".
- Structured rank-failure dump via `TORCHELASTIC_ERROR_FILE`: the only way we get per-rank context on a multi-GPU death.
- Pre-copying shards to the local data volume: parallel FUSE reads fail, and we stopped trying to make them not fail.
- `TORCH_DISTRIBUTED_DEBUG=INFO`, never `DETAIL`: DETAIL turns FSDP2 init from "works" into "deadlocks on Modal containers".

What did not survive: relying on the default NCCL heartbeat, `DETAIL` distributed debug, FUSE parallel reads, subprocess autotune under FSDP, one-shot cache seeding without a size gate, and trusting Modal's default observability to tell us which rank actually failed.

On topology: Modal exposes multi-GPU nodes as peers on a single host, not as a multi-node InfiniBand fabric. That is strictly simpler than our own H200 hosts, which have IB between nodes, and it means we do not have to fight NCCL's IB transport selection on Modal. NCCL falls back to NVLink + shared-memory intra-node transports, which on 8x H200 is exactly what we want for FSDP2 and DDP. We deliberately do not publish cross-node multi-GPU claims from Modal; those live on our own fabric.

## Guide: when a Modal multi-GPU job hangs

When an 8x H200 FSDP2 job goes silent, we follow this order:

1. Check `modal app list` / `modal app logs <app_id>` for the last collective or compile log line on any rank. "Last line is a Triton compile message on some ranks and a collective on others" is the canonical compile-divergence signature.
2. Pull the benchmark record from the checkpoint volume. If there is a benchmark record, `distributed_mode`, `inductor_cache_size_mb_start`, and per-rank first-forward wall times tell us whether this is compile divergence or a single-rank death.
3. Check `TORCHELASTIC_ERROR_FILE` if any rank reached the error handler. One rank with a real traceback plus N ranks with NCCL timeouts is a single-rank death; N ranks all still compiling is compile divergence.
4. Inspect the inductor-cache volume size. Under the production threshold means the seed step did not run or did not find a seed source; the job should not have started.
5. If it is compile divergence, stop the app, confirm the cache is warm (pre-seed from the last good tarball if needed), and relaunch.
6. If it is a single-rank death (OOM, disk-full, kernel import failure), fix that cause first. The rank-death signature in `_runtime_missing_fused_kernels` output or the torchelastic dump almost always tells us which.
7. If neither fits, downgrade to the diagnostic reduced-complexity preset to confirm whether the hang reproduces on a simpler graph. If the diagnostic preset also hangs, the issue is infrastructure, not model; file a Modal support ticket with the app id, function call id, GPU spec, and the benchmark record.

We do not use `modal run ... --detach` for these launches; the supported contract is explicit `app.run(detach=True)` from Python launchers with saved app and call identifiers so the collector can find the run later.

## Production checklist

- Multi-GPU launches refuse to start on an inductor-cache volume under the production size threshold. Seed first, launch second.

Environment settings we enforce on multi-GPU Modal launches:

| Variable                              | Value      | Why                                  |
|---------------------------------------|------------|--------------------------------------|
| `TRITON_DEFAULT_NUM_STAGES`           | `2`        | caps shared-memory budget on SM90    |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM`     | `0` (FSDP) | autotune subprocess OOMs at 8x       |
| `TORCH_NCCL_ENABLE_MONITORING`        | `0`        | avoid 10-min heartbeat kill on compile |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`    | `7200`     | safety net for long compile warmup   |
| `TORCH_DISTRIBUTED_DEBUG`             | `INFO`     | `DETAIL` deadlocks FSDP2 init        |
| `TORCH_NCCL_AVOID_RECORD_STREAMS`     | `1`        | standard FSDP2 hardening             |
| `FSDP_USE_ORIG_PARAMS`                | `true`     | standard FSDP2 hardening             |
| `TORCHELASTIC_ERROR_FILE`             | set        | structured per-rank failure dumps    |

The helper that applies them before importing torch:

```python
# shared entry helper (sketch)
def harden_multi_gpu_env(n_gpus: int) -> None:
    if n_gpus <= 1:
        return
    os.environ["TRITON_DEFAULT_NUM_STAGES"]        = "2"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"]     = "0"
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"]  = "1"
    os.environ["FSDP_USE_ORIG_PARAMS"]             = "true"
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
    os.environ.setdefault("TORCHELASTIC_ERROR_FILE",
                          "<runtime error-report path>")
    if is_fsdp():
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
```

- Bake a last-known-good inductor cache into the base image as a seed directory for fresh-deployment recovery.
- `TRITON_DEFAULT_NUM_STAGES=2`, autotune off under FSDP, `TORCH_NCCL_ENABLE_MONITORING=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` generous, `TORCHELASTIC_ERROR_FILE` set. These are applied by a shared helper, not copy-pasted.
- `TORCH_DISTRIBUTED_DEBUG=INFO`. Never `DETAIL` on Modal multi-GPU.
- Pre-copy training shards to the local data volume on first use. Never read training data through FUSE in parallel.
- Each multi-GPU lane writes a structured benchmark record including `distributed_mode`, `gpu_type`, `gpu_count`, inductor-cache size, per-rank first-forward wall time, and NCCL env snapshot. Benchmark records are written on every shutdown path, not just success.
- One detached app per multi-GPU benchmark; saved `app_id` and `function_call_id`. No grouped spawning for heavy benchmarks.
- A diagnostic reduced-complexity preset (fewer experts, no MoD) lives next to the production preset for debugging hangs.
- Cross-node multi-GPU claims come from our own IB fabric, not from Modal.

## References

- training and benchmark entrypoints
- FA4 scaling and parity helpers
- detached benchmark and sparse-validation launchers
- watchdog and benchmark-record reader utilities
- multi-GPU status and benchmark planning notes
- [NCCL documentation — docs.nvidia.com/deeplearning/nccl]
- [PyTorch FSDP2 design notes — pytorch.org/docs]
- [Modal Labs documentation — modal.com/docs]
