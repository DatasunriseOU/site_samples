---
title: "Modal as a Third Training Surface Alongside Our H200 Hosts and TPU Slices"
description: "Why MegaCpp uses Modal for ad-hoc training and benchmark jobs, how the image, GPU, volume, and secret story is wired, and when Modal wins against reserved H200/TPU capacity."
date: "2026-04-18"
tags: ["modal", "infrastructure", "h200", "b200", "tpu", "benchmarks"]
---

We run training and benchmarks on three surfaces: our own reserved H200 hosts, our TPU v6e slices, and Modal. Modal is not the cheapest per GPU-hour and it is not the fastest to warm up, but it is the only one of the three that lets one engineer launch a clean multi-hundred-GB training or benchmark job in under a minute without coordinating with anyone. This post is about where that trade actually lands for us: how we wire Modal into the POC, what the image / GPU / volume / secret model looks like, and the specific regimes where Modal beats reserved capacity and where it loses.

## Why MegaCpp cares about this

Our training fleet is a mix of long-horizon runs on reserved capacity and short, bursty, embarrassingly-parallel work: kernel ablations, FA4 promotion checks, sparse-attention backend parity sweeps, preset sanity smokes before a long run, and regression benchmarks after every kernel change. The long runs belong on our own H200 hosts and TPU slices, where we control the NCCL topology, the inductor cache, the checkpoint fabric, and the scheduler. The bursty work does not; it lights up for an hour, needs a clean environment, and wants to scale out to N independent containers rather than up to one big one.

Modal is good at exactly that second shape. It gives us a programmatic app surface, detached execution, persistent named volumes, and on-demand H100/H200/B200 class GPUs without us having to reserve them. The argument for paying the premium was simple: one engineer hitting `app.run(detach=True)` from a workstation and getting a clean, reproducible H200 in seconds is worth more than a 20% hourly markup versus reserved hosts, as long as we confine Modal to the work shapes where that matters.

## What we built in the POC

The Modal surface in the POC is not "one harness". a Modal benchmark plan enumerates three distinct lanes, and we honor that separation in code:

1. Whole-model training benchmarks via `modal_train.py` and `modal_matrix.py`.
2. Exact-token sparse detached benchmarks via `modal_bench_dsa_backend_detach.py` and `modal_bench_dsa_backend_collect.py`.
3. Sparse validation / FA4 promotion via `modal_sparse_validation_detach.py` and `modal_sparse_validation.py`.

Each lane has its own manifest contract, its own notion of "success", and its own bookkeeping (`app_id`, `function_call_id`, launch flags, steady-state metrics, backend identity, promotion status). The plan doc is explicit that collapsing these into one generic "Modal story" is how you lie to yourself about what the receipts mean, and we wrote the launchers accordingly.

The whole-model training lane is anchored in `modal_train.py`. A single `modal.App` owns the image, the GPU spec, the volume mounts, and the entrypoint function. The GPU class is chosen by environment variable (`H100`, `H100:8`, `H200`, `B200`, `A100-80GB:4`, and so on), which means the same code path covers everything from a 1x H100 smoke to an 8x H200 multi-GPU launch. We default to H200 for training because its 140+ GB HBM lets a depth-52 hybrid preset with MoE plus Mamba blocks actually fit on a single device; H100 is the default for cheaper ablations; B200 shows up when we care about absolute throughput (our own measurements put it around 1.5x H100 on the same preset).

Volumes are where Modal earns its keep. `modal_train.py` mounts four:

- a read-only cloud bucket for training data and pre-built wheels,
- a persistent checkpoint volume,
- a persistent inductor-cache volume,
- a persistent local data volume that we pre-copy shards into for multi-GPU runs.

The separation is deliberate. The cloud-bucket mount is fine for single-GPU reads but suffers under parallel FUSE traffic, so multi-GPU launches pre-copy shards into the local data volume on first use and then read from there. Checkpoints and inductor cache live on their own named volumes so that a new container starts warm: `reload()` before the run, `commit()` after, and the next launch inherits both the autotune cache and the last checkpoint.

Secrets are boring on purpose. Cloud-bucket HMAC credentials live in one named Modal secret, referenced by name from `modal_train.py`, and are wired into the bucket mount via Modal's `CloudBucketMount`. We do not bake credentials into the image, we do not read them from the repo, and we do not pass them as function arguments.

The benchmark lanes layer on top. `modal_matrix.py` is the declarative matrix harness for the whole-model lane: it defines labeled cases ("depth-52 full current", "depth-52 no-autotune", "depth-52 no-gradient-checkpointing", DDP, FSDP2, TP=2, and so on), sets `NANO_... _MODAL_GPU` before importing the training module, enters `app.run(detach=True)`, spawns each case, and persists launch args plus parsed metrics to a report file. The sparse-bench lane (`modal_bench_dsa_backend_detach.py`) and the sparse-validation lane (`modal_sparse_validation_detach.py`) follow the same detach-and-collect pattern but with their own manifest schemas. We deliberately do not use Modal's `modal run ... --detach` local-entrypoint surface for these lanes; the supported contract is explicit `app.run(detach=True)` plus a collector that queries `function_call_id` later.

Observability is a mix. `modal_watchdog.py` polls Modal apps, detects `running -> stopped` transitions, pulls receipts off the checkpoint volume, and alerts on failure. `read_modal_receipts.py` and `check_modal_apps.py` are thin CLI wrappers over the same receipts. None of this replaces the broader training dashboards; it just closes the loop on "did that detached Modal job actually finish, and if not, why".

## How it lands in MegaCpp

The production codebase keeps Modal as a peer surface, not a replacement. A few concrete shifts:

- The image story moves from per-app ad-hoc builds to a single pinned base image published to our registry, rebuilt only when the training stack bumps. `modal_train.py` already supports this: the fast path pulls a pre-built base (`from_registry(...)`) and only overlays fresh repo code; the slow path (`..._MODAL_BUILD_FROM_SCRATCH=1`) reinstalls torch nightly, Triton, Mamba SSM, Flash Attention, Cut Cross Entropy, and the fused-kernel wheels from scratch and exists purely as a debug fallback. In MegaCpp, only the fast path is supported; the from-scratch builder stays in-tree as a recovery tool.
- GPU class selection becomes a typed enum on the launch side rather than a free-form env var, and the allowed set is pinned to H100 / H200 / B200 plus a short whitelist. We still let operators pick the class; we do not let them invent new ones.
- The four-volume layout is kept as-is: cloud bucket read-only, checkpoints, inductor cache, local data. The only change is that the cache-seeding logic in `modal_train.py` (`_seed_inductor_cache`) is lifted into a small shared module and reused by the nightly runner (`modal_train_nightly.py`), the FA4 receipt runner (`modal_fa4_receipt.py`), and the sparse validation lane.
- The three benchmark lanes stay separate. The production version enforces the separation with types rather than convention: each lane has its own manifest dataclass, its own collector, and its own report schema.
- `modal_watchdog.py` is promoted from "cron job on a laptop" to a proper service that writes receipts into our normal alerting pipeline rather than a local log.

What we are dropping: the one-off convenience harness (`modal_benchmark.py`) is documented in `MODAL_BENCHMARK_PLAN.md` as "not the source of truth for distributed H200 claims" and does not survive the move. The `modal run ... --detach` local-entrypoint path stays unsupported for the detached lanes.

## Ablations and what we kept

The CHANGELOG is unkind about this surface, and we want to be honest about what it showed.

A March audit of 20 whole-model Modal H200 receipts had every single run fail with zero training steps completed. The breakdown, by root cause:

- Inductor-cache disk full during `torch.compile`: 9 runs.
- Single-GPU OOM on H200 with a depth-52 4B hybrid preset (MoE with 64 routed experts, Mamba SSM forward): 6 runs.
- DDP 2-GPU crashes cascading from disk or OOM: 4 runs.
- FSDP2 4-GPU crash: 1 run.

The top blocker was not multi-GPU at all; it was ephemeral-storage pressure on `/inductor_cache` while compiling a 4B hybrid. The fix that actually stuck was the persistent inductor-cache volume plus the layered seeding in `_seed_inductor_cache` (checkpoint-volume tarball first, then a pre-warmed tarball from the cloud bucket, then a Docker-baked seed dir, then a per-blob API download as a last resort). A later measurement compared cold versus warm starts on the same preset and same H200: step 0 went from roughly 27 minutes on a cold cache to about 2 minutes on a warm one, and the total time-to-first-useful-step dropped more than an order of magnitude. That single change is the difference between Modal being usable for iterative work and Modal being a trap.

On the kernel side, `modal_fa4_receipt.py` is the canonical "is FA4 CuTe actually live on this image" smoke. The receipt is a 10-step training smoke on a depth-52 hybrid preset with MTP disabled to isolate the attention kernel, and it passes or fails on runtime backend detection rather than on throughput. `modal_fa4_backward_parity.py` extends that with a back-to-back comparison against a dense baseline and records loss/gnorm trajectories. Both exist because we learned the hard way that a clean import of FA4 is not the same thing as FA4 actually running in the backward pass.

What survived contact with real multi-tenant GPU scheduling, roughly in order of importance: persistent inductor cache, detached `app.run` with explicit `function_call_id` tracking, one detached app per heavy benchmark, pre-copying shards to the local data volume for multi-GPU, and strict per-lane manifest contracts. What did not survive: grouped spawning of heavy benchmarks in one app, relying on cloud-bucket FUSE for parallel reads, and trusting a single convenience harness to answer both "is throughput healthy" and "is this backend promoted".

## Production checklist

- Pin Modal to three workloads: ad-hoc training smokes, benchmark matrices, and kernel/backend promotion checks. Long production runs stay on our own H200 hosts and TPU slices.

Three lanes, three manifest contracts:

| Lane                              | Launcher                                  | Success criterion              |
|-----------------------------------|-------------------------------------------|--------------------------------|
| Whole-model training benchmarks   | main training and benchmark entrypoints   | steady-state tok/sec + benchmark record |
| Exact-token sparse detached bench | `modal_bench_dsa_backend_detach.py`       | per-backend parity + tok match |
| Sparse validation / FA4 promotion | `modal_sparse_validation_detach.py`       | promotion status recorded      |

The Modal image and volume wiring, sketched:

```python
# main training entrypoint (sketch)
app = modal.App("training-runner")
image = modal.Image.from_registry("training-base:<pinned-digest>")

@app.function(
    image=image,
    gpu=os.environ.get("GPU_SPEC", "H200"),
    volumes={
        "/data":           modal.CloudBucketMount(...),
        "/checkpoints":    modal.Volume.from_name("ckpt"),
        "/inductor_cache": modal.Volume.from_name("ind"),
        "/local_data":     modal.Volume.from_name("local"),
    },
    secrets=[modal.Secret.from_name("bucket-hmac")],
    timeout=60 * 60 * 6,
)
def train(cfg: dict) -> dict: ...
```

- One base image per training-stack version, published to our registry, with fast-path overlay for fresh repo code and a documented from-scratch fallback.
- GPU class is an enum; the allowed set is H100, H200, B200 plus a short whitelist.
- Four persistent volumes, no exceptions: cloud bucket read-only, checkpoints, inductor cache, local data.
- Secrets are named Modal secrets referenced by name; never passed as function args, never baked into images.
- One detached app per heavy benchmark. Record `app_id`, `function_call_id`, launch flags, and lane-specific manifest fields.
- Keep the three benchmark lanes separate in code, in manifests, and in reports. Do not collapse them.
- Watchdog the fleet: poll app state, pull benchmark records from the checkpoint volume, alert on failure with the benchmark record attached.
- Modal wins when the job is short, independent, and needs a clean environment. Reserved capacity wins when the job is long, tightly coupled, or depends on a warm topology.

## References

- training and benchmark entrypoints
- detached launcher and collector entrypoints
- sparse-validation entrypoints
- FA4 validation and parity helpers
- watchdog and benchmark-record reader utilities
- benchmark planning and multi-GPU status notes
- [Modal Labs documentation — modal.com/docs]
- [Flash Attention 3 — Dao et al., 2024]
