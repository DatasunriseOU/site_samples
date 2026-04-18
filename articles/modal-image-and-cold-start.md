---
title: "Modal Image Construction and the Cold-Start Tax We Actually Pay"
description: "How we layer the Modal training image, why every wheel is pinned to the training stack, how persistent volumes absorb the inductor-cache hit, and the 30-90s startup tax we accept as the price of burst compute."
date: "2026-04-18"
tags: ["modal", "docker", "cold-start", "inductor-cache", "triton", "h200"]
---

A Modal container is only "fast" if you already paid for everything expensive somewhere else. For a 4B hybrid training image built on a torch nightly plus Triton plus Mamba SSM plus Flash Attention plus FA4 CuTe plus Cut Cross Entropy, "somewhere else" is a registry-hosted base image and a small set of persistent volumes. This post walks through how the POC constructs that image, why every wheel is pinned to the training stack, how the inductor cache volume turns a 27-minute cold start into a 2-minute warm one, and which parts of the 30-90 second residual startup tax we just accept.

## Why MegaCpp cares about this

We use Modal for work shapes where the startup-to-useful-work ratio matters: a 10-step FA4 smoke, a 100-step preset sanity check, a throughput regression after a kernel change. If container startup costs 20 minutes on a 10-minute job, Modal is a worse version of reserved H200 systems. If startup is under 90 seconds, Modal is a strictly better tool for that shape of work. The entire image-and-volume design in the POC is organized around pushing the expected startup below that threshold for the jobs we care about, while accepting a worse tail for the first run after any stack bump.

Secondarily, the training stack is a minefield of implicit compatibility constraints. Torch nightly version X wants Triton version Y; Mamba SSM expects a specific CUDA toolchain; Flash Attention wheels are not ABI-stable across Python minor versions; FA4 CuTe DSL wheels are built against a specific torch+cu combo. The image is the single place where all of that lock-in happens. Breaking it on Modal while leaving the long-lived H200 training systems pinned to the working combination is a recipe for two different definitions of "the model" depending on which surface launched the run. We do not want that.

## What we built in the POC

The canonical Modal training image lives in `modal_train.py`. There are two paths, and only one of them is used in anger.

The fast path pulls a pre-built base image from our registry via `modal.Image.from_registry(...)`. The base image contains Python 3.13, the pinned torch nightly on cu130, the pinned Triton nightly, Mamba SSM, Flash Attention, causal-conv1d, the FA4 CuTe DSL wheels, Cut Cross Entropy, and the usual cast of training dependencies (tokenizers, datasets, wandb, google-cloud-storage, transformers, numpy, pyarrow, einops). On top of that, the fast path installs `git`, pip-installs the accelerated model-architectures package directly from git, nukes any stale repo copy baked into the image, and then `add_local_dir(".", "/app/<repo>", copy=True, ignore=_ignore_modal_repo_path)` layers the current working tree in. The `ignore` predicate strips `.git`, `.venv`, `.cache`, `.codex`, `.omx`, `data`, `docs`, `experiments`, `reports`, `tests`, `tools`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, large log files, and a short list of runtime-generated files. The goal is that only source code diffs ship across the wire between runs, so the repo-code layer is small and cache-friendly.

The slow path, guarded by an environment variable, builds the image from `ubuntu:24.04` with Python 3.13, apt-installs `gcc`/`g++`, pip-installs the torch and Triton nightly wheels from a pinned `download.pytorch.org` URL (the exact cu130 cp313 manylinux filename is hard-coded), then the core deps, then the training deps. It exists for two reasons: so we can reproduce the base image from scratch when we do need to bump a pin, and so we have a recovery path when the registry is temporarily unreachable. It is explicitly not the "exact cu130 + FA4 parity image" and we do not use it to produce training receipts.

The wheel pinning matters. The Mamba SSM Triton kernel, the Flash Attention 3 wheel, and the FA4 CuTe wheels all have to match the exact torch / Triton / CUDA combination, and at the same time the benchmark scripts (`modal_fa4_receipt.py`, `modal_fa4_backward_parity.py`, `modal_fa4_fsdp_scaling.py`) assume specific fused-kernel imports succeed at runtime. `modal_train.py` has an explicit `_runtime_missing_fused_kernels()` helper that tries to import `causal_conv1d`, `mamba_ssm.ops.triton.ssd_combined`, and `flash_attn` at the start of a run and surfaces any import failure in the receipt rather than letting it turn into a mystery crash 30 seconds into step 0. If a wheel did not actually install or the ABI shifted, the receipt says so.

The persistent volumes are the other half of cold-start engineering. The training function declares four:

- A read-only cloud bucket mount for datasets and pre-built wheels (including the FA4 CuTe wheel directory and the torch-nightly wheel mirror).
- A named checkpoint volume for model checkpoints and run logs.
- A named inductor-cache volume mounted at `/inductor_cache` and wired into `TORCHINDUCTOR_CACHE_DIR`, with `TORCHINDUCTOR_FX_GRAPH_CACHE=1` and `TORCHINDUCTOR_AUTOGRAD_CACHE=1`.
- A named local-data volume that we pre-copy training shards into on first use so that multi-GPU workers do not beat on the cloud-bucket FUSE in parallel.

The inductor-cache volume is the one that actually bends the cold-start curve. Compiling a depth-52 hybrid model with attention, MoE, Mamba-3, and M2RNN blocks produces thousands of inductor entries (fx graphs, AOT-autograd decompositions, Triton kernels) and doing that from scratch on an H200 takes upwards of 25 minutes. `modal_train.py` keeps the volume warm across runs and, when the volume is empty on a fresh deployment, seeds it via `_seed_inductor_cache` in a layered fallback order:

1. A tarball written to the checkpoint volume by the previous run's `finally` block.
2. A pre-warmed tarball on the cloud bucket mount (single-file read, which the FUSE layer handles fine).
3. A seed directory baked into the Docker image itself (`/opt/inductor_cache_seed`).
4. A per-blob API download as a last resort.

Each fallback is an actual code path, not aspirational. The first one that produces a non-empty cache wins and the rest are skipped. The contract is that tarballs are created with `tar czf ... -C / inductor_cache` so the archive members are rooted at `inductor_cache/` and we extract to the parent of the cache directory.

## How it lands in MegaCpp

The production version keeps the overall shape and tightens the edges.

The pre-built base image becomes the only supported path. It is versioned by training-stack hash (torch nightly + Triton + Mamba SSM + Flash Attention + FA4 CuTe + Cut Cross Entropy), published to our registry under that hash as a tag, and pinned from the Modal app by digest rather than by `:latest`. The from-scratch builder stays in-tree as a reproducibility and recovery tool but is not used to produce receipts.

The `add_local_dir` overlay grows a tighter ignore predicate and a small, typed allow-list of directories that must be copied. Shipping a stray 300 MB report tree or a forgotten `.beads` cache into the image is a real way to turn a 10-second code overlay into a 90-second one, and the only defense is to keep the allow-list boring and regression-tested.

The fused-kernel import probe (`_runtime_missing_fused_kernels`) moves from a print-side-effect into a structured field on the run receipt. The existing ad-hoc backend-detection regex in `modal_fa4_receipt.py` (which had to gain a fallback matcher for the DSA runtime observation JSON when the original pattern missed `"actual_backend": "fa4"`) is replaced by a single typed runtime-observation record emitted by the training loop itself, so receipts do not have to scrape logs to decide whether FA4 actually ran.

Volume layout is unchanged. Cache seeding is extracted out of `modal_train.py` into a small shared module so the nightly runner, the FA4 receipt runner, and the sparse-validation lane all use the same layered seeder, and so new launchers cannot accidentally skip it.

What moves to kernel/Triton/Pallas paths is not the image itself but the work the image has to compile: as more blocks graduate to hand-written Triton or Pallas kernels, the inductor surface shrinks and the cold-start tax shrinks with it. The image design is deliberately symmetric so that an image built with more kernelized blocks is simply smaller at compile time, not restructured.

A new feature flag, off by default, lets operators skip the cloud-bucket tarball fallback and the Docker-baked seed and go straight to "empty cache, compile from scratch". It exists so we can measure cold-start in isolation on demand; it is not intended for production use.

## Ablations and what we kept

The honest version of the cold-start story is in the CHANGELOG. The 20-run H200 training audit in mid-March had 9 of 20 failures attributed to `/inductor_cache` disk-full during `torch.compile`, not to bad model code. That is what motivated the persistent inductor volume in the first place; before that, ephemeral container storage was simply not large enough to hold the compiled cache for a 4B hybrid preset.

The cold-versus-warm comparison we ran on the same preset and same H200 class is the cleanest measurement we have. Cold (empty cache): step 0 around 27 minutes, dominated by inductor compilation and autotune. Warm (volume populated from the previous run): step 0 around 2 minutes, dominated by CUDA context setup, container boot, and a small amount of lazy backward-graph compilation that `torch.compile(fullgraph=False)` defers from step 0 to roughly the first 100 steps. That lazy-backward behavior is the reason the fresh-image path shows a second compile spike around step ~100 even when step 0 looks warm: NAM-style hybrid models introduce graph breaks at every MBlock (`torch.compiler.disable` on the Mamba path), so each subgraph's backward is a new FX graph with new matmul shapes, and AOT-autograd compiles them incrementally. FX graph cache prevents re-tracing but not re-autotuning, because autotune is keyed by kernel source and backend hash.

Multi-GPU autotune was its own separate lesson. On H200 DDP with 8 GPUs, the Triton autotune subprocess spawned its own CUDA context, competed with the main training process for HBM, and started requesting shared-memory blocks beyond the SM90 hardware limit. The receipts looked like "No valid triton configs". The fix in `modal_train.py` is to force `TRITON_DEFAULT_NUM_STAGES=2` on multi-GPU, disable autotune entirely under FSDP (`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0`), set `TORCH_NCCL_ENABLE_MONITORING=0` plus a generous `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` so the NCCL heartbeat does not kill the process during the 15-minute compile warmup, and pointedly not set `TORCH_DISTRIBUTED_DEBUG=DETAIL` (which wraps NCCL process groups in a Gloo-backed validator that deadlocks FSDP2 init under Modal's restricted container IPC).

What survived: a pre-built pinned base image, a typed fused-kernel import probe, `add_local_dir` with an aggressive ignore predicate, a layered inductor-cache seeder with four fallbacks, and single-file tarball reads off the cloud-bucket mount. What did not survive: building from scratch as a default, relying on ephemeral `/tmp` for the inductor cache, trusting FUSE for parallel shard reads, and any form of DETAIL-level distributed debug on multi-GPU.

The residual 30-90 second tax we accept is roughly: container schedule and boot (a few seconds to tens of seconds depending on GPU-class availability), image pull for any unchanged layers (seconds on a warm registry), local-code overlay (a few seconds for a clean overlay, tens of seconds if the ignore predicate drifted), CUDA context init and torch import, and the irreducible minimum of lazy compile. We do not try to beat that; we try to keep it inside the 90-second upper bound.

## Production checklist

- One pre-built base image per training-stack hash, pinned by digest, published to our registry. No `:latest` in production.

Cold vs warm startup on the same H200 preset:

| Phase                         | Cold cache | Warm volume |
|-------------------------------|------------|-------------|
| Container schedule + boot     | seconds - tens of seconds | same |
| Image pull (unchanged layers) | seconds on warm registry  | same |
| Local-code overlay            | seconds                   | same |
| CUDA context + torch import   | seconds                   | same |
| Inductor compile + autotune   | ~27 min at step 0         | deferred  |
| Lazy backward compile         | spike around step ~100    | spike ~100 |
| Steady-state step time        | recipe-dependent          | recipe-dependent |

The four-volume mount layout the production image assumes:

```python
# modal_train.py (sketch)
volumes = {
    "/data":            modal.CloudBucketMount(...),      # read-only corpus + wheels
    "/checkpoints":     modal.Volume.from_name("ckpt"),   # persistent checkpoints
    "/inductor_cache":  modal.Volume.from_name("ind"),    # torch.compile cache
    "/local_data":      modal.Volume.from_name("local"),  # pre-copied shards
}
```

- The from-scratch builder stays in-tree as a reproducibility tool; it is never the path that produces receipts.
- Wheels for torch, Triton, Mamba SSM, Flash Attention, causal-conv1d, FA4 CuTe, and Cut Cross Entropy are pinned to exact filenames matching the training stack. Bumping one means rebuilding the base image and revving the hash tag.
- `add_local_dir` uses an aggressive, tested ignore predicate. The code-overlay layer must be small enough that it does not dominate startup.
- Four volumes, always: cloud bucket read-only, checkpoints, inductor cache, local data.
- Seed the inductor cache via the layered fallback (checkpoint-volume tarball, cloud-bucket tarball, Docker-baked seed dir, per-blob API download). Never start a production run on an empty cache unless you are deliberately measuring cold start.
- Pre-copy shards to the local data volume on first use; do not read training data in parallel through FUSE.
- Fused-kernel imports are probed on entry and recorded in the receipt. A "healthy" run with a silently missing kernel is a bug.
- Multi-GPU environment: `TRITON_DEFAULT_NUM_STAGES=2`, autotune off under FSDP, NCCL heartbeat monitoring disabled, `TORCH_DISTRIBUTED_DEBUG=INFO` (never `DETAIL`).
- Budget: aim for sub-90-second warm-start for iterative work; budget 20-30 minutes for the first run after a stack bump and treat it as a cache-seeding pass.

## References

- `modal_train.py`, `modal_train_nightly.py`, `modal_matrix.py`, `modal_benchmark.py`
- `modal_fa4_receipt.py`, `modal_fa4_backward_parity.py`, `modal_fa4_fsdp_scaling.py`
- `modal_bench_dsa_backend.py`, `modal_bench_dsa_backend_detach.py`, `modal_sparse_validation.py`
- `MODAL_BENCHMARK_PLAN.md`, `MODAL_MULTI_GPU_STATUS.md`
- [Modal Labs documentation — modal.com/docs]
- [Flash Attention 3 — Dao et al., 2024]
- [Mamba-3 — Gu et al., arXiv:2603.15569]
