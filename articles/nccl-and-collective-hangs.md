---
title: "NCCL and collective hangs: the H200 multi-host timeout playbook"
description: "Allreduce stragglers, NCCL deadlocks, P2P env vars, ibverbs quirks, and the liveness/timeout playbook we run on MegaCpp's H200 multi-host CUDA lanes."
date: "2026-04-18"
tags: ["nccl", "h200", "distributed", "megacpp"]
---

Most of the genuinely expensive debugging on our H200 fleet was not about the model. It was about NCCL: allreduce stragglers, bootstrap failures, plugin regressions, watchdog timeouts firing in the wrong place, and recovery paths that replaced one failure mode with another. This post is the playbook we landed on: the env vars we set, the ones we unset, the retry logic in the training entrypoint, and the liveness rules we enforce before declaring a run healthy.

## Why this matters

NCCL failures are bimodal. Either the run never starts (you bisect an env var that should have been unset at the launcher), or it starts, looks healthy for some minutes, and then a collective on rank 3 quietly waits forever for rank 5 to finish a kernel that is itself waiting for rank 3. Both look identical in the launcher: the parent process sits there with no useful message. The watchdog either fires too early (during compile warmup) or too late (after a real hang). Every fix has a blast radius: an env var that saves a single-host lane breaks the multi-host one, a heartbeat extension that survives compile lets a real hang sleep through the night.

Compile warmup on our dense+MoE preset takes long enough that NCCL defaults assume something has died. We had to teach the watchdog the difference between "compute is busy" and "the communicator is dead," teach the launcher to scrub plugin envs that leak in from the host shell, and teach the retry path which failure classes are eligible for a lazy-init second attempt.

## 1. The operating environment

The CUDA side runs on H200 hosts with 8 GPUs each. Training ranges from single-host `H200:8` to multi-host jobs of up to 4x`H200:8`. The fabric differs by host: some sit on plain IP with NCCL's default socket path; others have a vendor multi-node plugin present in `LD_LIBRARY_PATH`. Both had to work, and both had to fail gracefully when the plugin environment leaked into a single-host run.

PyTorch is 2.12 nightly (`2.12.0.dev20260304+cu130`), NCCL is whatever ships with that wheel, Triton 3.6, Python 3.13. DDP is the production path for most rungs; FSDP2 lanes and expert-parallel MoE lanes add their own collective patterns on top.

## 2. The five hang classes we learned to name

Everything we dealt with falls into one of these buckets:

1. **Communicator bootstrap failure.** `ncclRemoteError`, `socketPollConnect ... Connection refused`, `remote process exited`, `Failed to initialize any NET plugin`. The ranks never get a working communicator; the job dies before step 0.
2. **Watchdog firing during compile.** `torch.compile`'s Triton JIT takes 15-20 minutes on the larger dense+MoE preset, and NCCL's default 600 s heartbeat watchdog kills ranks that do not run a collective during compile. The surviving ranks then hang on the next collective.
3. **Allreduce stragglers.** One rank is 100-300 ms slower than the others per step, usually on a specific GPU, sometimes correlating with a specific PCIe slot. The collective blocks the fast ranks; effective throughput falls to the straggler's rate.
4. **Plugin / env mismatch.** A multi-node NCCL tuner config leaks into a single-node bench and the bootstrap fails with the plugin error above. Or the opposite: a single-node config runs on a multi-host lane and IB never gets used.
5. **Coalesced-op unsupported.** `Backend nccl does not support allgather_into_tensor_coalesced`. Not a hang, but it looks like one: the run gets past bootstrap, into compile warmup, and then dies a few tens of seconds later with that string. We saw it on replay rungs more than on mainline.

## 3. Env vars: what we set, and why

The default environment policy lives in the main training entrypoint and applies to every CUDA rank unless the operator has set the variable by hand. The defaults are:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1
TORCH_NCCL_AVOID_RECORD_STREAMS=1
NCCL_NVLS_ENABLE=0
NVTE_DP_AMAX_REDUCE_INTERVAL=0
NVTE_ASYNC_AMAX_REDUCTION=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_NET_CHUNKSIZE=524288
TORCH_NCCL_HIGH_PRIORITY=1
TOKENIZERS_PARALLELISM=False
```

`CUDA_DEVICE_MAX_CONNECTIONS=1` serialises CUDA streams so NCCL overlaps with compute deterministically; without it we saw intermittent backward stalls. `TORCH_NCCL_AVOID_RECORD_STREAMS=1` removes a sync per collective. `NCCL_NVLS_ENABLE=0` is the single most impactful line for bootstrap reliability on hosts without NVLink Sharp; the probing stall masquerades as a hang. `NCCL_P2P_NET_CHUNKSIZE=524288` matters more for pipeline parallelism than for pure DDP, but the cost of setting it defensively is zero. `TORCH_NCCL_HIGH_PRIORITY=1` puts NCCL streams on the high priority so compute/comm overlap is not starved.

Two more env vars are set specifically for regional-compile FSDP2 runs (`_configure_regional_compile_nccl_timeouts`):

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=3
TORCH_NCCL_BLOCKING_WAIT=1
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=<compile_timeout_minutes*60>
NCCL_TIMEOUT=<compile_timeout_minutes*60>
```

`ASYNC_ERROR_HANDLING=3` aborts on NCCL error rather than raising asynchronously: fail-fast, not a ghost collective. `BLOCKING_WAIT=1` makes collectives block instead of spin; combined with the long heartbeat timeout, it gives us clean tracebacks instead of watchdog-abort mystery. We also monkey-patch `torch.distributed.new_group` so every process-group created by FSDP2 or Megatron-style optimisers inherits the compile-era timeout; without the patch, internal groups got the default short timeout and fired the watchdog during compile warmup.

## 4. The env vars we unset

Bench hosts often inherit a multi-node IB tuning env from the login shell. On a single-node 2-GPU lane that env is poison. The relevant symptom is a `Failed to initialize any NET plugin` early in bootstrap. The fix is a `sanitize_single_node_nccl_env` function that unsets the leaked vars and forces a plain intra-node path:

```bash
unset NCCL_NET
unset NCCL_CROSS_NIC
unset NCCL_NET_GDR_LEVEL
unset NCCL_TUNER_CONFIG_PATH
unset NCCL_IB_ADAPTIVE_ROUTING
unset NCCL_IB_FIFO_TC
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_TC
unset NCCL_NVLS_CHUNKSIZE
unset NCCL_P2P_NET_CHUNKSIZE
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
```

And we scrub the vendor IB lib path out of `LD_LIBRARY_PATH` before launching. `IB_DISABLE=1` and `NET_PLUGIN=none` are aggressive; they are right for single-node lanes and wrong for multi-host. The launcher knows which it is, so the sanitiser only runs where it belongs. The multi-host flavour keeps IB on and the plugin present but still sets `NVLS_ENABLE=0` and the heartbeat envs from the defaults. We do not set `NCCL_DEBUG=INFO` by default; only on bench quartet runs where we are actively chasing a bootstrap issue.

## 5. The watchdog timeout story

This is the one we got wrong first, more than once. The default 600 s heartbeat kills ranks during compile. Classic fix:

```bash
TORCH_NCCL_ENABLE_MONITORING=0
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
```

These three land in `base_train` automatically when `LOCAL_RANK` is detected. With them, the DDP+compile lane went from "hanging and NaN" to a steady-state DDP-on-MoE throughput band on the recompilation-fix receipt.

What went wrong first: we disabled monitoring on lanes that should have kept it on. `TORCH_NCCL_ENABLE_MONITORING=0` makes NCCL stop exporting the heartbeat signal, so a real hang looks exactly like a slow compile. On a two-hour replay we only learned that the wrong way. Current policy: monitoring is off only during compile warmup, and we re-enable it after. The long heartbeat window (7200 s) stays on for the whole run; cheap insurance, and compile can re-trigger on a retry.

Second mistake: applying these vars uniformly to retry re-execs. Plain DDP lanes were fine; expert-parallel lanes needed `MEGACPP_SKIP_CUDA_BOOTSTRAP_BARRIER` and lazy NCCL init, and generic CUDA retry re-execs crashed early under lazy init when eager would have worked. The landed policy in the main training entrypoint is:

| Retry class | NCCL init | Bootstrap barrier | Watchdog envs |
| --- | --- | --- | --- |
| Generic CUDA retry child | Eager | Standard | Standard |
| Expert-parallel retry child | Lazy | Skipped | Standard |
| Known retry-eligible startup error | Lazy | Skipped | Standard, retry once |

The narrow eager-NCCL retry matcher now treats `ncclRemoteError`, `remote process exited`, `socketPollConnect`, and `Connection refused` as lazy-retry-eligible bootstrap failures. That list was built from reproducible bootstrap receipts on H200 hosts.

## 6. Liveness checks we actually run

The rules we enforce before calling a run "healthy":

- **Compile warmup completed.** The log contains a `compile_warmup` timing line. Without it, no step-0 claim is valid. Typical warmup values on the 8-GPU DDP lane after cache warm are in the low tens of seconds; cold is a couple of minutes.
- **Step 0 is compile-contaminated.** We classify `step 00000` as contaminated by default. It routinely shows numbers in the low thousands of tok/sec on a stack that will steady-state in the high six figures of tok/sec. Any bisect that uses step 0 as a data point is discarded.
- **Step 1 is the first real number.** Step 1 + step 2 + step 3 is the minimum receipt. A single-step receipt is a rumour.
- **Peak memory printed.** If the receipt does not include peak memory, we do not use it for throughput bisects. Peak memory catches the "accidental FSDP resharding" class that looks fine on tok/s but shows a 2-3x memory swing.
- **End marker present.** Runs that die inside compile often leave plausible-looking partial logs. A run without a clean end marker is classified as a startup/compile stall, not a throughput receipt.

The contract is visible in concrete receipts. Successful historical replays come back with a `compile_warmup` line in the tens of seconds, a contaminated step 0 in the low thousands of tok/sec, then steps 1, 2, and 3 climbing into the steady-state band, with peak memory in the high teens of GiB on the H200:8 lane. Without the NCCL timeout fixes and the compile-warmup tolerance we would have classified some of those receipts as hung.

## 7. Stragglers, the coalesced-op class, and multi-host

Stragglers we handle structurally. We keep `TORCH_NCCL_HIGH_PRIORITY=1` and `TORCH_NCCL_AVOID_RECORD_STREAMS=1` on, wrap reduce-scatter chains in `OverlappedGradReducer.wait_all()` with the coalescing manager (opt-out via `MEGACPP_DISABLE_COALESCING_MANAGER=1`), and pad bucket sizes via `pad_buckets_for_high_nccl_busbw=True` aligned to 65536 elements. When a specific rank is consistently slow we swap the GPU on the host; PCIe contention and cooling variance are real, and "re-rank" is not a software fix. We watch per-rank step time in the log and the `/status` API; when the delta exceeds roughly five percent steady-state we investigate. Rare, and always physical.

The `allgather_into_tensor_coalesced` class looks like a hang and is not. Some historical commits called the coalesced variant; the PyTorch+NCCL combo on the replay host did not support it. The process crashes cleanly after compile warmup with a clear backend message. Fix: commit-family classification, not an env var. Bisects landing in this class are treated as known-bad; the replay moves to an earlier rung.

Multi-host lanes add the trouble you would expect: bootstrap takes longer (NVLS off, IB on, init timeout matched to heartbeat); `NCCL_P2P_NET_CHUNKSIZE=524288` is a reasonable starting point and on narrower inter-host links we have tuned down, never up; `TORCH_NCCL_BLOCKING_WAIT=1` matters more than on single-host because async errors across hosts are harder to attribute. We do not publish a production multi-host receipt; that work is still exploratory.

## 8. Operator rules we now enforce

1. Never inherit IB env into a single-node lane. Scrub at the launcher, always. The sanitiser runs unconditionally; if the lane needs IB, the launcher re-enables it explicitly after.
2. Never kill a run while a collective is outstanding. Ranks left mid-collective take the NCCL communicator with them and the next attempt fails bootstrap. Always prefer the control-plane stop (the `/control` API) over `kill -9`.
3. Never trust a single-step receipt. Step 0 is compile-contaminated, step 1 is the first real number, three steps is the minimum.
4. Always print the stack line at start: `torch.__version__`, `flash_attn.__version__`, `triton.__version__`, and the NCCL major version from `torch.cuda.nccl.version()`. Without it, post-mortem is guesswork.
5. Retry on known bootstrap errors only. The eager-NCCL retry matcher is the source of truth. New failure classes get a regression test before they get added.
6. Keep the watchdog window greater than compile warmup. 7200 s is overkill for most runs and the right number when warmup is variable and the cost of being wrong is a full restart.

## What we kept and what we threw away

We kept the env defaults, the regional-compile timeout overlay, the `new_group` monkey-patch that propagates the long timeout to internally-created groups, the single-node sanitiser and its multi-host counterpart, the narrow eager-NCCL retry matcher, the lazy-init retry path for expert-parallel children, the long heartbeat window for the whole run, and the five-rule liveness contract. We never `kill -9` a rank mid-collective.

We threw away disabling monitoring globally (now back on after warmup), uniform retry policy across lane types, blanket `NCCL_DEBUG=INFO`, and chasing stragglers in software when the real fix is a GPU swap. We do not currently run PyTorch's NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`); it would have shortened several "which rank hung first" investigations and we should wire it in. A small per-rank step-time watchdog emitting a structured event on deviation is the other obvious next step. The rest of the playbook stays.

## References

- https://docs.pytorch.org/docs/stable/distributed.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
