---
title: "Dynamo and torch.compile Breakage on a Mamba-3 Hybrid"
description: "Graph breaks, recompile storms, guard explosions, and cache-hygiene rules we landed while keeping torch.compile useful on MegaCpp's hybrid Mamba-3 + Transformer stack."
date: "2026-04-18"
author: "David Gornshtein"
tags: ["torch.compile", "dynamo", "mamba", "MegaCpp"]
---

MegaCpp's training core is a hybrid: attention blocks (`ABlock`), Mamba-3 SSM blocks (`MBlock`), expert blocks (`EBlock`, MoE), and Engram blocks in a repeating pattern across 52 layers. The TPU/XLA compile story lives in its own post; this one is specifically about Dynamo and Inductor on CUDA. Which graph breaks we silenced, which we accepted, which dynamic-shape guards we set by hand, and the compile-cache hygiene we now treat as non-negotiable.

## Why this matters

On a hybrid model the difference between a compiled run that merely boots and one that actually pays off is tens of minutes of first-compile wall clock and a low single-digit percent steady-state tax that compounds across the whole training wave. Dynamo's defaults assume small, homogeneous models; a 52-block mixed-architecture net with an opaque Triton kernel per Mamba layer, top-k MoE routing, and a padded expert dispatch path is the exact shape those defaults were not designed for. The failure modes are easy to misread - a recompile storm looks like a hang, a guard explosion looks like a NaN, an autotune OOM looks like a kernel bug - and the tooling tells you about each one only by printing at the wrong level. What follows is the set of knobs, disable points, and cache rules that finally made `torch.compile` a net positive for us on CUDA.

## 1. Ground rules we ended up with

Six configuration lines in the main training entrypoint do most of the work. They are set before any compile happens, before Dynamo is allowed to trace anything:

```python
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.enable_compiler_collectives = False
```

Each line is a scar.

`capture_scalar_outputs=True` lets Dynamo trace through ops that return Python scalars (`.item()`-adjacent patterns in MoD/MoE bookkeeping) without forcing a graph break on the site. We do not want that - we want the break to move into a `torch.compiler.disable` region we control - but we want the choice, not whatever Dynamo would have done by default.

`cache_size_limit=64` and `accumulated_cache_size_limit=256` exist because our deep hybrid preset has 52 compiled blocks; the per-callsite default of 8 guarantees a cache-eviction storm. The accumulated cap is the total budget across every compiled callsite in the process. Hitting either triggers the `recompile_limit=64` log line, and once you see it the next step will be slow: Dynamo silently falls back to eager on the callsite that hit the limit.

`automatic_dynamic_shapes=False` plus `assume_static_by_default=True` is the main load-bearing pair. Letting Dynamo auto-promote dims to dynamic after it sees two values is how you get a run where step 1 compiles for 20 minutes, step 2 recompiles for 12, and step 3 recompiles again because `dbs` wiggled by one during a gradient-accumulation warmup. We mark dynamic dims by hand, explicitly, and only where we want them. Concretely, the first batch axis is marked with `torch._dynamo.maybe_mark_dynamic(t, 0)` on warmup inputs; everything else stays static by construction.

`enable_compiler_collectives=False` disables the 2.12-era experimental knob that tries to coordinate Dynamo across DDP ranks. It interacts poorly with our regional compile setup and produced ranks with divergent guard trees, which then deadlock on a collective that one rank decided to inline and another did not.

## 2. The graph breaks we accepted

We run `fullgraph=False`. That is a choice.

`MBlock.forward` is permanently wrapped in `@torch.compiler.disable`. We tried the alternatives - per-block compile with the Mamba kernel allowed to recompile, `allow_in_graph(mamba_chunk_scan_combined)`, splitting MBlock into a compiled outer shell and disabled inner - and each one eventually failed. The Dynamo-traces-through-it path crashed on `allow_in_graph` in torch 2.10 because Dynamo still walked into the Triton kernel and hit `.data_ptr()` on a FakeTensor proxy. Per-block compile made the Mamba graph breaks worse, not better, because each block carried its own guard tree and the guards did not all match across blocks of the same type.

So: `MBlock` is black-boxed. The whole-model compile runs with breaks at each Mamba layer. We count them and cap them. On our deep hybrid preset that is 13 breaks per forward. Each break costs a little (sync plus Python dispatch reentry); collectively, on H200:8 DDP, it is a fixed one to two percent tax measured steady state.

Four other disable points exist inside the main model runtime module and friends, gating things Dynamo cannot safely see:

- state mutations in the API server hand-off surface
- the DTensor-safe embedding dispatch (`dtensor_safe_embedding = torch.compiler.disable(...)` in `dtensor_utils.py`)
- a fused MoE entry that forces a graph break because `F.grouped_mm` goes through a path Inductor cannot fuse across EBlock boundaries
- the `score_mod` wrapper that adapts around the current FA4 CuTe softcap ABI mismatch

None of these are "nice to have" disables. Each earned its decorator by crashing a run.

## 3. The graph breaks we fought

### The MoE overflow counter

The worst graph-break incident was not a graph break. It was a recompile storm that looked like one.

`MoE._overflow_total` was a Python `int`. It incremented every `forward()`. Dynamo specialized on its value. Every step produced a new guard, a new cache key, a new compile. On 8-GPU DDP the behavior manifested as "compile hangs and NaN", and the team misdiagnosed it as a Muon bf16 interaction and worked around it for weeks with `MEGACPP_NO_COMPILE=1`.

The real root cause was Dynamo hitting `recompile_limit=64` on the MoE callsite, falling back to eager for MoE, disagreeing with DDP's reducer about which parameters had run, and producing silent grad-sync drops. Converting the counter to `register_buffer` (the Megatron-Core pattern) makes it a tensor, takes it off the guard path, and restores a stable compile cache. The fix recovered thousands of tokens per second on 8-GPU DDP that had previously been hanging or NaN-ing.

The lesson: any Python scalar touched by compiled code becomes part of the guard tree. If it increments, you have a time bomb.

### The padded MoE path

MoE dispatch wants dynamic shapes (per-expert token counts). Dynamo with `automatic_dynamic_shapes=False` does not want them. We reconciled the two with a padded dispatch: tokens are bucketed to the next power of two of expert capacity, the dense matmul runs on the padded shape, and a mask selects valid outputs. The padded path has a static shape, which means it compiles once per bucket size instead of once per observed distribution.

The tradeoff is explicit: roughly 25 percent padding overhead in the worst case, for a fully compilable graph that does not recompile when the routing distribution shifts. We measured the alternative (data-dependent shapes with `torch.compiler.set_stance("lazy")`) and it was worse in every dimension: slower steady state, longer first compile, unpredictable tail.

### The one dynamic axis we do mark

The global-batch dimension is the only genuinely dynamic axis in the training graph. Gradient accumulation, auto-fit retries, and the final-batch-of-epoch case all vary it. We mark it with `torch._dynamo.maybe_mark_dynamic(t, 0)` on the warmup step, exactly once, and `automatic_dynamic_shapes=False` prevents Dynamo from inferring any other dim as dynamic.

When Dynamo sees `mark_dynamic` on a tensor whose shape happens to match another tensor's shape it might have inferred as dynamic earlier, it will create a new symbolic int and try to reconcile. With `automatic_dynamic_shapes=False` that reconciliation does not happen, and the run stays on the static path. This is exactly what we want: the one dynamic axis is opt-in, not Dynamo-inferred.

## 4. Compile cache hygiene

### Where the cache lives

`TORCHINDUCTOR_CACHE_DIR` is set explicitly at import time in the main training entrypoint. It defaults to the project cache root, falls back to a process-private temporary cache if that is unwritable, and is reported in the status API.

`TORCHINDUCTOR_FX_GRAPH_CACHE=1` and `TORCHINDUCTOR_AUTOGRAD_CACHE=1` are enabled in the H200 bench launchers. The autograd cache is the difference between a 15-minute first compile and a 30-second second compile for the same model on the same host.

### Ephemeral storage bites

On Modal H200 the inductor cache previously filled the on-pod ephemeral mount - tens of GB - when compiling the deep hybrid preset with full enriched features. Padded MoE is still compilable, but the cache footprint is large enough that a single cache-clear run on a fresh host can take an hour of lazy backward compile before steady state. We moved the cache to a persistent Modal volume and wired an explicit `sync_inductor_cache` step into the exact-matrix launcher so that bench hosts inherit a warm cache from the object store instead of recompiling from zero.

### Cache sync across hosts

The public cache-plumbing examples pin the contract: a bench host starting up should refresh `tokenizer.json` from the checked-in tokenizer artifact, set `TORCHINDUCTOR_CACHE_DIR` to the expected shared path, and skip re-seeding if the hash matches. That contract exists because concurrent cache-sync on the same host can trash a warm cache; the safe path is "skip if already synced" and "refuse to sync into a non-writable path."

### Reset discipline

`torch._dynamo.reset()` is called at exactly one site - after a CUDA retry re-exec that rebuilds the model. Anywhere else it is a bug. Resetting Dynamo invalidates all cached graphs, and on the deep hybrid that is 15 to 20 minutes of re-compile. We once had a helpful piece of auto-fit code that called `reset()` on every shape-change candidate, and it made the retry loop feel like it was hung.

### Suppressing errors

`torch._dynamo.config.suppress_errors = True` and `.disable = True` are used in exactly two places, both behind `MEGACPP_NO_COMPILE`-like guards. They exist for operator footguns (a donor path that does not compile, a fallback in the distributed optimizer stress test) and not as a general "make compile problems go away" switch. We do not ship with suppress in the default hot path - if something does not compile, we want the error.

## 5. The NCCL heartbeat interaction

`torch.compile`'s Triton JIT on the deep hybrid preset takes 15 to 20 minutes on H200:8 cold. NCCL's default heartbeat monitor (600 s) kills any rank that does not run a collective during that window. The symptom is a process torn down with a timeout error deep inside a compile pass, and a remaining rank hanging on the next collective.

Fix is three env vars we set automatically when `LOCAL_RANK` is detected:

```bash
TORCH_NCCL_ENABLE_MONITORING=0
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
```

Plus `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0` because the autotune subprocess OOMed on large matmuls (a 32768 x 65536 bf16 benchmark tensor is 8.6 GB on its own) and returned `inf ms` for legitimate configs, which then poisoned the cache with a bad pick. The compile-side lesson is that compile warmup and the distributed watchdog have no native handshake, so we impose one.

## 6. Noise we learned to ignore

Dynamo prints a lot. Some of it matters, most does not. We keep a short allowlist:

| Log signal | Action |
|---|---|
| `triton._C.libtriton.native_specialize_impl` warnings during warmup | Ignore - expected, not a break |
| `graph break` log lines matching known Mamba sites | Count; ignore if within expected bound |
| `accumulated_cache_size_limit` hit | Always a regression, alert |
| Autotune "Ignoring this choice" | Ignore unless correlated with a step-time jump; if correlated, autotune OOMed |

Anything above the expected Mamba break count is a regression. Cache-limit hits are treated the same way; we have an alert on the log line.

## Current compile policy

The current policy keeps the six Dynamo config lines, the `MBlock` disable, the four surgical disable points around DTensor/MoE/score_mod/API, the padded MoE path, the single explicit dynamic axis, the separate Inductor and autograd caches, the reset-exactly-once rule, and the NCCL heartbeat trio. The buffer-not-int rule is treated as a hard lint item on anything compiled code touches.

The policy does not treat `fullgraph=True` as a near-term goal; the Mamba chunk-scan custom op it would require to close the one to two percent break overhead is substantial work and remains deferred. It also excludes `enable_compiler_collectives`, any use of `torch._dynamo.reset()` outside the retry re-exec, and any uncomputed dynamic axis. The MoE-counter-as-int pattern is gone from the codebase, and compile-disable guards remain scoped rather than broad.

`torch.compile` is genuinely load-bearing once these rules are in place. Without them it is a liability. The difference is not the compiler; it is the stance you compile with.

## References

- [PyTorch `torch.compiler`](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [PyTorch recompilation model](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html)
- [PyTorch Inductor](https://docs.pytorch.org/docs/stable/inductor.html)
- [Mamba-3 Trapezoid Porting Notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/mamba3-trapezoid-porting.md)
- [Mamba-3 TP partition size excerpt](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/cppmega/megatron/tensor-parallel-and-sharding__mamba3_tp_partition_sizes__v1.py)
- [Regional compile ordering sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/regional_compile_ordering_sample.py)
- [Compile warmup policy sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/compile_warmup_policy_sample.py)
- [Dynamic batch compile policy sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/dynamic_batch_compile_policy_sample.py)
- [Compile runtime receipt sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/compile_runtime_receipt_sample.py)
- [Opaque kernel compile wrapper sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/opaque_kernel_compile_wrapper_sample.py)
