---
title: "The Compile-Time Tax We Accept for Runtime Speed"
description: "Why MegaCpp pays first-compile and recompile costs in exchange for steady-state throughput, and the operational rules that keep torch.compile, torch_xla and Triton caches honest across runs."
date: "2026-04-18"
tags: ["torch.compile", "torch_xla", "triton", "inductor", "training-infra"]
---

Compilation is not a performance feature, it is a debt contract. We pay it once at session start (autotune, JIT, kernel selection), we pay it again every time a guard fires or a shape changes, and in exchange we are allowed to ship a steady-state step that is meaningfully cheaper than the eager one. This post is the trade-off rationale we use to decide where that contract is worth signing inside the MegaCpp training stack, what we instrument so the bill stays bounded, and the operational rule we ended up writing in blood after the famous "compile hangs and NaN" episode. It is intentionally separate from `dynamo-and-compile-breakage` (which is the bug catalogue) and `graph-recompilation-hell` (which is the TPU side); this post is about why we accept the tax at all.

## Why MegaCpp cares about this

A representative hybrid stack runs across two very different toolchains: PyTorch 2.x with Inductor and Triton on H200 and B200, and `torch_xla` plus a JAX path on TPU v6e. Both buy us steady-state throughput by burning wall clock at start and risking wall clock at every recompile. On a depth-52 hybrid preset with Mamba-3 SSM blocks, an MoE expert tail and an MLA/DSA attention minority, the compile surface is huge. Inductor wants to fuse, but a Triton SSM kernel is opaque to it. Dynamo wants to specialize, but MoE routing is shape-dynamic by construction. XLA wants a single HLO, but optional adapters and MTP heads add side branches. If we let the compilers run with defaults, first-step time blows up to tens of minutes, and a single mis-typed counter can trigger a recompile storm that looks indistinguishable from a hang on a multi-rank job.

We accept the tax because the alternative is worse. Eager-mode steady state on the same model leaves a low-double-digit percentage of step time on the table, mostly in elementwise glue around the SSM and attention paths and in the per-microbatch overhead of small Python dispatches. We have measured this on multiple presets and the answer has not changed: compiled is faster as long as the cache is warm, the guards are stable, and the recompile budget is bounded. The whole job of the rules below is to make those three preconditions hold.

## What we built in the MegaCpp training stack

MegaCpp is where we discovered all the failure modes; production then encodes the survivors. Five surfaces matter.

The first is `TORCHINDUCTOR_CACHE_DIR` plumbing. Every launch script we still ship sets it explicitly, alongside `TORCHINDUCTOR_FX_GRAPH_CACHE=1` and `TORCHINDUCTOR_AUTOGRAD_CACHE=1`, before Python starts. The runtime should pin the cache to a writable persistent volume, never to a small root disk where Inductor can silently fill the partition mid-run. We learned this the hard way in an early ablation pass when nine consecutive runs failed with "DISK FULL (inductor cache)" and were initially misclassified as model bugs. Public change notes show that the fix was operational, not algorithmic. The practical rule is: export `TORCHINDUCTOR_CACHE_DIR` to a per-run subdirectory on a large persistent volume, and refuse to start training if the directory is missing or unwritable. Inductor's persistent FX-graph cache then survives across runs and across processes, which is the thing that actually moves first-step wall clock from "tens of minutes" to "tens of seconds" on a warm host.

The second is the choice between `regional_compile` and a full-graph compile. A representative model runtime module exposes `regional_compile: bool` at the model config level, and the hot path is structured so that the *block* is the compile unit, not the whole model. Mamba blocks, attention blocks, MoE blocks and the hyper-connection plumbing all carry comments that explicitly call out "regional_compile region" boundaries; the variadic unpack at the block boundary is intentionally kept in eager Python so Dynamo does not have to trace through `*args, **kwargs`. We tried full-graph compile on the depth-52 preset and rejected it: the guard surface across 52 mixed blocks is too large, autotune time on H200 SM90 with `TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM=1` ran the per-rank Triton subprocess into OOM, and one shape mismatch anywhere in the graph forces a full recompile of all 52 blocks at once. Regional compile localises both the autotune cost and the recompile blast radius: a guard miss in MoE only invalidates the MoE region, not the SSM region next door.

The third is the dynamo guards we actively avoid. The pattern we now treat as a code smell is any Python-level state that a compiled function can read on the hot path: an `int` counter, a `bool` flag, a `getattr(self, ...)` lookup, an environment variable read inside a method that is later compiled. Each of these becomes a guard, and a guard that flips even once can trigger recompilation. The MoE module carries explicit comments at the points where this matters: loss accumulators are stored as Tensors not as `None`-or-Tensor unions (so Dynamo does not synthesise a type-change guard), buffer caches are looked up via direct attribute access instead of `getattr` (no attribute-lookup guard), and routing branches are pre-baked into module-level booleans at construction time so Dynamo can specialize without a per-call guard. Where we genuinely need a Python branch we mark it with `@torch.compiler.disable`, accept the graph break, and document it in place.

The fourth is the Triton-kernel compile wrapper choice for Mamba. The Mamba-3 SISO kernel and the older Mamba-2 SSD kernel are both `torch.autograd.Function` implementations whose backward dereferences Triton-specific APIs. Dynamo's FakeTensor proxy crashes when it tries to trace those backward bodies. We wrap both forward and backward in `torch.library.custom_op` (the public Mamba compile wrapper sample, `mamba_compile_wrapper.py`), provide `register_fake` shape-only stubs, and wire autograd via `setup_context` plus `register_autograd`. The kernel becomes opaque to Inductor; the rest of the M-block (`in_proj`, `conv1d`, `out_proj`, RMSNorms) compiles around it. The trade-off is explicit: we forfeit any Inductor fusion across the SSM boundary in exchange for keeping the surrounding linear/norm work compiled. Repeated measurements show this is a win at the target shapes; a fully eager M-block lost more on the surrounding glue than it gained on the kernel.

The fifth is XLA persistent cache discipline. The TPU side keeps its own JAX persistent compilation cache (`jax.config.update("jax_compilation_cache_dir", ...)`) and its own `torch_xla` HLO cache. Both must be pinned to a writable, persistent directory before any JAX or `torch_xla` operation runs. Representative TPU benchmark scripts set this in the very first imports, configurable via `JAX_COMPILATION_CACHE_DIR`. Without it every run paid the full HLO compile cost over again, which on a hybrid preset is not a few seconds. The same precondition applies on the CUDA side: a missing or non-writable cache directory should abort launch.

## How it lands in MegaCpp

In the deployed MegaCpp stack the rules become non-optional.

We lift the cache plumbing as-is. Every preset launcher exports `TORCHINDUCTOR_CACHE_DIR`, `TORCHINDUCTOR_FX_GRAPH_CACHE`, `TORCHINDUCTOR_AUTOGRAD_CACHE` and the JAX/XLA cache directory before Python is allowed to start. The MegaCpp bring-up script verifies free space and write permission on the cache volume before allocating GPUs.

We rewrite the autotune surface. On H200 SM90 we keep `TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM=0` because the distributed-autotune subprocess OOMs against a concurrent FSDP/EP layout; on B200 and on single-node H200 with `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1` we widen the autotune backends to `ATEN,TRITON` and accept the longer first-compile in exchange for better steady-state kernels. The `TORCHINDUCTOR_MAX_AUTOTUNE_SUBPROC_RESULT_TIMEOUT` is pinned to a value that survives our slowest kernel without false-failing.

We drop the `NO_COMPILE` escape hatch. That switch existed only because of the regression that gave this post its operational rule; it is no longer needed and is not exposed in the deployed MegaCpp stack.

We move the Mamba kernel wrappers under a feature flag. The `custom_op` wrappers are the production default; the eager fallback exists only for the SM<80 development boxes (where bf16 Triton codegen is broken and we fall back to fp16) and is gated behind a single env switch.

We move the FIRE/DASH plasticity hooks out of the regional-compile region. The hyper-connections and FIRE-orthogonalisation passes have explicit `regional_compile` boundaries that stay eager-only; in the deployed MegaCpp stack those boundaries are enforced by startup checks instead of comments alone.

The whole set can be summarized as a compile contract enforced at startup; if any contract item is violated, the run should abort before touching an accelerator.

## Ablations and what we kept

Ablation history tells the trade-off story more honestly than any micro-benchmark. The throughput investigation entry from this spring is the canonical reference: the 8-GPU DDP path with `torch.compile` enabled was hanging or going NaN, and the diagnosis bounced through three wrong root causes (Muon, bf16, network) before landing on the real one. The MoE module's `_overflow_total` counter was a Python `int` that incremented every `forward()`. Dynamo specialized on the value of that counter and recompiled. With the recompile limit at 64 it took only a handful of microbatches before we hit the limit on every rank, and the symptoms — a multi-minute stall followed by NaN — looked exactly like a numerical instability, not a compiler bug.

The fix is the operational rule we shipped after that episode and the single most important paragraph in this post:

> **No Python-level mutable state on the compiled hot path.** Counters, accumulators and flags that change across forwards must be `register_buffer` Tensors mutated in-place under `torch.no_grad()`. Anything else is a recompile waiting to happen.

The fix in the main MoE runtime module was mechanical — convert `_overflow_total` and the related accumulators to Tensor buffers, mutate them with `.add_()` — but the rule generalises. We now grep for `int` and `bool` attributes on any `nn.Module` that lives inside a `regional_compile` region and treat each one as a code-review block. The same rule eliminated three other near-misses: a `getattr` lookup that Dynamo treated as opaque (rewritten as direct attribute access set in `__init__`), a `None`-or-`Tensor` loss accumulator (always-Tensor with a zero default), and an env-var read inside `forward()` (cached at construction time, with a comment that changing the env now requires recreating the module).

The other ablation worth keeping is the regional vs full-graph comparison. We re-ran the depth-52 hybrid preset under both modes after the MoE counter fix landed: full-graph had a longer first-compile, hit autotune OOM under DDP unless we disabled distributed autotune, and recompiled the entire 52-block graph on every shape change. Regional kept the recompile blast radius local. Steady-state throughput was within noise; first-compile and recompile cost was strictly worse for full-graph. We kept regional.

## Production checklist

- Export `TORCHINDUCTOR_CACHE_DIR`, `TORCHINDUCTOR_FX_GRAPH_CACHE=1`, `TORCHINDUCTOR_AUTOGRAD_CACHE=1` before Python starts; pin to a persistent volume; abort the run if the directory is missing or unwritable.
- Pin `JAX_COMPILATION_CACHE_DIR` and the `torch_xla` HLO cache the same way; no JAX or `torch_xla` import is allowed before this is set.
- Default to `regional_compile` on the depth-52 hybrid preset; full-graph is opt-in for small experiments only.
- Wrap Triton autograd kernels (the public Mamba compile wrapper sample, `mamba_compile_wrapper.py`) in `torch.library.custom_op` with `register_fake`; never let Dynamo trace into a Triton backward.
- No Python-level mutable state on the compiled hot path. All accumulators are `register_buffer` Tensors mutated under `torch.no_grad()`.
- Mark all Python branches that must stay eager with `@torch.compiler.disable` and document why in place.
- Disable `TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM` on multi-rank H200 SM90; re-enable in subprocess mode on B200 with a bounded result timeout.
- Track the per-run unique program count (CUDA: Inductor cache misses; XLA: distinct HLO keys). Recompile budget is bounded; exceeding it pages a human.
- Doctor script verifies cache plumbing, dynamo recompile counters and Triton autotune timeouts before any rank claims a GPU.
- The compile contract doc is the source of truth; launchers read it at start and refuse to run on violation.

## References

- [PyTorch `torch.compile` documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch compiler troubleshooting guide](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [PyTorch custom operators landing page](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/)
- [FIRE plasticity toolkit article](https://megacpp.com/blog/fire-plasticity-toolkit.md)
