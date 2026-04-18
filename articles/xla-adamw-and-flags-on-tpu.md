---
title: "XLA-safe AdamW and the libtpu Flag Matrix on v6e"
description: "How our TPU AdamW avoids .item() and graph breaks, what libtpu flags actually move perf on v6e, and the memory-calibration loop we ship alongside them."
date: "2026-04-18"
tags: ["tpu", "v6e", "xla", "adamw", "libtpu", "calibration"]
---

On CUDA, the AdamW step is a compiled graph of tensor ops and nobody cares how Python spells it. On TPU v6e, the AdamW step is the single piece of code most likely to silently force a host-device sync, break the XLA graph, or trigger a long recompile. This post is how we got our AdamW step XLA-safe in the POC, which libtpu flags actually move v6e perf, and the calibration loop we use to keep the memory envelope stable across torch / torch_xla / libtpu upgrades.

## Why the AdamW step is the canary

The MegaCpp training mix lives on TPU v6e and H200 simultaneously. The optimizer step has to be numerically identical across devices, fit inside `torch_xla.compile()` on v6e, and stay competitive with the fused CUDA path. That is harder than it sounds: the reference `torch.optim.AdamW` calls `_get_value(step_t)` which calls `.item()`, which on XLA is a host-device sync that forces a graph break. Every graph break changes the next graph's Python-level scalar constants (`bias_correction`, `step_size`), and XLA hashes those constants, so you recompile on every step. We measured that as a roughly 48-minute recompile tax on the depth-52 hybrid preset before we fixed it.

The libtpu flag matrix matters for a different reason. On v6e the flags decide whether the scheduler overlaps collectives with compute at all, whether SparseCore is used on future generations, and whether a known MSA crash on MoE training manifests. The right flags are not defaults.

## The two AdamWs and one flag library

The POC has two AdamW implementations and one flag library, and the split between them is the interesting part.

the public AdamW sample is the distributed AdamW with a fused step. The fused step body, `_adamw_step_fused_impl`, is a single function wrapped with `torch.compile` on CUDA and called eagerly on XLA. All scalar hyperparameters (`lr`, `wd`, `beta1`, `beta2`, `eps`, `step`) arrive as 0-D tensors. The function casts them to the parameter dtype at the top, computes `safe_step = max(step, 1)` as a tensor, applies decoupled weight decay in-place, updates the running averages with `lerp_` (so the gradient square is fused into the second-moment update without a temporary), computes the bias corrections as tensor exponentiations, and does the final update with a single `p.sub_((exp_avg / denom) * step_size)`. No `.item()`, no Python scalars, no `if step == 0` branch. On CUDA this lets Inductor emit one graph for the step. On XLA the JIT does the equivalent. Two critical knobs: `_should_compile()` returns False when `PJRT_DEVICE=TPU`, and `_uses_xla_runtime_scalar_cache(device)` flips the class into a mode where 0-D scalars come from a stable per-device cache.

That scalar cache is the core trick. `prepare_xla_step_scalars(device)` materialises 0-D tensors for every changing value and stores them under a per-device key in `_xla_runtime_scalar_cache`. At step time, inside `torch_xla.compile()`, the optimiser reads those tensors and mutates them via `fill_()`. With `XLA_NO_SPECIAL_SCALARS=1`, `fill_(value)` generates a `DeviceData` IR node whose hash depends on shape and dtype only, not on the filled value. That is the difference between recompiling every step and compiling once.

the public XLA AdamW sample is the stripped-down sibling for callers that do not need the distributed slice machinery: a plain `torch.optim.Optimizer` subclass that uses the same tricks (pre-allocated 0-D tensors for `bc2_sqrt`, `step_size` and the weight-decay scale, moved to the device on the first step, filled in place each step). The comment in that file says `capturable=True` on the stock AdamW segfaults on TPU inside `libtpu`'s `ConvertFromCppChunk`. That failure mode was reproducible enough that the safer path was to avoid `capturable=True` here and use a dedicated class instead.

Two more rules we learned the hard way and encoded as invariants:

- Decoupled weight decay is applied by multiplying the parameter by `1 - lr*wd`, with `lr` and `wd` as tensors, not via the optimizer-provided path that uses `_get_value`.
- `step` itself lives on the host as a Python int. The optimiser increments it in Python and then fills a 0-D device tensor with it. That sounds paradoxical (isn't a Python scalar the problem?) but the Python-side step is not inside `torch_xla.compile()`; only the per-parameter math is. The 0-D tensor is what the traced graph sees, and its IR is stable.

```python
# Stylised XLA-safe step body.
def adamw_step_xla(p, g, exp_avg, exp_avg_sq, scalars):
    p.mul_(scalars["one_minus_lr_wd"])              # decoupled WD
    exp_avg.lerp_(g, scalars["one_minus_b1"])       # m_t
    exp_avg_sq.mul_(scalars["b2"]).addcmul_(g, g, value=scalars["one_minus_b2"])
    denom = exp_avg_sq.sqrt().div_(scalars["bc2_sqrt"]).add_(scalars["eps"])
    p.sub_((exp_avg / denom) * scalars["step_size"])
```

## The libtpu flag library that actually matters

the public XLA flags sample is adapted from MaxText's flag library and split into groups. The important groups:

| Group | Representative knob | What it does |
|---|---|---|
| VMEM limits | `xla_tpu_scoped_vmem_limit_kib` | 98304 KiB dense, 81920 KiB MoE; spilling to HBM costs prefetch room |
| Continuation Fusion | `xla_tpu_enable_async_collective_fusion` | On for TP AllGather / FSDP ReduceScatter; off for MoE alltoall |
| DCN PP chunking | `xla_tpu_iova_dma_chunk_size_bytes=16777216` | Opt-in for pipeline parallel over DCN |
| Layout / scheduling | various | Conservative defaults; aggressive variants reserved for dense ablations |
| libtpu compile mode | Mode A vs Mode B | Mode A for dense, Mode B forced for MoE (Mode A SIGKILLs MSA on alltoall) |

The CF group is the one that produces visible perf swings. CF on for AllGather/ReduceScatter overlaps the collective with the next matmul; CF on for MoE alltoall produces longer compiles for no measurable runtime improvement. We turn CF on selectively per collective type rather than as a global switch. SparseCore and CF are mutually exclusive for the same collective type, but on v6e SparseCore is not available so CF wins by default.

`LIBTPU_INIT_ARGS` is owned by our code. the TPU training launcher sets these before importing `torch_xla` because `torch_xla`'s `_set_missing_flags()` only fills values not already present. Letting `torch_xla` set them first leaves you with the wrong defaults for the MoE MSA workaround.

## The memory calibration loop

the public XLA memory-calibration sample runs a short calibration on the first few steps of every launch and writes the result next to the checkpoint as `xla_startup_calibration.jsonl`. It records the predicted vs measured per-component sizes (the same components the analytical estimator carries: parameters, gradients, optimizer state, activations, dispatch buffers, scratch, allocator overhead). The auto-fit retry ladder reads the most recent calibration before the next launch. If predicted activations were within 5% of measured, the next launch trusts the estimator; if they were off by more, the next launch widens the safety margin and may step `dbs` down without rediscovering the memory cliff.

This loop is what makes the flag matrix stable across upgrades. When `libtpu` bumps and the per-collective working set changes, the calibration record diverges from the estimator, the auto-fit ladder catches it, and the next launch retries with a more conservative shape. We have done two `libtpu` bumps under live training using this loop and lost no run time on either.

## What we kept and threw away

We kept the per-device scalar cache, `XLA_NO_SPECIAL_SCALARS=1` as a hard requirement, the rule that every changing scalar is a 0-D tensor filled in place, the per-collective CF flag selection, Mode B `libtpu` for MoE, the calibration loop as the auto-fit input, and our own the public XLA AdamW sample rather than chasing `capturable=True` on stock AdamW.

We threw away `capturable=True` on stock AdamW (segfaults inside `libtpu`), CF on every collective (longer compiles for no runtime gain on alltoall), `XLA_USE_BF16=1` (incompatible with current Pallas flash kernels), `XLA_USE_SPMD=1` as a shell toggle (set by runtime startup; the env var has no effect and confuses the control path), and global flag profiles (per-workload VMEM limits move perf more than any single global flag).

The throughline is small. On TPU, the optimiser step is the most fragile part of the graph; if it is XLA-safe, almost everything else falls into place. If it is not, you will spend training afternoons recompiling.

## How the flag profiles compose

the public XLA flags sample exposes a small number of named profiles (`none`, `auto`, `dense`, `moe`, `offload`) and a smaller number of overrides. The profiles are not orthogonal: enabling `moe` selects Mode B `libtpu`, drops the VMEM limit to the MoE-friendly value, and disables CF for the alltoall collective; enabling `offload` opts into the DCN DMA chunking and is only meaningful when pipeline parallel is also on. The auto profile picks dense or moe based on the model config and is the recommended default for new operators.

Two operator footguns. The first is shell-exported `LIBTPU_INIT_ARGS`. Anything in the shell environment overrides what the public XLA flags sample can set, because `torch_xla`'s missing-flags helper only fills in unset values. The launcher unsets that variable before applying the profile; we recommend the same pattern in any wrapper. The second is mode swaps mid-run. Mode A versus Mode B is a `libtpu` init-time decision; you cannot switch between them across micro-steps. The launcher records the mode in the receipt and the calibration record so a regression can be attributed if the mode differs from the previous run.

## What the calibration record actually records

The calibration record is a small JSONL file with one entry per launch. Each entry carries: torch SHA, torch_xla SHA, libtpu version, jax version, the active flag profile, the active mode, the predicted per-component memory sizes from the analytical estimator, the measured per-component sizes from the per-chip pybind, the cold and warm step-0 times, the steady-state tok/sec window, the persistent cache hit rate on the first 100 steps, and the mark_sharding fingerprint over the parameter tree. Reading two consecutive entries tells the operator everything they need to attribute a regression: did the cache hit drop, did a flag profile change, did a memory term blow up, did the fingerprint change.

The autopilot retry ladder reads the most recent calibration entry before launching. If the previous launch had a low cache hit rate, the next launch widens the compile budget. If the previous launch had measured activations above prediction, the next launch widens the safety margin. If the previous launch had a fingerprint change, the persistent cache is invalidated for the affected key. None of those decisions is heroic; they are arithmetic on the calibration record.

## What we still cannot do

The XLA-safe AdamW pattern handles per-step varying scalars. It does not handle per-step varying shapes. Anything that produces a different tensor shape on different steps still recompiles, and we live with that by padding aggressively at the dataloader and by forbidding runtime tokenisation in the hot path. The other thing the pattern does not handle is the dual-compile window on step 0, where the `grad create` and `grad accumulate` variants compile separately. We mitigate that by warming the cache deliberately on bring-up, but the dual-compile remains a real cost on the first run after a `libtpu` bump.

The flag matrix has a similar limit. Flags help when the bottleneck is overlap or scheduling; they do not help when the bottleneck is HBM. For the deep MoE preset on v6e the bottleneck is HBM, and no flag profile will fix that. The fix is topology (FSDP, EP) or model shape, not a flag.

## References

- adamw.py
- xla_adamw.py
- xla_flags.py
- the public XLA memory-calibration sample
- the TPU training launcher
- TPU_SETUP.md
- the public engineering changelog
- CURRENT_STATE.md
