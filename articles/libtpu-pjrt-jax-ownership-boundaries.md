---
title: "libtpu, PJRT, JAX, and ownership boundaries"
description: "Why a shared TPU substrate still leaves distinct ownership lines across PJRT, torch_xla, JAX, and libtpu, and where MegaCpp saw the real failure boundaries."
date: "2026-04-19"
tags: ["libtpu", "pjrt", "jax", "torch-xla", "tpu", "MegaCpp"]
---

The easy story is that TPU software is one stack, so mixed tooling should behave like one system. The local evidence says otherwise. MegaCpp's TPU work repeatedly converged on a stricter rule: `libtpu` may be shared substrate, but frontend ownership, runtime policy, cache policy, and backend proof still need to stay separate.

That separation was not aesthetic. It was forced by failure boundaries that kept reappearing in code, reports, and runtime receipts.

## The stack is shared, but the owners are not

One local architecture note already shows the layering clearly:

```text
MegaCpp PyTorch code
  -> torch_xla
  -> PJRT runtime
  -> libtpu
  -> TPU hardware
```

That same note also records the practical complication: JAX is imported directly for some TPU kernel paths, while the main model path remains PyTorch/XLA-owned. In other words, one process can contain more than one frontend authority even when the lower TPU runtime substrate is shared.

The launch surfaces make this explicit. The TPU CLI documents `--splash_attn` as a JAX Pallas kernel reached through the `torch_xla` bridge, while `--chunked_attn` is documented as a pure PyTorch fallback that does not require JAX. The same CLI also exposes `--xla_flag_profile` as an opt-in `XLA/libtpu` launch-policy knob, and treats `PJRT_DEVICE=TPU` as the relevant gate for the XLA SPMD lane. That is not one interchangeable software stack. It is one shared substrate with multiple frontend-owned entry points.

## Why the ownership split became necessary

The TPU failures in local docs were usually not generic "TPU is broken" failures. They were boundary failures where one layer made an assumption that another layer did not honor.

One review note captured the risk directly: TPU autodetect could still force XLA on a broken or non-runtime host, and the recommended fix was to fail fast on a real runtime probe instead of trusting fallback signals. That is an ownership lesson. Device detection belongs to runtime truth, not to optimistic frontend inference.

The TPU backend provenance receipt shows the same lesson from a different angle. It records `torch`, `torch_xla`, and `jax` versions together, then logs which backend actually ran: `xla_flash_pallas`, `xla_flash_pallas_softcap`, or `xla_splash_via_trace_pallas`. That is the right model for a mixed TPU environment. The run has to record which frontend asked for work, which runtime contract was active, and which backend actually answered.

## The failure boundaries that mattered in practice

### 1. Runtime intent is not runtime proof

`PJRT_DEVICE=TPU` expresses intent. It does not prove that the runtime lane is healthy. The autodetect review showed that library presence and fallback logic could still push execution toward XLA on a host that was not actually ready. That failure belongs at the runtime-ownership layer.

### 2. JAX-backed kernel paths have different owners than pure torch_xla paths

The attention CLI makes the split explicit. `--splash_attn` is a JAX Pallas surface accessed through the `torch_xla` bridge. `--chunked_attn` is the pure PyTorch fallback. Those are different ownership domains with different failure modes, even though they meet on the same TPU runtime underneath.

### 3. Cache ownership is separate from frontend ownership

The training stack carries both `torch_xla.runtime.initialize_cache(...)` and JAX compilation-cache configuration. The reports and setup notes repeatedly emphasize that cache configuration must happen before computation starts. That is another boundary line: one shared accelerator does not imply one shared cache contract. PyTorch/XLA and JAX each carry their own cache-entry policy even when both eventually target the same TPU backend.

### 4. Some failures terminate inside libtpu-owned behavior, not OpenXLA-owned behavior

One local note on the 2 GB executable-proto limit is especially revealing. The split-proto fix landed for GPU in OpenXLA, but the TPU cache path still went through closed-source `libtpu` and therefore did not inherit the same fix. That is the clearest ownership boundary in the stack: even with PJRT and XLA above it, some TPU runtime behavior is still effectively owned by `libtpu`.

## What MegaCpp ended up owning explicitly

The local code and notes converge on four ownership buckets.

### Frontend ownership

The code keeps a PyTorch/XLA-owned main path and treats JAX-backed kernels as explicit opt-in surfaces. That prevents a JAX-dependent feature from masquerading as a generic TPU capability.

### Runtime-policy ownership

The launch layer owns `PJRT_DEVICE`, XLA flag profiles, and startup cache configuration. Those settings are treated as launch policy, not shell noise, because import order and early runtime state change behavior materially.

### Backend-proof ownership

The provenance receipt exists because statements like "Pallas ran" or "Splash ran" are not specific enough. The run has to record which backend actually executed.

### Failure-boundary ownership

When a fix belongs in launch policy, kernel routing, cache setup, or backend substitution, MegaCpp keeps it there instead of pretending the entire problem is a generic TPU-runtime fault.

## What "latest libtpu" really means

The local notes also make one more point that public TPU discussions often flatten away: "use the latest libtpu" is not the same statement as "all frontend/runtime combinations are compatible with the latest libtpu."

One version matrix records a preferred newer `libtpu` lane together with custom `torch_xla` builds. Another compatibility note records that older stock pairings stop at an earlier `libtpu` boundary because the PJRT execute-options ABI changed. That is exactly the kind of ownership split that gets lost when people describe TPU software as one blob. The backend may be "latest" for one validated pairing and still be outside the supported boundary for another.

So the practical question is not "what is the newest libtpu package?" The practical question is "which `torch_xla` and JAX frontend pairings have been proven against which `libtpu` and PJRT boundary?"

## Practical debugging order

If a TPU lane involves `libtpu`, PJRT, `torch_xla`, and JAX in one environment, the local MegaCpp evidence suggests debugging it in this order:

1. Which frontend owned the failing path: pure PyTorch/XLA, or a JAX-backed kernel surface?
2. Which runtime policy was selected before imports: `PJRT_DEVICE`, XLA flags, cache path, JAX platform policy?
3. Which backend actually ran according to the receipt?
4. Does the failure terminate in application routing, in the frontend bridge, or in a `libtpu`-owned runtime or cache boundary?

That is the useful meaning of ownership here. The substrate is shared. The accountability is not.

## References

- `CLAUDE.md`
- `docs/TPU_SETUP.md`
- `common.py`
- `reports/current_live_bugs_2026-03-07_strict_pass3.md`
- `reports/enrichment_framework_review_2026-02-23.md`
- `reports/tpu_backend_provenance_v6e8_2026-03-16.json`
- `scripts/base_train.py`
- `scripts/train_args.py`
