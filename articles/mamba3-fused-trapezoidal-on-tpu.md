---
title: "Mamba-3 fused trapezoidal scan on TPU v6e"
description: "How we took the Mamba-3 trapezoidal SSM update from a CUDA Triton kernel to a Pallas/XLA-friendly scan on TPU v6e, and what survived the deployment port."
date: "2026-04-18"
tags: ["mamba3", "tpu", "v6e", "pallas", "xla", "ssm"]
---

The Mamba-3 block is the cheapest sequence mixer we use in the MegaCpp hybrid, but only if the trapezoidal update is actually fused. On CUDA we lean on an official Triton SISO kernel wrapped as a custom op. On TPU v6e we had to fold the same trapezoidal math into a Pallas/XLA path that plays well with torch_xla, does not break `torch.compile`, and honours our document-masking contract. This post walks the POC modules that made that work and what is being lifted into deployment.

## Why MegaCpp cares about this

The hybrid recipe we use interleaves MLA blocks with Mamba-3 blocks. On the sequence mixer budget Mamba-3 is only competitive with attention when two things hold: the SSD scan is a single fused pass, and the "trapezoidal" discretisation is not a separate PyTorch elementwise stack. Profiling the v6e path with the naive decomposition of the Mamba-3 update showed the same story as on CUDA: something like sixty percent of block time living in elementwise ops around the scan, not in the scan itself ([Mamba-3 — ICLR 2026](https://openreview.net/forum?id=HwCvaJOiCj)).

Trapezoidal discretisation couples adjacent timesteps: each token needs a scaling factor that mixes `dt[t]`, `dt[t+1]`, and a learned gate `lam[t]`. Naive PyTorch turns that into nine elementwise ops per MBlock before the scan even starts, plus a second correction pass after the scan. On TPU v6e the MXU is not the bottleneck there — HBM bandwidth and kernel launch count are. Either you fuse the prologue and epilogue into the scan, or you lose the advantage of using an SSM at all.

The other reason this post exists: nobody else runs Mamba-3 on TPU. The upstream reference assumes Triton. We carry, on both sides of the codebase, the only working trapezoidal + document-masked SSM path on v6e that we are aware of.

The three paths at a glance:

| Lane              | Scan kernel                     | Prologue / epilogue           | `chunk_size` |
|-------------------|---------------------------------|-------------------------------|--------------|
| CUDA (H200/GB10)  | upstream Triton SISO            | fused Triton pre + diag corr  | variable     |
| TPU v6e (XLA)     | `mamba_scan_compiled` SSD       | plain XLA fusion              | 64           |
| TPU v6e (opt-in)  | `mamba_scan_compiled` SSD       | Pallas variant (experiment)   | 64           |

## What we built in the POC

There are three relevant modules in the POC and they split cleanly along the CUDA/TPU line.

The CUDA prologue collapses the trapezoidal arithmetic into one Triton pass:

```python
# _fused_trapezoidal_preprocess_kernel (sketch)
dt          = softplus(dt_raw + dt_bias)
lam         = sigmoid(lam_raw)
gamma       = lam * dt
gamma_shift = (1.0 - lam_next) * dt_next
scale_ratio = (gamma + gamma_shift) / tl.maximum(dt, 1e-6)
```

On the CUDA side, the public Mamba fused trapezoidal kernel sample defines two Triton kernels. The first, `_fused_trapezoidal_preprocess_kernel`, computes `dt = softplus(dt_raw + dt_bias)`, `lam = sigmoid(lam_raw)`, `gamma = lam * dt`, the shifted gamma `(1 - lam[t+1]) * dt[t+1]`, and the final `scale_ratio = (gamma + shifted_gamma) / max(dt, 1e-6)` in a single pass over `[B, T, H]`. Nine elementwise ops collapse into one Triton launch, and the per-block `BLOCK_T` is picked as `min(1024, next_power_of_2(T))`. The second kernel, `_fused_diagonal_correction_kernel`, rolls up the post-scan `y += x_ssm * (D + qk_dot - CB_diag * scale)` contraction so we never materialise the full `net_skip` tensor in HBM. That was bug #8 in the Mamba-3 adoption log: a double-count between the `qk_dot` skip and the trapezoidal diagonal correction, which we only caught once we stopped decomposing the epilogue into separate PyTorch lines.

The real SSM scan on CUDA is still the upstream Triton SISO kernel from the Triton Mamba3 SISO combined kernel. We do not rewrite it, we wrap it. the public Mamba compile wrapper sample registers the Mamba3 SISO forward kernel and the Mamba3 SISO backward kernel as `torch.library.custom_op`s under our library namespace, each with a `register_fake` implementation that returns the right shapes without executing Triton. This is the only way we have found to keep the rest of the MBlock — `in_proj`, conv1d (when it exists), `out_proj` — inside torch.compile's graph. Without the wrapper, dynamo's FakeTensor proxy trips on the SISO backward, which reaches into Triton-specific APIs that fake tensors do not model. The backward custom op re-runs the forward with `torch.enable_grad()` and uses `torch.autograd.grad` to extract the gradients. That doubles activation memory for that one op, which we pay happily in exchange for compile coverage on everything around it.

The equivalent wrapper for the classic Mamba-2 SSD path lives in the public Mamba compile-wrapper sample. It is the same pattern — fwd/bwd custom ops with fake-tensor shapes, empty-tensor sentinels for the `D` and `seq_idx` `None` arguments because `custom_op` cannot accept `None`, and a `_normalize_cuda_bwd_operands` helper that forces the transient operands used by the Triton backward to match the output dtype. The trapezoidal compiled path can widen `A`/`B`/`C` to fp32 while `dout` stays bf16, and Triton's `tl.dot` asserts on mismatched dtypes. We normalise transients for the kernel call and cast gradients back to the original dtypes afterwards. The `_setup_context` also clones `seq_idx` defensively, because with whole-model compile inductor has been seen to in-place a shared `doc_ids` tensor used by every Mamba block; the clone is tiny and defuses an autograd version-counter mismatch.

On TPU v6e the story is different. The Triton SISO kernel does not run there, and calling `call_jax` into a JAX implementation is a latency trap. The compile wrapper work still pays off, because our scan on v6e is the `mamba_chunk_scan_combined` SSD path running through `mamba_scan_compiled`, which is a torch_xla-friendly custom op with a `register_fake` shape contract. The Pallas lowering sees a stable set of tensor shapes per step, with `seq_idx` always materialised (never `None`), and XLA can fuse the trapezoidal prologue and epilogue around the scan body. On v6e we tile the scan by the fixed `chunk_size` — the author config default is 64 — which keeps every chunk's intermediates inside VMEM at our head dimensions and avoids spills onto HBM. We do not try to make `chunk_size` data-dependent: that was one of the first things we tried and it paid for itself immediately in recompiles.

Document-mask integration is the other axis where TPU forced the design. We had to reshape `seq_idx` to int32 at the custom-op boundary (`Triton 3.6` requires consistent loop-carried types, and torch_xla's Pallas lowering is no happier than Triton about silent dtype widening). The SSD scan itself already respects `seq_idx`, so document boundaries inside a packed sequence are honoured without a separate mask tensor. This is the same contract the attention side uses via Pallas FA `segment_ids`, which is not an accident: we want one document-masking story end to end, not one per block type.

## How it lands in deployment

In deployment, the trapezoidal Mamba-3 block becomes a Megatron-LM spec driven by `AuthorMamba3Config`. The config maps the Megatron Mamba surface onto the Mamba-3 contract (`d_state`, `headdim`, `ngroups`, `rope_fraction=0.5`, `dt_min/dt_max`, `A_floor=1e-4`, `chunk_size=64`) and a small set of toggles for MIMO and `outproj_norm`. Four things move with it.

The custom-op wrappers are lifted as-is. the public Mamba compile wrapper sample and the public Mamba compile-wrapper sample are the same bytes on both sides. They are the most boring part of the port and also the part that is hardest to reimplement correctly, so we resisted the temptation to rewrite them into a Megatron-shaped registry.

The Triton preprocess/epilogue kernels in the public Mamba fused trapezoidal kernel sample are lifted for the CUDA path and are gated by the public Mamba compile patch sample (the regional torch.compile patch). That kernel family is where our "5.93x data-dep-A" number comes from, and we are not reproducing it in eager PyTorch.

For the TPU path we do not port those Triton kernels at all. Instead, the prologue stays in pure XLA ops around the `mamba_scan_compiled` call, because XLA's elementwise fusion on v6e is genuinely good at this specific shape. The knobs that survive are the ones the SSD scan already exposes: `chunk_size`, `seq_idx`, and whether `D` is present. The deployment recipe pins `chunk_size=64` for the trapezoidal path.

What is being dropped: the `call_jax`-based experiments we wrote when we first tried to run Mamba-3 on TPU. Every one of them was either a correctness hazard (the bridge does not handle our backward shape metadata cleanly) or a latency hazard. The main path is trace_pallas or plain XLA, never `call_jax`.

What is becoming a feature flag: the fused diagonal correction. On CUDA we keep it on by default. On TPU we leave it to XLA fusion by default and ship an opt-in Pallas variant as an experiment, because the measured delta on our shapes is inside single-digit percent and we would rather not own another kernel on v6e yet.

## Ablations and what we kept

The Mamba-3 adoption log is long and loud. The trapezoidal-specific lessons:

1. Single-pass trapezoidal (B-scaling eliminates the dual-scan, halves scan compute) is on by default. This is the single biggest knob and the one we are least willing to turn off.
2. `qk_dot_skip` and the trapezoidal diagonal correction used to double-count `C·B * gamma * x`. Both were adding the same term independently; we removed one and wrote a regression test that computes the contribution both ways and asserts equality.
3. The SISO kernel wrapper used to double-sigmoid `lam`. The caller had already applied sigmoid, then the kernel applied it again. We now pass `lam_raw` through to the kernel's `Trap` argument and keep `sigmoid(lam_raw)` on the PyTorch side for everything else.
4. On XLA we initially tried to pass `A` in fp32 while `B`/`C` stayed in bf16. The backward then ran `tl.dot` against mismatched dtypes on the CUDA mirror path. `_normalize_cuda_bwd_operands` was the minimal fix; a bigger refactor was possible and not worth it.
5. On TPU the trapezoidal update interacts badly with FSDP2-style shard rearrangement if `A` is silently promoted to fp32 inside the scan body — we saw the promotion the first time we touched the CHANGELOG entry about "A(float32) × dt_soft(bf16) silently promoted to float32" and now cast `A` to `dt_soft.dtype` explicitly before the trapezoidal dA computation.
6. MIMO (rank-4) is off by default on TPU. It works, it passes our numerical tests, but the shape story made the Pallas FA side less stable than we wanted for launch.

The failed experiments that live in the CHANGELOG and do not live in the code: a fully data-dependent `chunk_size`, a Pallas rewrite of the trapezoidal prologue, and a monolithic "one custom op that owns the whole MBlock" approach. The first two cost compile time without earning runtime. The third blocked torch.compile from seeing the projection GEMMs, which was the whole reason we started this.

## Production checklist

- `chunk_size` is pinned at 64 on v6e; do not let it become data-dependent.
- `seq_idx` is always materialised as int32, not None, before calling `mamba_scan_compiled`.
- The custom-op wrappers are imported before any call site to register the ops; import-order changes go through a unit test that asserts the `mamba_scan_fwd` custom op is registered under the expected namespace.
- On CUDA the fused trapezoidal preprocess and diagonal-correction kernels are on; a config flag gates them off for triage.
- On TPU the trapezoidal prologue stays in XLA fusion; an opt-in Pallas variant exists but is not the default.
- Backward dtype normalisation (`_normalize_cuda_bwd_operands`) is invariant under whole-model compile and is covered by a regression test.
- MIMO and `outproj_norm` toggles travel with the config; TPU default is MIMO off.
- Document masking travels through `seq_idx`; no per-block alternative is allowed.
- the Mamba3 SISO backward kernel does not support `return_final_states=True`; the serving path uses a separate streaming impl.

## References

the public Mamba fused trapezoidal kernel sample, the public Mamba compile wrapper sample, the public Mamba compile-wrapper sample, the public Mamba mixer sample, the TPU attention dispatch layer, the public Pallas softcap kernel sample, the public Mamba compile patch sample, the public authored-Mamba spec sample, the public authored Mamba spec sample, the public Mamba mixer sample, the public Mamba recompute patch sample, the public startup-memory calibration sample, the public XLA memory-calibration sample. External: [Mamba-3 — ICLR 2026](https://openreview.net/forum?id=HwCvaJOiCj), [Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu & Dao](https://arxiv.org/abs/2312.00752), [jax.experimental.pallas — JAX docs](https://docs.jax.dev/en/latest/pallas/index.html).
