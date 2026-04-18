---
title: "Pallas kernels on TPU v6e: what we ship and what we deleted"
description: "Where Pallas beats the XLA lowering on TPU v6e, where it loses, the debugging workflow that keeps us sane, and the kernel deltas we kept versus the ones we reverted."
date: "2026-04-18"
tags: ["pallas", "tpu", "v6e", "jax"]
---

TPU v6e is the main TPU we run training on today, and Pallas is how we get kernel-level control on it. Over the last few months we ended up with a working taxonomy of "Pallas earns its keep here, XLA is fine there, and this third thing we deleted because maintaining it was not worth it." This is the honest breakdown, with the debugging workflow we actually use.

## Why this matters

Pallas is a precision tool. Every kernel we ship there is a kernel a future engineer has to maintain across `torch_xla`, `libtpu`, and `jax` upgrades, against an XLA lowering that genuinely keeps getting better. The cost of writing and debugging Pallas kernels on TPU is materially higher than writing Triton on CUDA: less ecosystem, sharper version pinning, harder to reproduce on a laptop, and silent correctness bugs hide better than performance regressions. So the question is not "what can we write in Pallas" but "what must we write in Pallas to clear a real bottleneck XLA cannot close." This post is the answer for our v6e training stack.

The decisions also have downstream consequences: when a Pallas kernel crosses the bar, our compile cache, our SPMD propagation, and our backward graph all become coupled to its quirks. Adding a Pallas kernel is closer to taking a dependency than to writing a function.

## 1. What the TPU stack actually looks like

When we say "Pallas" in this repo, we mean a few concrete things:

1. Native Pallas Flash Attention with softcap and local window. Implemented in `the POC/the public Pallas softcap kernel sample`, invoked via `torch_xla.experimental.custom_kernel.trace_pallas` for both forward and all three backward kernels (dQ, dK, dV). No `call_jax` bridge in the hot path.
2. A mask-composition layer (CausalMask, LocalMask, FullMask, ChunkedCausalMask, SinkMask and `&`/`|` composition) that is pure Python and NumPy, so the CUDA side can import the mask classes without pulling JAX.
3. A delegation path to the original native Pallas FA kernel (`torch_xla.experimental.custom_kernel.flash_attention`) whenever the request is plain causal or full, with no softcap, no local window, and no composed mask.
4. A legacy `call_jax`-based Splash fallback that only runs when the trace-pallas family is unavailable.
5. A sparse experiments tree under `experiments/sparse_pallas/` where we prototyped hierarchical and fused-scoring kernels. Most of that stayed experimental.

The backend-selection logic lives in `the POC/the TPU attention dispatch layer` behind `--xla_flash_attn` and `--splash_attn`, and the stoplight matrix in a backend stoplight matrix records the exact status each flag maps to.

## 2. Where Pallas actually beats XLA lowering

A few patterns where the Pallas kernel is meaningfully better than what XLA lowering produces on its own:

- Softcap + causal in one pass. XLA lowers the `tanh(x / softcap) * softcap` prologue into a separate fused HLO that still walks the attention matrix once more than strictly necessary. The Pallas kernel folds softcap after `sm_scale` directly into the flash attention inner loop, which means we do not re-read the score matrix. This is the primary reason we built the public Pallas softcap kernel sample at all.
- Local window with grid shrinking. A naive mask-times-scores approach pays O(T * W) mask memory and touches every block. Our Pallas kernel skips entire empty KV blocks at grid construction time and only applies the per-element mask inside the boundary blocks. This mirrors the Splash optimisation and makes sliding-window attention genuinely local in wall-clock terms, not just in math.
- Block-sparse attention with `scalar_prefetch`. A three-valued block_mask (0 skip, 1 partial, 2 full) dispatched via `scalar_prefetch` prunes the grid before the Pallas kernel launches. For patterns with moderate sparsity this is a real speedup over XLA chunked attention. XLA will happily run the full dense kernel and multiply by a mask.
- GQA and MQA via reshape/batch-fold rather than `repeat_interleave`. The `_native_fa_with_gqa_reshape` helper folds the K/V head replication into the batch dimension, so the native FA kernel sees the contract it expects. XLA's lowering of `repeat_interleave` is fine on paper and wasteful in practice.
- Document masking via `segment_ids` instead of a dense mask. This matters most for our throughput at long context: with doc-masked training, the Pallas kernel takes `segment_ids` directly and the SDPA fallback builds a dense mask that blows up memory.

Default block sizes for those kernels land at `block_q = block_k = block_k_major = 512` across the Q, dKV, and dQ passes, which is the set that both compiles on v6e-4 and v6e-8 slices and stays under the scratch memory ceiling for our head dimensions.

## 3. Where Pallas loses to XLA lowering

Equally honest about the other direction. Where we either deleted a Pallas kernel or never landed one:

- Plain causal attention without softcap and without a window. The native `torch_xla.experimental.custom_kernel.flash_attention` call is already calling into Pallas under the hood and we have no reason to write our own. The trace-pallas softcap kernel explicitly delegates to the original native FA when softcap is zero and no local window or mask is requested.
- Short sequences. The kernel launch and compile overhead on v6e is non-trivial, and for sequences below a few thousand tokens the XLA chunked attention fallback in `the POC/the TPU attention dispatch layer` is simpler, compiles faster, and runs within a few percent of the Pallas path.
- Anything that wants to dynamically change block sizes per step. Pallas likes static shapes; every time we introduced a data-dependent block size we paid for it with recompiles. We kept block sizes in `_active_block_sizes` as a global you can swap before warmup and left it alone inside the training loop.
- RMSNorm and other elementwise-plus-reduction ops. XLA's fusion here is actually very good on v6e, and every Pallas RMSNorm prototype we tried landed inside a percent or two of XLA. Not worth maintaining a separate kernel.
- Most fused-scoring sparse experiments. We prototyped a bunch of things under `experiments/sparse_pallas/` and only the pieces that clearly beat XLA survived. The remainder stayed in the experiments directory.

### A quick decision table

| Pattern | Pallas vs XLA | What we ship |
| --- | --- | --- |
| Softcap + causal | Pallas wins | the public Pallas softcap kernel sample |
| Local window with skip | Pallas wins | Same kernel, grid-shrunk |
| Block-sparse with `scalar_prefetch` | Pallas wins on real sparsity | Same kernel family |
| Plain causal, no extras | XLA / native FA wins | Delegation path |
| Short sequences | XLA wins | SDPA chunked fallback |
| Dynamic block sizes | XLA wins (or static-pin) | Static `_active_block_sizes` |
| RMSNorm-class | XLA wins | XLA lowering |

## 4. The debugging workflow that keeps us sane

Pallas debugging on TPU is the part that dominates engineering time, and our workflow is mostly built to keep us off the pod when something breaks:

- Mask classes are pure Python plus NumPy. Any composition bug reproduces in sanitized Pallas mask tests on a laptop. We never debug mask logic on a TPU pod if we can help it.
- Feature-by-feature correctness tests against an SDPA reference: basic, softcap-only, window-only, softcap + window, segment ids, backward. Every combination is its own test; they are cheap to run.
- Parity tests against Splash. If our kernel disagrees with Splash on the same inputs, Splash wins the argument until we can explain why it should not.
- Backend probing at startup. The training script emits which backend was selected per attention site and what the request looked like. A misrouted request (e.g. softcap-zero falling into the custom kernel instead of delegating) is visible in the first log line, not after a five-minute compile.
- Provenance JSON for backend selection during real runs (`tpu_backend_provenance_v6e8_2026-03-16.json`). When a configuration starts behaving differently we diff the provenance against last known good before we touch the kernel.

The native trace-pallas kernel set replaces the `call_jax`-based path for everything that needs softcap, local window, segment-ids, or composed masking. It delegates to `torch_xla.experimental.custom_kernel.flash_attention` when the request does not need any of the above. Splash is still there, under `--splash_attn`, but `enable_splash_attention()` first tries the same trace-pallas softcap/local-window kernel family, and only falls back to the legacy `call_jax` Splash if the trace-pallas path is unavailable. The legacy path is explicitly flagged as higher-overhead.

## 5. The kernels we deleted

Worth enumerating, because the deletions tell you as much about the tradeoffs as the keeps:

- Custom Pallas RMSNorm — deleted; XLA fusion on v6e was within 1-2 percent on every shape.
- From-scratch Pallas fused-scoring kernel — deleted from the main tree, kept as the public fused-scoring sample in experiments. Correct but not clearly faster than the XLA lowering.
- Pallas grouped-GEMM for MoE dispatch — deleted; cuBLAS/XLA lowering held its own and the Pallas variant cost compile-time on warmup.
- Early `call_jax`-based Splash as default — demoted to fallback only. The bridge overhead in the steady-state loop was not worth carrying.
- Hierarchical TPU attention prototype (sanitized hierarchical TPU tests) — stays in experiments; taught us where Pallas loses to XLA when per-block work shrinks.

## 6. Kernel numbers we believe

Headline characteristics we trust, all measured on v6e-4 or v6e-8 slices with real training shapes:

- Softcap + causal + doc-masked attention, long context. The Pallas path with `segment_ids` comfortably outruns the SDPA dense-mask fallback; the dense mask runs out of scratch memory before it runs out of time.
- Local window with grid shrinking. Roughly linear in window size rather than in sequence length, which is the entire point.
- Block-sparse with high zero-block fractions. Proportional wall-clock reduction once you cross the launch-overhead threshold; below that, the XLA dense path is faster.

Numbers we do not have clean comparisons for and refuse to make up: head-to-head Pallas vs Splash on every configuration (we have the parity tests, not a production bench matrix), and absolute tokens/sec deltas per feature toggle. Those are in the provenance JSON as backend identifiers, not as cleaned benchmarks. Don't cite a number you don't have.

## 7. Practical rules we ended up with

A short distillation of the v6e-specific things we had to learn:

Configuration we lock at warmup, not per step:

```python
# the public Pallas softcap kernel sample — module-level, swap before warmup, never inside step
_active_block_sizes = {
    "block_q": 512,
    "block_k": 512,
    "block_k_major": 512,
    "block_q_dkv": 512,
    "block_k_dkv": 512,
    "block_q_dq":  512,
    "block_k_dq":  512,
}
# Lazy JAX import — never at module load on a CUDA-only host.
def _ensure_jax():
    import jax  # noqa: F401
```

- Keep block sizes static and large. 512 for Q, K major, K, and the dKV/dQ blocks. Changing them per run costs recompiles; changing them per step costs retraces.
- Use `trace_pallas` for anything new. The `call_jax` bridge is still there as a fallback, but it adds latency and forces JAX to manage state we would rather manage ourselves.
- Do not import JAX at module load. Lazy `_ensure_jax()` only; our CUDA code imports mask classes from the same module.
- Write your mask composition as pure Python / NumPy. When a bug shows up, you want to reproduce it on a laptop in sanitized Pallas mask tests rather than on the TPU pod.
- Write feature-by-feature correctness tests. Basic, softcap-only, window-only, softcap + window, segment ids, backward. Every combination is worth its own test.
- Parity-test against Splash. If your kernel disagrees with Splash on the same inputs, Splash wins the argument until you can explain why it shouldn't.
- Delegate to the native kernel whenever you can. Our most-used code path is still a delegation call; we only pay the custom-kernel cost when softcap, local window, or composed masking is involved.
- When in doubt, delete the Pallas kernel and let XLA lower it. For v6e, XLA is fine at a lot of things, and maintaining a Pallas kernel that is within five percent of XLA is not worth it.

## What we kept and what we threw away

We kept the trace-pallas softcap+local-window kernel, the mask composition layer in pure Python/NumPy, the GQA reshape helper, segment-ids document masking, the `scalar_prefetch` block-sparse path, the static 512-block configuration, the delegation to native FA whenever the request is plain causal, and the layered debug workflow built on CPU mask tests plus Splash parity.

We threw away our custom Pallas RMSNorm, the in-tree fused-scoring kernel, the Pallas grouped-GEMM MoE experiment, the homegrown hierarchical TPU attention prototype, dynamic per-step block sizes, and the `call_jax`-based Splash as the default backend (it stays only as a last-resort fallback). We also threw away the temptation to maintain Pallas kernels that are merely tied with XLA; near-parity is not worth the maintenance bill.

Pallas on v6e is a precision tool. It earns its place for softcap attention, local windows, block-sparse patterns, document masking via `segment_ids`, and GQA without `repeat_interleave`. It loses to XLA for plain causal attention, short sequences, RMSNorm-class ops, and most dynamic-shape work. The kernels we ship today are the ones that crossed a measurable bar; the ones we deleted did not.

## References

- the public Pallas softcap kernel sample
- the TPU attention dispatch layer
- BACKEND_STOPLIGHT_MATRIX.md
- TPU_SETUP.md
- nsa_tpu_pallas_kernel.py
- tpu_backend_provenance_v6e8_2026-03-16.json
- TRAINING_STATUS.md
- attention_coverage_matrix_2026-03-16.md
- deferred_sparse_scope_decisions_2026-03-11.md
- test_the public Pallas softcap kernel sample
- test_splash_vs_pallas.py
- test_pallas_masks.py
- bench_pallas_fa.py
- bench_pallas_fa_v2.py
- fused_scoring.py
- importance_scoring.py
- test_hierarchical_tpu.py
- the public engineering changelog
