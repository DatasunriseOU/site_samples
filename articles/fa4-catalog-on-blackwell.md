---
title: "The FA4 Catalog on Blackwell: Variants, sm Guards, and Runtime Selection"
description: "Inside the Flash Attention 4 catalog MegaCpp ships: which kernel variants we keep, the sm_100 / sm_121a guards, the selection policy at runtime, and the validity checks that fail closed."
date: "2026-04-18"
tags: ["Flash Attention", "FA4", "CuTe", "Blackwell", "H200", "Attention Kernels"]
---

The cross-path "Flash Attention 4 in practice" post explained why we treat FA4 as four separate lines of work and not one migration. This post is the implementation companion to that one. It walks the actual catalog: which surfaces we register, what backends each surface accepts, the compute-capability gates that decide whether a request is even legal on a given device, the runtime selection policy, and the validity checks that turn a failed eligibility test into a typed fallback reason instead of a silent backend switch.

## Why MegaCpp cares about this

FA4 is two things at once. It is a CuTe DSL kernel family from the Flash Attention authors that brings real wins on Hopper and dense Blackwell ([Flash Attention 4 announcement and CuTe DSL release notes — Tri Dao et al., 2025-2026]). And it is a kernel surface that does not cover all the masks, layouts, and decode shapes our hybrid stack actually uses. If we wire it as a single backend flag we are guaranteed to either silently fall back (and lose the receipt that we were running FA4 in the first place) or to crash on a shape FA4 does not support. The catalog exists to make the first impossible and the second explicit.

The other reason this matters: our two deployment NVIDIA SKUs differ at the silicon level. Datacenter Blackwell (sm_100a, B200) has `tcgen05`, TMEM, and 2-SM UMMA. Consumer Blackwell (sm_121a, GB10 DGX Spark) does not — no `tcgen05`, no TMEM, no TMA multicast across cluster, ~128 KiB physical SMEM/SM versus 228 KiB on H100/H200. CUTLASS uses the `a` suffix (`sm_120a`, `sm_121a`) to gate block-scaled MMA paths; bare `sm_120` / `sm_121` does not have it. FA4 kernel variants land into different dispatch paths on each SKU, and the catalog has to know.

## What we built in the POC

The implementation is split across three public surfaces: the catalog itself, the dense full-attention dispatcher, and a typed validity layer that both consume.

The public FA4 catalog is pure data. Three frozen dataclasses, no side effects, importable on any Python. One describes a *surface*: a (family, stage) pair such as dense full-attention train/eval, dense prefill, dense decode, block-sparse train/eval, block-sparse shadow mode, or TPU dense runtime. One describes a *profile*: a specific contract test, observational benchmark, planner entry, or execute-proof record tied to that surface. The third describes a validation path: the build host class, test suite, and selection rule we use to validate a surface before we let it touch a training preset.

Small helper constructors keep the catalog literal readable. The two main dictionaries are built bottom-up, and a validator cross-checks that every profile points at a real surface and that every surface claiming opt-in status has at least one execute-proof profile attached. The validator runs in CI; a missing or mistyped surface ID fails the import-time test.

The fields that carry the operational contract:

- `fa4_relation`: how the surface relates to FA4 — `native_dense`, `blockized_flash`, `direct_fa4_eval_only`, etc.
- `architectural_fit`: `applicable`, `conditional`, `non_applicable`. This is the gate that says "this surface can in principle execute FA4 if the runtime allows".
- `implementation_state`: `opt_in`, `experimental`, `bounded_local`, `shadow_only`, etc.
- `evidence_state`: what kind of evidence we will accept, such as targeted tests, observed runtime, or a validated remote execution record.
- `proof_policy`: the rule for what counts as a real execute proof. The most common are: actual backend required; requested and actual backend both required; KV-cache and decode evidence both required; and "shadow mode never counts as execute proof."
- `requested_backend_aliases` and `accepted_actual_backend_values`: the names a user can request and the names the runtime is allowed to report.
- `constraint_tags`: the hard requirements of the surface — `cuda_only`, `causal_only`, `no_non_uniform_doc_ids`, `no_sliding_window`, `no_partial_valid_token_counts`, `fa3_checked_first`, `bounded_contiguous_or_paged_kv`, `no_general_varlen_dense_decode`, etc.
- `blocked_reasons`: surfaces that are recognized but explicitly not productized, such as `general_varlen_dense_decode_not_wired`.

The dense/full FA4 surfaces are where deployment lives. `dense_full.train_eval` and `dense_full.prefill` accept `dense_fa4` as the requested backend with `fa3` checked first; `dense_full.decode` is the bounded contiguous-KV plus bounded paged-KV serving path. The constraint tags are the contract: causal masking, no document-level masking, no sliding window, no valid-token-count trimming, CUDA-only. Anything else is a typed fallback.

The sparse surfaces are deliberately separated. `block_sparse_flash.train_eval` is the blockized sparse CUDA surface that *may* execute FLASH/CuTe kernels under the MoBA / BlockIndexer router; `block_sparse_flash.shadow` is the shadow path where we record the requested backend in telemetry while the actual runtime is forced to Triton — the proof policy `shadow_never_counts_as_execute_proof` exists because we kept getting asked to count shadow runs as deployment receipts. The exact-token DSA surfaces (`exact_token.eval`, `exact_token.train_opt_in`) are bounded to direct-FA4 with no general deployment decode. TPU dense and clustered sparse surfaces exist for cross-platform completeness but never claim FA4 execute proofs.

The public flash-attention module is the dense dispatch implementation. The relevant bits:

- `_load_flash_attention()` probes `torch.cuda.get_device_capability()` and only loads our local FA3 build for `sm in (90, 121)` — Hopper (H100, H200) and GB10 DGX Spark. Bare Blackwell (sm_100) and Ada (sm_89) get an SDPA fallback because our local FA3 build does not target them; FA4 is a separate path layered on top of FA3.
- `_validate_dense_fa4_eligibility()` is the typed gate. It returns `(False, reason)` with one of `dense_fa4_requires_causal`, `dense_fa4_no_doc_ids_support`, `dense_fa4_no_sliding_window`, `dense_fa4_no_valid_token_counts`, `dense_fa4_cuda_only`. These reason tags are the same strings the catalog enumerates as constraint tags, by design.
- `_flash_cute_runtime_status()` and `_dense_fa4_runtime_status()` are LRU-cached probes that check the CuTe runtime is actually loadable. The same probe gates dense full FA4 and the FlexAttention `BACKEND="FLASH"` path. The compute-capability check inside `_probe_flex_flash_backend()` accepts `major in (9, 10, 11)` — Hopper, datacenter Blackwell, consumer Blackwell — and rejects everything else with an explicit `unsupported compute capability sm{major}{minor}` message. The 9/10/11 split is what maps to "FA4 is allowed to dispatch here at all".
- `get_last_dense_fa4_result()` and `get_last_dense_fa4_kvcache_result()` expose the per-call dispatch outcome (`"executed"`, the eligibility reason, or `None` when not enabled) so receipt and measurement code can read why FA4 was bypassed without parsing logs.

The public attention-validity module is the small dataclass layer that normalizes the validity contract. `AttentionValidity` carries one of three modes — `none`, `token_prefix`, `slot_prefix` — plus the optional `token_prefix` tensor for trimming the last partial slot. `normalize_attention_validity()` collapses legacy `attention_meta` keys and explicit kwargs into a single typed object so the dispatch layer never has to guess. The `has_partial_token_prefix()` helper is the one the FA4 eligibility check calls to decide whether the request hits the `dense_fa4_no_valid_token_counts` rejection.

## How it lands in MegaCpp

The catalog lifts as-is. It is the contract; rewriting it would mean re-litigating every surface decision, and the data is what protects against drift.

The dense dispatch layer is the file that gets the most rewriting in MegaCpp. It becomes the only entry point; the FlexAttention and FA3-direct paths stay as fallbacks, but the public API is a single attention call plus the catalog-driven backend selector. Per-call dispatch state moves from module-level globals into a structured backend-dispatch record that the training loop attaches to each step's metrics, so the per-step evidence is structured rather than scraped from logs.

Compute-capability gates are tightened. The local FA3 build floor remains `sm in (90, 121)` for the FA3 path. Dense FA4 is allowed only on `sm in (90, 100, 121)` — Hopper, datacenter Blackwell, GB10 DGX Spark — with two carve-outs:

- On sm_100a (B200) we also wire the `tcgen05` / TMEM-aware FA4 backward adapter; the cute_dsl_mimo backward kernels we have prototyped target this surface and graduate behind a separate flag.
- On sm_121a (GB10) the FA4 path uses the BF16/FP16 CuTe DSL kernels that we have validated on consumer Blackwell. There is no `tcgen05` and no FP4 tensor-core path, and the catalog enforces this by rejecting the `dense_fa4_fp4` requested backend on this SKU.

Several FA4 variants get rewritten at the kernel layer: the dense-full backward fused epilogue moves from the prototype CuTe DSL adapter to a proper CuTe build that lives in the MegaCpp Megatron extension. The forward dense path is whatever upstream `flash_attn.cute.interface` ships once we have a stable Tri Dao tag pinned.

What we drop: the four old `fa4_*`-named presets that ran Triton FlexAttention rather than FA4 CuTe (because they did not set `moba_flex_backend="flash"`). They survive in the catalog as documented historical entries with the explicit note that they did not actually execute FA4. The real FA4 CuTe presets are the depth-52 variants pinned to `moba_flex_backend="flash"`. The mislabeled presets do not graduate.

What becomes a feature flag: every dense FA4 surface is opt-in (`--dense_fa4`); every blockized sparse FLASH path is opt-in via `moba_flex_backend=flash`; every exact-token direct FA4 path is `eval`-only by default and requires `--exact_token_fa4_train` to enter the training surface. The shadow path is gated separately and never counts as an execute proof.

What moves to a kernel/Triton path: the blockized sparse path uses FlexAttention's `BACKEND="FLASH"` kernel options when CuTe is available, with Triton as the fallback. The block-mask construction stays in Python. The exact-token DSA path uses direct FA4 for eval and a chunked-gather Triton kernel for the rest.

## Ablations and what we kept

The catalog itself is the ablation history compressed. Every entry exists because we burned cycles on a backend choice that someone else wanted to make a one-flag decision. A few specific lessons:

- Keeping `dense_full.decode` *separate* from `dense_full.prefill` was the difference between a clean serving rollout and a panic. Decode lives on a different runtime path with KV-cache contiguity guarantees, and "FA4 train works, so FA4 decode should too" is the kind of false symmetry that wastes weeks. The bounded contiguous-KV and bounded paged-KV decode receipts on H200 are real (March 2026); a general varlen dense decode is explicitly `blocked_reasons=("general_varlen_dense_decode_not_wired",)`.
- The CuTe FA4 backward parity smoke against the dense FA3 baseline on the depth-52 preset showed loss and gnorm trajectories within noise across 8 steps; that was the gate to promote dense FA4 from shadow to opt-in. Without backward parity, no execute receipt counts.
- The FA3 control-ppath regressions that killed early FA4 candidates were all on the eligibility side: a CuTe variant that swallowed a non-uniform `doc_ids` tensor and produced silently wrong outputs (caught by `dense_fa4_no_doc_ids_support`); a tuple-vs-tensor return-type mismatch (caught by `dense_full.df01_cpu_tuple_unwrap`); a sliding-window request that was masked correctly by FA3 and ignored by an early FA4 build (caught by `dense_fa4_no_sliding_window`). Each of those is now a contract test in sanitized dense FA4 tests and sanitized flash-attention tests.
- The exact-token `fa4_gather` smoke on H200 ran clean once we unwrapped the CuTe tuple before dtype casting; that fix moved from a per-call workaround into the dispatch layer and the catalog now requires `actual_attention_path='fa4_gather'` plus an empty fallback reason for the proof to count.
- The FlexAttention `BACKEND="FLASH"` path needed a stricter capability probe than the Inductor default; the shared `_probe_flex_flash_backend()` exists because Inductor was happy to import and then fail at codegen on unsupported devices. Failing earlier is the whole point.

What we tried and dropped:

- A "smart" auto-selector that picked FA4 over FA3 based on shape heuristics. The selector landed on the wrong side of the eligibility cliff for several specialist configurations and there was no operator-readable reason for the choice. We replaced it with explicit opt-in and the typed eligibility check.
- A unified blockized sparse plus dense FA4 dispatch path on the theory that the kernel runtime is the same. The semantics are different (sparse versus full attention), the proof contracts are different, and conflating them was how shadow runs leaked into deployment receipts.
- A speculative-decode path built on dense FA4 decode. The catalog has `non_applicable.na03_speculative_decode_gate_closed` documenting why the gate stays closed: the bounded decode receipts do not transfer to the speculative path's variable-length KV pattern.

## Production checklist

- Use the FA4 catalog validator as an import-time test in CI; catalog drift fails the build.
- Keep dense, blockized sparse, and exact-token FA4 lines separate; never share a single `--fa4` flag across them.
- Gate dense FA4 to `sm in (90, 100, 121)`; reject everything else with the explicit capability message.
- Pass `_validate_dense_fa4_eligibility` as the only entry to the dense FA4 dispatch; the typed reason strings are the contract.
- Treat shadow paths as telemetry, never as execute proofs (`shadow_never_counts_as_execute_proof`).
- Require `actual_backend_required` (or stronger) proof policies before promoting any FA4 receipt.
- Emit a structured per-step backend receipt; do not rely on log scraping.
- Keep the FA3-checked-first rule in dense surfaces; FA4 is the upgrade, FA3 is the fallback floor.
- On GB10 (sm_121a), reject FP4 and `tcgen05` requests at the catalog layer; only the BF16/FP16 CuTe DSL paths are valid.
- On B200 (sm_100a), wire the `tcgen05` / TMEM backward adapter behind a separate feature flag; keep it off the default opt-in path until parity receipts land.

## Catalog surfaces

| Surface | Backends | sm guard | Proof policy |
|---|---|---|---|
| dense FA4 | CuTe DSL | sm in {90, 100, 121} | `actual_backend_required` |
| blockized sparse FA4 | CuTe + sparse mask | sm in {100, 121} | shadow != execute |
| exact-token DSA | DSA indexer + FA4 | sm in {100, 121} | typed eligibility check |
| FA3 fallback | FA3 | sm >= 80 | always available |

```python
# Typed eligibility - failure returns a reason string, never silently switches.
ok, reason = _validate_dense_fa4_eligibility(shape, dtype, device)
if not ok:
    return dispatch_fa3(shape, dtype, device, reason=reason)
```

## References

- public FA4 catalog, dispatch, validity, backend-matrix, and sparse-attention modules in the MegaCpp project
- [Flash Attention 4 announcement and CuTe DSL release notes — Tri Dao et al., 2025-2026]
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — Shah et al., 2024]
- [FlexAttention with FLASH/CuTe backend — PyTorch blog]
- [NVIDIA CUTLASS sm_120a / sm_121a guidance — NVIDIA CUTLASS issues #2800, #2947, #3044, #3100, #3144]
