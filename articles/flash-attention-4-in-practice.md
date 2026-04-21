---
title: "Flash Attention 4 in practice: what we shipped and what we cut"
description: "Our hybrid stack's applicability matrix for Flash Attention 4, the Build4 validation profiles, the dense-full rollout gates, and the regressions that killed the first FA4 variants before they reached deployment."
author: "David Gornshtein"
date: "2026-04-18"
tags: ["Flash Attention", "FA4", "CuTe", "H200", "Attention Kernels", "Dense Attention"]
readTime: "13 min read"
---

# Flash Attention 4 in Practice: What We Shipped, What We Cut

Flash Attention 4 is not a drop-in replacement for Flash Attention 3. In our
hybrid Mamba 3 plus Transformer stack, the dense Transformer blocks are a
minority of compute but most of the attention-shaped risk, and FA4 changes
enough of the kernel contract that "just turn it on" is the wrong verb. What
we actually shipped is a bounded opt-in path on the canonical H200 stack, with
a fail-closed applicability matrix, a staged rollout, and a list of variants
that looked fine in microbenchmark and were rejected on contract grounds.

This post describes that work the way we track it in code and public receipts: applicability
split, Build4 test profiles, the dense/full rollout manifest, the hybrid
prefill/decode plan, and the FA3 control-ppath regressions that killed the
early FA4 candidates before we ever let them touch a main training run.

## Not One Migration, Four Separate Lines

The first non-negotiable rule we wrote down is that "FA4 migration" is not a
single line of work. Collapsing the lines is how you end up promoting sparse
donor-side evidence as if it were dense deployment proof. The taxonomy we
actually track splits along five axes:

- `dense/full FA4`: the upstream `flash_attn/cute` dense full-attention path
  in our main dispatcher.
- `blockized sparse CUDA`: blockized sparse routing that may execute on
  Triton or on CuTe/FLASH, with MoBA-style block selection.
- `exact-token DSA`: token-topk sparse semantics including the bounded
  eval/no-grad `fa4_gather` path.
- `serving/decode`: product-surface decode truth, which is stricter than
  model-level plumbing.
- `non-applicable`: Mamba-3 and Gated DeltaNet surfaces, which are not
  attention and must stay labeled that way.

Each line gets its own execution evidence, its own promotion gate, and its own
fail-closed guards. A checked-in execution record on one line never implies parity on
another. We encode this in a machine-readable catalog so that the planner, the test runner, and the evidence grader all agree on surface IDs, profile IDs, and family boundaries, and we use a companion stoplight matrix for prose synthesis. Truth order when they
disagree: current code on `main` wins, then open task state, then the
catalog, then the matrix, then dated reports and execution records. Docs never
outrank code.

## Applicability: What Actually Counts As Proof

Applicability is where most FA4 enthusiasm goes to die. Our working matrix
is roughly this:

- Dense/full FA4 applies now to bounded CUDA causal fixed-length
  train/eval/prefill, plus bounded contiguous-KV and paged serving/runtime
  decode. Proof requires `actual_backend=dense_fa4` on a real bounded
  execute surface. A `--dense_fa4` CLI flag, a `prefill_backend=dense_fa4`
  config entry, or a planner manifest does not count.
- Blockized sparse FA4 applies to bounded CUDA blockized sparse train/eval.
  Proof requires the full four-tuple: `requested_backend`, `actual_backend`,
  `runtime_mode`, and `fallback_reason` together. Preset names and `shadow`
  runs do not count.
- Exact-token DSA applies only to exact-token semantics, including bounded
  eval/no-grad `fa4_gather`. Donor/runtime compare and blockized sparse
  FLASH execution records are not substitutes.
- Serving/decode applies to bounded dense-first serving with contiguous-KV
  and to paged or scheduler-managed bounded serving paths. Proof requires
  engine/runtime evidence on the real bounded serving path; control-ppath
  config acceptance does not count.

The non-negotiable guards behind the matrix: `shadow` never counts as sparse
FA4 execute proof; donor runtime comparison never counts as mainline execute
proof; TPU rows stay TPU-native and non-FA4; Mamba-3 and Gated DeltaNet stay
non-applicable; and a `ServingConfig(prefill_backend="dense_fa4")` plus a
bounded `decode_backend="dense_fa4"` still do not imply paged or
scheduler-managed productization. Those knobs are control-ppath surface
until an engine-level FA4 prefill path actually executes.

## Build4 Validation Profiles

Build4 is our local validation host for the FA4 code and test substrate. The
profile catalog is code, not prose: a machine-readable wave structure lives
alongside the dispatcher and is rendered into pytest commands by a helper
script instead of being hand-maintained in shell history. Each profile
carries four schema fields that keep overclaim out of the evidence stream:

- `proof_class`, one of `planner`, `observational`, `validation`, or
  `execute`.
- `artifact_kind`, the concrete substrate: `contract_test`,
  `planner_manifest`, `launch_manifest_pending`,
  `observational_profile_record`, `matrix_summary_record`,
  `microbenchmark_case_record`, `validated_remote_execution`, or
  `negative_guard`.
- `evidence_grade`, matching `proof_class`.
- `counts_for_status`, which is `True` only for real execute-grade rows.

Wave 1 is dispatcher-contract coverage on the fixed-length no-KV slice. It
runs CPU and small-CUDA tests on the public flash-attention backend matrix,
serving config rejection of `prefill_backend="dense_fa4"` with paged-KV, the
train-args wiring, and the Modal CuTe contract. Wave 2 crosses into bounded
H200 canonical execution records: a CuTe import-plus-tiny-forward smoke, an H200
dense-decode smoke, an exact-token `fa4_gather` H200 smoke, and the
small-shape CuTe train ladder evidence family. Wave 2 does not produce promotion rows;
it produces the evidence that wave 3 (rollout executions) is allowed to run at
all.

## The Real Blocker: Dispatch Order

The single biggest reason dense/full FA4 spent a month as "real code, not yet
a runtime runtime" is dispatch ordering. Our `flash_attn_func` checks FA3
before dense/full FA4 on CUDA paths where FA3 is available. That means
`--dense_fa4` is not equivalent to "must execute dense/full FA4" on a normal
H200; it means "dense/full FA4 is enabled as an optional branch" while FA3
usually wins first. The fix is not a `raise` on FA3 availability; the fix
is a dedicated selector:

- `enable_dense_fa4_attention()` and `disable_dense_fa4_attention()` as
  explicit entry points.
- A `_use_dense_fa4(device)` helper that stays separate from the
  sparse backend knobs (`moba_backend=fa4`,
  `donor_runtime_compare=fa4`, `block_sparse_runtime_mode`).
- A `doc_ids` contract that is an explicit policy decision and not an
  implicit fallback. Uniform rows normalize away upstream and may still
  dispatch on dense FA4; non-uniform rows are rejected with a
  machine-readable tag (`dense_fa4_no_doc_ids_support`) and fall through to
  another dense backend rather than raising.

Per-call truth is recorded in `_last_dense_fa4_result`, so a test or a
execution evidence can assert that the dispatcher actually took the FA4 branch rather than
inferring it from presets.

## Dense/Full Rollout: L0 to R7

The rollout is a manifest, not a vibe. The machine-readable companion lives
in the project and drives the helper that emits manifests and runnable commands
templates. The rungs:

- L0: reference dispatcher contract is green on the fixed-length no-KV slice.
- L1: comparison/profile evidence schema is stable; candidate rows fail
  closed when FA4 did not actually execute.
- R1/R2: bounded H200 canonical execute proof with explicit
  environment-bound rows.
- R3: Modal detached execute proof with the same shape (batch=1, seq=128,
  n\_head=8, head\_dim=64, bf16, causal-only, no KV cache, CuTe interface
  import check).
- R4: no-KV dense comparison matrix, six configs plus baseline, with
  complete perf/memory/diff fields and preserved actual-backend truth.
- R5: profiler evidence on the canonical stack, driven by
  the public FA4-vs-Triton dense profiling harness.
- R6: bounded 2-step train and 100-step short-train execution records, no NaN loss,
  loss-convergence ratio not worse than 1.05 vs the dense reference preset.
- R7: hybrid prefill execution record with explicit decode truth; promotion gate
  passes with explicit execute-proof evidence.

Stop rules are concrete: `max_abs_diff > 0.01` on any required row, any
NaN/Inf output, throughput regression over 10% versus the dense H200
reference, memory increase over 5%, a failed 2-step train, or any execution record
claiming decode/KV-cache support for dense/full FA4 that has not actually
executed that path. The last one is the rule that catches the most eager
reports.

## Hybrid Prefill, Explicit Decode

The first honest hybrid plan is narrow: prompt-prefill may use upstream
`flash_attn/cute`; decode stays on an already-supported path. Receipts at
this rung record machine binding, the exact prompt length, causal policy,
`doc_ids` presence, the observed prefill result (including the dispatcher's
requested-vs-effective backend truth and any fallback reason), and an
explicit decode field: either `decode_executed=false` or
`decode_backend="fa3"` with the name spelled out. An execution record that silently
mixes FA4 prefill with unlabeled decode fallback is not valid evidence.

Decode itself is still blocked on prerequisites we refuse to wave through:
a public decode/KV-cache contract for dense/full FA4 in the dispatcher, a
paged-KV decode semantics proof on that line, a fail-closed `doc_ids` and
packed-doc policy for decode, a bounded canonical-H200 token-by-token
decode execution record, and a parity matrix after the prefill handoff. The
promotion-gate config keeps `no_kv_cache_support` active for both
`prefill_promotion` and `full_train_promotion`, so the gate itself blocks
honestly when these are missing.

## Regressions That Killed Early Variants

Three classes of regression accounted for nearly every rejected candidate.

The first was fail-open inference in the promotion gate. An earlier gate
marked a dense/full candidate ready as soon as all comparison rows passed,
without requiring explicit execute-proof rows for both `h200_canonical` and
`modal_h200`. That is exactly how "all green" promotions happen without a
single real execute record. We removed the inference, forced
environment-bound rows, and renamed the canonical metrics to
`median_tok_per_sec` and `peak_memory_mib` so legacy field names
(`throughput_toks`, `peak_memory_gb`) can still parse but cannot be the only
evidence.

The second was a control-ppath dtype regression on the exact-token FA3
path. Our chunk-metadata candidate vectorized planning work and restored
`row_cu_seqlens` to `int64` on the layout-facing side while keeping `cu_k`
`int32` on the kernel-facing side. Baseline: 485,376 tok/s; candidate:
552,397 tok/s; delta: +13.8%; `peak_memory_mib` unchanged. Same exact-token
backend, same packer path, same `chunk_plan_count=64`, same
`runtime_row_metadata_prepared_once=true`. We kept it because the runtime
identity did not drift. The earlier non-dtype-correct candidate looked
similarly good in microbenchmark but silently changed the metadata contract
consumed by tests and chunk-layout code, and that is the kind of
"improvement" that turns into a week of triage two refactors later.

The third was serving-config overclaim. An older shape of
`AdapterServingEngine.from_config()` accepted
`prefill_backend="dense_fa4"` combined with paged-KV and continuous
batching, because the engine routed prefill through a replay path that
never asked the dispatcher which backend actually ran. We made
`ServingConfig` fail closed on those combinations and made
`AdapterServingEngine.from_config()` also fail closed until an engine-level
FA4 prefill path exists. Now the config surface blocks the claim instead
of enabling the illusion.

## The Honest Verdict

Dense/full FA4 is real code, a real bounded opt-in path, and not yet a
general deployment-grade runtime. Code-wired: yes. Dispatcher-contract
tests: yes. Canonical H200 execute proof: yes. Modal execute proof: the
path exists and the helper emits the command, but the checked-in execution record is
still pending. No-KV dense comparison matrix, bounded prefill, and bounded
short-train: planned and gated, not executed on `main` yet. Decode and
KV-cache: blocked on the prerequisites above.

Shortest honest description for the rest of the team: dense/full FA4 is a
bounded experimental execution path with one canonical H200 smoke record
and a staged rollout, separate from exact-token DSA and from blockized
sparse CUDA/FA4. The value of writing it down this way, rather than
collapsing it into a single "FA4 is live" bullet, is that the next person
to touch the dispatcher can see exactly which line they are standing on and
which execution records they still owe.

## Lane status

| Lane | Code wired | Dispatcher tests | Canonical H200 execution record | Modal execution record |
|---|---|---|---|---|
| dense / full FA4 | yes | yes | yes (smoke) | helper emits, evidence pending |
| blockized sparse FA4 | yes | yes | partial | not started |
| exact-token DSA + FA4 | yes | yes | yes | partial |
| FA3 fallback | yes | yes | yes | yes |

```python
# Fail-closed applicability - dense FA4 is opt-in only.
def select_backend(req):
    if not req.opt_in_fa4:
        return "fa3"
    ok, reason = _validate_dense_fa4_eligibility(req.shape, req.dtype, req.sm)
    return "fa4_dense" if ok else ("fa3", reason)
```

## References

- [Dense FA4 execute proof sample](../examples/kernels/dense_fa4_execute_proof_sample.py)
- [Dense FA4 KV-cache decode sample](../examples/kernels/dense_fa4_kvcache_decode_sample.py)
- [FA4 receipt summary sample](../examples/distributed/fa4_receipt_summary_sample.py)
- [Kernel examples index](../examples/kernels/index.md)
- [Kernel examples README](../examples/kernels/README.md)
