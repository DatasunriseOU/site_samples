---
title: "Attention Validity and Structure-Aware Attention"
description: "A packed-row validity regression, the clustered-sparse follow-up it forced, and the structure-aware attention plan we are integrating into the MegaCpp training stack."
author: "David Gornshtein"
date: "2026-04-18"
tags: ["Attention Validity", "Packed Rows", "Clustered Sparse", "Structure-Aware Attention", "Code Modeling"]
readTime: "13 min read"
---

# Attention Validity and Structure-Aware Attention

A lot of model quality on long C++ contexts hides inside boring metadata:
which tokens in a packed row are actually valid, which blocks a clustered
sparse router is allowed to attend to, and whether a piece of structural
information is expected to be present, absent, or explicitly zero. When
any one of those three states gets silently collapsed into another, the
consequences are not noisy - they are quiet and plausible, and they
survive through training because nothing obviously explodes. This post
walks through one regression we fixed, the clustered-sparse follow-up
that regression forced into the open, and how the result is feeding the
structure-aware attention plan we are integrating next.

## The Packed-Row Validity Regression

We train on packed rows where multiple documents share a single
fixed-length sequence. Attention-validity metadata tells downstream
kernels which tokens in the row are real, where each document starts and
ends, and how to mask across document boundaries. On current main, the
training script canonicalized that metadata for two paths in particular:
CUDA + FSDP distributed, and XLA. Both canonicalizers had a seemingly
harmless rule: if any validity tensor is present in the batch,
zero-fill all the other validity fields that are missing, so that every
path downstream sees a uniformly shaped dict.

For packed-row metadata that is slot-prefix-only - meaning the batch
supplies block-level slot counts but deliberately omits token-level
counts - the "zero-fill missing" rule rewrote several absent fields into
zero tensors:

- missing `row_valid_token_counts` became a zero tensor.
- missing `row_valid_block_counts` became a zero tensor.
- missing `row_block_size_tokens` became a zero tensor.
- missing `base_block_tokens` became a zero tensor.

The function is called shape canonicalization, but the semantic content
it was producing was not shape. It flipped "token-prefix is absent" into
"token-prefix is explicitly zero." Downstream consumers of
`normalize_attention_validity()` read that as a real, zero-length
token prefix, which is exactly the same as telling the kernel "this row
has zero valid tokens." No crashes, no NaNs, just silently-masked rows
inside training batches.

The affected code was upstream of the attention path itself. The bug
was not in `attention_validity.py` or
`flash_attention.py` or the clustered sparse module; it was in
the main training entrypoint, in
`_canonicalize_structure_meta_for_fsdp_cuda` and
`_canonicalize_structure_meta_for_xla`. The fix is one-line philosophy
applied in two places: missing validity fields stay absent. Structural,
platform, and tree metadata continue to be shape-stabilized as before
- those were never the problem - but optional validity fields are now
preserved in their original present/absent state. Slot-only metadata
now stays slot-only, and `normalize_attention_validity()` sees
`slot_counts` present and `token_prefix` absent unless the loader
actually supplied one.

The fix landed with regression tests across the obvious surfaces:
attention-validity tests, targeted coverage in
training entrypoint regression tests for the CUDA and XLA
canonicalizers preserving absent token-validity fields while still
injecting the missing required CUDA/XLA keys, the
attention-validity integration tests, and a
flash-attention test run to confirm nothing downstream
regressed. The durable rule for future canonicalizers is now written
down: missing validity fields must stay absent unless the runtime is
intentionally deriving them from a stronger contract. Shape
stabilization and semantic injection are different operations and
should never share a code path.

## The Clustered-Sparse Follow-Up

Fixing the canonicalizer closed the source of the regression but did not
close an ambiguity it exposed downstream. Our clustered sparse
attention contract has a function
(`_resolve_attention_validity_contract`) that keeps `slot_prefix`
metadata only when the batch's `base_block_tokens` equals the attention
kernel's `query_tile_size`. When those do not match, the code had a
fallback path, and the fallback had never been covered by a test.

Enumerating the cases: `slot_prefix` is present, but `base_block_tokens`
does not match `query_tile_size`. Two subpaths:

- If an auxiliary `token_prefix` is present in the batch, clustered
  sparse falls back to `token_prefix` and proceeds with token-level
  validity.
- If no auxiliary `token_prefix` exists, the contract degrades to
  `AttentionValidity(mode="none")`. Clustered sparse attention then
  behaves as if no validity metadata was ever supplied.

That second branch is what the packed-row fix made visible. Before the
fix, slot-only rows with mismatched block sizes would pick up a
zero-filled `token_prefix` from the canonicalizer and hit the
"fall back to `token_prefix`" branch, with `token_prefix` semantically
meaning "zero valid tokens." After the fix, they hit the
`mode="none"` branch, which is also the explicitly documented behavior.
Neither behavior is a crash; both are product decisions about what
"slot-only metadata, wrong block size, no token fallback" should mean
on the clustered sparse path.

The follow-up added one targeted regression test that pins the current
explicit behavior: slot-prefix-only metadata, mismatched
`base_block_tokens`, no auxiliary `token_prefix`, the contract degrades
to `mode="none"`. The residual risk is honest and written into the
report: this is now explicit, but it is still a product decision rather
than a final semantic contract. A future follow-up may choose stricter
strict fallback behavior, or preserve a coarse slot contract instead. The
value of the test is that whichever choice we make later, we will make
it on purpose.

## Attention Validity, Presence, and Absence

The general rule the packed-row incident forced us to write down:
attention validity metadata has three states, not two.

- Present and populated. Use it. The kernel gets real counts and
  masks.
- Absent. Do not invent values. Downstream code selects a lower-signal
  contract (for example, degrading from token-prefix to slot-prefix, or
  from slot-prefix to `mode="none"`), and does so explicitly.
- Present and zero. This is a real semantic state that means
  "zero valid tokens," and it must only ever arise because the loader
  or runtime intentionally produced zero. It is never the result of a
  missing tensor being normalized to zero.

Canonicalization code at any level of the stack must preserve the
absence vs present-zero distinction. That is the rule we now enforce
in tests, and the rule we require of any new metadata path.

## Why This Matters for a Code Model

The failure mode here is not theoretical. For C++ code packed into
long training rows, slot-only metadata is common: we often know block
boundaries from the chunker without materializing token-level counts in
the loader. A quiet degradation to "zero valid tokens" on a fraction of
those rows looks, in loss curves, like a subtle data-quality problem or
a subtle learning-rate problem, depending on your priors. It does not
look like a canonicalizer bug, which is what it is.

This is also why we are paying more attention to attention-coverage
receipts. Current truth from the coverage matrix on our reference accelerator lane:
blockized sparse FA4/FLASH has a bounded execute receipt on the pinned
support region `{8, 9, 10, 11}` with `actual_backend=fa4` checked in for
2/8/20-step single-GPU and 8-step FSDP2 2-GPU, but `shadow` and `off`
still execute Triton, and `active_eval` still lacks a checked-in
passing execute receipt. Triton FlexAttention remains the broadest
block-sparse reference. Exact-token DSA keeps `dsa_backend=sdpa` as
the literal default, with `chunked` as the production-safe sparse path
when explicitly selected and `fa3_gather` bounded to eval/optimization.
Dense/full CuTe FA4 is real code on a bounded opt-in lane with one
canonical accelerator smoke, no KV-cache proof, no broad train/prefill
acceptance receipt. Sparse decode exists in
`sparse_attention.py` but serving config still rejects it.
These are the surfaces the validity contract has to stay correct on.

## Recommendation Hierarchy Before We Change the Kernel

The packed-row fix plus the clustered-sparse test kept the existing
dense path honest. The next move is a post-attention gate, not a
kernel rewrite. Our review of the sink/spike, gated-attention,
streaming-sinks, and DynamicTanh papers produced an explicit
recommendation order that lines up with what our backends can actually
accept without breaking contracts:

- First: an optional query-dependent sigmoid gate after the attention
  output, applied before `c_proj`. This addresses attention sinks more
  directly than bounded softcap squashing, stays backend-agnostic
  across FA3, CuTe FA4, FlexAttention, Pallas, and Splash, and
  preserves the current `qk_norm`, `qk_clip`, and softcap logic instead
  of replacing them.
- Second: instrumentation and a packed-doc audit. First-token
  attention mass, max and high-percentile hidden activations,
  prefix-vs-suffix usage on packed documents, and sink behavior per
  document rather than per row. Part of the observed bias likely
  comes from `best_fit` packing cropping document prefixes and
  oversampling document starts, which is a data question, not an
  attention-math question.
- Third: DynamicTanh as a separate research track. Larger
  architectural blast radius, initialization sensitivity, and full
  train-dynamics impact make it wrong as the first production-facing
  mitigation step.
- Fourth: sink-aware serving and KV-cache follow-up. Sparse decode
  can reduce KV reads; paged KV is the real route to storage savings;
  sink-window retention is a bounded serving heuristic, not a claim of
  long-dependency preservation for code.

The gated-attention V1 spec is deliberately narrow: one config group
(`attn_output_gate`, `attn_output_gate_granularity="head"`,
`attn_output_gate_bias`, `attn_output_gate_init="identity_bias"`,
`attn_output_gate_log_stats`). The gate initializes close to identity -
weight near zero, bias positive so `sigmoid(bias)` is near 1 - so V1
does not destabilize existing checkpoints. The dense module gains a
per-head `c_gate` linear and multiplies the per-head attention output
by `sigmoid(c_gate(x))` before flattening into `c_proj`. The sparse
module mirrors the same math through a shared helper in the
`_full_attention` path and the `_finalize_sparse_output` exit, so
dense and sparse do not drift semantically. Checkpoint compatibility
is explicit: gate parameters exist only when the feature is enabled,
old checkpoints load silently with the feature off, and enabling the
feature adds clean missing-key behavior if operators do it
deliberately.

## Structure-Aware Attention Integration

Longer-term, the attention-validity work feeds into the
structure-aware attention plan. The thesis is short: code structure
is mostly an offline fact. We already have tree-sitter chunks, clang
call/type edges, and token-aligned AST metadata available in the
enriched parquet contract; we just stop re-deriving weak versions of
them online.

The proposed end-state attention contract for code has three parts:
a local causal window, exact structural neighbors from an offline
graph IR, and a small learned overflow budget for edges the graph did
not capture. Execution stays block-friendly: semantic blocks derived
from `chunk_boundaries` replace fixed 128-token blocks, fixed-K
neighbor lists come from offline preprocessing, and the runtime layout
fits existing block-sparse plumbing. The four-path incremental rollout
sequences it as graph-bias first (relation-aware bias, no hard masking
beyond causal/doc masks), then semantic block sparse (chunk-aligned
blocks in place of fixed-size MoBA blocks), then offline sparse
structure IR (fixed-K neighbors fed to the model, eliminating the
online generic router for structure-driven layers), then a hybrid
overflow router on top.

The required integration contract is strict: training consumes token
IDs, token-aligned metadata, and precomputed sparse/graph IR. No
text-to-token, no int-to-string-to-AST, no runtime tree-sitter or
clang. Parquet rows used for training should already contain or
cheaply derive `input_ids`, `target_ids`, `doc_ids` (or enough to
reconstruct them from BOS), `token_structure_ids`,
`token_dep_levels`, optional `token_ast_depth`, `token_sibling_index`,
`token_ast_node_type`, token/chunk graph metadata (`token_chunk_starts`,
`token_chunk_ends`, `token_chunk_dep_levels`, `token_call_edges`,
`token_type_edges`), and, once available, the precomputed sparse
structure IR itself. The preprocessing parser stack branches cleanly:
tree-sitter v11 for the syntactic line, clang v12 for the semantic
line, both converging on one token-only consumer contract with
identical packed-row semantics.

## Where the Validity Rule Meets the Structure Plan

The packed-row fix is the reason the structure plan can move forward
without relitigating basic metadata semantics. Offline graph IR,
semantic blocks, and fixed-K neighbor lists all expand the set of
optional tensors each batch can carry. If the canonicalizer's old
"zero-fill missing" rule were still in place, every new optional
tensor would be one more silent zero-equals-absent trap waiting to
happen at scale. With the absence-preserving rule enforced and tested,
we can add structural fields to the contract confidently: present and
populated means use the graph, absent means degrade to a lower-signal
contract explicitly, and present-zero is reserved for the case where
the preprocessing actually emitted zero.

Two immediate next moves follow from this. The gated-attention V1
ablation runs on the current dense and sparse paths without touching
validity, to separate the architecture effect from the packing effect.
In parallel, the structure pipeline produces its first fixed-K
neighbor IR per chunk, with explicit presence/absence semantics,
loaded through the same canonicalization discipline. By the time the
semantic-block sparse path lands, the validity contract has already
been proven to survive two new optional-metadata introductions
without quiet regressions. That is the only scaffolding that makes
the structure-aware attention integration safe enough to run.

## The Short Version

A canonicalizer that zero-filled missing attention-validity fields
silently converted "absent" into "zero valid tokens" for slot-only
packed rows. The fix preserves absence; the regression tests pin it;
the clustered-sparse follow-up pins the explicit fallback to
`mode="none"` when `slot_prefix` metadata and `query_tile_size` do not
agree and no `token_prefix` exists. The durable rule - present,
absent, and present-zero are three distinct states and none of them
may be invented by shape canonicalization - is now enforced across
the CUDA+FSDP and XLA paths. On that foundation, the gated-attention
V1 work and the structure-aware attention integration can proceed
without paying rent to a metadata ambiguity we have already fixed.

---

## Validity states at a glance

| State | Meaning | What the kernel must do |
|-------|---------|-------------------------|
| Present, nonzero | explicit valid-token count | use as-is |
| Present, zero | row is intentionally empty | skip row, do not synthesise |
| Absent | metadata omitted for this row | fall back via contract, never invent |

```python
# fallback rule applied in the canonicalizer
if token_prefix is None and not slot_prefix_matches(query_tile_size):
    return ValidityMode.NONE  # explicit absence, not zero-fill
```

## References

Filenames only: `attention_coverage_matrix_2026-03-16.md`,
`attention_validity_clustered_followup_2026-03-10.md`,
`attention_validity_packed_rows_2026-03-10.md`,
`12-attention-sink-mitigation-rfc.md`,
`13-gated-attention-v1-spec.md`,
`14-structure-aware-attention-and-feature-integration-plan.md`.
