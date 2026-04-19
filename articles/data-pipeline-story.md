---
title: "Building the C++ Training Data Pipeline: What Worked, What Broke"
description: "An honest walkthrough of how the MegaCpp training data pipeline was built — source selection, filtering, dedup, tokenization, document masking, and the quality gates that catch our own mistakes."
date: "2026-04-18"
tags: ["data", "pipeline", "c++", "tokenizer", "quality"]
---

The most important data decision in MegaCpp is not a model hyperparameter. It
is deciding what bytes the model is allowed to see, how those bytes are pinned,
and what checks are required before a dataset snapshot is promoted into a real
training lane.

This article focuses on the public engineering contract behind that pipeline.

## Start with a small pinned operational slice

MegaCpp keeps a clear distinction between:

- the **operational slice** that is actively wired into training
- the **catalog** of additional sources that may become future inputs

That split matters because data-pipeline work fails differently from model work.
If the team tries to ingest every interesting repository on day one, most of
the debugging time is spent on storage, format drift, and tooling gaps rather
than on quality.

The public rule is simple: keep the active training slice pinned to explicit
revisions, and keep the larger catalog as metadata until it is needed.

## The pipeline shape that survived

MegaCpp's pipeline can be summarized in five stages:

| Stage | Output | What must be true before promotion |
| --- | --- | --- |
| collect | pinned public inputs | revision and license metadata recorded |
| normalize | cleaned source tree | encodings and obvious noise normalized |
| enrich | structure-aware records | provenance of syntax-only vs build-aware signals preserved |
| tokenize and store | explicit columnar artifacts | schema and token checks pass |
| verify | candidate training snapshot | round-trip decode and consumer smoke checks pass |

This is deliberately conservative. The pipeline is designed so that a broken
promotion fails on a measurable check rather than surviving as a vague feeling
that "the data looked fine."

## What we filter before chunking

Three filters do most of the work:

1. **language and structural filtering**  
   Keep the language mix intentional. Drop obvious binaries, blobs, and files
   that are clearly generated noise.

2. **license and provenance filtering**  
   Treat license metadata as structured data, not as a comment someone might
   remember to read later. SPDX expressions and REUSE-style headers are useful
   because they make this machine-readable.

3. **PII and secret scrubbing**  
   Secret-like tokens, direct personal addresses, and machine-local paths
   should be normalized before tokenization, not after.

The point is not to claim perfect safety. The point is to make the pipeline
less likely to promote obviously bad inputs.

## Deduplicate before you believe the token counts

Deduplication is valuable for code data, but it is easy to describe too
strongly. MegaCpp uses dedup as a mitigation, not as proof that memorization
risk or contamination risk is gone.

The safest public version of the claim is:

- exact duplicates should be removed before chunking
- near-duplicate handling is valuable for vendored and lightly modified code
- dedup helps training quality, but it does not by itself prove the corpus is safe

That wording is closer to what public code-model literature supports and avoids
promising more than the data pipeline can actually guarantee.

## Why build-aware enrichment stays in the loop

For C++, plain lexical chunking is not enough. MegaCpp therefore keeps a
structure-aware enrichment lane that can use build context when it exists and
syntax-only structure when it does not. That is how the data story connects to
the semantic-indexing story: broad coverage and semantic trust are different
axes, and the pipeline records which one produced a given artifact.

The practical effect is that later training or evaluation code can treat a
build-aware slice differently from a syntax-only slice instead of pretending
they are the same kind of evidence.

## Tokenization and storage are part of the contract

Tokenizer reproducibility is not just "use the same tokenizer name." The safer
rule is:

- pin the tokenizer artifact by revision or saved files
- record special-token and normalization settings
- store the resulting dataset in an explicit schema

MegaCpp uses explicit columnar artifacts for this reason. Columnar storage is
not the schema itself, but it is a good fit for large corpora because it keeps
the stored contract visible: token columns, structure columns, metadata
columns, and per-snapshot versioning.

## Long-context training made document masking non-optional

Once documents are packed into long sequences, document boundaries stop being a
nice-to-have. They become part of the correctness story. A long packed row that
does not preserve boundaries can quietly teach the model relationships between
unrelated files.

That is why MegaCpp treats document masking as a first-class data contract. The
public point is not one exact implementation. The public point is that packing,
masking, and evaluation must agree about where a document ends.

## The real moat is quality gates

The pipeline only becomes believable once promotion is blocked by explicit
checks. MegaCpp's checks include:

- schema validation
- token-range and dtype validation
- round-trip decode checks
- sample-level sanity checks
- a small consumer smoke run before a snapshot is promoted

The exact thresholds may change. The idea should not. A dataset snapshot either
survives promotion checks or it does not.

## What the public claim should be

The strongest defensible public claim is:

- the active corpus is pinned
- license and provenance metadata are recorded explicitly
- dedup happens before promotion
- build-aware enrichment is kept separate from syntax-only coverage
- tokenizer and dataset revisions are versioned
- long-context packing requires explicit document-boundary handling
- dataset snapshots must pass promotion checks before training uses them

That is a stronger and more useful story than listing many sources without
explaining the contract that ties them together.

## References

- [MegaCpp public repository](https://github.com/DatasunriseOU/cppmega/tree/main)
- [MegaCpp article samples](https://github.com/DatasunriseOU/site_samples/tree/main/articles)

- [The Stack paper](https://arxiv.org/abs/2211.15533)
- [The Stack v2 paper](https://arxiv.org/abs/2402.19173)
- [SPDX license expressions](https://spdx.github.io/spdx-spec/v2.2.2/SPDX-license-expressions/)
- [REUSE specification](https://reuse.software/specifications/)
- [Parquet concepts](https://parquet.apache.org/docs/concepts/)
- [HumanEval](https://github.com/openai/human-eval)
- [BigCode evaluation harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
