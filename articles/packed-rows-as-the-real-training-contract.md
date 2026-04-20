---
title: "Packed rows as the real training contract"
description: "Why packed rows are the real boundary between the data pipeline and the model, and why MegaCpp treats row packing as a schema contract rather than a storage detail."
date: "2026-04-19"
tags: ["data", "packing", "long-context", "fim", "training-contract"]
---

The most honest place to describe a training data pipeline is not the crawler
and not the tokenizer. It is the packed row. That is the first format the model
actually consumes as a stable contract.

MegaCpp's public data examples make that unusually clear. They show enriched
records, masking transforms, schema samples, and row builders separately, but
all of them are converging on the same operational boundary: one row that tells
the runtime what tokens are valid, where document boundaries are, what the loss
should ignore, and which structure fields are still aligned to those tokens.

## Why the packed row is more important than the intermediate artifacts

Intermediates matter, but they are not the model contract. The model does not
consume a repo clone, a build graph, or a raw enriched JSONL record directly.
It consumes packed, token-aligned rows with explicit masks and metadata.

That is why the local example pack is structured the way it is:

- one fixture for enriched records
- one row-builder example
- one schema sample
- one masking excerpt that preserves alignment through transformations
- one loader-side sample for reading optional enriched columns

Taken together, these files say something stronger than "we have a data
pipeline." They say the pipeline is only finished once all those fields survive
packing in a form the model can actually train on.

## What a packed row has to preserve

The public examples support a practical row contract built around a few durable
surfaces:

- token ids and target ids
- a loss mask
- valid-token accounting
- document-boundary or segment information
- optional enriched columns that still line up with token positions

That is already enough to explain why packed rows deserve their own article.
Once the row exists, many earlier data-pipeline arguments stop being abstract.
You can now ask whether masking survived, whether enrichment stayed aligned, and
whether the loader still knows how to read the optional columns without turning
everything into an opaque sidecar.

## Packing is not just an efficiency trick

Sequence packing is often described as a throughput optimization, and that is
true but incomplete. In MegaCpp it is also a correctness boundary.

If the packed row fails to encode boundaries and masks correctly, long-context
training will teach the model false relationships across unrelated documents.
If the row loses alignment between tokens and enriched structure fields, later
structure-aware work becomes guesswork. If the row builder and schema disagree,
the loader can still run while silently training on the wrong contract.

That is why the row builder, schema sample, and masking excerpt belong in the
same public surface. They are three views of the same training boundary.

## Why this article belongs next to FIM and long-context notes

The packed row is where fill-in-the-middle, document masking, and long-context
packing finally meet one another. FIM is not just a transform on raw text. It
changes what part of the row carries loss. Document masking is not just an
attention idea. It depends on boundaries that the packed row still needs to
preserve. Long-context packing is not just "put more text in one example." It
is the discipline of packing without losing row-level semantic truth.

This is exactly why the public examples in `examples/data` and
`examples/long_context` reinforce each other. The row contract is what lets
those two families talk to each other honestly.

## Prior art and context

The general idea of packing sequences efficiently without cross-sample leakage
is well established. There is prior art on efficient sequence packing with
attention isolation, official Megatron and Torchtune docs on packed sequence
formats, canonical FIM work, and broader long-context papers that explain why
middle-position and boundary effects matter. MegaCpp's local contribution is
the public-safe contract surface: examples that show how enriched records,
masking, and row building remain aligned all the way to the model-facing row.

## References

- [Packed rows schema sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/packed_rows_schema_sample.py)
- [Packed row builder example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/packed_row_builder_example.py)
- [Packed row fixture](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/packed_row_fixture.json)
- [Enriched record fixture](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/enriched_record_fixture.json)
- [Masking pipeline excerpt](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/masking_pipeline_excerpt.py)
- [Loader enriched columns sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/loader_enriched_columns_sample.py)
- [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
- [Megatron Bridge packed sequences docs](https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/training/packed-sequences.html)
- [Torchtune packing docs](https://docs.pytorch.org/torchtune/0.6/basics/packing.html)
- [Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [Structured Packing in LLM Training](https://arxiv.org/abs/2312.17296)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
