---
title: "SLM data: what the pipeline optimizes for and why the loader contract matters most"
description: "A grounded walkthrough of the MegaCpp data path: parquet shards, split logic, packed rows, metadata columns, and the interface choices documented in the public sample corpus."
date: "2026-04-18"
tags: ["data", "slm", "training", "dataloader", "dataset", "packing"]
---

Small-model data discussions often stay too abstract. People argue about corpus mix, synthetic ratios, or token budgets without showing what the training input actually looks like. The public MegaCpp sample packs are useful because they keep the discussion on concrete surfaces: pinned-input rules, masking examples, compile-command examples, and notes about structure-aware metadata. Taken together, those files show that in this stack, “SLM data” is not just a bag of documents. It is the whole path from pinned source inputs to packed training rows with explicit metadata and compatibility rules.

That framing matters because most data failures are interface failures. A pipeline can store tokens in a perfectly reasonable format and still train on the wrong thing if split rules, metadata defaults, or schema evolution are underspecified.

## The base contract starts with pinned inputs and explicit splits

The first useful constraint is boring on purpose: inputs should be pinned and described as structured data. The public data-prep notes say to pin every upstream input to an explicit release tag, commit hash, or dataset revision, and the pinning note gives concrete examples such as `llvm-project@llvmorg-19.1.0` rather than a floating branch. That is the right starting point because reproducible data work begins before tokenization.

The same notes also make the split story more concrete than many training writeups do. They describe a staged pipeline: collect public inputs, normalize them, attach license and provenance metadata, deduplicate, extract structure-aware metadata, write explicit columnar artifacts, and only then promote a snapshot after schema and consumer checks. In other words, a split is not just a train/validation percentage; it is part of a larger contract about what qualifies as a publishable snapshot.

| Layer | What it does | Why it matters |
| --- | --- | --- |
| input pinning | records exact source revisions and licenses | makes the corpus auditable and repeatable |
| preprocessing | normalization, masking, deduplication | removes accidental noise before tokenization |
| row materialization | writes columnar artifacts with explicit fields | defines what the trainer can actually read |
| schema checks | validates shape and field compatibility | prevents silent drift between producer and consumer |

## The real training unit is the packed row

The strongest theme across the public notes is that the important unit is the row consumed by the trainer, not the raw source document. The data-prep note explicitly calls out deduplication before chunking, keeping build-aware metadata separate from plain lexical chunks, and running consumer smoke checks before promotion. That is exactly the mindset you want for an SLM pipeline: optimize for what the loader sees, not for a storage format headline.

Even the small runnable examples reinforce that point. The masking example is tiny, but it is conceptually important because it treats document structure as something the pipeline preserves and edits intentionally rather than as an accident of text concatenation.

```python
def mask_document_sections(tokens: list[str], mask_token: str = "<mask>") -> list[str]:
    masked = []
    for token in tokens:
        if token.startswith("DOC_"):
            masked.append(mask_token)
        else:
            masked.append(token)
    return masked
```

The point is not that this sample is a full trainer. The point is that the public example already encodes a loader-facing assumption: document markers survive far enough into preprocessing to be masked deterministically.

## Enriched metadata turns the loader into a feature boundary

The public notes also make clear that “data” means more than token IDs. The pipeline description explicitly separates build-aware metadata from plain lexical chunks, and the semantic indexing note describes structure-aware metadata as a first-class export surface rather than a side channel. That changes what the loader boundary is responsible for.

Once batches may contain token-aligned structure IDs, chunk boundaries, or graph-derived relations, the loader is no longer a passive transport layer. It becomes the feature boundary between corpus construction and model consumption. That is why schema discipline matters more than format branding. You can store rows in Parquet, Arrow IPC, or another columnar format and still fail if the meaning of a metadata field is unstable across versions.

The compile-command example is a good illustration of why this boundary matters:

```json
{
  "directory": "/workspace/build",
  "file": "src/indexer.cpp",
  "arguments": [
    "clang++",
    "-std=c++20",
    "-Iinclude",
    "-Igenerated",
    "-c",
    "src/indexer.cpp"
  ]
}
```

This is not training data by itself. It is build context. But it is exactly the kind of structured input that can be threaded into chunk metadata or later retrieval features. If that context is kept separate, typed, and pinned, it can enrich the corpus. If it is smeared into free-form text, it becomes hard to validate and harder to evolve.

## Most failures in a pipeline like this are boundary failures

The public files do not claim access to every internal failure mode, and they do not need to. They already point to the likely weak points.

One weak point is split integrity. If a promoted snapshot does not define train and validation materialization rules clearly, later comparisons become meaningless.

Another is metadata decoding. The more structure-aware fields a corpus carries, the more important it becomes to define canonical missing values and canonical field shapes before rows reach a model-facing loader.

A third is resume compatibility in the broad sense: when a dataset snapshot evolves, consumers need a stable rule for what happens to old rows, new rows, and missing fields. The data-prep note’s instruction to run schema and round-trip checks before promotion is a compact way of stating that requirement.

| Failure surface | Publicly grounded signal | Why it matters |
| --- | --- | --- |
| floating inputs | `docs/reference-corpus-pins.md` forbids floating revisions | prevents irreproducible corpora |
| mixed metadata shapes | `docs/semantic-indexing-notes.md` treats structure metadata as explicit export data | avoids consumer ambiguity |
| lossy preprocessing | `docs/data-prep-notes.md` separates normalization, dedup, and metadata extraction | keeps transformations inspectable |
| build-context drift | `examples/data/compile_commands.sample.json` shows typed build inputs | keeps structure features reproducible |

## What a robust SLM data pipeline should preserve

The public MegaCpp sample packs support a fairly strict checklist.

- Pin every upstream repository, dataset, and tokenizer artifact to an exact revision.
- Keep license metadata and provenance records as structured side data, not prose.
- Deduplicate before chunking when possible.
- Keep lexical content, build-aware metadata, and structure-aware metadata as distinct layers.
- Validate snapshots with schema checks and at least one consumer smoke pass before promotion.
- Treat missing metadata as a typed compatibility case, not as a reason for ad hoc loader branching.
- Keep masking and similar preprocessing transforms deterministic and inspectable.

That is the useful summary. The important claim is not “this project uses Parquet” or “this project has metadata.” The useful claim is that the published samples define a pipeline where pinned inputs become columnar training rows through explicit preprocessing, explicit metadata boundaries, and explicit compatibility checks. That is what makes the data path legible enough to debug.

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/semantic-indexing-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands.sample.json
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/masking_pipeline_sample.py
- https://parquet.apache.org/docs/
