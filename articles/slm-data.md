---
title: "SLM data: what the current pipeline optimizes for, where it still fails, and why the loader contract matters more than the storage format"
description: "A grounded walkthrough of the small-language-model data path in the research repo: parquet shards, split logic, best-fit packing, enriched metadata, resume behavior, and the live failure modes documented in current reports."
date: "2026-04-18"
tags: ["data", "slm", "training", "dataloader", "dataset", "packing"]
---

**TL;DR:** the current SLM data path is not just “read parquet and tokenize.” It is a layered training contract: shard discovery, packed-row and metadata-aware loading, optional structure-aware features and FIM augmentation, and a set of failure reports showing exactly where that contract breaks. The strongest pattern in the code is that data quality problems come less from file format choice than from boundary mistakes: split assumptions, metadata parsing, packed-row accounting, and resume/schema drift.

A lot of discussions about small-model training data become abstract too quickly. People debate web corpora, synthetic ratios, or token budgets without checking what the trainer actually consumes. The current project code is a useful corrective because it exposes the mechanics directly. The shard-discovery layer defines how parquet shards are located, opened, downloaded, and split into train versus validation. The dataloader layer is where those rows become packed sequences, telemetry, FIM-augmented batches, and structure-aware features. Public bug notes and training retrospectives then record where the real contract is still weak.

That combination is enough to say something concrete: in this stack, “SLM data” means the entire path from shard naming and cache directories to packed rows, metadata columns, and resume state. If any of those surfaces drift, the model may keep training while silently learning from the wrong thing.

## The base contract starts with parquet shards and an explicit split rule

The foundation is plain and intentionally boring. The dataset layer defines the pretraining dataset as a set of parquet files. It contains helpers such as `index_to_filename`, `list_parquet_files`, `open_parquet_file`, and `get_parquet_paths_for_split`. The simplest but most important rule is the split policy: the last parquet file is validation, and the rest are training.

That split rule sounds trivial until you read the guardrails around it. `get_parquet_paths_for_split(split, data_dir=None, require_train_shard=True)` explicitly raises an error when there is only one parquet file and the caller asks for the train split. The error message is intentionally blunt: with one shard available, the last file would be consumed by validation, leaving nothing for training. That is a small example of a good data-system habit. The loader refuses to pretend that a degenerate split is acceptable.

| Layer | What it does | Why it matters |
| --- | --- | --- |
| dataset layer | shard discovery, cache-directory resolution, parquet opening, split rules | prevents bad assumptions before training begins |
| dataloader layer | packing, metadata loading, augmentation, telemetry, state dict | determines what the model actually sees |
| reports / plans | document where the data contract currently fails | keeps the failure modes visible |

The same file also reveals a second important property: the data source is treated as a cache-backed remote dataset, not as a fixed local monolith. `BASE_URL` points to a public shard location, `download_single_file` downloads into a temporary file before rename, and `_resolve_default_data_dir` falls back to a safe temporary-root cache when the default base path cannot be initialized. The storage layer is therefore deliberately resilient. If the system fails later, it usually will not be because the repo forgot how to name a shard.

## The real training path is packed rows, not raw documents

The more distinctive part of this stack lives in the dataloader layer. Its module docstring makes the priorities explicit: BOS-aligned best-fit packing, preallocated buffers, a single host-to-device transfer, and optional enriched metadata for structure-aware training. That is already a strong clue that the project does not think of “the data” as raw documents streaming one by one. The operative training unit is the packed row.

The telemetry helper `PackedDocTelemetry` shows what the loader considers important enough to measure continuously:

- utilization, defined as `valid_token_count / T`
- average documents per packed row
- average tokens per document inside packed rows
- the fraction of rows whose last document likely got cropped

Those metrics are worth paying attention to because they reveal the actual optimization target. The loader is trying to maximize useful tokens per training row without hiding when documents are being truncated. That is exactly the right orientation for an SLM pipeline: the main concern is not whether the format is elegant, but whether the trainer is consuming dense, interpretable token batches.

A compact slice of the code says a lot:

```python
if valid_token_count >= T and n_docs > 0:
    self._total_cropped_rows += 1
```

That line sits inside the cropped-document heuristic. It is not mathematically perfect, but it shows the design intent. Packing efficiency is being tracked together with the risk that the final document in a row was cut at the boundary. This is the sort of tradeoff a serious data path must surface rather than hide.

## Enriched metadata turns the loader into a feature gateway

Another reason the data path matters is that it now carries more than token IDs. The dataloader layer imports a large set of metadata columns: structure IDs, chunk boundaries, AST depths, call edges, type edges, repository-level stable IDs, change masks, and more. It also imports helpers from structure-embedding utilities and the semantic-block exporter. That means the loader is not merely ferrying text into the model. It is the gateway for a broader structured-training story.

This is why “SLM data” in this project cannot be reduced to corpus size. Once a batch can contain structural metadata, the loader becomes the place where representation experiments either remain faithful or get corrupted. One of the live bug reports spells that out: missing or malformed metadata can collapse to zero-like IDs and become trainable signals rather than explicit absence. That is worse than a hard crash because the model can silently internalize the wrong feature semantics.

The code reinforces that risk by how much responsibility sits in the loader boundary. It is reading enriched columns, deciding which columns are present, resolving packed-row validity counts, permuting metadata for FIM, and constructing relation masks. A pipeline with that much semantic work concentrated in one place needs boring, explicit contracts. If it relies on optimism, the model will learn from accidental conventions.

## FIM and structure-aware augmentation belong to the data system, not to model folklore

The imports around FIM make another project choice clear. The dataloader layer brings in `StructuredFIMDataset`, `apply_fim_batch`, `apply_fim_mixed_batch`, and `permute_metadata_for_fim`. That is the correct architecture if you want FIM to be a repeatable data transformation rather than a mystical model feature. The same principle applies to structured augmentation: the place to reason about it is the loader and the schema, not only the forward pass.

This design is also reflected in the feature-plan tests on the MegaCpp side. sanitized Megatron-args tests checks that a `NAM56R` feature plan with `use_fim=True` emits the `--fim-rate` flag along with MoE, DSA, MLA, and MTP toggles. The important point is not the exact flag spelling. It is that data augmentation choices are treated as part of the launch contract. That keeps them close to the reproducible recipe instead of burying them inside hidden defaults.

A good SLM data stack should work that way. Data transformations that change the effective supervision should be visible in the same place where sequence packing and feature selection are declared.

## Most live failures are boundary bugs, not storage bugs

The most valuable source for this topic is a live bug audit report because it records what actually went wrong. Several of the issues are directly about the data path.

One class of failures is split and iteration behavior. The report calls out a single-shard parquet split regression in the dataloader layer, exactly the kind of issue that can make a training run look alive while producing no useful train data. Another class is enriched-metadata parsing. The report notes that `_read_enriched_columns()` does raw `json.loads()` on metadata, which means malformed values can blow up the loader path instead of degrading gracefully. A third class is semantic fallback behavior: metadata absence can effectively collapse into a trainable zero ID rather than a distinct “missing” state.

These are not random edge cases. They all show the same structural truth: data pipelines fail at boundaries. The shard format itself is rarely the hardest part. The dangerous parts are assumptions about how many shards exist, what schema version is present, whether metadata is valid JSON, and whether resume state matches the current loader expectations.

| Failure mode | Grounded source | Why it is dangerous |
| --- | --- | --- |
| single-shard split regression | live-bugs report plus `get_parquet_paths_for_split` | trainer can end up with no meaningful train split |
| raw metadata JSON parsing | live-bugs report | malformed metadata crashes or poisons the loader path |
| zero-like fallback semantics | live-bugs report and structure-aware imports | missing metadata can become learned signal rather than absence |
| resume-schema drift | live-bugs report | training resumes with incompatible loader expectations |

This is why the article’s main claim is that loader contract matters more than storage format. Parquet is fine. The hard part is whether the semantics around it stay honest.

## Resume behavior is part of data quality

The live bug report also contains a point that teams often miss: resume compatibility is a data-system problem. One issue is titled “Resume path still hard-requires the newest metadata schema,” and the proposed fix is to use `.get(...)` defaults for `loop_state` and `dataloader_state_dict`. That may sound like plain training engineering, but it directly affects what data the model sees after restart.

If the dataloader state contract changes and the resume path cannot interpret older checkpoints, then the model is not simply resuming optimization. It may be resampling data differently, replaying rows, or skipping expected metadata state. For long-running SLM experiments, that is a data-integrity failure.

The same logic applies to token counts. The dataset layer includes an `_extract_texts_and_valid_lengths` helper, which reads an optional `actual_token_count` column when present. That is a subtle but important detail: the system is willing to preserve exact valid lengths from parquet metadata instead of assuming every row’s text has to be reinterpreted in one generic way. Whenever exact token-count information exists, the loader should prefer it. That is how you keep accounting honest.

## Planning documents matter because they bind data assumptions to execution

The code is not the only useful source. The project plan and changelog material also matter because they show which data assumptions were important enough to document. The public training plan notes and GB10 release notes are valuable here not because they are perfect truth, but because they keep tokenization, scheduler, and loader choices close to the launch surface.

That is healthy. A serious SLM pipeline should make it easy to answer questions such as:

- what shard source the run expects
- whether packed rows are active
- whether FIM is enabled
- whether enriched metadata is required or optional
- what resume assumptions hold for the dataloader state

Once those assumptions stay visible, the data stack becomes debuggable. When they are scattered across ad hoc scripts and undocumented defaults, every training run becomes a forensics exercise.

## What a robust SLM data pipeline should preserve in this codebase

The current code already points to a clear standard.

1. Keep shard handling boring and explicit.
2. Keep packed-row accounting visible through telemetry.
3. Treat metadata columns as part of a schema contract, not best-effort decoration.
4. Tie augmentation features such as FIM to the launch plan.
5. Treat resume compatibility as a data-integrity requirement.

A small config-style excerpt captures the operational mindset better than slogans do:

```python
parquet_paths = get_parquet_paths_for_split(split, data_dir=data_dir)
for filepath in parquet_paths:
    pf = open_parquet_file(filepath)
    for rg_idx in range(start, pf.num_row_groups, step):
        rg = pf.read_row_group(rg_idx)
```

That loop is the real heart of SLM data work here. Everything else, from packing efficiency to structure-aware metadata, depends on whether this boundary stays truthful.

So the right summary is not “the project uses parquet” or “the project has best-fit packing.” The grounded summary is that the stack turns shards into packed semantic training rows, and the current live reports show that most remaining risk comes from schema, metadata, and resume boundaries. That is exactly where future hardening effort should stay focused.

## References

- public dataset-layer notes
- public dataloader-layer notes
- public bug notes and launch retrospectives
- public training-plan notes
- GB10 release notes
- sanitized Megatron-args tests
