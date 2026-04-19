# Data Preparation Notes

This note records the public rules MegaCpp uses when describing its
C/C++ data pipeline. The bullets below are policy and public process notes, not
a canonical public inventory of every pinned source.

## Core rules

- Pin every upstream input to an explicit release tag, commit hash, or dataset revision.
- Treat license metadata as structured data, not prose.
- Deduplicate before chunking whenever possible.
- Keep build-aware metadata separate from plain lexical chunks.
- Promote a dataset snapshot only after schema checks and a smoke consumer pass.

## Public pipeline shape

1. Collect public source code, build metadata, and supporting documentation.
2. Normalize encodings, line endings, and obviously generated noise.
3. Apply license and provenance tagging.
4. Deduplicate exact and near-duplicate content.
5. Extract structure-aware metadata from syntax and, when available, build context.
6. Tokenize and write explicit columnar artifacts.
7. Run schema, round-trip, and consumer smoke checks before promotion.

## Why the pipeline exists

- Raw corpora over-reward repeated boilerplate.
- Build-blind code chunks miss real cross-file structure.
- Documentation-only corpora miss executable constraints.
- Floating revisions make regressions impossible to explain.
- Long-context training makes document-boundary mistakes more expensive.

## What we publish

- A pinned-input policy.
- Small public samples instead of private storage layouts.
- Notes describing how schema and quality gates work.
- Runnable examples for masking and build-aware metadata collection.

## Related public files

- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands.sample.json
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/masking_pipeline_sample.py
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/semantic-indexing-notes.md
