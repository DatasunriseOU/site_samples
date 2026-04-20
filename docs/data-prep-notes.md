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

## Public-safe artifact families

- `examples/data/compile_commands_fixture.json` shows the build-database shape we rely on for semantic indexing without exposing any real workstation or CI paths.
- `examples/data/masking_pipeline_excerpt.py` shows how masking and lightweight enrichment can be expressed as explicit stages rather than hidden preprocessing magic.
- `docs/reference-corpus-pins.md` describes the policy for pinning public upstream inputs by revision, tag, or release line.
- `docs/semantic-indexing-notes.md` explains why compile-aware structure is higher trust than syntax-only extraction when the build graph is available.

## Data enhancements we describe in public

- Build-aware fields that preserve compile flags, include roots, and translation-unit boundaries when they are available.
- Documentation masking and curriculum shaping so the training mix can emphasize actionable C++ structure instead of repeated prose boilerplate.
- Deduplication layers that separate exact duplicates from near-duplicate generated expansions.
- Schema versioning that keeps structure-aware metadata, provenance, and tokenization outputs legible across snapshot upgrades.
- Small sanity samples that can be linked from articles without exposing private directory layouts, machine names, or internal storage conventions.

## Inspection checklist

- Confirm every public sample can be traced back to a named input class.
- Check that provenance and license fields survive normalization.
- Verify deduplication happens before downstream chunk statistics are reported.
- Keep schema checks separate from quality judgments so failures stay legible.

## Related public files

- `examples/data/compile_commands_fixture.json`
- `examples/data/masking_pipeline_excerpt.py`
- `examples/data/packed_row.sample.json`
- `examples/data/enriched_record.sample.json`
- `examples/long_context/doc_mask_segment_ids_sample.py`
- `examples/long_context/fim_long_context_metadata_sample.py`
- `docs/reference-corpus-pins.md`
- `docs/semantic-indexing-notes.md`
