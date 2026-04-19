---
title: "Building a C/C++ Corpus for Training: What We Actually Keep, What We Throw Away, and Why"
description: "A detailed walkthrough of how MegaCpp builds a C/C++ corpus: source selection, pins, deduplication, compile-command metadata, chunking, structure-aware exports, and refusal rules."
date: "2026-04-18"
tags: ["cpp", "corpus", "dataset", "training", "data", "tokenizer"]
---

A usable C/C++ training corpus is not just a dump of repositories. The work is in deciding which public inputs are eligible, how they are pinned, which metadata survives preprocessing, and which sources should stay out until they can be described cleanly. The public `site_samples` files are enough to outline that process without leaning on unpublished inventories.

## What the corpus story should keep explicit

The data-prep and pinning notes define the important parts of the construction story.

- Every promoted input is pinned to a tag, commit, or dataset revision.
- License metadata is treated as structured data.
- Deduplication happens before chunking when possible.
- Build-aware metadata stays separate from plain lexical chunks.
- A snapshot is promoted only after schema checks and a smoke consumer pass.

That is already a stronger corpus story than most model cards provide. It says the corpus is a versioned build artifact with entry criteria, not an ever-shifting collection of repositories.

## Source selection is narrower than source discovery

A corpus builder should distinguish between three things: sources worth considering, sources worth pinning, and sources that are currently eligible for promotion. Public discussions often collapse those categories and make the resulting corpus sound more settled than it is.

The public notes support a cleaner rule. Discovery can be broad. Promotion should be narrow. A source becomes promotion-eligible only after it has a stable revision, acceptable license metadata, and a place in the schema and verification flow.

That distinction matters most for awkward hosts, mirrored repositories, and access-gated inputs. A link is not the same as a reproducible source record.

## Build context belongs in the corpus pipeline

The compile-command sample shows why C/C++ corpora need more than raw files.

```json
[
  {
    "directory": "/workspace/build",
    "file": "src/parser.cpp",
    "arguments": [
      "clang++",
      "-std=c++20",
      "-Iinclude",
      "-DMEGACPP_EXAMPLE=1",
      "-c",
      "src/parser.cpp"
    ]
  }
]
```

This kind of metadata matters because C and C++ meaning is partly build-defined. Include roots, language mode flags, generated directories, and compile units all shape what a parser or structure extractor can see. The public data-prep note reflects that by explicitly separating build-aware metadata from plain lexical chunks.

That separation is one of the most important design choices in the corpus pipeline. If build metadata is flattened into prose or discarded too early, later structure-aware features lose their anchor.

## Deduplication and normalization come before chunking

The public pipeline shape is explicit about order: normalize encodings and line endings, remove obviously generated noise, apply license and provenance tagging, deduplicate exact and near-duplicate content, then extract structure-aware metadata and write columnar artifacts.

That order is not cosmetic. Deduplicating after chunking is weaker because template boilerplate, vendored code, and near-clone headers have already been allowed to dominate chunk statistics. Doing it earlier keeps repeated infrastructure from overwhelming the rarer patterns a specialist model actually needs.

Normalization also has to stay conservative. Line endings, encodings, and obviously generated noise are good normalization targets. Semantic rewrite of code style is not. The point is to remove accidental variation, not to erase meaningful formatting or build distinctions.

## Structure-aware exports should stay typed and separate

The semantic indexing note and the data-prep note both point in the same direction: structure-aware metadata is part of the export contract, not a vague aspiration.

That means chunk rows should keep their main lexical content separate from additional fields such as structure IDs, chunk boundaries, compile-command-derived context, or graph-style relations. Typed side fields are easier to validate, easier to evolve, and much easier to consume than one overloaded text field that tries to carry everything.

This is also where many C/C++ corpus projects quietly fail. They gather rich parser output, then collapse it back into lossy text before the loader boundary. The public MegaCpp materials argue for the opposite choice: keep the richer metadata explicit and versioned.

## Versioning is part of corpus construction, not post-processing

The reference pinning note includes schema version as minimal metadata per input. That is important because schema versioning is not an afterthought once rows are already written. It is part of how the corpus is built.

If a chunk row gains a new metadata field, the pipeline should have a canonical way to represent older rows, newer rows, and missing fields. Otherwise every consumer becomes a schema detective. Public sample notes cannot prove every downstream implementation detail, but they clearly endorse the right discipline: explicit schemas, round-trip checks, and consumer smoke tests before promotion.

## What we keep and what we throw away

The public materials imply a straightforward keep/discard policy.

Keep:

- pinned public source files
- structured license and provenance metadata
- build-aware metadata that affects parsing or chunk meaning
- typed structure-aware exports
- versioned columnar artifacts that pass schema and smoke checks

Throw away or keep out of the promoted snapshot:

- floating revisions
- sources that cannot be pinned or described cleanly
- obviously generated noise
- duplicate or near-duplicate content that would dominate training statistics
- ambiguous metadata that cannot be represented in the current schema without ad hoc interpretation

## Practical checklist

- Start from a revision ledger, not a clone directory.
- Treat build metadata as corpus input, not incidental tooling output.
- Deduplicate before chunking whenever possible.
- Keep lexical, structural, and provenance fields separate.
- Promote only snapshots that pass schema and consumer checks.
- Do not describe review inventory as if it were already promoted training data.

That is the detailed corpus-construction story the public files support. It is narrow enough to defend, concrete enough to implement, and much more useful than a generic claim about “training on lots of open-source C++.”

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/semantic-indexing-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands.sample.json
- https://clang.llvm.org/docs/JSONCompilationDatabase.html
