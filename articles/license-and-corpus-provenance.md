---
title: "License Hygiene and Provenance for a C++ Training Corpus"
description: "How MegaCpp handles mixed BSL/Apache/GPL/MIT licenses across its C++ corpus, detects SPDX headers, pins sources by commit SHA, and maintains a refusal list for anything we will not train on."
date: "2026-04-18"
tags: ["corpus", "license", "provenance", "spdx", "data"]
---

Most of the C++ code on GitHub is not "yours to train on" in the unqualified sense the marketing suggests. It is a mix of Apache 2.0, BSL-1.0, MIT, GPL-2.0, LGPL, mixed headers with no `SPDX-License-Identifier` at all, and vendored blobs whose upstream license nobody remembers. If you treat the corpus as a single bag of tokens you will quietly train on the wrong thing, and you will not find out until someone asks why a model trained on an 8B-line catalog is reproducing a GPL header verbatim under the name `absl::`. This post is the license-hygiene and provenance layer we built around the MegaCpp corpus.

## Why this matters

License hygiene is not a compliance ceremony. It is a correctness property of the data pipeline. The corpus is the product; the license posture is the contract that lets us ship that product to anyone who asks where the tokens came from. Treating SPDX detection, commit-SHA pinning, and a refusal list as load-bearing infrastructure is what keeps "specialist trained on permissively licensed C++" from becoming a slogan that does not survive a five-minute audit.

The practical payoff is operational, not legal. When a specialist regresses, the difference between "we know exactly which Boost commit changed and can revert" and "we shuffled some files and don't know what happened" is a one-day investigation versus a one-week one. The provenance sidecar exists for that reason as much as for the audit reason, and the refusal list keeps the catalog honest about what we are not training on.

## 1. The two-tier corpus

We run two distinct corpora.

The **reference corpus** is eight public C/C++ repositories, documented in `data_preparation.md`: `llvm/llvm-project` at `llvmorg-19.1.0`, `boostorg/boost` at `boost-1.86.0` with submodules, `torvalds/linux` at `v6.10`, `fmtlib/fmt` at `11.0.0`, `google/googletest` at `v1.15.0`, `abseil/abseil-cpp` at tip, `facebook/folly` at tip, and `grpc/grpc` at `v1.67.0`. Combined, about 15 GB of shallow clone. This is the corpus that actually feeds the Megatron launchers.

The **extended catalog** is 142 repositories across 16 categories, documented in the public corpus catalog notes. It is not wired into training. It is a pre-vetted menu for future specialists: kernels, compilers, databases, graphics, desktop toolkits, ML frameworks, crypto, embedded. Each entry carries a size bucket and a note for awkward sources (SQLite Fossil, Chromium and V8 and Fuchsia on third-party Git hosts, VLC and x264 on VideoLAN's GitLab, Unreal behind Epic org access).

The two-tier split is itself a license-hygiene decision. The moment a repository graduates from catalog to operational, it picks up a license audit, an SPDX sweep, a provenance pin, and a place in the refusal-list review. Keeping the working set small is what makes that audit tractable on a single workstation.

## 2. The license mix we actually have

For the operational eight, the header-level license mix is:

| Repository group | License |
| --- | --- |
| LLVM, abseil-cpp, grpc, folly | Apache 2.0 (LLVM with LLVM exception) |
| boost super-project + submodules | BSL-1.0 |
| fmt, googletest | MIT |
| Linux kernel headers we touch | GPL-2.0 with syscall note |

None of that is an accident. The operational list was chosen, in part, because none of those licenses is viral against model weights in a way that is practically disputed today, and because every one of them carries a clear attribution expectation we can satisfy through a references file rather than inline. That is a deliberate, conservative reading. It is also why `torvalds/linux` is scoped to headers and well-defined subsystems rather than drawn in wholesale: once you move past the syscall note into GPL-2.0 full source the governance conversation changes, and we did not want that conversation gating the operational pipeline.

The extended catalog is more mixed: Apache-heavy corners (LLVM, ONNX Runtime, protobuf, abseil, grpc, Envoy, Bazel); BSD/MIT-heavy corners (the BSDs, NVIDIA PhysX, Dear ImGui, nlohmann/json, CMake, Ninja); GPL/LGPL-heavy corners (Linux, GCC, binutils, Emacs, Wine, GIMP, GNOME, KDE, GnuPG, GnuTLS); dual-licensed and special cases (MySQL with the FLOSS exception, MariaDB, PostgreSQL, SQLite public domain amalgamation, FFmpeg LGPL default with `--enable-gpl`, x264/x265 GPL); and access-gated source (Unreal Engine via Epic EULA).

Calling any of this "permissive" in aggregate is a lie. Our rule is simple: at ingestion time, every file carries the license of the repository it came from, propagated through the pipeline, and a specialist trained on it must declare its license mix. The refusal list is how we keep that declaration honest.

## 3. SPDX detection and the "wastes context" problem

Raw files carry license information in three places: an `SPDX-License-Identifier:` marker, a free-form header comment, or nothing at all. The data-pipeline design doc (a data-pipeline design note) is blunt that inline license headers are "wastes context, no semantic value" and proposes stripping them. That is true for the tokens the model sees during training, but it is the wrong thing to do for metadata. We treat those as two separate jobs.

At scan time, for every file we:

1. Look for `SPDX-License-Identifier:` in the first 200 lines. Parse the identifier under the SPDX expression grammar. Normalize composites such as `Apache-2.0 WITH LLVM-exception` and `GPL-2.0-only WITH Linux-syscall-note`.
2. If no SPDX marker, run ScanCode Toolkit's signature-based detector (mentioned in a data-pipeline design note and a stepping-stones design note). Keep both the detected identifier and ScanCode's confidence score.
3. Fall back to the repository-level `LICENSE`/`COPYING` file parsed at clone time; record that the file inherited its license from the repo root.
4. Emit one JSONL record per file with `{path, repo, commit, license_spdx, license_source, license_confidence}`.

Only then do we run the strip-for-context pass: the tokenizer never sees the SPDX prefix, but the metadata sidecar retains it forever. License metadata changes when ScanCode improves; the parquet shards do not.

## 4. Provenance: pinned, not branched

Every operational repository is pinned by commit SHA, not by branch. A "tip of main" pin is not a pin; it is a moving target that silently changes the corpus between two specialist runs and makes regression bisection impossible.

```text
For each pinned public repository:
1. fetch the exact tag or commit
2. resolve the detached commit SHA
3. record the repository name and SHA in the provenance ledger
4. refuse promotion if the ledger does not match the published training manifest
```

The `.provenance` file becomes the input to every downstream stage. Parquet shards produced by stage 2 carry per-file `source_repo`, `source_commit`, and `source_path` columns. The enriched-doc path in `tools/cpp_chunker/` threads `filepath` plus per-file compile-command provenance through materialization (noted in the public changelog entries around the build3 and clang-pipeline work), so provenance survives chunking, merging, and two-pass cross-file assembly rather than collapsing at the final write step.

Two payoffs we actually use:

- When a specialist regresses, we can difference its last-good and current parquet by `source_commit` and find out whether a repository bump introduced the regression. That happened on a Boost bump during curriculum testing; the fix was to revert the pin, not the model.
- When someone asks "did this specialist see file X version Y," the answer is a `DuckDB` query against the provenance sidecar, not a guess.

For the extended catalog we also record ingest intent — the commit we would use if this repo became operational tomorrow — even though we do not actually download it into the training cache. That keeps the catalog honest.

## 5. The refusal list

A corpus is defined as much by what we will not ingest as by what we do. The refusal list is a small, boring document we review when the catalog changes.

- EULA-gated or source-available-but-not-open projects (Unreal Engine is the obvious one; any "shared source" style license goes here too).
- Anything without a clear repository-level license, even if individual files look permissive. "Probably MIT" is not a license.
- Crypto implementations where the repository combines permissive and export-controlled code without clear separation; BoringSSL, LibreSSL, OpenSSL, GnuPG, libsodium are fine because the licensing is clean, but ad-hoc "crypto experiments" are not.
- Vendored blobs inside otherwise-clean repos (a vendored copy of a GPL library inside an Apache project, for example). We detect these with vendor-dir heuristics (the "Vendored deps" entry in the quality-filter table of a data-pipeline design note) and refuse them at the file level, not the repo level.
- Auto-generated files with clear "DO NOT EDIT" markers that lift their content from elsewhere — same reasoning as vendored blobs.
- Anything under `test/` or the public test suite that looks like captured third-party test data rather than test code. Data with unclear licensing hides inside code-licensed repos all the time.
- Secret-bearing files flagged by the secret scan (we run `Yelp/detect-secrets`). The refusal here is not really about licensing, but the refusal list is the right place for it.

When a file is refused, we record why — which rule fired, with what confidence — in the provenance sidecar. A specialist's model card can then say, truthfully, "N files refused under rule R," instead of the more common "trained on all of it."

## 6. What actually happens at scale

Some honest numbers and one non-number:

- Operational corpus: 8 repositories, all SPDX-marked at the repo root, about 15 GB after shallow clone (from `data_preparation.md`). License distribution by source repo is known exactly and is reported by `verify_dataset_megacpp.py` at the end of stage 5.
- Per-file SPDX coverage across the reference corpus: high in LLVM, abseil, grpc, folly (near-universal `SPDX-License-Identifier:` headers); patchy in older Boost submodules and in some fmt/googletest files where the repo-root LICENSE is the authoritative record. Exact ratio: n/a — we did not publish it, and we should.
- GPL-2.0 exposure in operational specialists: limited to kernel headers with the syscall note. Non-header kernel source is routed to its own build line.
- Refusal-list rejections during catalog-to-operational promotion: most rejections are not licenses but vendored blobs and auto-generated files; pure license rejections are a small fraction.
- Extended catalog footprint if we actually downloaded all 142 repos: several hundred GB even with shallow clones, dominated by the H-sized kernels, browsers, and ML frameworks. The catalog does not exist to be downloaded wholesale; it exists to make graduation decisions cheap.

## What we kept and what we threw away

Kept: the two-tier split, commit-SHA pinning with a `.provenance` sidecar, the SPDX-then-ScanCode-then-repo-root cascade, license metadata in a sidecar (not in the parquet shards), the refusal list with per-file rule annotations, and the rule that token streams are stripped of license headers while metadata retains them forever.

Threw away: a learned per-file license classifier (high false-positive rate on headers with unusual wording — SPDX plus ScanCode plus repo root beats it operationally); a scheme that stored licenses as a per-row column inside parquet shards (one iteration; immutable shards plus a license metadata that changes when ScanCode improves was a bad pairing); and a "strip all comments, no SPDX retention" plan that died when reviewers needed to see SPDX headers at eval time for spot checks.

The directive for future modifiers is short: do not add a new repository to the reference corpus without a commit-SHA pin, an SPDX sweep, and a refusal-list review. Do not merge full-GPL or LGPL source into a specialist's token stream without declaring a new build line. Treat "probably permissive" as a refusal until the scan confirms it.

## References

- cpp-training-corpus-repos.md
- corpus_curriculum_mapping.md
- 02-data-pipeline.md
- 07-stepping-stones.md
- DATA_PIPELINE.md
- data_preparation.md
- CHANGELOG.md
- README.md
