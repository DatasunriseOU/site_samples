---
title: "License Hygiene and Provenance for a C++ Training Corpus"
description: "How MegaCpp describes source provenance, revision pinning, SPDX metadata, and refusal-list rules for a public C/C++ corpus narrative without overstating legal certainty."
date: "2026-04-18"
tags: ["corpus", "license", "provenance", "spdx", "data"]
---

A C++ training corpus should not be described with one blanket licensing sentence. Public source corpora mix Apache-2.0, BSL-1.0, MIT, GPL-family licenses, exception clauses, repository-level `LICENSE` files, and files that only carry provenance through version control history. If a corpus description ignores that mix, it stops being a provenance statement and turns into marketing.

The public MegaCpp sample pack supports a stricter and more useful story: pin every input, treat license metadata as structured data, keep a refusal list for sources that do not fit the current policy, and make provenance auditable enough that a regression can be traced back to a concrete revision.

## Why this matters

License hygiene is not only about compliance review. It is also a data-correctness property. If a model output or regression sends you back to the corpus, you need to know which repository, revision, and file family produced the relevant tokens. That is why the public notes put revision pins, license metadata, and schema checks on the critical path.

The practical payoff is operational. A pinned ledger makes it possible to answer questions such as “which revision changed?” or “which input class introduced this header style?” without reconstructing the corpus from memory.

## 1. A provenance-first corpus story

The public story does not need to be an exhaustive inventory of every candidate source. It does need a strong admission rule. The public data-prep note provides one: pin every upstream input to an explicit tag, commit hash, or dataset revision, and treat license metadata as structured data. The reference pinning note adds the minimum fields that make such a story auditable: source id, revision or tag, license metadata, retrieval date, schema version, and optional SWHID.

That is enough to define a useful working rule: a source is not “in the corpus” in any meaningful reproducible sense until it has those fields.

## 2. Describe the license mix honestly

The public pinning and data-prep notes do not claim a single-corpus license. They imply the opposite: the corpus is a set of pinned inputs, each with its own metadata. That is the right framing.

For a C/C++ corpus built from common public infrastructure projects, a realistic mix will often include categories like these:

| Source family | Common license patterns |
| --- | --- |
| toolchains and infra libraries | Apache-2.0, BSD-3-Clause, MIT |
| Boost-family inputs | BSL-1.0 |
| test and utility libraries | MIT, BSD-style |
| kernel-adjacent headers and systems code | GPL-family, LGPL-family, or exception-bearing variants |

The exact operational set should be reported from the ledger, not reconstructed in prose. That is the key public discipline: name the policy, publish the pinning fields, and avoid pretending that a mixed-source corpus is “all permissive” just because some important inputs are.

## 3. SPDX detection should stay in metadata even if training text is normalized

The data-prep note says to treat license metadata as structured data, not prose. That one line carries an important consequence: license information should survive preprocessing even when other text normalization steps strip comments, boilerplate, or repeated headers from the training-facing representation.

A practical scan order looks like this:

1. Check the file header for `SPDX-License-Identifier` markers and parse them as SPDX expressions.
2. If that fails, fall back to repository-level license metadata or a dedicated license scanner.
3. Record both the detected expression and the source of the detection.
4. Keep that metadata outside the model-facing token stream.

That is consistent with the SPDX specification itself, which treats license expressions as structured identifiers rather than free-form prose, and with tools such as ScanCode Toolkit, which are designed to produce machine-readable license findings.

## 4. Provenance means pinned revisions, not floating branches

The public pinning note is explicit on this point: do not publish training or evaluation claims against floating `main`, `master`, or `tip`. That rule matters as much for provenance as it does for benchmarking. A floating branch is not a reproducible input.

Software Heritage identifiers are useful here as a second anchor. They do not replace repository commits, but they do provide a stable cross-host provenance pointer when one is available. That makes them a good optional field in a public corpus ledger.

## 5. Build metadata needs provenance too

The compile-command sample in `examples/data/compile_commands_fixture.json` is small, but it shows an important part of provenance work that is often missed: build context is data. Include paths, language mode flags, and the compilation unit path are all part of what later structure-aware stages may consume.

If build context is used during chunking or metadata extraction, it should be pinned and versioned like any other source input. Otherwise a corpus can drift even when the source repository revision stays fixed.

## 6. A refusal list is part of the public contract

The public notes do not provide a giant exclusion table, but the policy they describe clearly implies one. Some sources should remain outside the current operational corpus because they are gated, ambiguously licensed, dominated by generated code, or difficult to pin in a way that supports public claims.

That is not a weakness. It is a sign that the provenance story is honest enough to say “not yet” or “not under this policy.”

## Operational checklist

- Pin every source to an exact tag, commit, or dataset revision.
- Store license metadata as structured side data.
- Keep provenance fields alongside schema version and retrieval date.
- Preserve build metadata when it affects downstream extraction.
- Report the actual license mix from the ledger, not from memory.
- Use a refusal list for sources that do not satisfy the current provenance rules.
- Prefer optional SWHIDs when available for stable public reference.

The useful public claim is therefore narrow and defensible: the corpus is described by pinned inputs plus structured provenance metadata, not by a vague sentence about “public C++ code.” That narrower claim is much easier to audit and much harder to misuse.

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands_fixture.json
- https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions/
- https://scancode-toolkit.readthedocs.io/
- https://docs.softwareheritage.org/devel/swh-model/persistent-identifiers.html
