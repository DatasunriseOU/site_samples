# Architecture Notes

This note summarizes the public architecture themes that appear across the
sample training, indexing, and accelerator notes in this repository.

## Design posture

- keep ownership explicit between data preparation, model blocks, runtime, and evaluation
- prefer pinned inputs and narrow receipts over broad undocumented claims
- separate compile-time structure from execution-time measurements
- make optional features visible as named policies rather than hidden defaults

## Repeated system layers

| Layer | Public role |
| --- | --- |
| data preparation | normalize, deduplicate, tag provenance, and emit inspectable artifacts |
| semantic indexing | recover syntax and compile-aware structure with explicit coverage labels |
| model layout | describe block families and sequence layout without pretending all layers are interchangeable |
| runtime policy | keep sharding, precision, and auxiliary-feature gates explicit |
| verification | promote notes only when a small reproducible receipt exists |

## Public documentation rules

- publish inspectable notes instead of private runbooks
- describe mechanisms before reporting outcomes
- prefer local cross-links inside this repository over raw infrastructure references
- remove machine-specific details that do not help a reader understand the method

## What these notes are for

These files are meant to be stable citation targets for articles. They should
explain why a design exists, what it owns, and how to inspect the public sample
surface without depending on private machines, local layouts, or internal IDs.

## Related local notes

- `docs/data-prep-notes.md`
- `docs/semantic-indexing-notes.md`
- `docs/hybrid-layout-notes.md`
- `docs/distributed-debugging-notes.md`
- `docs/reference-corpus-pins.md`
