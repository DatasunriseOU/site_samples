# Reference Corpus Pinning Notes

This note records the public rules MegaCpp uses for source, benchmark, and
tokenizer pinning.

## Pinning policy

- Never publish a training or evaluation claim against floating `main`,
  `master`, or `tip`.
- Record the exact revision for every source repository, dataset, tokenizer, and
- benchmark harness.
- Keep license metadata alongside the revision record.
- Store optional provenance identifiers such as SWHIDs when they are available.

## Minimal metadata per input

| Field | Why it matters |
| --- | --- |
| source id | tells readers what the input actually was |
| revision or tag | makes the snapshot reproducible |
| license metadata | keeps legal review machine-readable |
| retrieval date | explains time-sensitive drift |
| schema version | explains downstream format changes |
| optional SWHID | provides a stable provenance pointer |

## Public examples

| Input class | Good pin style | Bad pin style |
| --- | --- | --- |
| source repository | `llvm-project@llvmorg-19.1.0` | `llvm-project@main` |
| dataset | named release plus dataset revision | `latest` |
| tokenizer | saved artifact plus commit hash | "current tokenizer" |
| benchmark harness | tagged release or commit hash | "recent harness checkout" |

## Rule of thumb

Pinning does not guarantee legal or scientific perfection. It does make claims
auditable, which is the minimum requirement for a public training narrative.
