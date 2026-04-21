---
title: "Verifier-first C++ evals: why compile-and-test owns the metric"
description: "What the C++ evaluation stack teaches about deterministic extraction, sandbox contracts, pass@k, and why benchmark tables only become trustworthy after the verifier owns the pass label."
date: "2026-04-18"
tags: ["evaluation", "cpp", "verifier", "benchmarking", "human-eval", "MegaCpp source repository"]
---

Executable evaluation only becomes honest when the verifier is the authority. The current C++ stack is valuable because it does not treat compilation as a cleanup step after generation. It treats deterministic extraction, declared sandboxing, compile/test outcomes, and failure bucketing as the source of truth, then computes summary metrics on top. That order matters more than any single leaderboard number.

Evaluation systems for natural-language tasks can get away with fuzzy matching, grader models, or human preference signals. C++ cannot. There is no meaningful notion of "almost correct" once the deliverable is code that must compile and satisfy tests. For that reason the verifier is not auxiliary tooling. It is the measurement device.

The current codebase makes that visible in several places. The harness layer is responsible for orchestrating datasets, samples, results, and aggregation. The verifier layer implements the language-specific authority, including how source is extracted, compiled, and executed. Public-facing documentation explains the operational contract. Together they imply a discipline that is easy to state and surprisingly rare to follow: first recover the intended program deterministically, then run it inside a declared verification environment, then compute metrics like pass@k from those verifier-backed outcomes.

## Why verifier-first matters specifically for C++

C++ evaluation fails in two distinct ways. The obvious one is that the model produces wrong code. The subtler one is that the harness misidentifies the intended code or evaluates it under an unstable contract. The second problem is more dangerous because it can fabricate research conclusions.

If a model emits a correct solution wrapped in extra markers, reasoning text, or multiple fenced blocks, a weak harness may choose the wrong region and report a compile failure. If the sandbox changes compiler flags, timeout windows, or filesystem assumptions between runs, the same candidate can pass on Tuesday and fail on Wednesday for reasons unrelated to model quality. Once that happens, benchmark tables stop describing capability and start describing evaluator drift.

That is why a verifier-first stack is not just "strict." It is epistemically cleaner. It makes the question precise: under this extraction policy and this compile-and-test contract, did the candidate solve the task?

| Layer | What it must decide | What breaks if it is sloppy |
| --- | --- | --- |
| Extraction | Which code region is authoritative | Format noise becomes fake capability loss |
| Sanitization | Which wrappers are allowed or removed | Harmless scaffolding becomes a compile error |
| Compilation | Which toolchain and flags define validity | Results become non-comparable |
| Execution | Which tests and timeouts define success | Syntactic validity is mistaken for correctness |
| Aggregation | Which labels count toward pass@k | Summary metrics drift away from evidence |

That table is the philosophy of the stack in one place. Every interesting metric question sits downstream of those decisions.

## The repo already encodes the right authority boundary

The verifier layer matters because it is where success stops being rhetorical. It defines the logic that converts a model response into a verifier outcome. It is the place where extraction policy, compile orchestration, and test execution are tied together. Even if different task families require different details, the design pressure is the same: the verifier owns the label.

The harness then sits one level up and does the work an evaluation harness should do: load tasks, call the verifier consistently, record outcomes, and summarize them. That separation is healthy. It prevents the benchmark loop from smuggling in ad hoc per-task decisions that would make historical comparisons noisy.

Public evaluation documentation is also more important than it looks. A good evaluation README is not onboarding fluff. It is the human-readable form of the contract. If the harness is supposed to support deterministic extraction, separated compile and run timeouts, stable result schemas, or pass@k built on verifier labels, that documentation is where those expectations become legible to the next engineer.

One practical benefit of this shape is that regressions become localizable. If scores move, you can ask whether the model changed, the extraction logic changed, the compile contract changed, or the aggregation changed. A weaker setup would collapse all of that into a single opaque number.

## Deterministic extraction is not polish; it is part of the metric

C++ generations increasingly arrive with extra structure. Some models emit analysis before code. Others produce multiple code fences or tool-style wrappers. A verifier-first stack has to answer a very basic question consistently: what exact text becomes `main.cpp`?

That is why deterministic extraction belongs inside the metric path rather than in a cleanup script somebody runs later. If the extraction rule is unstable, the metric is unstable. If the extraction rule is implicit, the metric is not reproducible.

The practical requirement is simple.

```text
1. identify the authoritative code region deterministically
2. normalize only known wrappers or boilerplate
3. preserve the candidate source exactly after normalization
4. compile under a declared contract
5. record extraction failure separately from compile failure
```

Those steps matter because they prevent evaluator folklore. Without them, two engineers can look at the same raw sample, pick different code blocks manually, and report different pass rates. Once that happens, there is no real benchmark anymore.

There is also a strategic reason to be strict here. Structured-output checkpoints are common in modern training pipelines. If a checkpoint switches from raw code completion to a tool-style format, a raw benchmark may show a catastrophic drop even if the underlying code ability is unchanged. The right conclusion is not automatically "the model regressed." The right first question is whether the verifier contract still matches the output contract.

## Pass@k only means something after the verifier owns the label

`pass@k` is a useful summary because executable tasks are inherently stochastic. A model may produce several candidate programs, and the engineering question is often whether at least one of them is correct within a limited sample budget. But `pass@k` becomes meaningless if the underlying pass label is noisy.

That is why the ordering matters so much:

1. verifier determines pass, fail, timeout, extraction failure, or sandbox error;
2. harness aggregates those stable labels;
3. metrics summarize the already-grounded outcomes.

If you reverse that order and let heuristic parsing or soft matching leak into the label itself, `pass@k` turns into polished ambiguity. You still get a number, but it no longer tracks what engineers care about.

The current design avoids that trap. The harness works because it sits on top of verifier-backed truth rather than replacing it. That is the correct architecture whether you are measuring HumanEval-style tasks, broader C++ suites, or checkpoint-to-checkpoint drift.

| Metric view | Good use | Bad use |
| --- | --- | --- |
| compile rate | measure syntax + toolchain compatibility | treat it as complete task success |
| test pass rate | measure executable correctness | ignore extraction instability |
| pass@k | summarize verified multi-sample success | compute over heuristic or manually corrected labels |
| failure buckets | diagnose regressions | hide all failures inside one blended score |

A verifier-first setup does not reject summary metrics. It gives them a foundation.

## The sandbox contract is part of the benchmark, not an implementation detail

Evaluation people often talk about models and datasets while quietly changing the environment underneath them. For executable tasks that is a mistake. The compiler, flags, timeout limits, filesystem view, and allowed includes are all part of the benchmark contract.

The stack is strongest when those assumptions are explicit. A verifier should not just say "compile failed." It should know what compiler was used, which phase timed out, whether the failure happened before tests began, and whether the task was single-file or multi-file. That kind of reporting is not bureaucracy. It is what makes results comparable across weeks of research.

The deployment checklist is short, but it does need to be explicit:

| Requirement | Why it matters |
| --- | --- |
| fixed toolchain and flags | prevents accidental benchmark drift |
| separate compile and execute timeouts | distinguishes language validity from runtime behavior |
| stable memory/filesystem limits | keeps results comparable |
| explicit single-file vs multi-file policy | avoids hidden task-shape bias |
| structured result schema | makes regressions diagnosable |

Once those pieces are stable, the benchmark becomes instrumentation instead of theater.

## The harness should preserve evidence, not just publish scores

One of the best lessons from the current design is that final numbers are not enough. Engineers need to inspect the path from raw sample to extracted source to compile result to test verdict. The harness is useful precisely because it can sit above those details without erasing them.

That evidence chain is what lets a team tell the difference between four very different events:

- the model generated wrong logic;
- the model generated correct logic wrapped in unsupported formatting;
- the toolchain or sandbox changed;
- the verifier itself regressed.

Those are not edge cases. They are routine failure classes in real model-eval work. A verifier-first stack makes them visible enough to debug.

This is also where contract-oriented documentation helps. When a report tells readers to inspect the verifier logic, the harness behavior, and the public evaluation contract, it is pointing them toward the actual rules rather than asking them to trust a dashboard. That habit is worth preserving in MegaCpp too.

## What MegaCpp should institutionalize

The long-term rule should be simple: for executable tasks, the verifier owns the success label and every aggregate metric must be downstream of that fact.

That implies a concrete standard:

- deterministic extraction is mandatory;
- compile and run phases are reported separately;
- sandbox assumptions are declared and stable;
- pass@k is computed only from verifier-backed labels;
- reports include failure buckets, not only one top-line score.

This is stricter than leaderboard culture, but it is also more useful. Training stacks need metrics that survive contact with debugging. If a score drops, the team should be able to tell whether the model regressed, the format changed, or the verifier moved. The only way to make that possible is to treat verification as the authority from the start.

In other words, verifier-first is not a preference about style. It is the minimum architecture for honest C++ evaluation.

## References

- [Compile commands and semantic graphs](https://megacpp.com/blog/compile-commands-and-semantic-graphs/)
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [LiveCodeBench](https://livecodebench.github.io/)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
