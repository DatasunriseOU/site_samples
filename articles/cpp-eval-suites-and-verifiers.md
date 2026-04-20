---
title: "The C++ Eval Suites, Verifiers, and the Compile-Then-Test Wall"
description: "The C++-specific eval surface we actually run: problem sets, the compile-then-test verifier sandbox, header and include coverage, and how per-specialist scorecards fall out of the same harness."
date: "2026-04-18"
tags: ["evaluation", "C++", "verifier", "benchmarks", "testing"]
---

Nobody ships a code model on perplexity. We ship on whether the generated code compiles, whether it runs, and whether it passes tests written in advance by somebody who was not the model. The C++ eval surface is what turns those three questions into numbers, and the verifier is the wall between "the model emitted plausible-looking C++" and "we counted it as correct." This post is the C++-specific half of the eval story: the suites we run, how the verifier sandbox is wired, how header and include coverage gets measured, and how the scorecard for each specialist comes out of the same harness.

## Why MegaCpp cares about this

A C++ model that scores 85% pass@1 on a single-file Python-style benchmark is not a C++ model. It is a single-function generator that happens to emit C++ syntax. Real C++ work is translation units with includes, headers with template instantiations, macro-gated build configurations, call graphs that cross files, and compiler diagnostics that are load-bearing feedback. Any eval that grades on stdout-matching, or on LLM-as-judge narrative review, is grading the wrong thing. We built the eval suites around a compile-then-test wall because it is the single source of truth everyone — training, SFT, RL, release — can agree on.

The secondary reason: our product is an ensemble of specialists, not a single model. Each specialist has a different generation profile, a different failure mode, and a different operational budget. A shared harness with shared verifier semantics is what lets us compare them fairly. The scorecards fall out; we do not design them separately.

## How the eval stack is structured

There are two main layers on the C++ eval side: a HumanEval-style benchmark runner and a general-purpose verifier used during RL and SFT reward computation. They share a compile-then-test architecture, run through `g++` with tight timeouts, and treat exit code as truth.

### The problem set

The base C++ benchmark is 163 problems adapted from HumanEval and translated to C++. The difficulty distribution is 35 easy, 118 medium, 10 hard. Each problem is a line of JSON with `name`, `prompt`, `test`, `difficulty`, `task_id`, and `entry_point`. The prompt is a complete translation unit up to and including the opening brace of the function under test: required `#include` headers, a docstring with example inputs and outputs, the function signature, and `{`. The model's job is the function body. It is not a chat format; it is function completion.

Example shape:

```cpp
#include<algorithm>
#include<stdlib.h>
#include<vector>
using namespace std;
/* docstring with examples */
bool has_close_elements(vector<float> numbers, float threshold){
```

The model generates raw C++ continuing from the opening brace, and generation stops on one of three heuristics: a closing `}` that balances the opening brace, a `\nint main(` or `\nvoid main(` prefix that signals the model started writing the test harness, or the token limit (512 by default). The completion is truncated via a brace-depth parser that handles strings, line comments, block comments, and nested braces correctly.

### The verifier sandbox

The verifier is the compile-then-test core. It does the following, in order:

1. Strips thinking blocks. Content between `<THINK_*>` and `<THINK_END>` is removed pre-compilation; a token-mode variant (`strip_think_token_ids`) does the same on raw ID sequences.
2. Writes the source into a temp directory with `TMPDIR`/`TMP`/`TEMP` all scoped to that dir so compiler intermediates are garbage-collected on context exit.
3. Decides compile-only vs compile-and-link by scanning for `int main(` after stripping comments. Present -> binary; absent -> object file. That rule lets the verifier accept function-only snippets without caller-side wrapping.
4. Compiles with `g++ -std=c++20 -Wall -Wextra` plus caller flags (usually `-O1` for eval; fuzzer and ASan flags stay out of the reward path for timing reasons). Benchmark-runner defaults use `-std=c++17 -O1`.
5. Enforces a wall-clock timeout on compile and execution (default 10 s): generous for HumanEval shapes, intentionally tight for pathological metaprograms.
6. If compile succeeded and a `main` exists, runs the binary under a second timeout.
7. Parses test output as `PASSED X/Y` or individual `PASS`/`FAIL` lines. No arbitrary stdout matching.
8. Returns structured results (`compile_ok`, `warnings`, `run_ok`, `stdout`, `stderr`, `returncode`, `tests_passed`, `tests_total`) plus an optional `error_class` and a parsed GCC `diagnostic` block.

The sandbox itself is `subprocess.run` inside a tempdir with a `ulimit`-bounded pod entrypoint on the infra side. When `bwrap` is available, the binary runs under bubblewrap with a read-only bind for the executable and a minimal rootfs; when it is not, there is an explicit fallback to the unsandboxed run. This is deliberately boring. Exotic sandboxing adds failure modes without improving the signal; `g++` on a tempdir with tight timeouts has worked reliably for every benchmark we have run.

### The compile-then-test wall

We grade on three outcomes, defined by the verifier:

- **PASS**: compiled, ran, and either asserts passed (exit code 0) or parsed `PASSED X/X`.
- **COMPILE_FAIL**: `g++` returned non-zero. Syntax error, type error, missing header, missing include guard, ODR violation. The model failed at the language frontend.
- **RUNTIME_FAIL**: compiled but `assert` tripped or the process exited non-zero. Logic was wrong.

Pass@k is computed using the unbiased estimator from the Codex paper (`CppBenchmark.pass_at_k`): for `n` samples at temperature > 0 with `c` correct, `pass@k = 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)`. We compute `pass@1` and `pass@10` and never quote a sample-size-of-one number at temperature > 0.

For training checkpoints we use greedy decoding — `temperature=0`, one sample per problem — because the metric we want to track is deterministic across checkpoint comparisons. Stochastic decoding with multiple samples is for the release-candidate layer.

### Reward shaping for RL

The RL loop calls `compute_reward(verify_result)` with a deliberately coarse schedule:

- `0.0` — does not compile.
- `0.3` — compiles but does not run cleanly, or ran with no tests provided.
- `0.5` — compiles and runs cleanly, no tests declared.
- `0.3 + 0.7 * (tests_passed / tests_total)` — with test results.

The coarse bands exist to keep the reward signal robust under verifier noise. A model that learns to game a finer-grained schedule — say, to produce output that compiles with warnings but no runtime errors — already did most of what we want. The 0.3/0.5 floor rewards "at least it compiles" more than nothing but less than a working program, which is the ordering we want during early RL.

### Multi-file and real translation units

`CppBenchmark.compile_and_run_multifile` accepts a `{filename: content}` dict, writes it into the tempdir, collects every `.cpp`/`.cc`/`.cxx`, and compiles with `-I<tmpdir>` so includes resolve naturally. This is the hook for benchmarks that evaluate header coverage and cross-file reasoning.

### Header coverage and include-graph sanity

For the benchmarks that come out of our bounded-context-graph data (v4/v5), the eval harness records which `#include` directives the model emitted and checks them against the ground-truth include set extracted by Tree-sitter from the source translation unit. A model that produces a plausible-looking function but omits a required `#include <algorithm>` will fail COMPILE_FAIL, but the scorecard breaks that down further: header-missing errors, header-extra (invented) errors, and header-correct-but-code-wrong are separate rows. That decomposition is what tells us whether a specialist's headline compile-fail rate is dominated by include-graph drift or by actual code-level mistakes, and the answer has shaped more than one ablation.

### Eval watcher and pipeline architecture

During pretraining we keep the training TPU slice uninterrupted. Checkpoints are written to object storage on a cadence; a separate watcher on a cheap inference GPU polls for new checkpoints, pulls them, loads the model in float32 for T4 compatibility, runs all 163 problems with greedy decoding, writes the results JSON locally, and mirrors it back to object storage. The watcher state is namespaced by a SHA-1 of the absolute checkpoint directory path so two watchers pointed at different SFT runs whose paths share the same trailing component do not confuse each other's state. That isolation bug shipped once and is pinned now.

Results are written with the `.tmp + os.replace` pattern everywhere an atomic promotion matters; partial JSON cannot be mirrored. Upload failures retry on the next poll.

## How it lands in MegaCpp

In production, the eval stack is the same code, wired into the release pipeline instead of a bespoke watcher. The verifier is imported by the RL and SFT reward paths directly. The benchmark runner is called by the release-candidate harness that every specialist checkpoint passes through before it is admitted to the ensemble.

What is lifted as-is: the compile-then-test wall, the exit-code truth rule, the pass@k estimator, the brace-depth stopping rule, the `PASSED X/Y` parser, the thinking-block stripper, the reward schedule. These have been stable for several release cycles and changes go through contract tests (see `eval-harness-plumbing` for the full list).

What is rewritten: the sandbox entry point. Production runs verifiers inside per-pod resource envelopes with a bounded compile RAM ceiling (`ulimit -v`) so that `g++` OOM on a pathological template becomes a clean compile failure instead of a pod-kill; that ceiling was added after a real incident.

What is dropped: LLM-as-judge graders for the context-adherence and hallucination axes. A static parser against the Tree-sitter symbol table is zero-cost and reproducible, and judge-API drift is not a thing we want in a regression dashboard. The judge still exists for the release-candidate correctness axis on long-context benchmarks (where the ground-truth test set is sparse), but it is capped by compile status — a non-compiling diff caps its maximum score.

What is a feature flag: dynamic flaky-task retry. Off by default. Our rule is that if a task flakes, we either fix it or deprecate it, not retry it.

## Ablations and what we kept

The eval ablations worth keeping:

- We tried grading on stdout-matching for a subset of benchmarks. It was noisy (trailing newlines, locale, printf formatting) and it silently penalized correct outputs that formatted differently. We moved every benchmark to `#undef NDEBUG` plus `assert()` grading, with exit code as the signal.
- We tried LLM-as-judge for the axes perplexity cannot see (context adherence, hallucination). Judge drift between runs — self-preference, length bias, sycophancy — made the numbers noisier than the underlying signal. The Tree-sitter-extracted `Callee` list is a deterministic oracle for those axes. We kept the judge only for the correctness axis on hard cross-file benchmarks.
- We tried a T4-sharing scheduler across multiple watchers. It was fast and debug-hostile; we went back to round-robin and accepted the throughput hit for the operability win.
- We tried auto-retry on "Compilation timeout." It made numbers noisier without improving them. Timeouts are now a clean failure.
- We kept the three-layer eval cadence (in-line perplexity, 4K functional on every promoted checkpoint, 16K and 64K cross-file on release candidates). Each layer earns its cost by catching regressions that the cheaper layers miss.

## Production checklist

- Every grade is exit-code based. Stdout matching is not accepted as a pass signal.
- `g++` is the reference frontend. `-std=c++17` for the HumanEval-adapted base; `-std=c++20` for everything that uses the verifier directly. Other compilers are not part of the grading contract.
- Timeouts are enforced at two levels: compile and execute. Default 10 seconds, configurable per-benchmark. Timeouts are clean failures, never auto-retried.
- Think-block stripping runs before every compile on the reward path.
- Pass@k uses the unbiased estimator. We do not quote pass@1 at temperature > 0 with n = 1.
- Multi-file benchmarks go through `compile_and_run_multifile` with a real include path; header coverage is a first-class scorecard row, not a footnote.
- Eval watcher state is namespaced by SHA-1 of the absolute checkpoint directory path; path-suffix collisions do not corrupt state.
- Result JSON is written with `.tmp + os.replace` atomic-promotion and mirrored to object storage; partial files cannot be mirrored.
- A new benchmark passes the contract test set (pass@k edge cases, brace-depth tests, `CODE_START` tool-call extraction, checkpoint-dir isolation, end-to-end bands on known checkpoints) before it enters rotation.
- LLM-as-judge is capped by compile status and is used only where a deterministic oracle is infeasible. Judges are rotated and spot-checked across runs.

## Eval surface snapshot

| Suite | Verifier | Signal |
|-------|----------|--------|
| Single-file pass@k | compile + unit tests | correctness under k samples |
| Multi-file projects | `compile_and_run_multifile` | header/include coverage |
| Brace-depth and extraction | deterministic oracle | parser-sensitive regressions |
| LLM-as-judge (narrow) | rubric + compile gate | judged only where oracle absent |

## References

- [Verifier-first C++ evals](https://megacpp.com/blog/verifier-first-cpp-evals.md)
- [LiveCodeBench](https://livecodebench.github.io/)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [HumanEval benchmark dataset](https://github.com/openai/human-eval)
