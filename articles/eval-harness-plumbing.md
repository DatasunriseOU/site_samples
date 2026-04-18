---
title: "Eval Harness Plumbing: The Parts That Are Not the Benchmark"
description: "The four-axis eval harness plumbing under our C++ benchmarks: sandboxing, compile walls, timeouts, parallel runners, flake isolation, and the contract tests a new benchmark has to pass before it goes into CI."
date: "2026-04-18"
tags: ["evaluation", "testing", "infra", "c++"]
---

This post is not about eval results. It is about the mechanics that sit under every eval number we publish: how a model completion gets from tokenizer output to an exit-coded pass/fail, what isolates that from everything else running on the box, and why adding a new benchmark is not "write a scorer and point it at the checkpoints." Most eval regressions we see are harness regressions, not model regressions, and the plumbing here exists so that distinction can be made fast.

## Why this matters

A four-axis eval whose harness is sloppy will produce a four-axis lie. We grade C++ generations on compile probability, context adherence, hallucination rate, and end-to-end correctness, and the last two require actually running model output as a program. Running adversarially-shaped C++ on shared infrastructure with naive `subprocess` plumbing leaks zombies, miscounts compiler errors as model failures, and silently drifts numbers across watcher restarts.

The plumbing decisions here are the boring ones that prevent that. Hard timeouts at three nesting levels, atomic result writes, brace-aware stop tokens shared across benchmarks, namespaced watcher state keyed by a digest of the checkpoint path, and a contract test set that a new benchmark must pass before it gets a slot in rotation. None of it is novel. All of it is required to make a number worth quoting.

## 1. The four axes, briefly

We measure code generation on four axes because perplexity does not tell you whether the output is code:

1. Compile probability against the original translation unit.
2. Context adherence — does the model call functions that are actually in the provided call graph, or does it invent them?
3. Hallucination rate — references to non-existent symbols, headers, or overloads.
4. Correctness, graded against a held-out test set of cross-file prompts.

Only (1) and (4) require running generated code. (2) and (3) are static parses against a Tree-sitter symbol table extracted from the source TU. From a plumbing point of view that split is the whole design. Axes (2) and (3) are pure CPU, stateless, and trivially parallel. Axes (1) and (4) need g++, a filesystem, a process budget, and a way to survive model output that does not merely fail — it hangs, fork-bombs, or tries to open `/dev/zero`.

## 2. The sandbox (such as it is)

"Sandbox" overstates it. What we actually run is a disciplined use of `tempfile.TemporaryDirectory` plus `subprocess.run` with hard timeouts and captured output. The logic in the public C++ compile-and-run evaluation helper is roughly a hundred lines and does five things:

1. Writes `prompt + model_completion + test_code` to a fresh `solution.cpp` in a per-task tempdir.
2. Shells out to `g++ -std=c++17 -O2 -o solution solution.cpp` with a 30-second compile timeout.
3. If compilation fails, returns `(False, "Compilation error: " + stderr[:500])`.
4. If compilation succeeds, runs `./solution` with a 10-second runtime timeout (configurable per benchmark).
5. Returns `(passed, reason)`, where `passed` is the exit-code-zero check.

There are things that look missing: no chroot, no seccomp filter, no cgroup memory cap. We chose not to add them. The model is ours, the machine is ours, the checkpoints are ours, and the worst realistic failure from model output is a test binary that loops. Timeouts plus a `TemporaryDirectory` context that deletes its contents on exit were enough. The real isolation boundary is that the harness runs on dedicated eval machines and not on training hosts, and that boundary is the one that matters.

### Empirical gotchas

- The 30-second compile timeout matters. A completion that defines an accidental recursive template can take minutes to compile and will. The cap is not about security; it is about not burning a T4 hour on one broken sample.
- `g++ not found` is a real failure mode on freshly-scaled pods, and the path distinguishes it from "compiled and failed." Counting "no compiler" as "did not compile" cascades wrong into the compile-rate.
- Compile and runtime error strings are truncated to 500 bytes. Model completions can produce multi-megabyte g++ error blasts (unresolved templates), and we were paying for log storage we did not need.

## 3. The compile wall

Compilation, not inference, is the most expensive part of a C++ eval. On our T4 pods a checkpoint finishes generating 163 HumanEval-C++ completions in roughly two to three minutes; compiling and running those samples takes fifteen to forty-five minutes depending on what g++ gets handed.

That asymmetry — inference fast, compile slow — drives the rest of the plumbing. A naive harness that serially iterates "generate, compile, check" per problem burns GPU idle while g++ chews on template instantiations. We split the two phases:

1. Generation pass: load the model, walk the benchmark, store `completion_{task_id}.cpp` on local scratch. GPU is the bottleneck.
2. Compile pass: a multiprocessing pool over the completions, each worker shelling out to `g++` with its own tempdir. GPU is idle, which is fine because this is the cheap machine.

The eval watcher runs them back-to-back per checkpoint. For richer evals with multiple samples per task (`pass@k` with `k=5`, `num_samples_per_task=5`), generation runs long enough that we gated it on `temperature > 0`; below that we do greedy, one sample, deterministic. That is what every training-checkpoint eval run uses.

## 4. Timeouts, layered

Timeouts nest, and the wrong nesting silently changes your numbers.

| Scope | Limit | On expiry | Counted as |
| --- | --- | --- | --- |
| Per compile | 30 s | `TimeoutExpired` | "Did not compile" |
| Per run | 10 s default | `TimeoutExpired` | Runtime failure |
| Per task end-to-end | ~45 s + generation | implicit | bounded |
| Per benchmark | watcher-configured | watcher kill | "degraded" |
| Per watcher poll | 120 s between polls | next poll | n/a |

We do not separately report "timed out" vs. "assert failed" because we explicitly do not want to reward infinite loops. The per-benchmark wall-clock budget fires roughly never; it has fired when we broke the tokenizer and every generation hit the max-token path.

The thing we got wrong once and now test: a timeout that fires inside `subprocess.run(..., capture_output=True)` can leave the subprocess running if you do not explicitly kill the group. The default behavior was enough on our production runners; on one machine class it was not, and zombie g++ processes piled up. We keep a cleanup sweep on the pod that kills any `g++` older than five minutes as a belt-and-suspenders.

## 5. Parallel runners and watcher state

Parallelism happens on two layers.

Inside one checkpoint, compilation parallelism. We fan out across a multiprocessing pool sized at `min(n_cores, 8)`. On a T4 pod, eight concurrent g++ processes saturate the CPUs without tempdir disk I/O becoming the bottleneck. We measured this; at twelve workers throughput went down, at four it went down differently, eight was the knee.

Across checkpoints, parallelism is the eval-watcher fleet. Each member owns a named bucket prefix and polls independently, using `.watcher_state.json` to track which steps it has already assessed. If two watchers race for the same step, both produce results, both upload with the same filename, and one wins the overwrite. That is deliberate: results are idempotent enough that double-running is harmless, and we would rather spend T4 minutes than add a coordination layer.

The assessed-steps state is persisted per watcher, namespaced by a SHA-1 of the absolute checkpoint directory path:

```python
normalized = os.path.abspath(checkpoint_dir).rstrip(os.sep)
digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
```

That ten-character suffix is what prevents two watchers, pointed at different SFT runs whose paths share the same last component, from confusing each other's state. That bug shipped; the namespacing is the fix and sanitized checkpoint-isolation tests pins it.

## 6. Flake isolation

Flakes in an execution-based eval come from three places.

### Model-side flakes

A non-greedy decode at `temperature > 0` is explicitly stochastic; that is intentional for pass@k. We pin the data-ordering seed so the per-task sample set is deterministic across a single run, and we log the seed in the result JSON. If two runs differ, we check the seed first.

### Generated-program flakes

Generated C++ can be non-deterministic on its own: `time(NULL)`, uninitialized memory, races in threaded code. We mitigate by constraining the test framework. Every prompt ends `test_code` with `#undef NDEBUG` plus explicit `assert()`s, and we never grade on stdout matching. Either the asserts pass (exit 0) or they do not. That removes a class of timing, locale, and formatting flakes that plague string-match graders.

### Harness-side flakes

The ones we have actually seen:

- Tempdir cleanup racing with an open file descriptor. Fix: `ignore_errors=True` on cleanup; a leaked one-KB file is not worth caring about.
- `g++` OOM on pathological templates. Cap compile RAM with `ulimit -v` in the pod entrypoint so OOM becomes a clean compile failure instead of a kernel kill.
- Result JSON not flushed when the upload fires. Same `.tmp` plus `os.replace` pattern as the checkpoint code; partial result files cannot be mirrored. Failed uploads retry on the next poll.

The flake we still have not tamed: g++ occasionally produces non-deterministic diagnostics for the same input under different system states (PCH caches, filesystem case). This affects only the human-readable error text, not the pass/fail signal, but it breaks result diffs across runs. The workaround is to compare on `(compiled, passed)` and not on stderr text when bisecting a regression.

## 7. Contract tests for a new benchmark

A new benchmark module has to pass a contract test set before it enters rotation. We added this after two benchmarks shipped with subtly wrong accounting.

Required surface:

- `load_examples()` returning a list of `{name, prompt, test, difficulty, task_id, entry_point}` dicts.
- `generate_completion(model, tokenizer, prompt, ...)` using the standard brace-depth stopping rules.
- `compile_and_run(code, test) -> (passed: bool, error: str)` that classifies errors so the `cpp_compile_rate` and `cpp_pass_rate` arithmetic holds. "Timed out" is a runtime failure (compiled, then hung); "g++ not found" is a harness failure (not a model failure); "code_start block missing" is a parser failure (fall back to stripped code or count as compile fail, document which).
- A `problem_set.jsonl` schema test that `load_examples` actually returns the schema we say it does. An empty set is an error, not "0% pass."
- A difficulty-distribution check, so a benchmark that is secretly 95% easy gets labeled.

Required tests before the first checkpoint is allowed to use it:

- `test_pass_at_k_known_value`, `test_all_correct`, `test_none_correct`: edge-case sanity checks on the shared pass@k estimator.
- `test_pass_at_k_monotonic_in_c`: property test — as `c` rises with `n` fixed, pass@k must not decrease. Small but saved us once.
- Brace-depth and stop-token tests (`test_simple_function`, `test_nested_braces`, `test_string_with_braces`, `test_line_comment_with_braces`, `test_block_comment_with_braces`, `test_empty_body`, `test_no_closing_brace`, `test_escaped_quote_in_string`). The stopping rule is shared across benchmarks, so a failure here blocks every benchmark.
- Tool-extraction and `CODE_START` parser tests (`test_code_start_end_block`, `test_longest_code_block_selected`, `test_query_tool_run`, `test_fallback_to_stripped_code`). SFT completions are wrapped in tool-call markers, raw completions are not. The SFT-vs-raw mismatch — 0% compile because the model was emitting `<THOUGHT_START>` blocks on a raw-completion benchmark — happened because the benchmark skipped extraction. Pinned now.
- `test_checkpoint_dir_isolation`: watcher state files do not collide across named runs.
- End-to-end smoke on two fixed checkpoints (known-bad, known-reasonable) with expected `(compile_rate, pass_rate)` ranges. We test bands, not exact numbers; anything outside the band is a harness regression.

Only after those pass does the benchmark get a slot in the watcher's rotation.

## What we kept and what we threw away

We kept the split between static axes and execution axes, the simple `subprocess.run`-plus-tempdir sandbox, the layered timeouts, the SHA-1-suffixed watcher state, the contract tests, and atomic-rename writes mirrored to object storage. We kept the rule that most eval regressions are harness regressions until proven otherwise.

We threw away a T4-sharing scheduler in favor of round-robin across watchers (easier to debug); auto-retry on "Compilation timeout" (made numbers noisier without improving them); LLM-as-judge graders for axes (2) and (3) (a static parser against the Tree-sitter symbol table is zero-cost and reproducible, and judge-API drift is not something we want in a regression dashboard); and a dynamic flaky-task ignore list (if a task flakes we either fix the benchmark or deprecate it).

The harness is, in total, a few hundred lines of Python: `subprocess.run`, a consistent stopping rule, tight timeouts, an isolated pass@k estimator, a multiprocess pool around `g++`, and a result artifact mirrored with the same atomic-rename pattern the checkpoint code uses. It works because each part is boring and individually testable. The benchmarks themselves cannot be boring; that is exactly why they need contract tests before they ship.

## References

- MegaCpp evaluation harness architecture
- compile-then-test benchmark orchestration patterns
- contract-testing patterns for executable evals
