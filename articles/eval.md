---
title: "How We Evaluate the MegaCpp SLM Ensemble on Real C++ Work"
description: "The evaluation design, verifier stack, and release gates we use to measure C++ model quality without collapsing everything into a single leaderboard number."
date: "2026-04-18"
tags: ["evaluation", "benchmarks", "slm", "c++", "cost-per-quality"]
---

The MegaCpp evaluation story only matters if the measurement is strict enough to reject plausible-looking but wrong code. This post is about that measurement design. It explains what we score, how the harness runs, why compile-and-test signals outrank generic text metrics, and how we compare release candidates across quality, cost, and operational complexity.

The central idea is simple: a C++ model should be judged on repository-grounded work, not on whether it can produce a nice-looking snippet in isolation. That means evaluating generated patches against real translation units, real build settings, real symbol sets, and held-out correctness checks.

## Why this matters

A broad general-purpose model can look strong on short, self-contained coding prompts and still fail on the things that matter in a real C++ repository. The job is not to emit a plausible `std::vector` example. The job is to take a translation unit with its headers, macros, call graph, and template instantiations, then produce a change that compiles, links, passes tests, and avoids inventing APIs.

If the evaluation stack cannot tell the difference between "the patch landed" and "the snippet sounded reasonable," it will promote the wrong checkpoints. That is why the harness keeps compiler-backed and verifier-backed signals ahead of presentation quality.

## 1. What we are actually measuring

Our public evaluation design frames code-generation quality along four axes that text-only metrics do not capture well:

1. Compilation probability of the generated diff against the original translation unit.
2. Context adherence: did the model stay within the provided symbols, callees, and include structure?
3. Hallucination rate: references to symbols, headers, or overloads that are not actually present.
4. Correctness against held-out tests and repository-grounded task checks.

Perplexity still matters during pretraining because it is cheap and useful for early regression detection, but it is not the product metric. Release decisions lean on verifier-backed outcomes.

### Why these four, not pass@k alone

Pass@k is useful, but by itself it hides failure modes that matter in C++. A candidate can compile accidentally while still violating context, and a candidate can stay syntactically neat while inventing APIs that do not exist. Keeping the axes separated makes it easier to spot where a checkpoint is getting better and where it is only getting luckier.

## 2. The harness: cheap inference, compiler-grounded judging

The harness is designed to mirror the product problem more closely than a synthetic sandbox does.

- Candidate checkpoints generate completions against held-out cross-file C++ prompt graphs.
- Generated diffs are checked against compile and verifier rules before any summary score is computed.
- Runs are fanned out by variant and seed so the team can compare distributions rather than a single convenient point estimate.
- External judge models can help with structured review, but deterministic repository signals stay in charge for compile validity and symbol adherence.

LLM-as-a-judge systems have known biases, so the harness leans on deterministic oracles wherever possible. The compile axis is run through a declared C++ frontend and build contract before a reviewer model sees the result. Symbol and callee checks are matched against repository-derived context rather than taste-based grading.

```yaml
# Sketch of the per-variant job shape.
spec:
  parallelism: 8
  template:
    spec:
      containers:
        - name: eval-worker
          image: eval-worker:<release-tag>
          args: ["--variant", "$(VARIANT)", "--seed", "$(SEED)"]
```

## 3. Three benchmark layers

We run three layers, from cheapest to most release-like.

Layer 1: held-out loss and perplexity checks during training. These are early warning signals, not headline product metrics.

Layer 2: shorter-context functional evaluation for ablation sweeps and rapid checkpoint comparison.

Layer 3: longer-context evaluation on repository-grounded bounded-graph tasks, where cross-file reasoning and context discipline matter more.

| Layer | Context | Primary purpose | Cost class | Cadence |
| --- | --- | --- | --- | --- |
| 1 | 1K-4K | regression detection during training | low | every few thousand steps |
| 2 | 4K | rapid functional comparison across variants | medium | every promoted checkpoint |
| 3 | 16K-64K | release-candidate validation on long-context tasks | higher | before release decisions |

The key point is not the exact hardware mix. The key point is that cheaper layers are used to filter candidates early, while the more expensive layers are reserved for decisions that justify the additional evaluation cost.

## 4. Why verifier-first scoring wins

The strongest lesson from the current stack is that verifier-backed evaluation is more reliable than pure aesthetic judging.

A compiler can reject a patch even when the prose around it sounds convincing. A symbol checker can catch invented APIs even when a judge model finds the explanation persuasive. A held-out test can fail even when the patch looks stylistically correct. Those are not edge cases in C++; they are the point of the evaluation.

That is why the harness treats compile-and-test outcomes as the authority and uses softer review signals only as supporting structure.

## 5. Release gating and reproducibility

Training throughput is not a user-facing quality metric, but the training stack still matters because unstable training receipts produce unstable evaluation results. Public reproducibility notes therefore focus on configuration discipline rather than on marketing-friendly peak numbers.

The release process keeps a few principles fixed:

- candidate checkpoints must come from pinned, reproducible training lanes;
- smoke variants must converge into an expected steady-state band before they are admitted to larger runs;
- superseded or invalidated measurements are retired instead of silently mixed into newer summaries;
- evaluation artifacts should record checkpoint identity, harness revision, verifier settings, and seed distributions.

That discipline matters more than any single throughput screenshot because it is what turns an evaluation table into something the team can trust.

## 6. Cost per quality point

The cost argument is easiest to reason about when quality and operational complexity are reported together.

A smaller or more specialized serving target can make it practical to evaluate more variants, run more seeds, and keep repository-grounded checks in the loop. A larger general-purpose baseline can still be a useful comparison point, but it often raises evaluation cost enough that teams are tempted to cut corners on repeatability or depth.

For MegaCpp, the practical comparison is therefore not just "small versus large model." It is:

1. how much evidence can we afford to collect per candidate release;
2. how faithfully can we keep compile-and-verifier checks in the loop;
3. what serving and iteration costs follow from the chosen architecture.

That framing is less dramatic than a simple model-size slogan, but it is much more useful for engineering decisions.

### Why ensemble-style evaluation is still useful

A specialist ensemble is interesting only if the evaluation can show where specialization helps and where it does not. The current design is intended to make that visible across context adherence, hallucination control, and long-context repository work rather than collapsing everything into one headline number.

That also means the public claim has to stay narrower than marketing language usually prefers. The evaluation stack can support statements such as:

- verifier-backed C++ tasks are a better release metric than generic text scores;
- cross-file and long-context tasks expose different failure modes than single-file prompts do;
- smaller, task-focused systems can make deeper and more repeatable evaluation affordable.

It does not, by itself, support sweeping claims about universal superiority over every larger general-purpose model without publishing the underlying comparison tables.

## What we kept and what we threw away

We keep the four-axis methodology, compiler-grounded checks on structure and validity, verifier-first release gates, and the practice of comparing distributions rather than cherry-picked single runs.

We threw away single-number reporting, pass@1-only storytelling, and benchmark summaries that are not traceable back to a declared verifier contract. Superseded measurements belong in reproducibility notes, not in the headline argument.

For every serious evaluation cycle, the goal is to publish enough metadata that another engineer can understand which checkpoint was tested, under which harness revision, with which verifier settings, and over which seed set. That is what makes the measurement defensible.

## References

- [Verifier-first C++ evals](https://github.com/DatasunriseOU/site_samples/blob/main/articles/verifier-first-cpp-evals.md)
- [C++ eval suites and verifiers](https://github.com/DatasunriseOU/site_samples/blob/main/articles/cpp-eval-suites-and-verifiers.md)
- [SLM data pipeline](https://github.com/DatasunriseOU/site_samples/blob/main/articles/slm-data.md)
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [LiveCodeBench](https://livecodebench.github.io/)
- [vLLM documentation](https://docs.vllm.ai/)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
