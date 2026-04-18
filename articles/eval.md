---
title: "How We Evaluate the MegaCpp SLM Ensemble Against 70B Generalists"
description: "The benchmarks, harness, and cost-per-quality math behind our claim that a focused ensemble of small C++ specialists beats 70B+ generalist models on real C++ work."
date: "2026-04-18"
tags: ["evaluation", "benchmarks", "slm", "c++", "cost-per-quality"]
---

Our pitch — that an ensemble of small C++ specialists outperforms a 70B generalist on real C++ work — only holds if the measurement holds. So this post is the methodology, not the marketing. It describes what we benchmark, how the harness runs, how we price each quality point, and why a narrow ensemble is a better product than a wide generalist for C++ specifically. Numbers are anchored to our POC eval design and to pinned configurations in the MegaCpp production stack.

## Why this matters

A 70B generalist is trained to be acceptable at everything: Python, TypeScript, SQL, prose, image captions, light C++. "Acceptable at C++" in benchmark terms usually means it can write a compiling `std::vector` example and explain `const`. That is not the job. The job is: given a real translation unit with its headers, macros, call graph, and template instantiations, produce a patch that compiles, links, passes tests, and does not hallucinate APIs.

If our eval cannot tell the difference between "the patch landed" and "the snippet looked plausible," we will ship the wrong model. The four-axis methodology, the compiler-grounded scoring, and the per-variant distribution reporting all exist to keep that distinction sharp. The throughput-pinning discipline on the training side exists for the same reason: our eval is only as honest as the checkpoints feeding it.

## 1. What we are actually measuring

our POC eval design (`architecture_and_eval_en.md`, §3) frames code-gen quality as four axes that perplexity cannot see:

1. Compilation probability of the generated diff against the original translation unit.
2. Context adherence — did the model call the `Callee` functions actually present in the provided call graph, or did it invent new ones?
3. Hallucination rate — references to non-existent symbols, headers, or overloads.
4. Correctness, graded against a held-out test set of cross-file prompt graphs.

Perplexity still matters during pretraining because it is cheap and monotonically useful, but it is not a product metric. A 70B with broad web pretraining can look excellent on MBPP-style single-file prompts and still fail every one of the four axes on a real C++ repository.

### Why these four, not pass@k alone

Pass@k collapses the four axes into one number and rewards a model that compiles by accident as much as one that actually understood the call graph. We keep the axes separated because the failure modes diverge: a generalist tends to fail on (2) and (3) — it invents APIs that "feel" plausible — while a small specialist tends to fail on (4) when prompts are deliberately tricky. Reporting them jointly is the only way to see that.

## 2. The harness: cheap inference, compiler-grounded judging

The harness mirrors what we ship, not a synthetic sandbox.

- A Kubernetes node pool of T4 GPUs is scaled on demand. T4s are cheap, plentiful, and more than enough for SLM inference; using them instead of H100s collapses eval cost by roughly an order of magnitude per checkpoint.
- Each pod loads a candidate checkpoint (one of our specialists, or a generalist baseline served via vLLM) and generates completions against a held-out set of cross-file C++ prompt graphs derived from our v4/v5 Bounded Context Graph dataset.
- Generated diffs are piped to a frontier judge model (currently Gemini 3.1 Pro Preview) acting as an expert C++ reviewer. The judge grades each output on the four axes above. Auth uses cloud Application Default Credentials, so there are no raw API keys baked into the worker image.
- Jobs are fanned out as Kubernetes `Job`s, one per variant by seed, so we get full per-variant distributions rather than single-point estimates.

LLM-as-a-judge has well-known failure modes — length bias, self-preference, sycophancy. We mitigate them three ways. Every judged prompt is paired with a ground-truth `Callee` list extracted by Tree-sitter from the original repo, so axes (2) and (3) are scored against a deterministic oracle, not the judge's taste. The compile axis is run through an actual C++ frontend (clang with the repo's real build flags) before the judge ever sees the diff; a non-compiling diff caps its maximum score. And we rotate judges periodically and spot-check with a second model to detect drift between eval runs.

```yaml
# Sketch of the per-variant Job spec we fan out.
spec:
  parallelism: 8
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-t4
      containers:
        - name: eval-worker
          image: public-eval-worker:<pinned-tag>
          args: ["--variant", "$(VARIANT)", "--seed", "$(SEED)"]
```

## 3. Three benchmark layers

We run three layers, from cheap to expensive.

Layer 1 — perplexity and loss curves on held-out C++. Computed every few thousand steps during training. Used to catch regressions early, never to claim product quality.

Layer 2 — 4K-context functional eval on the 20 ablation variants. Each variant — Dense baselines, Hybrid Mamba-3 + GQA, Engram, mHC, Fine-Grained MoE (64 experts, Top-4, one shared expert), Ultra-Fine MoE (128 experts, Top-8), plus routing, capacity, and curriculum ablations — is trained on TPU v6e-x4 slices in parallel and scored by the T4-plus-judge harness. 4K is chosen because it is short enough to sweep in hours and is where generalists look strongest; any specialist win at 4K is not an artifact of long context.

Layer 3 — 16K and 64K-context eval on v4/v5 Bounded Context Graphs, run on TPU v6e-x8 slices. Cross-file reasoning, template instantiation across headers, and repo-level refactors get tested here. Intra-document masking, YaRN / RNoPE scaling, and our content-dependent sparse-attention prototype (Pallas) are all tuned for the C++ long-context regime. Generalists trained on a web-heavy mixture get no such tuning.

| Layer | Context | Hardware | Cost class | Cadence |
| --- | --- | --- | --- | --- |
| 1 | 1K-4K | TPU training slice | Free, in-line | Every few thousand steps |
| 2 | 4K | T4 inference + judge | Low | Every promoted checkpoint |
| 3 | 16K-64K | T4 + judge, larger TPU eval | Medium | Per release candidate |

For each variant we report not a single number but the joint distribution across the four axes, plus compile-rate and judge-agreement intervals. A variant does not "win" by topping one axis; it wins by Pareto-dominating.

## 4. Training methodology feeds the eval

The eval only tells the truth if the training pipeline is honest, and earlier training review work was explicit that some early TPU runs were not. An older `tpu_full_pipeline.py` script trained on `torch.randint` noise with hardcoded rewards — any "benchmark" off those checkpoints was measuring nothing. We treat that finding as the floor for what counts as a valid eval candidate.

A checkpoint is admitted into Layer 2/3 eval only if it was produced by a pipeline that:

- uses real distributed dataloaders with SPMD sharding, not random tensors;
- uses GQA (`num_kv_heads = num_heads // 4` or `// 8`), not default MHA;
- ties input and output embeddings;
- disables weight decay on 1D tensors, biases, and embeddings in SFT and GSPO, not just base training;
- enables gradient clipping (`--max_grad_norm=1.0`) and Gemma-style logit softcapping (30.0 at the LM head, 50.0 on attention);
- uses intra-document masking on packed sequences and a stepped context curriculum (4K -> 16K -> 64K -> 128K), not a flat `max_seq_len=1024`;
- uses RoPE + YaRN or RNoPE for long-context, not bare RoPE with `rope_theta=10000.0`.

These deltas come from earlier training and evaluation review notes. Each is a small correctness fix on its own; together they are the difference between a checkpoint whose eval numbers mean something and one whose numbers are training-noise artifacts.

## 5. Production-side reproducibility

Training throughput is not a quality metric, but it bounds how many eval candidates we can produce per unit cost. The MegaCpp training and correctness stacks are pinned to reproducible configurations in the public reproducibility notes:

- A multi-GPU BF16 training lane: TP=1 PP=1 EP=4 DP=2, MBS=8 GBS=64, seq=4096, MTP_DEPTHS=2. Throughput sits in the high-280s TFLOP/s per GPU at roughly 29% MFU, with peak memory around 127 GiB per rank on the reference setup. FP8 regresses by a mid-double-digit percent on this fabric, so BF16 stays canonical.
- A multi-GPU FP8 training lane: same model, TP=1 PP=1 EP=8 DP=1, MBS=10 GBS=80, FP8 tensorwise. Steady-state sits in the high-260s TFLOP/s at roughly 27% MFU. `CG_FLAGS=NONE` is mandatory at MBS=10 because the default TransformerEngine CUDA-Graph memory pool can hold tens of GiB and OOM on iter 1.
- A short smoke variant: 7-iter run that must converge into the steady-state TFLOP/s band by iter 4-7 before any run is admitted to training, let alone eval.
- A single-GPU correctness lane (sm_121-class, unified memory, BF16, MBS=1 seq=2048). Not a throughput run; it exists so the TileLang kernels we ship stay under the shared-memory cap for that target and produce finite gradients. FP8 Mamba SSM is a dead path on that lane and we do not pretend otherwise.

Superseded measurements — an older Liger `reduction="none"` number (silent gradient corruption via a known Liger bug), an early PP=2 baseline, a never-real DualPipeV baseline — are retired in the public reproducibility notes and explicitly not cited. The active Liger workaround is `reduction="mean"` broadcast in the public linear-CE shim, and the native-Hopper CE path stays OFF because it produces `grad_norm=NaN`.

That discipline is the point. Our eval numbers are only as good as the checkpoints feeding them, and our checkpoints are only as good as the pinned, smoke-tested, no-silent-corruption training stack they came from.

## 6. Cost per quality point

This is where the ensemble argument actually lives. A 70B dense generalist at BF16 needs ~140 GB of weights alone; served at reasonable throughput it wants two high-memory accelerator GPUs per replica, plus KV cache. Our Dense 1B baseline fits in under 1 GB of VRAM and runs happily on a T4; the Fine-Grained MoE target is roughly 5B total / ~800M active, which still serves comfortably on a single mid-range GPU because only the active experts and the shared expert are on the hot path per token.

The comparison is not "small vs. large model" in the abstract. It is a cost-per-quality-point comparison across three regimes:

1. Per eval run. Scoring one checkpoint against the full Layer 2 suite on T4 pods is roughly an order of magnitude cheaper than scoring a 70B via H100 inference at the same batch and context. That lets us score every variant by every seed and get real distributions instead of single vibes-based numbers.
2. Per training run. A v6e-x4 slice sweeping 20 ablation variants in parallel at 4K context costs a small fraction of a single 70B pretrain epoch.
3. Per deployed quality point. On the four-axis C++ eval, the ensemble — Dense 1B + Fine-Grained MoE + long-context variant, routed per request — wins on context adherence and hallucination rate against generalists in our evaluation runs and is within noise on compile-rate for single-file prompts. Because it runs on roughly one T4 per replica rather than two large accelerator GPUs, the dollar-per-quality-point ratio is not close.

### Why ensemble beats generalist for C++

C++ rewards specialization. The grammar is huge, the type system is Turing-complete at compile time, the idioms (templates, SFINAE / concepts, RAII, ODR, ABI stability) are unforgiving, and the right answer for a given TU depends on headers and build flags the model has to actually look at. A generalist spends most of its parameters on things that are not C++. A fine-grained MoE with 64 tiny experts plus a shared expert lets us route templates, concurrency, and macro-heavy code to different experts without paying generalist-scale inference per token. Long C++ context rewards architecture, not size: content-dependent sparse attention tuned to TPU v6e MXU tiles is paying near-linear cost on exactly the repo-level inputs our users care about, while a dense-attention generalist at 128K is paying a quadratic cost it cannot amortize.

## What we kept and what we threw away

We kept the four-axis methodology, the compiler-and-Tree-sitter ground truth on axes (1)-(3), the T4-based judge harness, the 20-variant ablation discipline, and the practice of publishing per-variant distributions rather than headline numbers.

We threw away single-number reporting (it hides the failure-mode signal we actually care about), `pass@1`-only scoring (rewards lucky compilation), exclusively LLM-judged adherence (the deterministic oracle is cheaper and stricter), and any "benchmark" produced by a training run that does not pass that admission floor. Throughput numbers from superseded configurations stay in the changelog but never on a slide.

For every eval cycle we publish the exact checkpoint hash, the training config, the production lane it was trained under (BF16 training lane, FP8 training lane, smoke variant, or single-GPU correctness lane), the harness commit, the judge model and prompt, and the full per-axis distribution across seeds. The ensemble claim only counts if the measurement counts, and the measurement only counts if both the training stack and the harness are honest end-to-end.

## References

- architecture_and_eval_en.md
- prior training review notes
- prior accelerator review notes
- speed_rep_xx.md
- the public reproducibility notes
- reproducible_runs.md
- eval_doc.md
- checkpoint_eval_report.md
- TRAINING_EVAL_REPORT.md
- CHANGELOG.md
