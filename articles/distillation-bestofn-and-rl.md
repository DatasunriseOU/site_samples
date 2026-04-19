---
title: "Distillation, best-of-N, and verifier-grounded RL in the post-training loop"
description: "How distillation, best-of-N, GRPO, GSPO, and verifier-grounded reward shaping compose the MegaCpp post-training pipeline: what we ship, what we still iterate, and the RL recipes behind the C++ specialist."
date: "2026-04-18"
tags: ["distillation", "best-of-n", "grpo", "gspo", "rlvr", "cpp-verifier", "post-training"]
---

The pretrained C++ base is the cheap part. What makes the MegaCpp specialist useful at compiler-grounded code generation is the post-training loop on top: distillation from a stronger teacher, best-of-N with a real C++ compiler in the loop, and verifier-grounded RL (GRPO/GSPO) where the reward comes from compiler exit codes and test pass rates rather than a learned reward model. This post walks through how each component fits into the training stack, how the verifier produces a usable reward, what we ship today vs still iterate on, and the recipes that drive it.

## Why MegaCpp cares about this

Code is the rare domain where reward is grounded, cheap, and unambiguous. A C++ snippet compiles or not; with a test harness it passes or not; with clang-tidy and AddressSanitizer we get more layers of signal. Every one is a real verifier in the [RLVR - Reinforcement Learning with Verifiable Rewards] sense - no learned reward model, no preference data. We can do PPO-family RL on a single GPU with no human in the loop and no reward-hacking against a proxy. Distillation brings teacher behavior into the specialist; best-of-N adds inference-time gains without retraining; GRPO/GSPO drive policy-gradient training off the verifier. All three compose, and we use all three.

## How the post-training loop is structured

The load-bearing piece is a C++ verifier with an interface like `verify_cpp(code, test_code, timeout, std, extra_flags, return_structured, strip_think)`. It writes to a temp file, compiles with `g++ -std=c++17 -Wall -Wextra`, optionally links and executes if the source has an `int main(...)`, and parses `PASSED N/M` markers. It returns a `VerifyResult` with `compile_ok`, `warnings`, `run_ok`, `stdout`, `stderr`, `returncode`, `tests_passed`, `tests_total`, and, when `return_structured=True`, an `error_class` plus a structured `diagnostic`. Helper routines such as `strip_think_blocks()` and `strip_think_token_ids()` remove model reasoning wrappers before compilation so they do not break C++ syntax. Token-ID stripping mirrors string-level stripping for the in-training case.

`compute_reward(result)` is the canonical scalar. The schedule is deliberately simple: `0.0` if compile fails; `0.3` if it compiles but isn't run; `0.3` if it compiles but runtime errors; `0.5` if it compiles and runs cleanly with no tests; `0.3 + 0.7 * (tests_passed / tests_total)` with tests. The `0.3` floor for "compiles" is the structural-prior signal - we want a non-trivial reward for the first hard threshold (parses and type-checks) before tests come into play. The cliff between non-compile and compile trains the structural prior; the test-pass slope trains the semantic prior.

The reward-shaping layer is the multi-signal extension on top of `compute_reward`. `CompilerRewardShaper` adds warning penalties (capped), execution-success and speed signals, AddressSanitizer leak detection, and clang-tidy `modernize-*` / `bugprone-*` checks. Conservative weights: compile 0.3, run 0.2, test 0.3, warning 0.02 each capped at 0.15, speed 0.05. Per-build timeout, `ThreadPoolExecutor` for parallel verification, stdout truncation at 64 KiB, hard test-count cap at 1000 - all lessons from RL runs where adversarial completions tried to drown the verifier in output.

The best-of-N layer is the simplest verifier user. `best_of_n_generate(model, prompt_ids, tokenizer, n, ...)` samples N completions with different seeds, runs each through `verify_cpp`, and returns the best ranked by `(reward desc, code_length desc)`. The length tiebreak prevents trivially compiling stubs from outranking real solutions. It returns `(best_code, metrics)` with per-candidate rewards, the best index, and compile/run counts. Pure inference-time technique, no model changes, and roughly +10-14% pass@1 at N=10-16 per [XCoder - EMNLP 2024].

The GRPO layer implements Group Relative Policy Optimization from [DeepSeekMath / GRPO - arXiv:2402.03300]. Per prompt: generate `group_size` completions with different seeds, verify each, compute group-relative advantages `(reward - mean) / std` with a zero-variance guard, take a clipped-surrogate PPO step plus a KL penalty against the frozen reference. `GRPOConfig`: `group_size=8`, `kl_coeff=0.1`, `clip_eps=0.2`, `max_gen_len=512`, `temperature=1.0`. The reference model is held in eval with no gradients. Reward is `verify_cpp -> compute_reward`. Per-completion log-probs are computed in a single batched forward over the padded batch.

The GSPO layer is the sequence-level upgrade from [GSPO - Qwen3, arXiv:2507.18071]: sequence-level importance ratios instead of per-token, which lowers gradient variance and stabilizes MoE training. Three layers on top: [LUSPO - arXiv:2602.05261] adds length-weighted importance; [DAPO - arXiv:2503.14476] discards zero-variance groups; [StepCoder - ACL 2024] contributes CCCS (prepend a fraction of the known solution to the prompt as easy context, decay the prefix as `compile_rate` improves) and FGO (parse the first compiler error line, mask tokens after it, optimize only the prefix the compiler actually evaluated). The CCCS prefix decays by `cccs_decay_step` whenever group `compile_rate` exceeds `cccs_decay_threshold`, never below `cccs_min_prefix_frac`. FGO uses `_parse_error_line` to extract the error line as a fraction of completion length from `g++`/`clang++` stderr.

The distillation layer centers on `DistillationLoss(config)`: KL between student-soft and teacher-soft logits at temperature T, plus hard cross-entropy with weight `(1 - alpha)`, plus an optional feature-level cosine-distance loss between matched intermediate hidden states (forward hooks on specific block indices). KL is scaled by `T^2` to compensate for temperature-reduced gradients, all logit math runs in float32 for numerical stability (the MaxText pattern), and `extract_hiddens` registers/removes hooks inside a try/finally so we don't leak hook state. Teacher is loaded once via `load_teacher(...)`, frozen, run under `torch.no_grad()`. Two modes: train-time NTP distillation from a larger teacher, and the DCD online-adaptation path where teacher and student are the same model with different context windows.

The DSA training path uses the same two-stage protocol from the DeepSeek-V3.2 report (Sec 3.1) - not post-training but it composes with the stack. Stage 0 is normal. Stage 1 is dense warm-up: freeze everything except the DSA indexer's `q_proj`, `k_proj`, `k_norm`, `w_proj`, flat warm-up LR (default `1e-3`), KL loss between sparse output and a detached dense teacher - teaches the indexer to route to dense-equivalent positions. Stage 2: unfreeze everything but detach the indexer from the LM loss via `_detach_indexer_from_lm_loss` so the routing gate's gradient comes only from the KL term. `DSATrainingStageManager` is the controller; `apply_dsa_stage1_freeze` / `apply_dsa_stage2_unfreeze` are the parameter-set ops.

The eval harness loads function-signature prompts plus reference test harnesses, generates completions, compiles with `g++ -std=c++17`, runs them, and computes pass@k using the Codex unbiased estimator. The same surface drives every adapter, every RL run, and every base checkpoint to produce the `humaneval_cpp` number that ends up in registry metrics.

## How it lands in MegaCpp

Verifier and reward primitives live in the training stack today. Production MegaCpp does not run `g++` per request - the latency budget is wrong by orders of magnitude, and executing arbitrary model output on a serving box is non-trivial. What MegaCpp takes is the artifact contract and the eval contract. RL-trained adapters land via the same `save_adapter_artifact` / `load_adapter_artifact` path supervised adapters take; the registry records `training_method = "gspo" | "grpo" | "sft" | "dcd"`. Pass@k from the eval harness is the gating metric for promoting `draft -> active`.

Best-of-N is a feature flag on inference, not part of the model. Production exposes a `best_of_n` parameter (default 1, capped per tier) that routes through the same multi-sample path with a lighter verifier - typically a small learned classifier as a proxy for compile-ability, reserving `g++` for the offline loop. Compile-grounded BoN is offline, used to construct training data. There is an experimental dev-mode tier that does in-request compile + BoN at N=4 with a 5-second budget; not the default.

GRPO and GSPO are post-training pipelines, not production code. They live in the training stack and produce adapters that get promoted into the MegaCpp registry. The recipe that survived: GSPO + LoRA, NF4 base, rank 8 on `MLP_TARGETS | MAMBA_TARGETS`, LUSPO length weighting on, DAPO zero-variance discard on, StepCoder CCCS at 0.7 initial prefix with decay threshold 0.6, StepCoder FGO on, KL `0.1`, group size 8, temperature 1.0, max gen 512. Reward is `compute_reward(verify_cpp(...))` for fast iteration; `CompilerRewardShaper` is the slower, higher-density alternative for adapters the simple reward cannot discriminate. DSA two-stage is base-model only - one-shot at base training, not per adapter.

Kernel boundary: nothing here is kernel-heavy. Verifier is `subprocess`; reward is arithmetic; GRPO/GSPO losses are standard PyTorch. The only fast path that matters is the generation step inside the policy-gradient loop, the same one inference uses, just with `top_k`/temperature and a larger `max_gen_len`.

## Ablations and what we kept

The reward function got one decisive simplification. The original had separate weights for compiles, links, runs, some-tests, and all-tests; "links" was nearly identical to "runs" and added noise. We collapsed to four tiers: `0.0`, `0.3`, `0.5`, and `0.3 + 0.7 * frac`. Reward variance dropped enough that GRPO advantages became less noisy. The `+0.3` floor for "compiles" survived from the original and is the most important number in the schedule.

GRPO vs GSPO was a clear GSPO win at our shape. Token-level ratios were unstable when policy and reference drifted apart, especially under MoE - a single token's ratio could blow up and dominate. Sequence-level ratios collapse variance into one scalar per completion, which is what RLVR with binary-ish rewards wants. GRPO stays in the tree as the simpler reference; the production recipe is GSPO.

StepCoder CCCS and FGO both survived. Either alone was fine; both together was the cleanest curve. We tried step-based decay (every K steps) and metric-based decay (when `compile_rate >= threshold`); metric-based won because it self-paces against actual learning, not wall-clock.

Best-of-N at N=8 with the compile-grounded reward gave the +10-14% pass@1 the literature predicted. N=16 gave a small further bump but wasn't worth the wall-time. We use N=8 for offline data construction, N=4 for dev-mode in-request, N=1 for standard serving.

Distillation behaved as expected: alpha=0.5 KL+CE mix outperformed pure CE on every eval; temperature=2.0 was robust; the feature-matching loss helped at low budgets and hurt at high budgets (the student's hidden states diverge from the teacher's once it can outperform the teacher on some inputs). Feature loss is off by default, on per-recipe.

DSA two-stage is the standard receipt for any sparse-attention base. Skipping dense warm-up gives a model that won't converge - the indexer never learns dense-equivalent routing and the LM loss sees garbage attention. Flat warm-up LR (`1e-3`) and indexer-only freezing are the two non-negotiable parts of stage 1. The reward-shaper warning penalty is the part we still iterate on - aggressive penalties push the model toward cargo-cult `// NOLINT` and `(void)var` instead of fixing issues, so weights stay mild. AddressSanitizer is useful but slow; we sample 10% of completions for ASAN.

## Production checklist

- The C++ verifier writes to a per-call `tempfile.TemporaryDirectory(prefix="...verify_")` and the subprocess `env` carries that as `TMPDIR`. Do not let temp files leak into the host `/tmp` under load - that has bitten us.
- `strip_think_blocks` and `strip_think_token_ids` must run before any compile. Stripping at the wrong layer (after the compile fails) wastes a build and produces nonsense rewards.
- The reward floor at `0.3` for "compiles" is the structural-prior signal. Do not zero it out for "compiles but does not run" - that collapses the reward to a binary and the policy stops learning the cliff.
- Stdout truncation at 64 KiB and `tests_total` cap at 1000 are anti-adversarial. Do not raise them.
- GRPO/GSPO group size, KL coefficient, and clip-eps are all per-recipe; the production GSPO recipe is `(group_size=8, kl_coeff=0.1, clip_eps=0.2, temperature=1.0)`. Changes require an eval re-run.
- DAPO zero-variance group discard must stay on. Zero-variance groups produce zero gradient contribution but full memory usage; discarding them is free throughput.
- StepCoder CCCS prefix decay is metric-driven via `compile_rate`. The decay threshold and step are per-recipe; do not couple them to wall-clock.
- The reference model in GRPO/GSPO is frozen and held in eval mode; the KL term references it, never the live policy. Do not "save memory" by reusing the policy as the reference.
- DSA stage-1 dense warm-up freezes every parameter except the indexer's `q_proj`, `k_proj`, `k_norm`, `w_proj`. The frozen-set contract is strict; any new sparse-attention code must add itself to the freeze set or explicitly opt out.
- Best-of-N candidate ranking is `(reward desc, code_length desc)`. Length tiebreak prevents trivially-compiling stubs from outranking real solutions.

## Recipe surface

| Component | Module | Reward shape | Notes |
|---|---|---|---|
| Distillation | distillation module | KL to teacher logits | offline, fixed teacher |
| Best-of-N | best-of-N module | verifier pass/fail | sort by `(reward desc, len desc)` |
| GRPO | GRPO module | group-normalized | group size 8, KL 0.1, clip 0.2 |
| GSPO | GSPO module | sequence-level | matches GRPO defaults |
| C++ verifier | C++ verifier module | compile + tests | per-call temp dir, 64 KiB stdout cap |

```python
# Compile-grounded reward floor preserves the "compiles but fails tests" cliff.
def shape_reward(verdict):
    if verdict.compiled and verdict.passed_all:
        return 1.0
    if verdict.compiled:
        return 0.3  # structural prior - do not zero out
    return 0.0
```

## References

- the public verifier, eval, best-of-N, GRPO, GSPO, reward-shaping, distillation, DSA training, and online adaptation samples
- [DeepSeekMath / GRPO - Shao et al., arXiv:2402.03300]
- [GSPO - Qwen3, arXiv:2507.18071]
- [LUSPO - arXiv:2602.05261]
- [DAPO - arXiv:2503.14476]
- [StepCoder - ACL 2024]
- [XCoder - EMNLP 2024]
- [Codex - Chen et al., arXiv:2107.03374] (pass@k unbiased estimator)
- [DeepSeek-V3.2 technical report - arXiv:2512.02556] (DSA two-stage protocol)
- [Distilling the Knowledge in a Neural Network - Hinton et al., 2015]
