---
title: "Determinism and bit-exact runs: what we guard and where we accept drift"
description: "A grounded account of GPU and TPU determinism on our stack: the fast path we run in production, the bitwise path we keep for regression testing, and the tests that fire when silent nondeterminism creeps in."
date: "2026-04-18"
tags: ["determinism", "reproducibility", "training-infra", "testing"]
---

Full numerical determinism on a modern accelerator is a consultant's promise,
not an engineering target. We do not claim it. What we do claim, and what we
have tests for, is a narrower and more useful property: for any code path we
consider stable, turning a new feature flag off gets us byte-identical outputs
to the version of the code before that feature existed. We call that
"bit-exact default path" and it is the contract that lets us ship changes at
the pace we do without losing the ability to bisect a training loss
regression.

This post is about where we land on determinism in practice: what is bitwise,
what is "deterministic up to reduction order", what we have just accepted as
non-deterministic, and which tests catch the drift.

## Three Tiers of "The Same"

We separate three notions that people regularly conflate.

1. Bit-exact — the two runs produce the exact same tensors, byte for byte.
2. Numerically deterministic — the same run, run again, with the same inputs
   and the same seed, produces the same output. Two different runs are not
   promised to match.
3. Statistically equivalent — loss and eval metrics land within a small
   tolerance; specific tensors do not match.

Tier 1 is what we guard for the default code path and for anything downstream
of a checkpoint `torch.load`. Tier 2 is what we guard for kernel-level
changes that are "the same algorithm, different implementation". Tier 3 is
what we settle for on GPU training end-to-end, because nothing else survives
contact with cuDNN, NCCL all-reduce reduction order, and Flash Attention
backward.

## The Fast Path Versus the Bitwise Path

The simplest statement of our stance is in `common.py`, in the
shared runtime bootstrap:

```python
torch.manual_seed(42)
if device_type == "CUDA":
    torch.cuda.manual_seed(42)
# skipping full reproducibility for now, possibly investigate slowdown later
# torch.use_deterministic_algorithms(True)
```

That commented-out line is the honest summary. We seed everything that has a
global RNG, we do not call `torch.use_deterministic_algorithms(True)` for
production training runs, and we do not force `CUBLAS_WORKSPACE_CONFIG`. We
tried. The slowdown on the FA3 / FA4 backward path was not tolerable at our
scale, and the runs that require bitwise behavior are small, targeted, and
already run in a separate configuration.

The production fast path therefore:

- Uses `torch.set_float32_matmul_precision("high")` on CUDA, which lets
  cuBLAS pick TF32 tensor-core algorithms whose reduction order can differ
  across launches.
- Runs NCCL all-reduces in whatever order the collective scheduler picks,
  which for bf16 averages is not commutative to the last bit.
- Uses Flash Attention backward kernels that internally accumulate in
  fp32 but do so with block orderings that depend on occupancy and
  stream scheduling.
- Accepts that two identical-seed runs on the same GPU will match loss to
  roughly five or six decimal places for a few hundred steps and diverge
  from there.

The bitwise path is a different configuration we keep paved for regression
tests and for the cases where drift would hide a real bug. It is not a mode
a user flips at runtime; it is a specific set of feature flags set to their
pre-feature values plus a few environment knobs. In practice this means a
known baseline configuration whose outputs can be compared byte-for-byte when a
new feature needs isolation.

## The Default-Path Invariant

The rule we enforce on every merge: with all new flags off, the training
forward pass is byte-identical to the pre-change baseline.

Concretely, for the Nemotron Nano 3 iteration:

- `reset_position_ids_at_doc_boundary=False` degrades the new position-ID
  reset logic to a no-op, so the tensor passed into attention is the same
  tensor the old code passed in.
- The per-block FP8 autocast context factory defaults to `None`. When it
  is `None`, `_fp8_factory(i)` degrades to `contextlib.nullcontext()`,
  which is a zero-overhead pass-through. The forward loop then compiles
  to the exact same bytecode as the pre-FP8-scope version.
- The MoE TE permute fast path is gated on `MEGACPP_MOE_TE_PERMUTE=1`.
  Unset, the code flow enters the `index_select` branch it always used,
  and the argsort-inverse unpermute path on the combine side is bit-exact
  to the pre-change combine.
- `grad_reduce_in_fp32` is opt-in. With the flag off, the reducer keeps
  its bf16 flat grad buffers and its bf16 `reduce_scatter_tensor`.

This pattern is deliberate and it is tested. The validation sweep we run on
a default-path PR is:

1. Torch save a reference forward-pass output tensor from `main`.
2. Apply the PR, run the same forward pass with all new flags off.
3. `torch.equal` on the two. Not `allclose`. Equal.

This is slow to run in full, so we run it against a pinned tiny model
(`depth=4`, `n_embd=128`, vocab trimmed) on CPU in CI. The model is small
enough that the full forward takes milliseconds; the guarantee is the
same. Our `test_mhc_group_fp8_ctx` family of tests is one example: seven
tests that pin the "no factory installed means nullcontext, means bit-exact
fallthrough" property across the three autograd call-site branches we care
about.

## What We Can and Cannot Make Deterministic

On GPU, the honest breakdown is roughly:

- FP32 matmul forward in non-TF32 mode: deterministic within a run, and
  bit-exact across identical runs if you pin CUBLAS workspace and disable
  the TF32 path. In production we do neither, because the cost is real.
- FP32 matmul backward: same as forward in principle, but cuBLAS can pick
  a different algorithm for the grad input than for the grad weight path,
  and the two paths can share a workspace. Deterministic within a run.
- BF16 matmul: deterministic within a run; differences across runs come
  mostly from reduction order, which is bounded but nonzero.
- Flash Attention forward: deterministic by construction for the softmax
  pass. The numerics are the same kernel every time; small block size
  and num_warps changes (Triton autotune) can change the result because
  reduction order within a block changes. We pin autotune choices for any
  test that wants bit-exact behavior.
- Flash Attention backward: not deterministic in the general case without
  the explicit "deterministic backward" variant. FA4 CuTe we use in
  production does not offer a deterministic backward at the block sizes
  we want. We accept it.
- cuDNN dSwiGLU `atomicAdd` backward: we hit this one head-on; it is the
  classic atomic-accumulate non-determinism. The workaround and the
  vendor escalation are documented in a associated validation notes.
- NCCL collectives: tree reductions with ring order fixed at init are
  deterministic for a given topology. The moment the topology changes —
  different node counts, different NIC ordering — the order changes and
  bf16 bits drift.
- Dropout, any stochastic sampling: deterministic iff seeded, and iff the
  seed sees the same RNG consumption history. Our data pipeline consumes
  RNG in a fixed order per rank.

On TPU / XLA:

- XLA compiled programs are deterministic for a given HLO. Different HLO
  from a recompile means different numerics; see our graph-recompilation
  experience for why this matters.
- SPMD collective ordering is pinned once the mesh is built, so the bf16
  reduction-order issue is better than on NCCL.
- `torch_xla`'s `SPMDSavePlanner` / `SPMDLoadPlanner` are deterministic;
  the checkpoint round-trip is bit-exact, which is why the resume
  regression test uses `torch.equal` and not `torch.allclose`.
- Mamba SSM scans have an `xla_scan` code path for XLA determinism; we
  use it when we care about bitwise reproduction of a Mamba layer on
  TPU, and the more general Triton scan in other cases.

The routing-statistics determinism one deserves its own sentence:
gradient checkpointing around an MoE block can route tokens to different
experts on the recompute pass than it did on the forward pass, because
routing reads auxiliary state. Our code comments this explicitly
(`GPT` block guidance, "EBlock: MoE dispatch is deterministic with
static shapes") and keeps recompute off the MoE dispatch by default;
turning it on is a conscious decision to accept non-deterministic
routing decisions for the sake of memory.

## Tests That Guard Against Silent Non-Determinism

The interesting tests are not the "this function is deterministic" unit
tests. The interesting tests are the ones that pin an invariant we
otherwise would have lost.

- `test_perturbation_determinism` and `test_perturbation_restore_exact`
  in the RandOpt suite: the first asserts that the same `(seed, sigma)`
  pair produces the same perturbation tensor; the second asserts that
  `perturb` then `restore` yields bitwise-identical weights to the
  original, including for frozen LoRA params. These two tests gate every
  RandOpt change.
- `test_mhc_group_fp8_ctx`: seven tests that prove each branch of the
  per-block FP8 context path is `nullcontext`-equivalent when the factory
  is unset. This is the specific shape of "default path bit-exact"
  enforcement we described above.
- `test_checkpoint_manager::test_resume_weights_exact_match`: uses
  `torch.equal`, not `allclose`, on resume. Anything that quietly changes
  `.pt` serialization numerics fails here.
- `test_doc_relative_sinks`, `test_exact_token_dsa_packed_doc_isolation`,
  and the Mamba document-boundary isolation tests: all of them compare a
  packed-doc batch to the equivalent single-doc batch and require exact
  equality on the payload rows. Any cross-document numerical leak shows
  up immediately.
- `fail_closed_decode` invariants on H200 decode runs: the validation sweep
  explicitly verifies determinism under decode (same inputs to the same
  model produce the same tokens), KV cache consistency, and finite
  logits. This is Tier 2 determinism and we run it on every serving
  candidate.
- FA3 backward parity tests, with an explicit disclaimer in the report:
  identical seed plus deterministic data ordering are the preconditions,
  and "numerical bitwise identity" is flagged as NOT expected for bf16
  kernels because their rounding differs. This is exactly the Tier 3
  line we live on for attention backward.

The pattern we try to enforce on every new determinism-adjacent test is:
pick the tier you are asserting, put it in the name of the test, and use
`torch.equal` for Tier 1 / 2 and a tight bounded tolerance for Tier 3.
`allclose` with default tolerances hides real bugs.

## Things That Bit Us

A short gallery of silent non-determinism that slipped through before we
added the corresponding test.

- FP8 autocast factory that returned a decorated generator context
  manager instead of a plain `nullcontext`. The isinstance check in the
  group helper matched on the wrong branch, so the "default path"
  occasionally took the FP8 path, which produced different numerics than
  the baseline. Caught by pinning "factory unset returns nullcontext
  exactly".
- Triton fused RoPE kernel that assumed `(1, T, 1, hd)` cos/sin but
  silently tolerated `(B, T, 1, hd)` by reading the wrong stride. It
  was deterministic within a run, which is why it did not trip
  determinism tests; it was deterministically wrong. Defense added in
  the form of a `cos.shape[0] == 1` guard plus a fallback to the
  safe `apply_rotary_emb`, plus a shape contract test.
- Dataloader `set_to_none=True` causing different gradient-creation
  graphs on rank 0 vs the others, which caused XLA to recompile to a
  different HLO, which changed reduction order and therefore bit-exact
  behavior on resumed runs. We now pin grad accumulation to a single
  `torch_xla.compile()` call around all microsteps and log the number
  of compilations per step as a determinism canary.
- MoE capacity factor with dynamic shapes: tokens-per-expert varied
  slightly across steps, which broke Tier 2 determinism. We moved to
  Expert Choice with a fixed capacity so per-step shapes are constant;
  the comment chain in the main MoE runtime module around the deterministic
  padded grouped GEMM documents the fix.

## What We Gave Up, Knowingly

- Cross-node bit-exactness on bf16 end-to-end training. Reduction orders
  change with topology; we do not try to force them.
- Cross-run bit-exactness on the default FA4 backward path. We use FA4
  because it is fast; we take the bf16 rounding differences.
- Bit-exact MoE routing under gradient checkpointing. Memory wins.
- Bit-exactness of long-horizon eval generation across GPU generations.
  Our eval uses greedy decoding (temperature 0) on purpose — that makes
  eval Tier 2 deterministic per device, which is enough to compare
  checkpoints of the same architecture. It does not make a T4 match an
  H200 bit for bit; we do not need it to.

## The Rule of Thumb

The stance that has held up for us is: bit-exact the default path,
numerically deterministic within a run for everything we ship, and
explicit about every place we accept less. The test suite exists to
make that stance cheap to enforce; the bitwise path exists so that when
a loss regression lands, we can reach for a known-good baseline and
bisect against it instead of arguing about whether a 1e-4 delta is
"real". Everything else — full cross-run bitwise determinism, fully
deterministic cuDNN on GPU — we have looked at, priced out, and decided
the cost is not worth it for our workload. That is a trade-off, not a
virtue.

## What we promise vs what we do not

| Property | Status | Backed by |
|---|---|---|
| same code path, same seed, same hardware family -> same first-N-step loss | promised | public distributed-CUDA sample tests |
| bitwise weight equality across runs | not promised | numerical drift in BF16 reductions |
| deterministic cuDNN | off in production | costs more than it pays |
| MoE token order under EP | deterministic per rank | dispatcher contract |
| randopt perturbation reproducibility | promised under explicit seed | public randopt sample tests |
| FA4 backward parity vs reference | guarded | a dedicated FA4 backward parity validation |

## References

- [PyTorch randomness notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [cuDNN reproducibility and determinism](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html#reproducibility-determinism)
- [Distributed OOM Triage Sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/distributed/oom_triage_sample.py)
- [Local shard contract sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/fire/fire-plasticity-tests__local_shard_contracts__v1.py)
- [Gradient span contract sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/stp/stp-loss-tests__gradient_span_contracts__v1.py)
