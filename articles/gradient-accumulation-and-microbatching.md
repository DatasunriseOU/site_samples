---
title: "Gradient Accumulation and Microbatching Under FSDP2: How We Stopped Guessing the Knobs"
description: "Microbatch sizing under FSDP2, accumulation boundaries that respect TP/EP/SP, loss scaling under FP16/BF16, and the tuning loop that finally converged on H200."
date: "2026-04-18"
tags: ["fsdp2", "gradient-accumulation", "microbatching", "training", "h200"]
---

# Gradient Accumulation and Microbatching Under FSDP2

Most of the time, "tune the microbatch" sounds like a one-line job: pick the
largest `device_batch_size` that fits in HBM, divide your global token budget
by it, and call the quotient `grad_accum_steps`. On a clean dense model that
is roughly true. In a hybrid training stack, where FSDP2 sits next to
TP, EP, SP, MoE dispatch, fused attention, and mixed optimizer state, the
same one-line job can eat weeks. This post is the honest version of how
we sized microbatches, where the accumulation boundary actually lives, what
loss scaling looks like in a world that is mostly BF16 with FP16 and FP8
islands, and the inner loop we now use to converge on a config without
brute-forcing the search space.

## What Sets the Microbatch in Practice

The naive formula is `tokens_per_step = device_batch_size * max_seq_len * dp`,
and `grad_accum_steps = total_batch_size // tokens_per_step`. A useful planner
starts there and then layers on the constraints that actually decide whether a configuration survives:

- HBM headroom after FSDP2 sharded params, AllReduceState FP32 buffers, and
  optimizer state. With `TORCH_NCCL_AVOID_RECORD_STREAMS=1` and a periodic
  `gc.collect()` we hold ~100 steps stable; without it we OOM around step 60.
- MoE dispatch capacity. The dispatch buffer scales with effective tokens per
  step, not per microbatch, because XLA can fuse across grad-accum micro-steps
  into a single graph. The estimator carries a separate `BT = total_batch_size
  / seq_len` term for this case; we learned it the hard way when a depth-52
  64-expert MoE run crashed at `C_e = 122880` with `BT = 1M`.
- Comm shapes. `device_batch_size` controls the size of every reduce-scatter
  and all-gather inside the FSDP2 group, the all-to-all volumes inside EP, and
  the SP halo. Picking `dbs=4` over `dbs=2` is not just a memory choice; it
  changes the cuBLAS tile and the NCCL message size, both of which can pull
  throughput up or down by double digits.
- Compile graph identity. Under regional compile, every distinct
  microbatch shape is a new graph. We treat `dbs` and `seq_len` as part of
  the compile cache key and refuse to mix sizes inside one stage.

The auto-fit ranker scores candidates roughly in this order: fits at all,
then larger `dbs`, then `gradient_checkpointing` off, then fewer
`grad_accum_steps`, then more headroom, then smaller TP, then smaller EP.
The order matters: we explicitly prefer `dbs=2 / ga=8` to `dbs=1 / ga=16`
even when memory is identical, because the comm and kernel-launch overhead
per micro-step is non-trivial on H200.

## Where the Accumulation Boundary Actually Lives

Once you have FSDP2 and a real TP/EP/SP topology, "accumulate gradients for
N microbatches, then step" is no longer a single line of training code. The
boundary has at least four moving pieces.

First, FSDP2's delayed-gradient-sync switch has to be set on every
microbatch except the last, or the reduce-scatter fires N times and you pay
N times the comm. We treat this as a hard correctness invariant; missing it
is silent (the math is fine) but throughput collapses.

Second, the optimizer step has to sit outside the accumulation loop, after
the final reduce-scatter has produced sharded gradients. Under our Megatron
optimizer wrapper this is enforced by the wrapper itself, but the loss
scaling is divided by `grad_accum_steps` inside the loop. For pipeline
parallel, the divisor is `grad_accum_steps * num_chunks` instead, which is
a bug we hit and fixed in the pipeline integration layer.

Third, MoE auxiliary losses live on a different scaler. We follow Megatron's
`MoEAuxLossAutoScaler.set_loss_scale` pattern: the LM loss is divided by
`grad_accum_steps`, the auxiliary load-balancing loss is multiplied through
the same factor, and the two are summed before backward. Forgetting the
aux scaler turns a `1e-3` aux coefficient into something effectively `N`
times that, which routes experts to mush.

Fourth, the data loader has to advance one microbatch at a time, not one
training step at a time. Our pipeline-parallel path also has to advance the
chunk schedule independently. Conflating these has been the cause of more
than one "loss is fine but it does not learn" puzzle.

## Loss Scaling: BF16, FP16, and the FP8 Islands

We run BF16 by default and have for the entire H200 program. BF16 has the
exponent range of FP32, so we do not run a `GradScaler` for the BF16 path:
the optimizer wrapper accepts `grad_scaler: Optional[torch.amp.GradScaler]
= None` and just skips the scale/unscale dance.

FP16 is a different story. When we exercise the FP16 lane (some of the
precision-bridge experiments and a few legacy runs), we use a dynamic
`GradScaler` with the standard powers-of-two backoff and a 2000-step
growth interval. The interesting interaction is with `skip_nan_steps`:
the wrapper detects non-finite gradients before sanitization and, when the
flag is on, skips the optimizer step entirely (Megatron's pattern). With
FP16 + `GradScaler`, the scaler also halves the loss scale on the same
event. We make sure the two cooperate: the wrapper records `step_skipped =
True`, the scaler updates, and the LR scheduler does not advance for the
skipped step.

The FP8 islands (TE-wrapped linears under `fp8_autocast`, COAT FP8 AdamW
for the master copies) need their own discipline. FP8 master copies live in
FP32; the model still produces BF16 activations and gradients; the FP8
scales are owned by TE and updated each forward. None of this changes the
gradient-accumulation math, but it does change what "loss scaling" means at
the kernel boundary, and we keep a hard invariant that loss scale never
enters an FP8 GEMM directly: we apply it after the cast back to BF16/FP32.

## Pre-Reduce NaN Sanitization

There is one more piece that touches both accumulation and loss scaling: the
pre-collective NaN/Inf sanitization. In the optimizer layer we
`nan_to_num` the flattened gradient buffer before the reduce-scatter when
the pre-reduce NaN check remains enabled. The reasoning is concrete:
a single rank producing a NaN (MoE routing overflow, an FP16 underflow in
an experimental kernel, an Mamba conv edge case) will, without sanitization,
poison every other rank's gradient through the average. The post-collective
`nan_to_num` cannot recover from that.

The cost is real: an extra full-tensor pass per accumulation boundary. On
H200 it is in the noise; on smaller links it is not. We expose the env var
specifically so the cheap path is reachable when the operator knows the
stack is clean.

## The Tuning Loop That Actually Converged

After a year of ad-hoc spreadsheets we standardized on a five-step inner
loop. It is boring on purpose.

1. Run the planner. It proposes a ranked list of `(tp, ep, dp,
   dbs, ga, gradient_checkpointing)` tuples, scored by the rules above. We
   take the top three.
2. Smoke each candidate for four steps with `--save_every=0
   --core_metric_every=0 --sample_every=0 --eval_every=0
   --warmup_ratio=0.0`. This is the same shape we use for short H200 smoke
   runs; it catches OOM, NaN, compile-cache misses, and the
   "first step is 10x slower than steady state" tax in under two minutes.
3. If a candidate fails OOM, the retry policy now prefers the next ranked
   candidate that preserves the current `gradient_checkpointing` state
   before it considers dropping remat entirely. We added that after a wave
   where the retry ladder silently swapped `dbs=8 + ckpt=on` for
   `dbs=4 + ckpt=off`, which broke a compile-sensitive benchmark configuration that
   depended on a specific activation-memory budget.
4. Read the steady-state median of steps 1..3, not step 0. The first step
   pays compile, NCCL warm-up, and FP8 calibration tax. We have seen the
   same topology look 20% slower at step 0 and tie at step 3.
5. Validate `gnorm` against an EMA. Our trainer maintains `_gnorm_ema`
   with `alpha = 0.01`. Anything > 10x is a `[GRAD SPIKE WARNING]`,
   anything > 100x is a `[GRAD SPIKE CRITICAL]` and worth pausing the
   ladder. Some configurations are throughput winners but never settle:
   one of our `TP=1, EP=2, dbs=2` lanes held a `gnorm 1160 -> 3648 -> 7200`
   pattern across the early steps; same topology with `TP=2, EP=2` went
   straight to `clip_grad_norm_ encountered non-finite total norm: inf`
   on rank 3 and we removed it from the ladder.

The loop is short but it has rules. We do not change two knobs at once.
We do not promote a config to a long run on the strength of step 0
throughput. We do not accept a config that the planner did not score,
because the planner's score encodes our retry behavior.

## Batch Warmup

One subtlety we did not initially expect: a "batch warmup" schedule
sometimes outperforms a fixed accumulation count for the first few
thousand steps. The trainer exposes `batch_warmup_steps`; when set, the
grad-accum count ramps from `max(1, max_grad_accum_steps // 8)` to
`max_grad_accum_steps` over the warmup window. The math is in
`_resolve_grad_accum_schedule` and `_scheduled_grad_accum_steps`. The
motivation is twofold: smaller effective batches early reduce the
per-step memory pressure during the first compile/cache window, and they
let LR warmup interact with a real signal-to-noise ratio rather than an
over-smoothed gradient.

There is one important constraint: on XLA we force `batch_warmup_steps =
0`, because varying the per-step micro-step count would invalidate the
compiled graph at every change. CUDA is permissive; XLA is not. The
`_effective_batch_warmup_steps` helper enforces this.

## What We Threw Away

- Manual `dbs` selection by humans. The planner is better at it, and the
  retry ladder is better at recovering when it is wrong.
- `dbs=1` as a default. It is a fallback for OOM, not a goal. The kernel
  launch overhead and the NCCL message-size penalty are large enough that
  we now prefer `dbs=2 + ckpt=on` over `dbs=1 + ckpt=off` even at equal
  memory.
- Treating `grad_accum_steps` as independent of `gradient_checkpointing`.
  The two co-decide what fits; we score the pair.
- A separate FP16 default path. We kept FP16 reachable for experiments and
  for the TE bridge, but BF16 is the production default and the
  `GradScaler` path is no longer on the hot loop.

## What Still Hurts

The biggest remaining sharp edge is that `grad_accum_steps` and pipeline
parallel `num_chunks` co-divide the loss, and the divisor depends on which
stage and which microbatch you are in. We have a fix and tests, but a
future refactor that swaps in a different PP scheduler will need to revisit
the scaling at every accumulation boundary. We also still rely on a
stack-wide environment flag for the
sanitization toggle, where it ought to be a per-run flag.

The shorter version of the whole post: microbatching is a planning problem,
not a guessing problem; the accumulation boundary is a contract that
touches FSDP2, the optimizer, MoE aux losses, and the data loader; loss
scaling is mostly trivial under BF16 and only interesting at the FP16/FP8
seams; and the inner loop that converges is the one that fails fast and
changes one knob at a time.

## Inner-loop knobs at a glance

| Knob | Owner | Default | When to change |
|---|---|---|---|
| `dbs` (device microbatch) | `auto_fit` planner | planner top-3 | only via planner; manual selection is the anti-pattern |
| `grad_accum_steps` | trainer + PP scheduler | derived from target tokens/step | co-tuned with `gradient_checkpointing` |
| `gradient_checkpointing` | trainer | on for deep/large | retry ladder preserves it before dropping `dbs` |
| `batch_warmup_steps` | trainer | 0 on XLA, on on CUDA | early-step compile pressure |
| pre-reduce NaN-check toggle | optimizer env | unset (sanitize on) | only when stack is provably clean |

Smoke-test shape we use for every planner candidate:

```bash
python -m <training-entrypoint> --config <candidate-config> --dbs 4 --grad_accum_steps 8 --save_every 0 --core_metric_every 0 --sample_every 0 --eval_every 0 --warmup_ratio 0.0 --max_steps 4
```

## References

- the planner and capacity-estimation components
- the distributed optimizer and pipeline integration components
- the main training entrypoint
- the training configuration layer
- regression tests for training entrypoints and configurations
- public change notes and validation summaries
