---
title: "Distributed Optimizer Stress: Drift, All-Gather vs Reduce-Scatter, and Muon Gotchas"
date: "2026-04-18"
author: MegaCpp Engineering
tags: [optimizer, muon, adamw, distributed, numerical-stability, tpu, H200]
summary: >
  What we learned stress-testing DistAdamW and DistMuon on the MegaCpp trainer:
  numerical drift at grad-none asymmetry, the all-gather vs reduce-scatter
  trade for sharded updates, and the Muon-specific traps that have nothing to
  do with the parameter math.
---

# Distributed Optimizer Stress: Drift, All-Gather vs Reduce-Scatter, and Muon Gotchas

Optimizer bugs in a distributed trainer are the worst class of bug to
own. They do not fail fast. They do not raise. They compound silently
over thousands of steps and then surface as "the loss curve looks
weird around step 3000" on a long run. For the MegaCpp trainer
we run a hybrid `DistAdamW` + `DistMuon` optimizer across CUDA and
XLA backends, and we spent real time building a stress harness whose
entire job is to catch drift before it compounds. This post is about
what that harness does, what it found, and the Muon-specific
failure modes that have nothing to do with orthogonalization math.

## Why a dedicated stress harness

The unit tests pass. The integration tests pass. The training runs
also look fine for a while. So what does the stress harness do that
those do not?

It exercises the one failure mode that regular training never
reliably produces: **rank-asymmetric grad presence**. In real
training, every rank computes a forward and backward for every
parameter on every step, so every `p.grad` is non-None on every rank
simultaneously. The distributed optimizer code path that handles
`{rank 0: grad, rank 1: None}` almost never runs in practice. But it
exists. It has to exist, because conditional compute (MoD, ReDo,
MTP toggles, disabled experts, inactive adapters) lets a rank
legitimately have `grad=None` for a parameter while another rank
has a real gradient.

The stress harness runs the distributed optimizer stress harness
under a 1000-step schedule with `--sample-every=100`, explicitly
cycling through:

- `{0: 1.0, 1: None}`: rank 0 has grad, rank 1 does not
- `{0: None, 1: -0.5}`: the opposite
- `{0: 0.25, 1: 0.75}`: both have grads
- `{0: None, 1: None}`: both missing

Across both scenarios — the pure `DistAdamW + DistMuon` scenario and
the `megatron_conditional` scenario that toggles MoD/ReDo/MTP with
rank-asymmetric patterns — the latest TPU v6e-8 CPU/gloo run reports
`max_adam_abs_diff = 0.0`, `max_mu_a_abs_diff = 0.0`,
`max_mu_b_abs_diff = 0.0`, and `max_abs_param_diff = 0.0` at every
sampled checkpoint. That is not "within tolerance". That is
bit-identical parameter values on both ranks after 1000 steps.

Tolerances still exist: the harness thresholds are `<= 2e-5` for
Adam and `<= 3e-4` for Muon. We keep them because future numerics
changes (fp32 fallbacks, new precision policies) can legitimately
drop us into "within tolerance but not zero". The current state is
zero, and the day it stops being zero we want to find out on a
stress run, not on step 3000 of a real training.

## The all-gather vs reduce-scatter decision

`DistAdamW` is a ZeRO-2-style sharded optimizer: each rank owns a
slice of the optimizer state (`exp_avg`, `exp_avg_sq`) for each
large parameter, reduces grads into its own slice, updates the
slice, and all-gathers the updated parameter so every rank sees the
full tensor. For a parameter `p` with first dim divisible by world
size, the pattern is:

1. Gather grads onto the owner rank via `reduce_scatter_tensor` on
   a flattened view, producing one `grad_slice` per rank.
2. Each rank runs Adam locally on its slice, updating its shard of
   `exp_avg` / `exp_avg_sq` and the param shard itself.
3. `all_gather_into_tensor` rebuilds the full param into a shared
   buffer, which is sliced back into `p`.

The alternative, `all_reduce` on the full gradient followed by a
local update on the replicated optimizer state, is what
`DistAdamW` falls back to for small params or params whose first
dim does not divide the world size. The shape check is explicit:
`shape[0] % world_size != 0` forces the `all_reduce` path to avoid
a `reduce_scatter` crash on indivisible shapes.

Why keep both patterns? Because `reduce_scatter + all_gather` only
wins when the parameter is large enough for the bandwidth savings
to dominate kernel-launch and state-management overhead. On our
geometry the crossover is around 1024 elements. Below that, the
per-param kernel launches and the bookkeeping for a sharded
optimizer state cost more than the full-tensor all-reduce. The
`is_small` list in the optimizer step code is explicit about which
param took which path; we trace it on step 0 and confirm on
every regression that the split did not drift.

On DTensor-like params, both paths are short-circuited: the DTensor
the runtime handles the collective internally, so `DistAdamW` skips
its own and consumes the already-reduced local shard. This was not
obvious when we first integrated DTensor TP — our early
implementation double-reduced gradients for DTensor params and
produced exactly the kind of "loss drifts at step 3000" symptom
the stress harness now catches. The current code gates on
`_is_dtensor_like_param(p)` and takes the local shard directly.

## Grad-none symmetry: the quiet correctness property

The single hardest correctness property to preserve is that
collectives run in identical order on all ranks, even when local
grad presence differs. If rank 0 skips a `reduce_scatter` because
its local `grad is None` while rank 1 runs it, NCCL either hangs
or — worse — silently consumes the next unrelated collective and
produces garbage.

The fix in both `DistAdamW` and `DistMuon` is the same shape: build
a rank-symmetric "has grad" mask with a leading `all_reduce(MAX)`
at the top of the step, and then always run the same collectives
on every rank, substituting a zero-filled tensor for missing
local grads. This is slightly more work than strictly necessary on
the "everyone has a grad" case, but the cost is a single
`all_reduce` of an `int32` tensor the size of the param list. We
measured it and stopped worrying about it.

The stress harness's reason for existing is to verify exactly this:
with the mask in place, 1000 steps of rank-asymmetric grad presence
produce zero parameter divergence. Without the mask, the same
harness used to deadlock in ~50 steps.

## Muon-specific gotchas

Muon is a fundamentally different optimizer from Adam. It runs
SGD-momentum followed by a Newton-Schulz iteration to orthogonalise
the update matrix before applying it. The math is well-behaved on
paper. The distributed implementation has sharp edges that do not
show up in any of the Adam literature.

### 1. Two-dimensional only

`DistMuon` asserts `all(p.ndim == 2 for p in params)` at construction
time. This is not a performance optimization — it is a correctness
requirement. Orthogonalising a rank-3 tensor as if it were a matrix
silently mixes subspaces that should stay independent. The most
common offender in our stack was fused QKV. A tensor of shape
`[3*d, d]` is 2D, but orthogonalising it as a single matrix mixes
the query, key, and value subspaces through Newton-Schulz in a way
that regresses depth-52 runs by a visible margin on the loss curve.

The fix is explicit `qkv_split_sizes` metadata on the param group.
Muon respects the split and orthogonalises each sub-matrix
independently. Losing that metadata during a refactor was the most
recent regression we caught; it showed up immediately on the
depth-52 Muon sanity run, not on the stress harness, because
grad-none symmetry has nothing to do with the orthogonalization
path. That is itself an argument for keeping both kinds of
regression guard in place.

### 2. The cautious-update gate

Muon's fused step is `momentum -> polar_express -> variance_reduction
-> cautious_update`. The cautious update masks out components of
the orthogonalised update whose sign disagrees with the raw
gradient's sign. Earlier in the rollout we reintroduced a raw-grad
gate that looked equivalent but used the pre-momentum gradient for
the sign check. On depth-52 runs this regressed Muon back to
pre-cautious-update behaviour. The current code keeps the gate on
the post-variance-reduction update, not on the raw grad, and the
regression receipt is preserved as a test case.

### 3. The reduce-scatter + all-gather dance

`DistMuon` uses the same two-pass collective pattern as DistAdamW
but on **stacked** parameter groups. All params of the same shape
within a group are stacked along a new leading axis, and the
optimizer runs one fused kernel on the whole stack. The benefit
is a single Newton-Schulz launch per group instead of one per
param, which dominated step time on deep models.

The hazard is that the chunk math must be identical on every rank.
`chunk_size = (len(group_params) + world_size - 1) // world_size`
is computed at init time and burned in. Adding a parameter to a
group after init (for example, when an adapter gets enabled
mid-run) requires `add_param_group` to recompute `chunk_size`.
We had exactly one bug where the chunk size was stale and rank 1
operated on a different slice of the stacked tensor than rank 0.
It produced a single non-zero value in the stress harness at
step 100 and we caught it before it hit real training.

### 4. XLA runtime scalars

The `prepare_xla_step_scalars` method on both DistMuon and DistAdamW
exists because naive float Python scalars force XLA recompilation
every time the LR schedule advances. We cache materialised 0-D
tensors for `lr`, `momentum`, `weight_decay`, `beta2`, and
`lr_multiplier` on the XLA device, and rewrite them in place
between steps. This is not numerically interesting — the values
are the same either way — but it is the difference between a
compile cache hit and a full recompile on every LR-schedule
transition. On TPU v6e-8 that matters.

### 5. FSDP2-native Muon vs DistMuon

MegaCpp actually ships three Muon variants: single-device `Muon`,
ZeRO-2-style `DistMuon`, and a `FSDP2Muon` that consumes
FSDP2-sharded `DTensor` parameters directly. The last one is
mathematically equivalent to `DistMuon`: each rank owns a chunk
of the stacked parameters and orthogonalises its slice. The
reason it exists as a separate class is that FSDP2 hands
parameters to the optimizer as sharded DTensors whose local
shards do not match the naive "split along leading dim" layout
`DistMuon` assumes. The adapter layer is
`_match_grad_to_local_shard`, which maps the DTensor grad onto
the local shard shape `FSDP2Muon` consumes.

Keeping three Muon implementations is tech debt, and we are
aware of it. Collapsing them into one would require either
making `DistMuon` DTensor-native (costly rewrite) or abandoning
the non-FSDP2 path (premature, because TPU XLA SPMD does not
use FSDP2). For now the stress harness runs both `DistMuon` and
`FSDP2Muon` through the same grad-none asymmetry scenarios and
confirms bit-identical behaviour across them.

## What the TPU v6e-8 receipt actually covers

The most recent checked-in receipt is from the CPU/gloo leg on the
v6e-8 host, not the XLA leg. That is a deliberate scope
restriction: the stress harness currently supports `gloo` and
`nccl` backends, not XLA. The CPU/gloo run exercises the exact
same Python optimizer code (`DistAdamW`, `DistMuon`,
`_group_muon_params`, `prepare_xla_step_scalars` paths) on the
same host, just with collectives running over gloo instead of
over XLA.

This matters because the distributed logic and the grad-none mask
are backend-independent. If they pass on gloo, they pass on any
backend that correctly implements the MPI collectives. The XLA
distributed optimizer path is exercised indirectly through real
multi-chip training runs via the base trainer with
`--tensor_parallel`, where divergence would show up as a loss
curve regression.

Extending the stress harness to XLA is a tracked follow-up; the
main blocker is that the harness expects a `torch.distributed`
process group, and the XLA SPMD runtime does not expose one at
the same API level. We have not yet decided whether the right
fix is to wrap XLA's collective primitives or to run the stress
harness against a synthetic `gloo` surrogate that replays XLA
SPMD grad layouts.

## The cheap lesson

The single cheapest thing we did on the distributed optimizer
rollout was adding the rank-symmetric `all_reduce(MAX)` mask at
the top of every step. It is five lines of code. It removed the
entire class of "deadlock on conditional compute" failure. The
single most expensive thing we did was chasing a depth-52
regression caused by fused QKV being treated as a flat matrix
rather than three sub-matrices. It took a week.

The pattern is consistent: the bugs that hurt are the ones where
distributed correctness quietly degrades the optimizer math, not
the ones where the collective hangs. Hangs are loud. Drift is
silent. The harness above is the handful of silent-drift sources
we have actually hit.

MegaCpp, by David Gornshtein and Boris Tamarkin,
treats the stress harness as a gate. If `max_adam_abs_diff` or any
of the Muon diffs drifts above zero on the checked-in scenarios,
the build does not ship. That is the kind of contract that is
easy to write down, occasionally annoying to satisfy, and worth
every minute the first time it catches a real regression.

## Failure modes we test for

| Mode | Symptom | Trip wire |
|---|---|---|
| grad-none asymmetry | rank-divergent updates | per-rank grad-norm gate |
| AG vs RS skew | drift in master weights | bitwise compare every 1k steps |
| Muon `nuc_norm` underflow | loss spike on long warmup | clamp + telemetry |
| FP32 master corruption | silent diverge over hours | periodic `is_finite` sweep |

```python
# DistMuon: keep master in fp32, gather updates in bf16, scatter back fp32.
master_fp32 = shard.to(torch.float32)
update_bf16 = muon_step(master_fp32).to(torch.bfloat16)
torch.distributed.all_gather_into_tensor(buf_bf16, update_bf16, group=tp_group)
master_fp32.add_(buf_bf16.to(torch.float32))
```

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md
- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/distributed_optimizer.html
- https://docs.pytorch.org/docs/stable/distributed.optim.html
- https://docs.pytorch.org/xla/master/perf/spmd_advanced.html
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md
