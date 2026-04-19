---
title: "Checkpoint Format and Resume: What We Save, and What We Test"
description: "DCP vs per-rank checkpoints, async mirroring to GCS, resume tests, world-size changes on resume, and the corruption classes that need explicit detection."
date: "2026-04-18"
tags: ["checkpoints", "dcp", "training-infra", "resume"]
---

Checkpointing is the part of the training stack that nobody gets to write a
paper about. It only shows up as pain: a rank that went dark at step 37,812 and
came back with the wrong optimizer shards, a DCP directory whose `.metadata`
landed but whose tensor files did not, a rotation job that cheerfully deleted
the only surviving copy of a checkpoint because the background GCS upload had
not finished yet. This post covers the checkpoint manager design, its test
suite, and the remaining failure modes that still deserve explicit caution.

## Two Formats, One Manager

We run two checkpoint formats in the same training jobs:

1. A per-rank, rank-0-gathered format: `model_{step:06d}.pt` (full unsharded
   model on rank 0), `optim_{step:06d}_rank{r}.pt` (per-rank optimizer
   shards), `meta_{step:06d}.json` (config + step + schedule state), and
   optionally `extra_{step:06d}.pt` for scheduler / RNG / dataloader cursors.
2. A torch DCP directory format: `dcp_{step:06d}/` (or the older
   `dist_ckpt_{step:06d}/`) containing `.metadata` plus sharded tensor files,
   with `meta.json` and `extra_state.pt` written next to it or inside it.

The per-rank format is honest and debuggable. Any rank 0 box with `torch.load`
and a Python shell can inspect it, and the CPU-gathered model file is the
ground truth for eval. It pays for that simplicity with an all-gather of the
full model to rank 0 on every save, which at 270M is free and at 4B+ is
painful.

DCP (`torch.distributed.checkpoint`) is the opposite. Each rank writes its own
shard directly to the shared filesystem, so there is no gather. On SPMD/TP
runs that is the difference between a 4 second checkpoint and a 90 second one.
The cost is that DCP state is meaningful only in aggregate: you cannot load a
DCP dir on a single-GPU eval box without a matching world-size and a planner
that knows how to re-shard.

We did not pick one. The manager supports both, parallel, with a shared
`meta_{step:06d}.json` schema so any consumer can tell which payload landed
by probing for `.metadata` in a sibling `dcp_*` or `dist_ckpt_*` directory
first and falling back to `model_*.pt`. The `stage_checkpoint_step()` entry
point takes an optional `require_kind="model"` or `"dcp"` argument because
we had to stop the eval pipeline from silently promoting a DCP dir to a
model file mid-run.

### Why Both

The honest answer is: workload asymmetry.

- Training runs on TPU v6e SPMD or H200 TP/EP shard heavily; DCP is the only
  format that keeps save time off the critical path.
- Eval and inference run on a single T4 or H200 with no distributed
  initialization; they want a plain `model_*.pt` they can `torch.load` with
  `weights_only=True` and move on.
- Emergency preemption ("SIGTERM, you have 30 seconds") wants the fastest
  possible local-SSD write with no coordination, which is a per-rank dump
  into an `emergency_step_{step}_{ts}/` scratch dir.

The same code path produces a DCP dir for training, a per-rank `model_*.pt`
for eval, and an emergency snapshot for preemption. That is what `use_dcp=`
and `save_emergency_checkpoint()` select between. Three formats would have
been simpler; three formats would also have meant three resume code paths.

## Async Mirroring and the Rotation Trap

Local-SSD is fast and ephemeral. GCS is durable and slow. NFS is somewhere in
between. The manager mirrors every successful save to both, asynchronously,
so the training step does not block on a 200 MB upload of optimizer shards.

The first version of this looked fine in isolation and turned out to be
dangerous. The rotation job ran every save and kept the last two steps on
local SSD. On a healthy run that is correct. On a run where the background
GCS upload for step N had not finished by the time step N+2 started its
own rotation, rotation would cheerfully `os.remove()` the only copy of
step N.

Fix: a per-step pending-uploads table (`_pending_gcs_uploads: dict[int,
list[Thread]]`) that the rotation path consults before deleting. If any
upload thread for the old step is still alive, `t.join(timeout=300)`. If
the thread exposes a `gcs_upload_ok is False` attribute (set by the upload
worker on failure), or the join times out, the rotation skips deletion for
that step entirely and logs an error. We would rather run out of local
disk than lose the only copy.

That behavior is pinned by two regression tests that were added after we
almost lost a model file: one simulates a slow upload and checks the rotation
waits, one flips the thread's `gcs_upload_ok` to `False` and checks the local
file survives.

## Atomic Writes Everywhere

Every on-disk artifact — model, optimizer, meta, extra, emergency pointer,
DCP sidecar files — goes through the same two-step dance: write to
`<final>.tmp`, then `os.replace(tmp, final)`. The reason is one specific
failure mode we kept seeing on preemption: a SIGKILL (not SIGTERM) during
`torch.save`, leaving a truncated `model_000200.pt` on disk, which then
`torch.load` would gladly deserialize into garbage on resume.

The contract test for this is blunt:

```python
def crashing_save(data, path, *args, **kwargs):
    if str(path).endswith("model_000200.pt.tmp"):
        with open(path, "wb") as f:
            f.write(b"partial garbage")
        raise OSError("Simulated disk full")
    return original_torch_save(data, path, *args, **kwargs)
```

After the simulated crash, the test asserts `not model_path.exists()`. The
`.tmp` file may or may not exist, depending on where the crash landed, but
the final file must not. A matching test does the same for the DCP path,
crashing mid-write on `extra_state.pt.tmp` and asserting the final
`extra_state.pt` never appears.

`test_no_tmp_files_after_save` sweeps the checkpoint dir after a successful
save and fails on any surviving `.tmp` file. That is the other half of the
invariant: post-success, `.tmp` must be gone.

## The Resume Tests We Actually Run

The resume surface is wider than the save surface, so the test matrix is
wider too. The ones we care about most come from the public checkpoint sample set and the distributed-checkpoint integration examples:

- `test_resume_weights_exact_match`: save a model with a specific weight
  tensor, reload through `load_checkpoint`, assert `torch.equal`. No
  tolerances. We tolerate bf16 numerical drift in training; we do not
  tolerate it on deserialization.
- `test_resume_preserves_training_step`: step number in `meta.json` round
  trips byte-for-byte.
- `test_optimizer_round_trip` and `test_extra_state_round_trip`: the two
  sidecars that determine whether you actually continue training or
  accidentally cold-restart Adam moments and the dataloader cursor.
- `test_load_corrupt_model_file_raises` and `test_load_corrupt_meta_file_raises`:
  truncated / non-UTF8 inputs must raise, not return silently-wrong data.
- `test_load_missing_optimizer_returns_none`: missing optimizer is an
  allowed resume mode (cold optimizer restart after an optimizer schema
  change), but it must be loud in the logs.
- `test_dcp_directory_reference`, `test_resolve_checkpoint_reference_accepts_dcp_directory`:
  the `resolve_checkpoint_reference` helper has to accept a `model_*.pt`
  path, a DCP dir path, or a parent checkpoint dir, and normalize each
  to `(dir, step, kind)`. Every entry point — training resume, eval
  watcher, ad-hoc one-shot eval — goes through it.
- `test_eval_cpp_loader_preserves_explicit_dcp_intent_over_same_step_model_file`:
  this one exists because we shipped a bug where, given both a DCP dir and
  a sibling `model_*.pt` for the same step, the eval loader silently
  preferred the model file. That is wrong when the caller passed an
  explicit DCP path. `require_kind="dcp"` now enforces the contract and
  this test pins it.

Round-trip tests on CPU are cheap. We run them on every push. The DCP
integration tests fake the distributed layer with monkeypatched
`dcp.save` / `dcp.load` and a fake `FileSystemWriter`, which is enough
to exercise the path logic without spinning up a real process group.
Real distributed resume is exercised separately on H200 and TPU smoke jobs.

## World-Size Change on Resume

This is the case the per-rank format does not handle well and the one we
always have to think about.

The per-rank optimizer files are `optim_{step}_rank{r}.pt`. Saved from a
world size of 8, they are eight files. Resumed on a world size of 4, the
naive `optim_{step}_rank0.pt`...`rank3.pt` path loads only half the
optimizer state. Resumed on world size of 16 it cold-restarts any rank
with no matching file. We consider this broken on purpose: any run that
changes world size must go through DCP, not per-rank.

DCP handles world-size changes because shard boundaries live in
`.metadata`, not in the filenames. The planner (`SPMDSavePlanner` /
`SPMDLoadPlanner` on torch_xla, the default planners elsewhere) reads the
global tensor metadata and re-shards into whatever the current world
partition is. We have resumed from 8 chips to 4 and from 4 to 8 on v6e
and the model weights come back bit-equal; optimizer statistics come back
with the re-sharding the planner computes.

What we deliberately do not try to support on resume:

- Changing TP degree while reusing a non-DCP checkpoint. The manager
  has a `_patch_missing_config_keys()` compatibility shim for adding
  new config keys on old checkpoints (so Engram / Mamba / DSA flags
  default to baseline-safe off), but it cannot invent a different
  sharding.
- Changing the MoE expert count. We rebuild routing state cold in that
  case. Resuming routing statistics into a different `n_experts` is a
  silent correctness hazard we refuse to offer.
- Resuming an emergency checkpoint across hosts. Emergency checkpoints
  are SSD-local by design; the optional `persistent_dir` mirror is the
  only way they become cross-host.

## Corruption We Had to Learn to Detect

Over the life of the project, these were the corruption classes that
showed up and informed tests:

1. Truncated `.pt` from SIGKILL during save. Defense: atomic rename,
   test `test_crash_during_save_leaves_no_final_file`.
2. Truncated JSON meta from the same cause. Defense: same atomic rename
   for meta, test `test_load_corrupt_meta_file_raises`.
3. DCP dir with `.metadata` present but some shard files missing —
   typically a crashed rank. Defense: on load, DCP itself raises;
   `find_local_checkpoint_steps` requires `.metadata` plus a sibling
   `meta_*.json` before advertising the step as available. A DCP dir
   that never got its meta is silently ignored by the lister.
4. Rotation deleted local copy while GCS upload was in flight. Defense:
   pending-uploads table; see above.
5. Stale local model file co-existing with a fresher remote DCP dir
   (happens when you rerun a training job against the same local
   scratch). Defense: `stage_checkpoint_step(refresh=True)` deletes
   the stale local model before downloading the remote DCP; test
   `test_stage_checkpoint_step_refresh_replaces_stale_local_model_with_remote_dcp`.
6. DCP optimizer key-count mismatch on load. The number of optimizer
   key groups depends on how many optimizers the run built; we probe
   `optimizer_key_counts = [3, 2, 1, 0]` and pick the first that loads,
   to handle AdamW-only vs AdamW+Muon vs AdamW+Muon+scheduler-state
   histories.

The corruption case we still do not have a clean defense against is
"DCP wrote successfully, then a rare XLA quirk made `dcp.load` return
zero-init tensors for a specific key". That one we caught only because
`test_resume_weights_exact_match` in CI happened to include the
affected layer. The fix was upstream; the lesson is that exact-equality
resume tests across a model with non-trivial structure are worth the
few extra seconds they cost.

## What We Would Change

If we were starting over we would make DCP the only format and treat
the per-rank `model_*.pt` purely as an export artifact produced on demand
from a DCP dir, not as a first-class save format. We keep the dual path
today because eval, emergency, and ad-hoc debugging all depend on the
per-rank file being the canonical thing `torch.load` opens. That coupling
is the single biggest source of branching in the checkpoint manager and one of
the main historical sources of resume bugs.

## Format snapshot

| Path | Writer | Reader | Use case |
|------|--------|--------|----------|
| Per-rank `.pt` | `torch.save` via manager | `torch.load` | eval, emergency, debug |
| DCP shard dir | `torch.distributed.checkpoint` | DCP loader | large resumes |
| DCP -> per-rank rehydrate | offline script | `torch.load` | hand-off between lanes |
| FP8 amax sidecar | TE hook | TE hook | FP8 resume correctness |

## References

- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- https://docs.pytorch.org/docs/stable/checkpoint.html
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/distributed/oom_triage_sample.py
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md
