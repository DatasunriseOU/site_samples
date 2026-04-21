---
title: "Data Shuffling and Seed Discipline"
description: "Deterministic shuffles, seed plumbing across rank and stage, the reshuffle-per-epoch rule, packed-sequence ordering effects on loss curves, and the reproducibility bar we actually hold."
date: "2026-04-18"
tags: ["data", "reproducibility", "dataloader", "training"]
---

Reproducibility is one of those words that sounds like a property of the code and is actually a property of the whole training setup. In MegaCpp we use a specific, finite bar: two runs of the same config, on the same data, with the same seed, on the same hardware family, should produce the same loss curve for the first few thousand steps. Beyond that, numerical drift in bf16 reductions makes strict bitwise equality pointless.

The bar we do hold covers everything under our control: data order, document packing, batch composition, FIM splits, initial weights. This post is about what it took to get there.

## Where seeds actually get set

the shared randomness utilities sets the global seeds during device init:

```python
torch.manual_seed(42)
if device_type == "CUDA":
    torch.cuda.manual_seed(42)
```

That comment in the file is the one that matters: "we set the global seeds here, but most of the code uses explicit rng objects." The global seed is a fallback, not the primary contract. If you relied on the global state, a single `torch.randn` call anywhere upstream of the place you cared about would shift your run.

the main training entrypoint adds an optional user-provided override:

```python
if getattr(args, "seed", None) is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
```

Note the order. We set all four. If any one is skipped, reproducibility leaks through that channel. numpy in particular is easy to forget — several utility modules call into numpy RNGs for shuffles and masks, and a missing `np.random.seed` had us chasing a spurious difference between two "identical" runs for an afternoon.

For Megatron-style tensor parallel init, `init_megatron_parallel_state` takes an explicit seed argument: `_megatron_seed = getattr(args, "seed", None) or 42`. Same seed on every rank, by design — the init is deterministic function of rank plus seed, so every chip computes its own shard identically.

## Explicit RNG objects over global state

The rule we follow in the code: any randomness that matters takes an RNG as an argument, or seeds its own local generator.

Examples in the tree:

- the adapter-merge path takes `seed: int | None`; when non-None, it creates a `torch.Generator` and calls `gen.manual_seed(seed)`.
- the compact-activation path takes a `seed` and seeds its own generator per call site. Its reverse pass re-seeds the same generator from `ctx.seed`, which is why the compressed activation can be reconstructed without storing the projection matrix.
- the KV-quantization path's orthogonal matrix helper seeds a local generator with the passed-in seed, defaulting to 42.
- The sampling path in the main model runtime module uses a passed-in `seed` (default 42) for `rng.manual_seed(seed)` inside `generate(...)`.
- the best-of-n sampling path takes a `base_seed` and uses `base_seed + i` for sample `i`. Not `torch.manual_seed` — a per-call generator.
- the fill-in-the-middle packing path takes an optional `random.Random` instance. When the caller wants reproducibility, they pass one in.
- the GSPO training path uses `seed=42 + i` inside its loop, for the same reason as best-of-n.
- the compact-activation path's seeded layer computes `seed = self.seed_offset + self._step * 31337`. The layer index contributes via the offset (`count * 7919`), so layer 0 and layer 1 do not share a seed.

We keep seeds plumbed through as arguments so that nothing further upstream can poison them. It is ugly. It works.

## The multi-source dataloader and its RNG

The streaming dataloader over multiple parquet source families uses a single `random.Random(42)` to pick which source family a given batch comes from. That RNG lives on the main process.

```python
rng = random.Random(42)
if resume_state_dict is not None and "multi_source_rng_state" in resume_state_dict:
    rng.setstate(resume_state_dict["multi_source_rng_state"])
```

Two design choices are worth calling out:

1. The RNG state is serialized into the checkpoint on save and restored on resume. If you resume at step N, the source selection at step N+1 is the one you would have seen in a fresh run of identical config; we have verified this on restart smokes.
2. The RNG is a `random.Random` instance, not `random` module global state. Nothing else can touch it.

For the packed-row loader, the epoch counter itself is part of the resume state:

```python
resume_epoch = (
    resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
)
```

It starts at 1, increments when all shards of the parquet lineage have been consumed, and is written back into the state we persist for resume. The `epoch` field is part of the loader state tuple, next to the shard index and the per-shard offset.

## Shuffles happen where it is cheap and safe

The parquet write stage is where we shuffle at scale. In the JSONL-to-Parquet ingestion stage:

1. Read JSONL lines with `readline()`.
2. Accumulate documents in batches of 50000.
3. When a batch is full, shuffle it and write a parquet shard immediately.
4. Wait on idle, write a final validation shard when the writer is done.

Shuffling at this stage is cheap because the data is already in memory, and it is safe because it happens exactly once per shard, before anyone reads it. The shuffle inside `create_fim_example`'s corpus prep uses `random.Random(42).shuffle(texts)` for the same reason: local, seeded, one-shot.

Why not a streaming shuffle buffer at load time? Two reasons. The packed-row loader cannot reorder across shard boundaries without breaking the chronology collator that needs `file_local_commit_index` to be ascending within a packed batch (`_validate_packed_row_commit_window_batch` in the dataloader implementation raises if it is not). And shard-level entropy is already enough; across thousands of shards, per-shard shuffle plus multi-source RNG produces batch composition that does not benefit from another buffer pass.

## The reshuffle-per-epoch rule

The data-pipeline design doc spells out the multi-epoch strategy. With roughly 80B unique tokens and 300B target tokens, we cycle:

- Epoch 1-2: full corpus, standard order.
- Epoch 3-4: re-shuffled, different FIM splits.
- Never repeat the exact same FIM split of the same file.

The re-shuffle on epoch rollover is implemented at the shard-listing layer. When the loader wraps around `len(parquet_paths)`, it rescans the directory (new shards may have landed), and the loader bumps the epoch counter. The epoch-bump is the signal to the upstream prep that FIM splits should be regenerated; the regeneration is seeded off the epoch number, so epoch 3 and epoch 4 get distinct splits and epoch 3 restarted from checkpoint gets the same splits as the original epoch 3.

The reason to not repeat the exact same FIM split is straightforward: cross-entropy on a token position you have already seen in the same framing is closer to memorization than generalization. Re-framing the same document with a different FIM split forces the model to solve a different task at that position.

We have not shipped aggressive data augmentation beyond FIM; the scope of "re-shuffled, different FIM splits" is the envelope.

## Packed-sequence ordering effects on loss curves

Packing is where ordering subtly matters. We have three policies: `packed_rows`, `best_fit`, and `single_doc`. The loader tracks utilization telemetry (optional, via `--packing_telemetry`) including `valid tokens / total slots`, `docs_per_row`, `avg_doc_tokens`, and `cropped_doc_frac`.

The effect on the loss curve:

- `single_doc` gives the cleanest curve but the worst utilization. Padding dominates on short documents.
- `best_fit` produces the best utilization but reorders documents within the packing window. That reordering is deterministic (bisect into a sorted list keyed by order key) given a fixed input document stream, so it does not break reproducibility, but it does change which documents land next to which. We have seen `best_fit` vs `single_doc` produce visibly different first-epoch loss traces on the same seed, with `best_fit` running ~0.03 bpb lower in the early phase purely because fewer tokens are padding.
- `packed_rows` reads pre-packed shards from parquet. The ordering inside each row is fixed at write time. This is the fastest path and the one we use for the main training lane.

One subtle interaction with the chronology collator: for temporal commit-window collation, packed rows in a batch are sorted by `file_local_commit_index` so commit chains appear in temporal order within the batch. That sort is stable and deterministic. We validated it explicitly because the collator rejects any batch where `sorted(indices) != indices`, so a bug in ordering would crash loudly rather than silently shift the loss.

We also had one instance where a telemetry counter change — adding utilization tracking — briefly seemed to change loss. It did not; the utilization tracking is plain Python with zero device overhead. The apparent shift was a different run picking up a different shard set because a completion sentinels were not yet in place. That bug fixed, reproducibility was back.

## Resume determinism

A real reproducibility bar includes resume. If resume-from-step-N is not deterministic, long training runs cannot be compared to their restart-free counterparts after an inevitable interruption.

What we persist per step for the loader:

- Shard index (`pq_idx`).
- Row/byte offset inside the shard.
- Epoch counter.
- `multi_source_rng_state` for the multi-source selector.
- Per-source state tuples.

What we persist for the model:

- Weights.
- Optimizer state (including Muon momentum buffers, AdamW exp_avg/exp_avg_sq).
- Learning rate scheduler state.
- Step counter.

What we do **not** persist, and have chosen not to:

- Per-rank dropout RNG state. Dropout in bf16 on TPU already introduces noise we do not try to make bitwise-exact across resume; the cost of persisting and restoring is not worth it. We accept that resume-from-N produces a loss curve that tracks but is not bitwise identical beyond the first few steps.
- Python `random` module global state beyond the multi-source RNG. Anything that needs reproducibility uses an explicit RNG, so the global is irrelevant.

On resume smokes we check three things: the next `(shard, offset, doc)` triple matches what a fresh run would produce at the same step; the first few losses are within bf16 noise of the original; and the multi-source RNG has re-emitted the same source for the next batch.

## The reproducibility bar

Stated explicitly:

1. Data order is deterministic given the same seed and the same shard set. Tested by comparing the first 1000 `(shard, doc_id)` pairs across two runs.
2. Document packing is deterministic given the same input document stream. Tested by hashing the packed-row token sequences for the first N rows across two runs.
3. FIM splits are deterministic given the same seed and epoch number. Tested by comparing FIM markers in the first N splits.
4. Weight init is deterministic given the same seed (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all` all set). Tested by comparing parameter hashes immediately after init.
5. The first ~100 training losses match to within bf16 noise across two runs with identical config. We eyeball this on every new architecture change; a visible divergence in step 1 is always a bug somewhere.
6. Resume-from-step-N matches the RNG state and data position of a fresh run at step N. Tested on every major dataloader change.

What we do **not** promise:

- Bitwise identical loss beyond the warmup window. bf16 reductions, XLA graph partitioning across recompiles, and nondeterministic collectives at scale make this an expensive illusion.
- Cross-hardware reproducibility. A v6e run and an accelerator-backed run are not expected to match bitwise; they share tokenizer, shard set, and seed, and that is enough to reason about.
- Reproducibility under `torch.use_deterministic_algorithms(True)`. The line is commented out in the shared randomness utilities with the note "skipping full reproducibility for now, possibly investigate slowdown later." The slowdown is real, and we have not needed the extra guarantee.

## What we threw away

- The idea that a single `torch.manual_seed` up front would pin all randomness. It will not, and pretending it does only hides the first place it leaks.
- Any attempt at a streaming shuffle buffer on top of the packed-row loader. Breaks chronology, adds latency, has never been worth it.
- A "canonical" epoch ordering preserved across epochs. Re-shuffle per epoch is the rule; fighting it to preserve order across epochs makes the loss curve worse.
- The habit of logging seeds only at startup. They are now logged into the per-run metadata snapshot alongside the model config so that a bisect on a loss regression can start by confirming the seed did not change.

Reproducibility at this level is a maintenance cost. It is also what lets a single failed check at step 100 turn into "which commit between these two introduced the divergence" rather than "something is wrong somewhere in the last month."

## Seed-to-source map

| Source of randomness | Where it is set | Per-rank? |
|---|---|---|
| `torch.manual_seed` (global) | the shared randomness utilities device init | yes (rank-offset) |
| dataloader RNG | explicit `numpy.random.Generator` | yes |
| FIM split | per-doc seeded `random.Random` | doc-deterministic |
| Best-of-N sampling | explicit RNG passed in | yes |
| GSPO group sampling | explicit RNG passed in | yes |
| init weights | `torch.Generator` per module | rank-shared |

## References

- the shared randomness utilities
- the dataloader implementation
- the fill-in-the-middle packing path
- the best-of-n sampling path
- the main model runtime module
- the GSPO training path
- the compact-activation path
- the KV-quantization path
- the adapter-merge path
- the main training entrypoint
- the JSONL-to-Parquet ingestion stage
- the public data pipeline notes
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md
- https://megacpp.com/blog/cpp-data-versioning-and-schema/
