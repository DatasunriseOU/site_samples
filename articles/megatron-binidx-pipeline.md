---
title: "The Megatron .bin/.idx Pipeline: uint32, sharding, and why we do not prebuild the sample index"
description: "How parquet shards become a Megatron MMapIndexedDataset: the uint32 token width forced by a 131K vocabulary, the streaming writer, the brittle sample index we deliberately do not prebuild, and the stage-4 cache discipline around it."
date: "2026-04-18"
tags: ["megatron", "data", "binidx", "pipeline", "c++"]
---

The shape of training data a Megatron-style trainer wants is unromantic. A flat `.bin` of token IDs end to end, a sibling `.idx` with a small header, `sizes[]`, and `pointers[]` saying where each document starts. That is the interchange format. Everything sophisticated — semantic chunking, structure metadata, dependency ordering — happened upstream. By the time bytes reach `--data-path`, the trainer cares about an mmap-able blob and an integer table. Most of what follows is the operational discipline around it; the format is forty lines of `struct.pack`, the pipeline is where the lessons live.

## Where it sits in the pipeline

The five-stage pipeline (download, tokenize, format, cache, verify) sits in this post at stages 3 and 4: the formatting step (parquet -> `.bin`/`.idx`) and the cache-and-validation step (warm + validate). Both are thin wrappers over the Parquet-to-Megatron conversion step, where the real work happens.

```
<data-root>/parquet/${DATASET}/
  shard_00000.parquet ... shard_NNNNN.parquet
  val_shard.parquet
  _COMPLETE
        |
        |  formatting step
        v
<data-root>/megatron/
  ${DATASET}_train.bin   ${DATASET}_train.idx
  ${DATASET}_valid.bin   ${DATASET}_valid.idx
        |
        |  cache-and-validation step  (validate + memmap warm)
        v
[ training launcher: --data-path 1.0 .../${DATASET}_train ]
        |
        v
Megatron builds its sample-index cache on first launch
```

The launcher hard-codes `--data-path 1.0 "<remote-root>/data/megatron/clang_semantic_4k_v10_train"` and Megatron picks it up. `--split 98,1,1` and `--tokenizer-type HuggingFaceTokenizer` are launcher concerns; the on-disk format does not care.

## Why uint32

The single most consequential format decision in this stage is the token-ID width. Our production tokenizer has 131,072 entries — a hybrid fixed C++ vocabulary plus learned BPE with BERT-style whitespace handling — aligned with the configured vocabulary size used at training time. 131,072 does not fit in `uint16`; it barely fits in 17 bits. The next legal numpy dtype up is `uint32`, so `uint32` is what we use.

This sounds small. It is not. The legacy `v2`/`v3` datasets were tokenized with the older 32 K / 65 K tokenizers and live on disk as `uint16` binaries; every consumer that reads them assumes 2 bytes per token. The first time we ran the new tokenizer through the old converter without changing the dtype, the writer silently truncated to the low 16 bits. Token ID `131,005` became `65,469`. The `.bin` was structurally valid, mmap worked, training started, loss was nonsense. The model was reading text in a vocabulary where every other token was wrong.

Token-ID dtype decision table:

| Vocab size | Legal numpy dtype | Megatron dtype code | Bytes/token |
|------------|-------------------|---------------------|-------------|
| < 32,768   | `uint16`          | 1                   | 2           |
| < 65,536   | `uint16`          | 1                   | 2           |
| 131,072 (ours) | `uint32`       | 4                   | 4           |
| signed     | `int32`           | 5                   | 4           |

The fix threads `dtype_str` end to end through `_convert_parquet_to_numpy` and `convert_parquet_to_megatron` with an explicit allow-list:

```python
dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}
dtype_code_map = {np.uint16: 1, np.uint32: 4, np.int32: 5}  # Megatron codes
```

The formatting step defaults to `uint32` with help text saying exactly why (`uint16` is only valid for vocabularies below 65,536), and the verification step checks `max(token_id) < vocab_size` (131,072) on the produced `.bin` and exits non-zero on failure. No silent fallbacks; the `uint16` corruption would be caught before training started.

Cost: roughly 2x disk per token, landing the production `clang_semantic_4k_v10` `.bin` at 6–10 GB train+valid (the `.idx` is under 100 MiB). We considered packed 17-bit and rejected it: Megatron's `MMapIndexedDataset` reads through numpy dtypes, so a custom representation would mean patching megatron-core or adding a hot-loop decode in the data path. Not worth it for a 2x storage win.

## Sharding and discovery

The producer writes parquet shards of 50,000 docs each plus a dedicated validation shard, with a completion sentinel when the last shard is flushed. The Megatron conversion stage walks the directory and decides train vs val:

```python
if split == "train":
    if has_explicit_val:
        return [str(p) for p in all_parquets if p.name != "val_shard.parquet"]
    return [str(p) for p in all_parquets[:-1]] if len(all_parquets) > 1 else [str(all_parquets[0])]
elif split == "val":
    if has_explicit_val:
        return [str(val_shard)]
    return [str(all_parquets[-1])] if len(all_parquets) > 1 else [str(all_parquets[0])]
```

An explicit validation shard is preferred when present — that is the contract emitted by every recent producer. The "last shard is val" fallback exists only because some older datasets were written without it, and we did not want the formatting stage to fail on inputs that were structurally fine. Shard count is uncapped: small validation lanes may have only a few dozen shards, while larger corpus builds can run into the high hundreds or beyond. The converter holds the parquet file list in memory; it streams the rows.

## The streaming writer

Two implementations live in the parquet-to-Megatron conversion script. The preferred path uses `megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder`: given an output prefix and a dtype, it accepts documents via `add_document(np.ndarray)` and writes `.bin` and `.idx` together at `finalize()`. We iterate over parquet row groups, then rows within a row group, calling `add_document` per row. Memory pressure is bounded to one row group plus the builder's internal buffers, which matters on larger conversion jobs.

The fallback runs when megatron-core is not importable (typical on a dev laptop). It gathers documents into a list, then writes `.bin` and `.idx` directly:

```python
MAGIC = b"MMIDIDX\x00\x00"  # 9 bytes
VERSION = 1
with open(idx_path, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<Q", VERSION))
    f.write(struct.pack("<B", dtype_code_map.get(dtype, 1)))
    f.write(struct.pack("<Q", len(all_docs)))           # num sequences
    f.write(struct.pack("<Q", len(all_docs) + 1))       # num documents (sentinel)
    sizes.tofile(f)                                     # int32[N]
    pointers.tofile(f)                                  # int64[N]
    doc_idx.tofile(f)                                   # int64[N+1], one doc per sequence
```

That is the entire `MMapIndexedDataset` on-disk format: magic, version, dtype code, two count fields, three packed integer arrays. We write it ourselves only so a dev laptop can round-trip a tiny dataset without a megatron-core install. The explicit `dtype_code_map` (`1=uint16`, `4=uint32`, `5=int32`) is Megatron's wire-format; using a different dtype than the production reader expects either crashes or silently misinterprets bytes. The fallback materializes everything into RAM, which is fine for dev shards and unsuitable for production — exactly why the megatron-core path is preferred where it matters.

## Why we do not pre-build the sample index

This is the part that surprises people who have not spent time inside Megatron's data layer.

Megatron's `GPTDataset` does not consume `.bin`/`.idx` directly during training. It builds a *sample index* on top of them — a per-epoch shuffled mapping from training step to (document, offset) — and writes it into a dataset-local cache on first access. The sample index depends on `--seed`, `--seq-length`, `--global-batch-size`, and the number of training samples. Every one of those changes the contents.

We deliberately do not pre-build it. A pre-built sample index made for `--seq-length=4096 --global-batch-size=512 --seed=1234 --train-iters=20000` is a liability for a launcher with `--seq-length=4096 --global-batch-size=1024 --seed=42 --train-iters=10000`. Megatron either rebuilds it (slow, and the pre-built one was wasted) or — worse, in older versions — fails to detect the staleness and uses it anyway. The historical bugs in this area always came from someone pre-building for one launcher and reusing for another.

Stage 4, the cache-and-validation step, deliberately does not touch Megatron's sample-index cache. It does three things: confirm `.bin` and `.idx` exist with non-zero size; instantiate Megatron's `IndexedDataset` reader on the prefix to validate the `.idx` parses, the dtype code is recognized, and the `.bin` mmaps cleanly; and report `docs`, `tokens`, `dtype` from the materialized index.

```python
ds = IndexedDataset(str(prefix))
n_docs = len(ds.document_indices) - 1
total_tokens = int(ds.sequence_lengths.sum())
print(f"[cache-check] docs={n_docs:,}  tokens={total_tokens:,}  dtype={ds.index.dtype}")
```

Entire stage-4 contract: prove the artifacts are well-formed, prove they parse with the same library the trainer will use, then get out of the way and let the trainer build the sample index at first launch. Megatron rebuilds it in seconds; pre-building it in a separate stage would cache the wrong thing the first time someone changes a launcher flag.

On-disk layout after stage 4:

```
<data-root>/megatron/
  clang_semantic_4k_v10_train.bin
  clang_semantic_4k_v10_train.idx
  clang_semantic_4k_v10_valid.bin
  clang_semantic_4k_v10_valid.idx
  cache/<sample-index>/            # built at first training launch
```

The `cache/` directory is not our problem. It is Megatron's.

## Stage 4 cache discipline, in practice

Three rules around the cache. Never check it into version control. Keep it under the dataset-local Megatron cache, never alongside training checkpoints — rsyncing checkpoints between hosts is routine and pulling a 1 GiB sample index that is wrong for the destination launcher is a foot-gun we have stepped on. Treat it as cheap to delete: when the launcher changes a knob Megatron may or may not detect (particularly `--seed`), our practice is to clear the local sample-index cache before launch. Rebuilding costs seconds; debugging a stale index costs a day.

On a dev laptop without Megatron-Core installed, the cache-and-validation step logs a clear SKIP and exits 0:

```python
try:
    from megatron.core.datasets.indexed_dataset import IndexedDataset
except Exception as e:
    print(f"[cache-check] SKIP: megatron-core not importable ({e}).\n"
          f"  Run this stage on the training host instead.", file=sys.stderr)
    return 0
```

The exit-0 is intentional: stage 4 is a verification stage, and on a host where verification cannot meaningfully run, refusing to fail makes the multi-stage pipeline runnable end to end without forcing an unrelated install.

## Reproducibility, hashes, and the things we do not version

The pipeline is deterministic on the parts we control. The streaming parquet writer uses `--seed=42`. The Megatron converter writes documents in shard order, then row-group order, then row order; identical inputs produce an identical `.bin`. The `.idx` is deterministic by construction.

For audits, `sha256sum` the resulting `.bin` and pin the hash in the experiment log next to the checkpoint. That is the only thing that survives upstream-tag drift, library upgrades, and tokenizer churn. A hash of the `.bin` *is* the dataset.

Not versioned: the raw source clones (~15 GB, mirror to cold storage if you need post-mortem reproducibility), the transient JSONL between stages 2 and 3 (~20–30 GB, intentionally ephemeral), and the trainer-built sample-index cache created per launch. Versioned: the tokenizer artifact copied into `<data-root>/tokenizer/` at stage 2, the conversion logic, the launcher configuration, and the `_COMPLETE` sentinel that proves a parquet directory is fully published.

## What stage 5 catches

The verification step is the no-silent-fallback gate. It checks `.bin` and `.idx` exist and are non-empty, that `.idx` parses through the same `IndexedDataset` reader Megatron uses, that `max(token_id) < vocab_size` (the check that would have caught the `uint16` truncation), and that the first 64 tokens of document 0 round-trip through the tokenizer and look like C++. Any failure exits non-zero. Two classes of bugs caught here that would otherwise have wasted a training run: dtype mismatches, and dataset-name typos pointing `--data-path` at an empty or partial publish. Stage 5 does not validate semantic content; the producer is responsible for that. Stage 5 is the structural integrity check between producer and consumer.

## What we are choosing to live with

A few things we know about and do not plan to fix. The fallback writer holds documents in RAM — fine for the dev path, wrong for production, and precisely why the production path requires megatron-core; the maintenance cost of a streaming fallback is not worth it for a path that exists only so the laptop test passes. Producer-stage failures live upstream, not here: stage 3 trusts the parquet input, and if a producer published a partial dataset without a `_COMPLETE` sentinel, stage 3 will happily convert what is there — that is the producer's discipline, not ours. We do not gzip the `.bin`: Megatron expects an uncompressed mmap, and compressing on disk would defeat the entire point of memmap-based loading.

The summary is that stages 3 and 4 are intentionally small. Most of the engineering went into making them refuse to be clever: a deterministic shard-discovery rule, a `dtype` correct by default and validated by stage 5, a streaming writer that does not lie about the format, and a stage-4 cache discipline that consists almost entirely of "do not pre-build what Megatron rebuilds in seconds." The format itself is forty lines; the discipline is the work.

## References

- [Megatron-LM indexed dataset implementation](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/datasets)
- [MegaCpp MegaCpp sample pack articles directory](https://megacpp.com/blog)
- [MegaCpp MegaCpp sample pack docs directory](https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md)
