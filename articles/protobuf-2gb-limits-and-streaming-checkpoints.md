---
title: "Protobuf, the 2 GB Wall, and Why MegaCpp Prefers Shards Over Giant Messages"
description: "Why large-message serialization becomes fragile near protobuf's practical limits, and how MegaCpp's checkpoint and data paths avoid single huge payloads by using sharded files, streaming conversion, and explicit completion markers."
date: "2026-04-19"
tags: ["protobuf", "serialization", "streaming", "checkpoints", "data", "training-infra"]
---

If you move enough model state or dataset payload through one serialized object, you eventually stop having a software architecture problem and start having a failure-domain problem. Protobuf's well-known size ceiling near 2 GB is one expression of that boundary. Even below the hard cap, giant messages are painful: allocation spikes, retry cost, partial-write ambiguity, and miserable debuggability.

The interesting question for MegaCpp was not whether protobuf is "good" or "bad." The question was narrower: what should the training and data pipeline do so that single-message limits do not become the control plane for checkpointing or corpus transport? The answer visible in the repos is consistent: avoid monolithic payloads, write shardable artifacts, use atomic per-file promotion, and require explicit completion markers before consumers advance.

## The 2 GB limit is real, but the practical pain starts earlier

Protocol Buffers use a 32-bit signed size model for message length accounting in several implementations and APIs, which is why the ecosystem commonly treats serialized payloads near 2 GB as out of bounds. That is the headline limit. The operational limit is lower.

Long before a message reaches that ceiling, large-object serialization creates four problems:

- one failure can invalidate the whole transfer rather than one shard
- producer and consumer both need enough contiguous memory headroom for the whole object
- retries become expensive because they repeat the whole blob
- partial writes and partial reads are harder to distinguish from valid-but-incomplete state

That is exactly the class of pain that large checkpoints and streaming corpora trigger.

## What the checkpoint code avoids

The checkpoint manager in the training repo does not rely on one giant serialized checkpoint message. It keeps two storage shapes instead.

First, a distributed checkpoint path writes a directory of shards through `torch.distributed.checkpoint`, where each rank writes its own data directly and rank 0 separately writes small JSON metadata. The code comment is explicit about the reason: this avoids gathering the full model to rank 0 and is materially faster for sharded runs.

Second, the non-DCP path writes separate files for model weights, optimizer state, metadata, and optional extra state. Those files are promoted atomically via `*.tmp` plus `os.replace(...)`, including metadata sidecars and emergency-save pointers. That is a very different failure surface from "serialize everything into one message and hope the write completes."

The rotation logic also shows the same design instinct. Old local checkpoints are not deleted immediately if background remote upload threads are still alive or have marked `gcs_upload_ok` as failed. The code prefers disk pressure over permanent data loss. Again, the pattern is file-by-file durability, not trust in one monolithic transfer succeeding.

## What the data path avoids

The data-preparation scripts show the same bias away from giant serialized payloads.

The streaming JSONL-to-parquet converter tails a JSONL file that may still be growing, accumulates rows only up to a configured `rows_per_file` threshold, writes a parquet shard, validates that shard by reopening it, and only then moves on. When input growth stops for long enough, it writes the final train shard, a validation shard, and a `_COMPLETE` sentinel.

The companion upload and download scripts are also shard-oriented. They copy parquet files individually and use `_COMPLETE` as the promotion marker rather than assuming that seeing some files means the dataset is ready. On the MegaCpp side, the same logic is carried into the renamed `prepare_stream_jsonl_to_parquet_megacpp.py` helper: growing JSONL in, bounded parquet shards out, explicit sentinel at the end.

That choice matters because a corpus is not safer just because it is "serialized." A 100 GB corpus wrapped in a giant logical message is still a 100 GB problem. Columnar shards plus explicit end-of-stream signaling are much easier to retry, validate, and resume.

## Why this is better than one huge message

The repo evidence supports a narrow claim: MegaCpp repeatedly chose boundaries that keep failure local.

| Problem | Giant-message design | Sharded-file design used here |
| --- | --- | --- |
| checkpoint save | one large serialize/write step | DCP shard dirs or separate model/optimizer/meta files |
| interrupted write | ambiguous whole-object corruption | `*.tmp` then `os.replace(...)` per artifact |
| remote copy lag | easy to delete the only good blob | keep local copies while upload threads are pending or failed |
| dataset streaming | one huge export unit | parquet shards with row thresholds |
| end-of-stream detection | infer from transport success | explicit `_COMPLETE` sentinel |

None of this means protobuf should never be used. It means protobuf-sized thinking is the wrong abstraction for heavyweight training artifacts. Once payloads are large enough to make 2 GB even relevant, the safer design is usually to stop pretending the object is singular.

## What we changed or deliberately avoided

From the code and article history, the practical design choices are clear:

- avoid rank-0 gather as the only checkpoint shape for heavily sharded runs
- avoid single-file checkpoint authority when DCP shard directories are the natural write format
- avoid treating partial presence as readiness; use sidecar metadata and `_COMPLETE` markers
- avoid non-atomic final writes for large artifacts; write temp files and promote with rename
- avoid dataset transport that depends on one oversized serialized payload staying valid end to end

This is not an anti-protobuf argument. It is an argument for choosing the right serialization grain. Small control messages are fine. Giant model and data artifacts want chunking, validation, and promotion semantics that survive interruption.

## The broader lesson

The 2 GB protobuf limit is a useful warning sign because it forces an architectural question: if one message can brick the operation, why is the operation shaped like one message at all?

MegaCpp's repos answer that question with a boring but durable pattern: many files, explicit metadata, atomic promotion, resumable copies, and completion markers. That pattern is less elegant than "one object in, one object out," but it is much closer to how large training systems actually survive.

## References

- https://protobuf.dev/programming-guides/proto-limits/
- https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/io/coded_stream.h
- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- https://megacpp.com/blog/checkpoint-format-and-resume/
- https://megacpp.com/blog/data-pipeline-story/
- https://megacpp.com/blog/cpp-data-versioning-and-schema/
