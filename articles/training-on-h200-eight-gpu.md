---
title: "Training on 8x H200 SXM: the operator playbook"
description: "End-to-end operator notes for driving an 8x H200 SXM node: topology, NCCL tuning, storage layout, and the invariants that keep a run from silently drifting."
date: "2026-04-18"
tags: ["h200", "nccl", "nvlink", "fsdp2", "torchrun", "training", "operations"]
---

An 8x H200 SXM node is a practical unit for training a mid-sized specialist model from scratch. On paper it looks like a larger-memory Hopper system. In practice the gap between a fresh machine and steady-state high-throughput training with reliable checkpoints is a sequence of small operational choices that, taken in the wrong order, cost days. This post focuses on that operator surface: how to drive the node, what the topology forces on the launch flow, which NCCL settings are worth making explicit, and how we keep receipts comparable.

## Why the operator surface is the contract

MegaCpp training is not about one heroic run. It is about repeating comparable launches across the same hardware class. That is a discipline problem, not just a performance problem. The launch surface is the contract. If two operators can produce different steady-state throughput on the same configuration because one forgot to pin a compile cache or left a stale NCCL setting behind, the comparison stops meaning anything.

The H200 memory budget changes the shape of the decision surface, but it does not eliminate activation pressure. A model that fits comfortably at short context can still run into trouble once sequence length, checkpointing policy, or expert routing buffers move together. We want one launcher that lands cleanly in both regimes and escalates predictably when it does not.

## Topology, launcher, and what the wrapper owns

Each rank owns one H200. The eight GPUs sit on an NVLink and NVSwitch fabric, and NVIDIA's public H200 materials frame that fabric as a first-class part of the Hopper-era multi-GPU story. What the operator has to get right is that NCCL actually uses the expected topology and that nothing in the host image is silently routing traffic through a slower path.

A fresh node still deserves one hard preflight: `nvidia-smi topo -m` should show the expected NVLink or NVSwitch fabric rather than an accidental fallback path.

| Concern              | Knob                                               | What it does                                     |
| -------------------- | -------------------------------------------------- | ------------------------------------------------ |
| stream serialization | `CUDA_DEVICE_MAX_CONNECTIONS=1`                    | keeps communication and compute ordering legible |
| stream priority      | `TORCH_NCCL_HIGH_PRIORITY=1`                       | helps comm streams avoid starvation              |
| allocator policy     | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces fragmentation under changing workloads   |
| heartbeat budget     | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`            | survives long compile windows                    |

The broader lesson is more stable than any single knob: environment-sensitive runtime behavior belongs in the receipt surface.

## What the wrapper owns

A good launcher owns provenance, environment, the `torchrun` line, and log extraction. It should record the source revision, a short working-tree status, and the machine class into the log before the first Python import.

```bash
exec torchrun --standalone --nproc_per_node=8   -m <training-entrypoint>   --config "$CONFIG" --run_name "$RUN_NAME"   --device_batch_size "$DBS" --max_seq_len "$SEQ"
```

## State on disk and live monitoring

There are three categories of state and they should go to three places.

- persistent training state belongs on a durable high-capacity data volume
- per-process artifacts such as compile caches belong in per-run scratch space
- operator receipts belong with the launch materials for that run

The invariant that should not drift is simple: the training data root must point at the intended data volume, not at an incidental home-directory path.

## Takeaway

The H200 operator story is not that one knob solved training. It is that topology, launcher policy, and receipt discipline together make the node trustworthy enough to compare runs.

## References

- [NVIDIA Hopper architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA HGX platform](https://www.nvidia.com/en-us/data-center/gpu-cloud-computing/hgx/)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
