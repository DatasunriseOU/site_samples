---
title: "The MegaCpp precision recipe: FP16, BF16, FP8 and NVFP4 in one stack"
description: "How MegaCpp picks a numerical format per op, per device, and per phase: FP16 only as a floor, BF16 as the steady state, FP8 in selected GEMMs, and NVFP4 for Blackwell inference."
date: "2026-04-18"
tags:
  ["bf16", "fp16", "fp8", "nvfp4", "mixed-precision", "training", "inference"]
---

We use four numerical tiers in the MegaCpp stack and we use them deliberately. BF16 is the steady state for training compute. FP16 exists mainly as a fallback on older or development hardware. FP8 is opt-in on a curated set of GEMMs. NVFP4 is the inference target on Blackwell. The recipe is not a global flag. It is a per-op, per-device, per-phase contract.

## Why the generation boundary matters

The public architectural line is simple. Hopper-class H200 systems belong to the FP8, BF16, and FP16 training story. Blackwell adds NVFP4. If an article collapses those into one universal statement, it becomes misleading fast.

That is why the recipe is asymmetric. Training formats are chosen around numerical stability and kernel support on the current training device. Inference formats are chosen around the deployment device.

## The four tiers

| Tier  | Main role              | Safe public framing                                                          |
| ----- | ---------------------- | ---------------------------------------------------------------------------- |
| FP16  | fallback               | useful on older or development hardware                                      |
| BF16  | training default       | the steady-state training format on modern hardware                          |
| FP8   | selective acceleration | opt-in on large GEMM-heavy surfaces when the hardware and kernels support it |
| NVFP4 | low-precision serving  | a Blackwell-era inference format, not a Hopper training format               |

## BF16 is the default training floor

BF16 is the training default because it keeps the main training path simple and stable. That includes the optimizer master-precision story: optimizer-state precision should be discussed separately from the model's compute precision.

## FP8 belongs on selected surfaces, not everywhere

Selective FP8 rollout belongs on layer families where Hopper's FP8 path is publicly supported and where cast overhead does not dominate the kernel. That usually means large projection-heavy GEMMs, not every small or irregular layer in the model.

Checkpoint and recompute precision is a separate surface again. It should be treated independently from both forward GEMM precision and optimizer-state precision.

## NVFP4 belongs to the Blackwell inference story

NVFP4 is the inference target on Blackwell. The training master remains BF16; conversion to NVFP4 happens at quantization time rather than during Hopper-side training. That separation matters because public NVIDIA documentation places NVFP4 on Blackwell, not Hopper.

The practical takeaway is also narrower than a headline speedup claim. NVFP4 is useful because of footprint and Blackwell-era low-precision inference support, not because one universal speedup number applies across every device shape.

## Practical takeaway

A clean public precision story should say this:

1. H200 is a Hopper platform, so the public training story is BF16 plus optional FP8.
2. Blackwell introduces NVFP4, so the public inference story can add NVFP4 there.
3. Precision policy should stay per-surface rather than collapsing into one global "low precision" label.

## References

- https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/
- https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
