---
title: "vLLM on GB10: the overlay, the registration fixes, and the paths we kept off"
description: "How MegaCpp stabilized a GB10-oriented vLLM lane with an on-disk overlay, text-only model registration, and a deliberate keep-disabled list for serving paths that were not yet honest."
date: "2026-04-19"
author: "David Gornshtein"
tags: ["vllm", "GB10", "serving", "inference", "overlays", "Blackwell"]
---

vLLM support for the MegaCpp serving lane on GB10 was not one bug and one fix. It was a bundle of compatibility cuts: a packaging decision, a model-registration decision, a checkpoint-loading decision, and a few deliberate non-decisions where the honest answer was "leave this path off until it is actually correct." This post records that state from the bench lane itself.

The short version is simple. Runtime monkey-patching was not durable enough for a multiprocessing engine, so we moved to an on-disk overlay. The base Qwen3.5 registration was resolving to the wrong serving shape for the text-only checkpoints we were actually using, so we registered explicit text-only classes. Some serving paths stayed disabled because they were not yet operationally honest: they either depended on worker-init behavior we had not stabilized, or they introduced training/serving complexity without a validated payoff on the current lane.

## Why MegaCpp needed an overlay instead of a runtime hook

The first attempt was the tempting one: patch vLLM at runtime in the parent process, register the text-only class, and launch a smoke test. The parent process did call the patch code, and the first logs even showed the architecture resolving to `Qwen3_5ForCausalLM`. But the worker model was launched through `spawn`, which means child processes re-import Python modules from disk instead of inheriting the patched in-memory registry from the parent. In other words, the apparent success was misleading. The registration existed in the wrong process.

That is the key reason MegaCpp switched from a parent-process hook to an overlay strategy. Once the modified files live on disk inside the image, every worker process imports the same patched module graph. That solves the actual failure mode instead of only making the parent process look healthy.

The bench lane records that conclusion directly: the strategy-B smoke test was marked inconclusive, not successful, because the runtime patch did not survive worker spawn, and the follow-up options were all variants of the same lesson: use the official plugin mechanism or patch the installed module itself. The current GB10 stack chose the second route because it is simpler to audit and deterministic inside an image build.

## What the overlay actually changed

The GB10 image is built around a pinned toolchain, then overlays patched vLLM files into the checked-out source tree before installing vLLM editable. The Dockerfile documents the intent clearly: clone vLLM at a pinned commit, copy the overlay files into that tree, and verify that the text-only class is importable before the image is considered sane.

That choice matters for two reasons.

First, it makes the serving lane reproducible. There is no hidden startup hook whose success depends on import order or process topology. The image contains the exact model registry and loader logic that the workers will use.

Second, it makes the diff reviewable. MegaCpp can point to a bounded checked-in overlay bundle and say exactly which surfaces diverge from upstream. For a fast-moving upstream like vLLM, that is a much healthier operational posture than carrying an invisible runtime rewrite.

## The model-registration fix

The next problem was model shape, not process shape. The checkpoints under evaluation were text-only Qwen3.5 derivatives, but the default vLLM registry path for `Qwen3_5ForCausalLM` was still effectively the multimodal-oriented wrapper path. MegaCpp needed the architecture name to resolve to a text-only handler that understood the actual checkpoint naming and loading constraints.

The overlay does exactly that. In the patched registry, `Qwen3_5ForCausalLM` is mapped to `Qwen3_5ForCausalLMTextOnly`, and `Qwen3_5MoeForCausalLM` is mapped to `Qwen3_5MoeForCausalLMTextOnly`. The corresponding model module adds those text-only subclasses and marks them as MRoPE-capable so the text path still provides the three-axis position inputs expected by the model configuration.

This was not cosmetic registration cleanup. Without it, the engine could select a class that was structurally wrong for the text-only checkpoints even before weight loading began.

## The checkpoint-loading fix

Registration alone was not enough because the checkpoint names still did not line up cleanly with what vLLM expected internally.

The checked-in loader notes show the mismatch in plain terms. The available text-only checkpoints used nested `model.language_model.*` prefixes and unfused projection naming, while vLLM expected `model.*` names together with fused internal parameter groups such as `gate_up_proj`, `qkv_proj`, and the linear-attention fused projections. MegaCpp did not try to manufacture a new checkpoint format for that. Instead, it relied on vLLM's existing internal fusion path and only fixed the naming seam that blocked it.

The text-only loader subclasses apply a `WeightsMapper` that rewrites `model.language_model.` to `model.` and skips `mtp.` and `visual.` prefixes. That is the important boundary. The overlay does not reimplement vLLM's parameter fusion rules; it just delivers names to the existing loader in the shape that lets those rules fire.

That was the principled cut. A custom one-off checkpoint conversion would have increased maintenance burden and hidden future drift. Letting vLLM perform its own stacked-parameter assembly after a minimal prefix rewrite is much easier to defend.

## Why some serving paths stayed disabled

It is useful to separate "not fixed yet" from "intentionally off."

One disabled path was the runtime-patch approach itself. After the H100 smoke test showed that spawn-based workers did not inherit the parent-side registry mutation, keeping that path alive would have meant pretending the system was safer than it was. MegaCpp deferred that route and documented the real requirement: a plugin loaded in every worker or a patched module installed on disk.

Another path stayed off in the adjacent training lane: colocated vLLM during GSPO remained disabled and the stable run continued with `use_vllm=False`. That decision was operational, not ideological. The checked-in run notes are explicit that the non-vLLM run was stable, already showing the desired reward trend, and not worth destabilizing while the serving-side loader story was still under active repair.

There is also a more mechanical keep-disabled choice in the GB10 smoke harness: it uses a constrained configuration with `gpu_memory_utilization=0.70`, `max_model_len=2048`, and a switch between compiled and eager execution through `enforce_eager`. That is not a production tuning guide. It is a bounded validation lane. MegaCpp kept the larger, more aggressive serving envelope out of this smoke path until the basic registration and load semantics were honest.

The broader rule is simple: we do not enable a serving path just because a one-process smoke test can be made to print text. We enable it only when the process model, registry path, and weight-loading path all match the real serving topology.

## Why the GB10 lane needed its own explicit stack

The GB10 container is not a generic CUDA image with vLLM installed on top. It pins a specific CUDA 13.2 toolchain, a nightly PyTorch line, a flashinfer revision, a vLLM commit, and then applies the overlay. That matters because the goal of this lane was not merely to get text generation once. The goal was to make the serving lane repeatable on Blackwell-class hardware where upstream support was still moving quickly.

The image recipe makes that contract explicit: pinned revisions first, then overlay, then one import-level sanity check for the text-only class. MegaCpp treats that as a patch lane, not as a transient shell session. That is the right operational shape for infrastructure that will be rerun.

## What we would still change upstream

The current overlay is a practical answer, not the ideal endpoint.

The cleaner upstream shape would be one of these:

1. A plugin-based registration path that vLLM guarantees to load in every worker process.
2. A first-class text-only Qwen3.5 registration and loader path upstream so local remapping is unnecessary.
3. Better documentation around worker spawn semantics for anyone tempted to rely on parent-side runtime registration.

Until then, the on-disk overlay is the honest mechanism because it matches the real multiprocessing boundary.

## The MegaCpp takeaway

The interesting part of this work is not that a few files were patched. It is where the boundary was drawn.

MegaCpp did not fork the whole serving stack. It kept a narrow overlay that does three concrete jobs: resolve the architecture to the right text-only class, repair the checkpoint prefix mismatch, and make those fixes visible to spawned worker processes. At the same time, it kept unstable serving paths disabled instead of promoting a fragile smoke-test result into a production claim.

That is what a healthy patch lane looks like. Fix the real boundary. Do not hide the remaining ones.

## References

- [MegaCpp example index](../examples/megacpp/index.md)
- [GB10 repro bundle README](../examples/megacpp/gb10_repro_bundle/README.md)
- [GB10 public claims](../examples/megacpp/gb10_repro_bundle/public_claims.md)
- [GB10 arch patch probe sample](../examples/megacpp/gb10_arch_patch_probe_sample.py)
- [GB10 driver signal vs runtime proof sample](../examples/megacpp/gb10_driver_signal_vs_runtime_proof_sample.py)
- [vLLM project](https://github.com/vllm-project/vllm)
