---
title: "Why Driver-Visible Paths Can Look Like Hardware Support on GB10, Even When Silicon Proof Is Missing"
description: "A field report on GB10 reverse engineering: how libcuda tables, helper cubins, and signed capability metadata can make tcgen05 look reachable from software while still falling short of proving that the underlying silicon really exposes the same path as B200 or GB100."
date: "2026-04-20"
tags: ["GB10", "Blackwell", "CUDA", "driver-research", "reverse-engineering", "tcgen05", "libcuda"]
---

One of the easiest mistakes in GPU reverse engineering is to confuse a software-visible path with a hardware-proven capability.

That mistake is especially tempting on GB10.

Once you start reading `libcuda`, helper cubins, and capability metadata, the datacenter Blackwell path can look uncannily close. You find architecture tables. You find helper assets. You find capability descriptors. You can sometimes patch one layer and watch the failure move to the next one. It is very natural to tell yourself that you are one more patch away from exposing a physically present feature.

Sometimes that is true. Sometimes it is the precise moment where a project starts lying to itself.

Our GB10 `smtest` work is a good example of why this distinction matters.

## The misleading signal

The strongest misleading signal on GB10 is not one big thing. It is the accumulation of several smaller truths:

- a baseline `sm_100a` cubin can run after an architecture-field rewrite;
- the driver clearly contains multiple product-gating layers;
- `libcuda` ships capability-related machinery that is richer than a simple "unsupported GPU" branch;
- helper selection and architecture routing can be nudged by patching metadata rather than by modifying hardware.

If you line those up in the most optimistic order, the story writes itself: the path is present, the driver is the only blocker, and deeper patching will eventually reveal the real feature.

The problem is that this is still a story built from indirect evidence.

## What the baseline result really means

The baseline arithmetic probe is worth emphasizing because it is the cleanest fact in the entire chain.

After a low-bit `e_flags` rewrite from `sm_100a` to `sm_121a`, a trivial cubin loaded, launched, synchronized, and produced the expected output on GB10. That means GB10 is not rejecting all datacenter-targeted Blackwell SASS at the decoder boundary. It also means at least one visible part of the architecture split is enforced in software.

That is already interesting enough. It proves a real loader/runtime fact.

But it does not prove any of the following:

- that TMEM is physically present on GB10,
- that `tcgen05.mma` is physically present on GB10,
- that a deeper helper path would complete correctly if the driver accepted the image,
- or that a capability signal in the driver should be read as a silicon guarantee.

The baseline result is proof of a baseline result. Reverse-engineering gets messy exactly when we try to cash it out for more than that.

## The four software gates before execution

The GB10 lane exposed four separate gating surfaces before a `tcgen05`-oriented kernel reached anything close to normal execution:

1. ELF architecture validation in `e_flags`
2. reserved weak symbol handling
3. `.nv.info` capability records
4. `.nv.capmerc` plus `.nv.merc.rela.*` signed capability metadata

This gate structure is the backbone of the whole story because it explains why a driver-visible path can be so misleading.

Early gates are easy to misread. If you patch the architecture bytes and the error disappears, it is tempting to say "the driver was the problem." If you patch symbol handling and the error changes again, it is tempting to say "we are getting closer." If you strip or rewrite metadata and the failure moves again, it is tempting to say "the feature must be there."

What is really happening is subtler. You are moving through layers of software policy. You are learning how the driver packages and protects a capability. You are not yet learning whether the underlying hardware truly exposes the end state you care about.

Gate 4 is the most important example. Once the kernel image depends on integrity-protected capability metadata, the presence of that metadata tells you that NVIDIA cares about the capability boundary. It does not tell you whether the capability on the far side is guaranteed to work on this SKU.

## Why helper cubins and routing tables can overstate support

This was the second misleading signal in our GB10 work.

When a driver ships helper assets, architecture tables, or wrapper-selection logic that clearly knows about a more advanced path, engineers naturally treat that as evidence that the hardware path is nearby. Sometimes it is. But "the driver knows how to talk about the path" is still not the same claim as "this SKU can execute the path."

The cleanest way to think about it is to separate three layers:

1. **Routing knowledge**: the software knows names, tables, helper assets, or dispatch rules for a capability.
2. **Submission knowledge**: the software can package and submit something that looks like that capability.
3. **Runtime proof**: the hardware actually completes the exact instruction family in a stable, intended way.

Most false positives in capability research happen when layer 1 or layer 2 gets mistaken for layer 3.

GB10 is full of opportunities for that mistake because the surrounding software stack shares a family resemblance with datacenter Blackwell. That family resemblance is real. It is just not sufficient proof.

## What the stronger patching path did and did not show

One of the more interesting internal directions was the deeper `libcuda` patch path: if we patch far enough to bypass validation or helper selection, can we get the driver to submit the kernel in a way that tells us something conclusive?

That is a valid research question. It is not the same thing as a finished proof.

Even the strongest version of that patching story only tells us that byte-level or table-level driver controls matter a lot. It may get us from "immediate rejection" to "deeper submission behavior." It may even get us to a launch or a hang. But a launch without a clean, stable, intended completion is still not the same as proving shipping silicon support.

That is why we are deliberately framing the deep patch path as a research lane, not a publication-grade capability receipt.

## The practical rule for future bring-up

The rule we want future GB10 work to follow is simple:

Separate these statements every time you write them down:

- the driver accepted the image,
- the driver routed the request through a richer helper path,
- the driver submitted something that looked closer to the desired capability,
- the exact instruction family completed and produced the expected result on hardware.

Only the last statement is silicon proof.

That rule sounds obvious. In practice it is the thing most likely to erode during a long debugging session because every incremental patch feels like momentum. The more exciting the path looks, the more aggressively you need to defend the distinction.

## What this means for GB10 public claims

For public documentation we are intentionally using the stricter reading.

The driver-visible evidence on GB10 is real and worth publishing. It shows that:

- product gating is layered;
- baseline `sm_100a` SASS is not categorically rejected;
- capability metadata is protected much more heavily than a trivial unsupported-path check would require;
- software-visible signs of a datacenter path can persist even when a clean end-to-end proof is still missing.

What it does **not** justify is a claim that GB10 has already been shown to expose working B200/GB100 `tcgen05` parity.

That is why the safer operational conclusion is still the one we use elsewhere on this site: treat GB10 as a different kernel target and keep datacenter-only assumptions off until a real receipt exists.

## Why we are publishing examples instead of a triumphalist claim

The examples that accompany this post are intentionally boring in the best sense.

They publish the pieces we can actually defend:

- the baseline arch-patch probe;
- the driver-signal-vs-runtime-proof distinction;
- the four-gate matrix that shows where the `tcgen05` path still stops.

That is more valuable than a dramatic headline because it gives future engineers reusable decision tools instead of inheriting a claim they now have to unlearn.

The rule here is the same rule that should govern any GPU capability bring-up:

If you want to claim silicon support, publish the runtime proof.
If you only have richer routing evidence, publish it as routing evidence.

GB10 is interesting enough without collapsing those two categories.

## References

- [gb10_driver_signal_vs_runtime_proof_sample.py](../examples/megacpp/gb10_driver_signal_vs_runtime_proof_sample.py)
- [gb10_arch_patch_probe_sample.py](../examples/megacpp/gb10_arch_patch_probe_sample.py)
- [gb10_tcgen05_gate_matrix_nearcopy.py](../examples/megacpp/gb10_tcgen05_gate_matrix_nearcopy.py)
- [What Our GB10 Experiments Actually Prove About Blackwell Consumer vs Datacenter Tensor Paths](./gb10-blackwell-tensor-paths-what-we-actually-proved.md)
- [Training the MegaCpp SLM Ensemble on GB10: a Grace Blackwell war story](./gb10-journey.md)
- [NVFP4 Inference for the MegaCpp SLM Ensemble](./nvfp4-inference.md)
