---
title: "What Our GB10 Experiments Actually Prove About Blackwell Consumer vs Datacenter Tensor Paths"
description: "Our GB10 tests show that some Blackwell datacenter-targeted SASS can be accepted and executed on consumer silicon, but they do not prove that physical tcgen05.mma execution is available on GB10. Older stronger claims overstate what the evidence supports."
date: "2026-04-20"
tags: ["GB10", "Blackwell", "CUDA", "tensor-core", "tcgen05", "driver-research"]
---

The practical question behind our GB10 lab work was simple: when a Blackwell-datacenter cubin fails on GB10, is that because the silicon physically lacks the path, because the driver blocks it, or because we are mixing evidence from several different layers and telling ourselves a cleaner story than the data supports?

The answer is narrower than some early drafts made it sound.

Our experiments proved that GB10 can accept and execute at least some `sm_100a`-targeted SASS after a small architecture-field rewrite in the cubin. They also proved that `tcgen05` and TMA-adjacent probes hit additional driver-side capability gates that the baseline arithmetic kernel never touches. What they did **not** prove is that GB10 physically executes `tcgen05.mma` the same way B200 or GB100 does.

For engineering purposes, that means the safe rule is still the conservative one: treat `tcgen05` on GB10 as unavailable unless you have a real end-to-end execute receipt for that exact instruction family. Driver-visible hints are not enough.

## The narrow positive result

The strongest positive result in the GB10 test lane is also the cleanest one. A trivial arithmetic cubin compiled for `sm_100a` loaded and ran on GB10 after only the low architecture bits in ELF `e_flags` were rewritten from `sm_100a` to `sm_121a`.

The important part is not the patch itself. The important part is what happened after the patch:

```text
[cuModuleLoadDataEx]    CUDA_SUCCESS
[cuLaunchKernel]        CUDA_SUCCESS
[cuCtxSynchronize]      CUDA_SUCCESS
# out[0..7]: 00000001 00000003 00000005 00000007 00000009 0000000b 0000000d 0000000f
```

That is enough to establish three things.

First, the user-space driver really does contain software gating at the architecture-identification layer. Second, GB10's instruction path will accept at least some SASS originally emitted for the datacenter Blackwell line. Third, the absence of an immediate Xid or hard decoder fault means "consumer Blackwell" and "datacenter Blackwell" are not separated by one single binary hardware switch at the very first instruction boundary.

That is a meaningful result. It is just not the same as "GB10 has working `tcgen05.mma`."

## The four software gates we found

Once we moved from a baseline arithmetic kernel to `tcgen05` / TMA-oriented probes, the driver path turned out to be layered.

Our GB10 `smtest` lane consistently exposed four gates before the cubin reached a usable execution state:

1. ELF `e_flags` architecture validation
2. reserved weak undefined symbols such as `.nv.reservedSmem.offset0` and `.nv.reservedSmem.cap`
3. `.nv.info.<kernel>` capability records
4. `.nv.capmerc.text.<kernel>` plus `.nv.merc.rela.*` signed capability metadata

The important engineering lesson is that these are not all the same kind of obstacle.

Gate 1 is simple identity metadata. Gate 2 is loader-side symbol plumbing. Gate 3 is still mutable per-kernel metadata. Gate 4 is qualitatively different: it is integrity-protected capability data. That is the point where naive cubin surgery stops being enough.

This matters because public discussions about GB10 support often collapse all of these layers into one sentence like "the driver blocks it" or "the hardware lacks it." That hides the real structure of the problem. Some things are byte-patchable. Some things are not. And the fact that you can move through the earlier gates does not tell you what would happen if the exact `tcgen05` kernel reached a fully valid submission state.

## What the `tcgen05` probes actually hit

The conservative GB10 receipts stop at gate 4.

Kernels using `tcgen05.alloc`, `tcgen05.ld`, `tcgen05.mma`, and TMA multicast were not observed completing as normal working GB10 kernels in the clean evidence set we are willing to publish. The isolated `tcgen05.alloc` path, for example, could be moved through the earlier metadata gates and still ended in `CUDA_ERROR_INVALID_IMAGE` once `.nv.capmerc` integrity became the deciding factor.

That is the key distinction that older drafts blurred.

There is a big difference between:

- proving the driver can be pushed farther than its default routing policy,
- proving a helper or wrapper path exists inside `libcuda`,
- proving a kernel can be submitted and hang,
- and proving that `tcgen05.mma` is a stable, physically present, usable execution path on GB10 silicon.

Only the last one would justify a public claim of B200-style `tcgen05` availability. We do not have that proof.

## Why older stronger claims are stale

Some of our earlier exploratory notes leaned too hard on the most exciting interpretation of the data. That is normal in a live reverse-engineering session and unacceptable in a publication.

The strongest overreach looked like this:

- a helper or routing patch can be made to reach deeper driver paths,
- therefore the silicon probably carries the full capability,
- therefore GB10 effectively "has" the datacenter path if we patch enough bytes.

That leap is too large.

A more honest reading is:

- the software stack clearly contains layered product gating,
- at least some datacenter-targeted SASS decodes and executes on GB10,
- the driver ships enough capability-related machinery to make the path look tantalizingly close,
- but the publication-grade `tcgen05` proof is still missing.

That is why we are treating the earlier stronger wording as stale. It was useful as a research hypothesis. It is not the standard we want attached to a public article or a customer-facing example repo.

## Consumer Blackwell vs datacenter Blackwell

What should an engineer conclude from this if they just want to ship kernels?

The practical conclusion is not mysterious:

- GB10 is not a small B200.
- Driver-visible datacenter artifacts do not make it one.
- If a path depends on TMEM, `tcgen05`, or other datacenter-only assumptions, you should treat GB10 as a separate target with its own kernel contract.

That conclusion matches the rest of the GB10 bring-up story across this site. In inference, we already treat GB10's OMMA-based FP4 lane as real while keeping TMEM-coupled paths off. In the FA4 catalog, we already gate GB10 separately from the `sm_100a` line. The tensor-path experiments fit the same pattern: shared branding, partial decode overlap, different operational contract.

The main correction here is about proof discipline. A capability table, helper cubin, or partially patched submission path is not a shipping contract.

## The public-safe rule

The right public-safe rule is stricter than the most optimistic private lab note.

If you are writing documentation, examples, or runtime policy for GB10:

- you may say that some `sm_100a` SASS executes on GB10 after an arch-field rewrite;
- you may say that multiple driver-side gates exist before `tcgen05` probes can run;
- you may say that driver-visible capability machinery can make unsupported paths look deceptively close;
- you should **not** say that GB10 has proven working `tcgen05.mma` parity with B200 or GB100.

That is also why the new public examples for this topic focus on the baseline arch-patch probe, the gate matrix, and the difference between a software-visible signal and runtime proof. Those are the parts we can defend cleanly.

## What we are publishing instead

For this topic we are publishing three things and drawing one line.

The three things:

- a compact baseline probe showing what the positive `sm_100a` result really means;
- a compact example showing why driver-visible support is not runtime proof;
- a near-copy gate-matrix receipt showing exactly where the `tcgen05` path still stops.

The line:

We are not publishing "6-byte patch unlocks `tcgen05` on GB10" as a settled statement.

That line is worth keeping. It saves future engineers from inheriting an evidence problem disguised as a success story.

## References

- [gb10_arch_patch_probe_sample.py](../examples/megacpp/gb10_arch_patch_probe_sample.py)
- [gb10_driver_signal_vs_runtime_proof_sample.py](../examples/megacpp/gb10_driver_signal_vs_runtime_proof_sample.py)
- [gb10_tcgen05_gate_matrix_nearcopy.py](../examples/megacpp/gb10_tcgen05_gate_matrix_nearcopy.py)
- [Training the MegaCpp SLM Ensemble on GB10: a Grace Blackwell war story](./gb10-journey.md)
- [NVFP4 Inference for the MegaCpp SLM Ensemble](./nvfp4-inference.md)
- [The FA4 Catalog on Blackwell: Variants, sm Guards, and Runtime Selection](./fa4-catalog-on-blackwell.md)
