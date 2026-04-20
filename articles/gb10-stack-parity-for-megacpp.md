---
title: "GB10 Stack Parity for MegaCpp: Torch 2.13 cu132, GCC 15, CUDA 13.2, and the Nightly Constraint"
description: "Why MegaCpp mirrored the GB10 software stack so exactly: PyTorch 2.13 cu132 nightly, GCC 15, CUDA 13.2, rebuilt source dependencies, and the device-specific constraints that made parity operational rather than cosmetic."
date: "2026-04-19"
tags: ["MegaCpp", "GB10", "PyTorch", "CUDA", "Build Systems", "vLLM"]
---

# GB10 Stack Parity for MegaCpp: Torch 2.13 cu132, GCC 15, CUDA 13.2, and the Nightly Constraint

When people hear "stack parity," they often picture convenience: make one environment look like another so debugging feels tidier. In MegaCpp, the GB10 parity stack was not about tidiness. It was about keeping the same compiler, ABI, wheel, and source-build assumptions across two different execution environments so that a result meant the same thing in both places.

The core parity target was narrow and explicit:

- Ubuntu 24.04
- GCC 15.2
- CUDA toolkit 13.2.51
- PyTorch 2.13 `cu132` nightly
- source-built `flashinfer`, `vllm`, and selected kernel-side dependencies at pinned commits

That exact bundle shows up directly in the build recipe used to mirror the GB10 serving stack: the image installs GCC 15 from the Ubuntu toolchain PPA, installs CUDA 13.2, and then force-reinstalls PyTorch `2.13.0.dev*` from the `cu132` nightly index before rebuilding the rest of the stack from source where wheels are not a safe substitute.

## Why parity mattered

MegaCpp had a practical problem, not an aesthetic one. The GB10 stack was `aarch64`, while the mirrored environment used for external validation and portability work was `x86_64`. That means wheel-level parity is impossible in the naive sense: even if the package names match, the binaries do not. The bench-side parity recipe says this plainly: wheels are cross-incompatible, so source dependencies have to be rebuilt against matching versions instead of copied over blindly.

This is the real meaning of parity in a compiler-heavy ML stack. It is not "same `pip install` output." It is "same effective runtime contract": same PyTorch generation, same CUDA toolchain generation, same compiler family, same pinned source revisions for the packages that actually compile kernels or extend the runtime.

For MegaCpp that mattered because several behavior boundaries are stack-sensitive:

- whether a package can be installed from a wheel at all
- whether a source package builds cleanly against the chosen PyTorch and CUDA pair
- whether runtime-compiled kernels target the same capability and shared-memory assumptions
- whether a serving or benchmark receipt can be compared to device-local results without hidden environment drift

## The nightly wheel was a constraint, not a preference signal

The GB10 parity build pins PyTorch to `2.13` `cu132` from the nightly index. That decision is worth stating directly because it is easy to misread. Nightly here is not a style choice and not a vague desire to be "latest." It is a compatibility constraint.

The parity scripts install PyTorch from the nightly `cu132` channel and then build the rest of the stack around it. The reason is simple: the rest of the environment is already targeting CUDA 13.2 and a current Blackwell-oriented toolchain. Once that is true, the PyTorch choice is no longer an isolated package decision. It becomes part of one coupled compiler surface.

In practice, that surface included:

- `torch 2.13.0.dev*` from `cu132`
- `cuda-toolkit-13-2`
- `cuda-python` / `cuda-bindings` in the 13.2 line
- editable or source installs for `vllm`, `flashinfer`, and other extension-heavy packages

That is why the dependency notes distinguish between ordinary PyPI pins and "custom-built editable installs." The latter are the packages that cannot be treated as interchangeable wheels without losing control of the build surface.

## GCC 15 and CUDA 13.2 were part of the same story

The parity image does not only pin PyTorch. It also installs GCC 15 and CUDA 13.2 explicitly, then exports the expected CUDA paths before any source builds begin. That matters because MegaCpp was not just importing Python modules. It was building and rebuilding extension code, JIT-capable libraries, and runtime-sensitive CUDA packages on top of that environment.

In other words, the stack boundary was not "Python package management." The real boundary was C++ and CUDA compilation. If GCC, NVCC, and PyTorch are not aligned closely enough, the resulting environment may install successfully and still fail to produce comparable runtime behavior.

That is why the parity recipe is structured in the order it is:

1. install system toolchain
2. install CUDA toolkit
3. install the exact PyTorch nightly lane
4. rebuild kernel-sensitive dependencies from source
5. overlay the exact `vllm` modifications needed by the MegaCpp-serving path

This is less glamorous than benchmark plots, but it is the difference between a reproducible environment and an anecdotal one.

## Why rebuilding from source was unavoidable

The dependency inventory for the GB10 stack makes the constraint visible. Several key packages are not consumed as ordinary released wheels. They are installed editable or from source at pinned commits, including `vllm`, `flashinfer`, `mamba_ssm`, and other kernel-adjacent components.

That happened for two separate reasons.

First, architecture. `aarch64` on one side and `x86_64` on the other means prebuilt wheels are not portable across the two environments.

Second, stack shape. Even on one architecture, the environment carried source-level overlays and pinned revisions that were part of the actual working runtime. The dependency notes explicitly call out a `vllm` commit plus a file overlay, plus additional divergence between the GB10 stack and the Modal-side stack. Once you are in that world, "install the nearest stable wheel" is no longer a real parity strategy.

## Parity mattered because device constraints were real, not hypothetical

The strongest reason this mattered for MegaCpp is that GB10 was not just another CUDA machine with fewer SMs. The codebase carries an explicit GB10 shared-memory preflight because `sm_121` has a 99 KiB dynamic shared-memory limit per SM. The preflight module explains the operational impact directly: several TileLang kernels can naively emit 140+ KiB of shared-memory descriptors, which compiles but then fails at launch with an opaque runtime error unless the aggressive shared-memory merge flag is enabled.

That logic is not a comment-only note. It is enforced behavior.

The README summarizes the production reason for shipping the preflight: on `sm_121`, training is refused unless every relevant TileLang kernel declares `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True`, because otherwise kernels can compile and only fail at first launch.

This is exactly the kind of device-specific behavior that makes parity matter. If MegaCpp validated adjacent serving and benchmark layers on a stack that drifted away from the GB10 runtime assumptions, then a "working" result would not tell us much. It could be hiding a compiler difference, a dependency-build difference, or a missing device-specific guard that only the GB10-shaped environment would reveal.

## The point was comparability, not identical hardware

The parity work did not pretend that two machines become identical once their package lists look similar. The bench recipe says the opposite: one side is `aarch64`, the other is `x86_64`, so exact wheel reuse is off the table from the start.

The goal was therefore narrower and more useful: preserve the parts of the stack that determine build behavior and runtime expectations, then compare outcomes across environments with fewer hidden variables.

That is why the parity image keeps the same broad stack story:

- same Ubuntu generation
- same GCC generation
- same CUDA generation
- same PyTorch nightly lane
- same pinned source commits for extension-heavy packages
- same local overlay strategy where upstream releases did not yet match the working stack

For MegaCpp, that was enough to make benchmark and serving evidence interpretable. Without it, every receipt would have had a built-in disclaimer: maybe the result is real, or maybe it is just a toolchain mismatch.

## What this means operationally

The GB10 parity stack is a good example of a broader MegaCpp rule: version notes should describe the active compatibility surface, not just list packages.

A useful stack note answers questions like these:

- which parts were wheel-installed versus source-built?
- which CUDA generation was assumed by the runtime?
- which compiler generation built the local extensions?
- which packages were pinned to commits because released artifacts were insufficient?
- which device-specific constraints made that exact stack necessary?

That is the reason this parity work deserves its own write-up. It was not just a porting convenience for one benchmark harness. It was the environment contract that made later benchmark, serving, and kernel evidence comparable enough to trust.

## References

- PyTorch nightly wheels: https://download.pytorch.org/whl/nightly/cu132
- PyTorch installation guidance: https://pytorch.org/get-started/locally/
- vLLM project: https://github.com/vllm-project/vllm
- FlashInfer project: https://github.com/flashinfer-ai/flashinfer
- Mamba project: https://github.com/state-spaces/mamba
- [NAM56R launcher profile sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_launcher_profile_sample.py)
- [NAM56R CUDA graph launcher sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_cuda_graph_launcher_sample.sh)
- [NAM56R runtime patch surface sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_runtime_patch_surface_sample.py)
- Local evidence: `bench/modal_vllm_gb10stack.py`
- Local evidence: `bench/Dockerfile.vllm_gb10stack`
- Local evidence: `bench/vllm_repo/meta/dependencies.md`
- Local evidence: `cppmega/megatron/preflight_smem_check.py`
- Local evidence: `cppmega/tests/test_preflight_smem_check.py`
- Local evidence: `cppmega/README.md`
