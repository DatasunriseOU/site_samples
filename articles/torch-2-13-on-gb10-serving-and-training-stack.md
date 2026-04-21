---
title: "Torch 2.13 on GB10: the serving and training stack we actually chose"
description: "A public, evidence-based write-up of the stack choices around Torch 2.13, CUDA 13.2, GCC 15, GB10, and vLLM compatibility in the MegaCpp workflow."
date: "2026-04-19"
tags: ["PyTorch", "GB10", "vllm", "CUDA", "training", "serving"]
---

When people ask whether Torch 2.13 is “ready” on GB10, the useful answer is not a yes-or-no. The useful answer is: ready for which lane, with which compiler, with which CUDA toolchain, and with which serving engine constraints.

The MegaCpp evidence points to a very specific stack choice. For the GB10-shaped serving lane, we pinned Ubuntu 24.04, GCC 15, CUDA 13.2.51, Python 3.13, Torch 2.13 nightly for `cu132`, FlashInfer from source, `mamba_ssm` from source, and a source-built vLLM checkout with a local overlay. For the training lane, we kept the GB10 launcher aligned with that toolchain family but treated CUDA-graph capture and MoE shape behavior as the real compatibility boundary rather than pretending that “Torch 2.13 support” alone solved the whole stack.

That distinction matters because the serving problem and the training problem failed in different ways.

## The serving stack was a toolchain-compatibility problem first

The public bench bundle includes a dedicated image build for what it calls “full GB10 toolchain parity.” That image pins `gcc-15` and `g++-15`, installs CUDA Toolkit 13.2, moves Python to 3.13, and then installs Torch from the nightly `cu132` index with `torch>=2.13.0.dev0`. It also builds `causal_conv1d`, `mamba_ssm`, FlashInfer, and vLLM from source instead of relying on a mixed wheel stack. That is the most important signal in the whole record: MegaCpp did not treat GB10 as a place for opportunistic binary compatibility. It treated GB10 as a place where source-level rebuilds were the safe default. 

The same Dockerfile also sets `TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"`, which is a concise way to say that the stack was being kept compatible across Hopper, Blackwell, and GB10-class targets in one build recipe. That is not the same as claiming one wheel set magically works everywhere. It is the opposite: one recipe, multiple architectures, source rebuild where needed.

## Why GCC 15 and CUDA 13.2 were not optional details

The image comments are explicit about matching the GB10 toolchain: GCC 15 is described as matching GB10 `15.2.0`, and CUDA Toolkit 13.2.51 is described as matching `cuda-toolkit-13-2`. Those comments are not decoration. They explain why the image was built the hard way.

If you change two or three variables at once, it becomes impossible to tell whether a failure came from Torch, CUDA, the host compiler, or one of the source extensions under vLLM. The MegaCpp bench bundle avoids that ambiguity by aligning the major host-side toolchain choices up front. In practice, that means Torch 2.13 on GB10 should be discussed as “Torch 2.13 plus CUDA 13.2 plus GCC 15 plus source-built extensions,” not as an isolated library upgrade.

That is also why the stack uses Python 3.13 consistently in the serving image. Once the choice was made to rebuild the extension-heavy layer anyway, keeping the interpreter aligned with the image contract became less risky than depending on a looser prebuilt ecosystem.

## vLLM compatibility was the real forcing function

The best evidence that vLLM compatibility drove the stack shape is not a README claim. It is the amount of explicit patching around model registration and import surfaces.

The bench build recipe checks out vLLM at a fixed commit, overlays seventeen patched files, and then runs an import sanity test against `Qwen3_5ForCausalLMTextOnly`. The associated training-serving scripts say the patch exists because vLLM needed a text-only path and model-registry adjustments that survive subprocess re-imports. That is a stronger claim than “we tweaked config.” It means the serving lane was blocked at the model-executor layer, so the team chose to own a pinned vLLM overlay rather than wait for upstream drift to settle.

This is the correct engineering move when the incompatibility is narrow and reproducible. It keeps the stack legible. Torch 2.13 is not being blamed for every issue, and vLLM is not being treated as a black box either.

The portability notes around the image make the same point from another angle: GB10 is `aarch64`, Modal is `x86_64`, and the notes explicitly warn that wheels are cross-incompatible, so the Modal image rebuilds source dependencies with matching versions. That is exactly the kind of detail that gets lost in casual “works on my machine” summaries and exactly the kind of detail that should drive stack design.

## The training lane had a different boundary: CUDA graphs and dynamic shapes

The GB10 single-device training launcher in the MegaCpp tree is useful because it does not pretend the hard part is package installation. The script spends its explanatory effort on runtime behavior: stream-mismatch warning suppression, single-device distributed-optimizer avoidance, and most importantly the CUDA-graph boundary around dropless MoE.

The launcher says the dropless MoE path has dynamic shapes and cannot be fully captured in a CUDA graph, with the failure surfacing as a CPU-to-CUDA copy during graph capture inside the MoE all-to-all dispatcher. The resulting choice is to use Transformer Engine graph capture in attention-only scope, leaving the MoE MLP path uncaptured.

That is the exact kind of stack choice that matters more than broad version headlines. Torch 2.13 may be the right foundation for the lane, but the working training configuration still depends on where graph capture is scoped and which dynamic paths are left outside it.

In other words, the MegaCpp training evidence says: do not talk about GB10 readiness as if it were only a package-resolution question. The package set got the lane to runnable. The capture scope and runtime constraints made it stable.

## What the stack choice really was

From the public evidence, the cleanest summary is this:

- For GB10-shaped serving, MegaCpp chose a source-built stack around Torch 2.13 nightly `cu132`, CUDA 13.2.51, GCC 15, Python 3.13, FlashInfer from source, `mamba_ssm` from source, and pinned-overlay vLLM.
- For GB10-shaped training, MegaCpp stayed in the same toolchain family but treated CUDA-graph scope, MoE shape behavior, and single-device runtime details as first-class compatibility constraints.
- For cross-environment serving, MegaCpp explicitly avoided assuming wheel portability between `aarch64` GB10 targets and `x86_64` cloud builders.

That is a good stack because it is honest about where compatibility really lives. Not in one version number, but in the interface between compiler, CUDA toolkit, Torch ABI, extension builds, and runtime behavior.

## What I would not simplify away

There is a strong temptation to compress this story into “Torch 2.13 fixed GB10” or “vLLM is compatible now.” The bench and launcher history do not support that kind of simplification.

What they support is narrower and more useful. Torch 2.13 on CUDA 13.2 is a workable base for the MegaCpp GB10 lane when the compiler is kept at GCC 15, the extension-heavy packages are rebuilt from source, and vLLM is treated as a pinned integration surface rather than as an interchangeable wheel. Training then adds a second layer of constraints around graph capture and dynamic-shape MoE behavior.

That is not a marketing sentence. It is a reproducibility sentence. And for this topic, reproducibility is the only thing that matters.

## References

- [MegaCpp example index](../examples/megacpp/index.md)
- [GB10 repro bundle README](../examples/megacpp/gb10_repro_bundle/README.md)
- [GB10 repro walkthrough](../examples/megacpp/gb10_repro_bundle/README_walkthrough.md)
- [GB10 public claims](../examples/megacpp/gb10_repro_bundle/public_claims.md)
- [GB10 arch patch probe sample](../examples/megacpp/gb10_arch_patch_probe_sample.py)
- [GB10 driver signal vs runtime proof sample](../examples/megacpp/gb10_driver_signal_vs_runtime_proof_sample.py)
- [NAM56R launcher profile sample](../examples/megacpp/nam56r_launcher_profile_sample.py)
- [NAM56R CUDA graph launcher sample](../examples/megacpp/nam56r_cuda_graph_launcher_sample.sh)
- [NAM56R runtime patch surface sample](../examples/megacpp/nam56r_runtime_patch_surface_sample.py)
- [PyTorch nightly CUDA 13.2 index](https://download.pytorch.org/whl/nightly/cu132)
- [vLLM project repository](https://github.com/vllm-project/vllm)
