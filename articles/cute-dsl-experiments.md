---
title: "Our honest experience with CuTe DSL"
description: "What we tried to build with CuTe DSL, where it held up, where it lost to alternatives, and the chunks we rewrote back to Triton or kept in CUDA."
date: "2026-04-18"
tags: ["cute-dsl", "cutlass", "kernels", "gpu"]
---

We spent a meaningful chunk of Q1 2026 learning CuTe DSL on H200 and Blackwell-class hardware. Some of that work ended up in the tree, most of it did not, and a couple of experiments taught us enough to change how we think about kernel stacks. This post is the honest retrospective, because we kept getting asked "so is CuTe DSL worth it?" and the answer really depends on what you are trying to do.

## Why this matters

Kernel stack choices are sticky. Once a critical kernel is in CuTe DSL, your container builds, your test machines, your CI lanes, and your debugging muscle memory all bend around it. CuTe DSL is also the path NVIDIA itself ships its newest reference kernels through (FlashAttention-4 most visibly), so any team running on Hopper or Blackwell ends up with at least a transitive dependency on it whether they meant to or not. The question is no longer whether to learn it; it is which kernels to actually own and which to consume.

The other reason it matters: the per-kernel engineering cost is materially higher than Triton, and the `torch.compile` integration story still has sharp edges. Misjudging where on the cost curve a particular kernel lives is how you spend a quarter porting something that was already fast.

## 1. Context and scope

CuTe DSL, for our purposes, is the Python surface shipped as `nvidia-cutlass-dsl` (we used 4.4.1 and 4.4.2) that lets you write CUTLASS-style kernels in Python, JIT-compile them, and call them without an ahead-of-time CUDA build. The headline attractions were the FlashAttention-4 integration (`flash-attn-4` 4.0.0b4, a pure-Python CuTe DSL kernel set) and the ability to hand-write WGMMA-backed GEMMs for shapes that cuBLAS and Triton do not corner well.

Our two concrete workstreams:

1. Running FA4 on H200 and B200 via the CuTe DSL kernel, both as a drop-in attention backend and as a nested kernel inside `flex_attention(kernel_options={"BACKEND": "FLASH"})`.
2. Exploring whether hand-written CuTe DSL GEMMs could replace or augment our TileLang fused scan for the Mamba MIMO backward-backward kernel on H200 and GB10.

The rest of the post is what actually happened in those two tracks.

## 2. What the learning curve actually costs

Everyone says CuTe DSL has a learning curve. It does, and that curve is steeper than people who have already absorbed CUTLASS concepts tend to notice. For an engineer who is competent in Triton, comfortable in CUDA, and has never written CUTLASS kernels, the ramp looks roughly like this:

- Week 1: getting a trivial smem-tiled GEMM to compile and run, with correct results at one shape. Most of the time is spent reading CUTLASS documentation to translate "tensor of shape" into a CuTe layout with the right strides and swizzles, and understanding which atoms your architecture exposes (for H200 sm_90a that means `warpgroup.MmaF16BF16Op` for WGMMA).
- Week 2: getting the same kernel to run at a second and third shape without regressing, which forces you to understand stage/pipeline parameters and smem lifetime. We got `warpgroup.MmaF16BF16Op` correctly emitting WGMMA on H200 with a single-GEMM result in the low single-digit microseconds before we were confident we understood what we had written.
- Week 3: trying to fuse multiple GEMMs in one launch. The 3-GEMM fusion we built kept K in smem across GEMMs and accumulated into an LKQ tile, exact bf16 match against the reference. It ran at roughly half the wall time of three separate WGMMA launches: a clean ~1.78x fusion benefit that we can actually measure.
- Week 4+: trying to fuse ten or more GEMMs with elementwise ops and reductions in between. This is where the curve goes vertical. The manual smem-lifetime management grows nonlinearly, and everything we tried past the 3-GEMM fusion started eating working days without delivering measurable speedups.

Concretely: the 3-GEMM fused CuTe DSL kernel on H200 matched cuBLAS on the same chain. It did not beat it. At small shapes, launch overhead is comparable to compute time, so the fusion eliminates launch overhead but has no headroom to do more. That was a useful result, because it told us exactly where the CuTe DSL payoff curve starts and stops.

## 3. What we kept

1. The FA4 CuTe DSL kernel lane itself, as a first-class backend in the training stack. `docker/modal-base/build.sh` pulls `flash-attn-4==4.0.0b4` and the matching `nvidia-cutlass-dsl` wheels from pinned package sources, and the Modal images ship with the full dependency chain (`apache-tvm-ffi`, `torch-c-dlpack-ext`, `quack-kernels`, `cuda-python`, `cuda-bindings`, `cuda-pathfinder`). The kernels JIT-compile on first call, which adds a one-time warmup we account for in our throughput measurements.
2. The specific FlexAttention + FA4 wiring, where the backend is selected via `flex_attention(kernel_options={"BACKEND": "FLASH"})` and wrapped in `torch.compiler.disable()` at the call site. That single wrap is what made the preset lanes compile and run end-to-end without Inductor trying to lower a CuTe DSL kernel into its outer compile graph.
3. The empirical GB10 / sm_121a capability matrix. When we went in, the working assumption was that CuTe DSL was "fully blocked" on sm_121a. Our agents' actual probes found that the bf16 path works; fp16 and fp8 paths have sharper restrictions that we documented rather than guessed at. That pivot kept us from burning weeks on the wrong assumption.
4. The 3-GEMM fused WGMMA proof on H200. We do not ship that particular fused GEMM in training, but the code stayed around in `experiments/` as a reference for anyone who needs to cut a specific shape that cuBLAS does not serve well. It is the cleanest example in our tree of a hand-written WGMMA kernel that is correct at bf16 and inside the ballpark of cuBLAS on a real shape.

## 4. What we rewrote back

1. The fused Mamba3 MIMO bwd_bwd kernel stayed in TileLang. We explored porting it to CuTe DSL and hit three structural walls: smem-layout round-tripping (WGMMA reads swizzled smem; `StMatrix` writes plain layouts, so each inter-GEMM intermediate wants an extra smem-to-smem copy, roughly a few microseconds of overhead per hop), manual stage/pipeline management for ten-plus GEMMs, and the absence of the automatic multi-stage pipelining that TileLang provides for free.
2. A custom CuTe DSL fused softcap+causal kernel for H200. We built a draft that worked at one shape and lost decisively to FA4 with softcap on every other shape. We deleted it.
3. A second-attempt CuTe DSL grouped GEMM for MoE dispatch. cuBLAS grouped GEMMs caught up before we finished, and our Python-level dispatch logic was already eating most of the headroom.

## 5. Sharp edges we hit

- FA4 CuTe DSL standalone is not compatible with `torch.compile` today. Without custom-op registration (upstream RFC filed, not yet landed at the time of the fix), the only supported compile path is `flex_attention(BACKEND="FLASH")` via PyTorch Inductor. `dynamic=True` is not supported with the FLASH backend. We document that loudly in our backend matrix.
- CuTe DSL standalone with our pinned layer presets produced numerically divergent backward passes (gnorm in the millions after a handful of steps) on H200. The "passing" FA4 lanes that we originally thought were CuTe DSL were in fact Triton FlexAttention, because `moba_flex_backend` was unset and defaulted to Triton. Once we forced `moba_flex_backend="flash"` on the same presets, the CuTe DSL FLASH kernel diverged. We filed upstream and verified the MHA (non-GQA) path separately to rule out a GQA-only issue.
- Version pinning is sharper than for Triton or CUDA. `nvidia-cutlass-dsl` 4.4.1 vs 4.4.2, `flash-attn-4` 4.0.0b4, and the `apache-tvm-ffi` / `torch-c-dlpack-ext` pair have to line up exactly. Mixing wheels from two container builds gave us import-time failures that masqueraded as missing symbols. We put the exact versions in `docker/modal-base/build.sh` and stopped trying to upgrade any of them independently.
- JIT compile warmup is noisy. The first call into any CuTe DSL kernel takes noticeably longer than a steady-state call. If you benchmark without a warmup, or if your throughput plot includes step 0, you will get misleading numbers.
- sm_121a (GB10) has real, not imagined, restrictions. The bf16 path works; fp16 and fp8 have sharp edges. We rebuilt the capability matrix from actual probe runs rather than trusting the "fully blocked" priors we inherited.

## 6. Comparisons we actually measured

For the one configuration where we have clean head-to-head numbers, a 3-GEMM chain on H200 at small bf16 shapes:

| Variant | Wall time | Notes |
| --- | --- | --- |
| CuTe DSL fused 3-GEMM, one launch | ~36 microseconds | Our hand-written kernel |
| CuTe DSL, three separate WGMMA launches | ~64 microseconds | Same body, no fusion |
| `torch.bmm` / cuBLAS, three launches | ~36 microseconds | Stock baseline |

That is the reality check. Fusion buys ~1.78x by killing launch overhead. Matching cuBLAS at a single fused shape is not a win; it is a tie. The interesting fusion wins start appearing only when you cross ten-plus GEMMs with non-trivial intermediates, which is exactly where CuTe DSL's manual smem management collapses unless you have a large engineering budget.

Numbers we did not chase to a clean comparison: CuTe DSL vs TileLang full the Mamba MIMO backward-backward kernel on B200. We stopped early because the structural gap was obvious after the 3-GEMM fusion result and the missing multi-stage pipelining analysis. "n/a" is the honest answer; we did not do the 600-plus line port.

## 7. Where CuTe DSL earns its keep

After all that, we still think CuTe DSL is the right tool in two situations:

1. When you need to write a new atom or a GEMM kernel for a shape or dtype combination that cuBLAS and Triton cannot corner well, and you need Hopper or Blackwell features (WGMMA, TMA, swizzled smem) that Triton does not expose cleanly. For these, Python-level CuTe DSL is meaningfully more productive than writing CUTLASS in C++.
2. When you are consuming a third-party CuTe DSL kernel like FA4 that was written by people who have already paid the learning curve cost. In that case you are essentially using it as a prebuilt kernel, and the job is wiring it into your compile graph without triggering inductor lowering of the kernel body itself.

Where it does not earn its keep, for us:

1. Replacing a mature TileLang or Triton fused kernel that is already at or near speed-of-light for your shapes. The automatic fusion and pipelining you get for free in those compilers is worth more than the hand-written flexibility CuTe DSL gives you.
2. Writing simple elementwise-plus-reduction kernels, where Triton or Liger is strictly easier to author, test, and debug.

## What we kept and what we threw away

We kept the FA4 CuTe DSL backend, the FlexAttention+FA4 wiring with `torch.compiler.disable()` at the call site, the empirical GB10 capability matrix, the 3-GEMM WGMMA proof as a reference, and the strict version pinning of the CuTe DSL stack inside our container build. We kept the rule that benchmarks include a JIT warmup and that step 0 is excluded from any throughput plot involving a CuTe DSL kernel.

We threw away the hand-written CuTe DSL fused softcap+causal kernel, the second-attempt grouped GEMM for MoE dispatch, the assumption that CuTe DSL would beat TileLang on the Mamba3 backward, and the inherited belief that GB10 was "fully blocked" for CuTe DSL. We also threw away the previous mental model of "Triton for everything GPU-side, CUDA when Triton can't." CuTe DSL pushed us toward a more honest three-tier view: Triton for memory-bound and elementwise-heavy work, TileLang for large fused kernels that need automatic pipelining and deep fusion, and CuTe DSL for hand-written atoms, GEMMs, and integrating third-party kernels like FA4. CUDA C++ is still the escape hatch for things none of the above cover.

The short answer to "is CuTe DSL worth it?" is: for specific shapes and for consuming FA4, yes; as a general replacement for Triton or TileLang in an established training codebase, no. The learning curve is real, the per-kernel engineering cost is higher than in Triton or TileLang, and the ecosystem (especially around `torch.compile` integration) still has sharp edges that you will have to work around. We kept what earned its place and rewrote the rest back.

```python
# FA4 via CuTe DSL, called from a FlexAttention backend path
import torch
from cppmega.kernels.fa4_cute import fa4_forward

@torch.compiler.disable()
def attn(q, k, v, block_mask):
    # step 0 is excluded from throughput plots; warmup must run first
    return fa4_forward(q, k, v, block_mask=block_mask, causal=True)
```

## References

- Public benchmark notes and code receipts in `site_samples`
- build.sh
- Dockerfile
- modal_train.py
- sparse_attention.py
- triton_kernels.py
- fused_mla_rope.py
- kernels.py
- dsa_backend_options_2026-03-07.md
- H200_STACK_SETUP.md
