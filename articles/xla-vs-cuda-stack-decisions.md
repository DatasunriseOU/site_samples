---
title: "XLA vs CUDA: The Decision Matrix For Our Two Training Stacks"
description: "Where we keep one model definition, where the kernels diverge, what determinism we can give on each, how comms differ between NCCL and XLA collectives, and the operator surface that has to stay portable."
date: "2026-04-18"
tags: ["xla", "cuda", "tpu", "nvidia", "spmd", "nccl", "portability"]
---

We run the same model on two radically different stacks: TPUs through `torch_xla` / XLA SPMD, and NVIDIA GPUs through CUDA / NCCL / Transformer Engine / tensor-parallel training libraries. Keeping both alive is expensive and people reasonably ask why we do not pick one. The answer is that the paths give us different things, and the portability discipline is what makes the duplication sustainable. This post is the decision matrix we use day to day in a dual-stack training workflow: what we keep unified, what we let diverge, what determinism we can guarantee, and where the operator surface has to stay clean.

## Why two stacks at all

Recent TPU generations can be genuinely cheaper per training token for the right shapes, and the XLA compiler does things such as sparse offload, continuation-style fusion, and global memory scheduling that NCCL-based CUDA stacks do not replicate directly. Recent NVIDIA accelerators are where FP8 and modern FlashAttention-class kernels arrive first, and where architecture iteration often moves fastest because low-level kernel libraries evolve quickly. Picking only one cuts the cost-per-quality curve in half and locks the stack into a single vendor. We pay the portability tax on purpose. The interesting question is how much of that tax is necessary. Our answer: the model definition stays unified, the kernels live in two stacks, the communication pattern follows two worldviews, and the determinism story has to be written twice.

## What stays unified, what diverges

| Layer | TPU path | CUDA path | Shared? |
|---|---|---|---|
| Model definition | shared model runtime modules | same modules | yes |
| Config | `GPTConfig` | `GPTConfig` | yes |
| Sharding | XLA SPMD sharding annotations | `parallelize_module` plus tensor-parallel wrappers | no |
| Optimizer | XLA-safe AdamW variant with device-resident scalars | fused AdamW under `torch.compile` | shared math, two impls |
| Collectives | XLA-inserted HLO ops | NCCL launched from explicit Python scheduling | no |
| Attention | XLA Pallas flash | FA3 / FA4 CuTe | no |
| FP8 | not used | TE `fp8_autocast` per zone | no |
| Compile | `torch_xla.compile` per micro-step | regional `torch.compile` per block | no |

The boundary sits below `model.__call__`: anything inside the call is shared, anything below is path-specific. When we introduce a new operator (a new attention variant, a new norm, a new routing scheme) it goes into the shared modules with a pure-PyTorch reference, then picks up a Triton/Pallas kernel for CUDA and an XLA-blessed implementation for TPU. If either path needs bespoke Python plumbing inside `__call__` to make the op work, we refactor until it does not. We have broken this rule a few times and paid: the main MoE runtime module had an `.item()` on XLA that caused recompilations, `mamba3` had an f32 scan requirement that diverged between CUDA and XLA until we normalised the scan order, and the DSA indexer fused BF16 per-head accumulation lives only on CUDA because the XLA equivalent sits inside the compiler.

### Determinism is two stories

On TPU, XLA compiles a deterministic HLO graph under a fixed seed, a fixed mesh, and fixed runtime flags. Two runs with the same inputs and the same flags produce bit-identical outputs step to step, provided we do not hit the per-micro-step dual-compile window on step 0 when graph shape changes between gradient creation and accumulation. The gotchas are well understood: avoid warmup policies that change the graph, pin the TPU runtime version, avoid Python scalars that leak into the graph, and do not mix runtime barrier policies within a run.

On CUDA we get algorithmic determinism rather than bitwise determinism. Flash Attention is non-deterministic under `FLASH_ATTN_DETERMINISTIC=0`; NCCL reductions are order-sensitive at the least-significant bits; FP8 is a noisy format by design. For bit-exact runs we set `torch.use_deterministic_algorithms(True)` plus deterministic NCCL plus deterministic FA variants, and pay a throughput cost measured in tens of percent. We use this for golden regression tests, not for deployment training.

### Communication is two worldviews

NCCL collectives and XLA collectives are different worldviews. NCCL is explicit, synchronous by default, scheduled from Python, and overlapped with compute by deliberate hook placement and bucket batching. In practice that means real gradient-bucket state machines to make overlap happen. XLA collectives are implicit, scheduled by the compiler, and overlapped via continuation-style fusion plus global memory scheduling. We have repeatedly seen the same workload bottleneck solved with completely different knobs: on CUDA by retuning bucket sizes and overlap policy, on TPU by adjusting memory limits and collective-fusion behavior. Same problem, completely different tools.

## The XLA-safe optimiser pattern (and why it generalises)

Our XLA-safe AdamW exists because `torch.optim.AdamW` eventually materializes Python scalars inside `torch_xla.compile()`, which forces a graph break and then recompiles every step with different float constants. The fix is to replace those scalars with 0-D device tensors filled in place under an XLA-safe scalar policy:

```python
class XLAAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._bc2_sqrt_t = torch.tensor(1.0)
        self._step_size_t = torch.tensor(0.0)
        self._wd_scale_t = torch.tensor(1.0)

    @torch.no_grad()
    def step(self, closure=None):
        # compute bc1/bc2/step_size on host, then fill_() device tensors
        # and run the per-parameter update as pure tensor ops.
        ...
```

The pattern matters outside AdamW too: any scalar that varies per step has to become a 0-D tensor under `XLA_NO_SPECIAL_SCALARS=1` if you want a stable compile cache. That alone saved us roughly 48 minutes of compile-time cost per long run on the depth-52 hybrid preset.

## Operator-surface rules and the audit

Our portability discipline, learned the hard way: no `.item()` anywhere that runs under `torch_xla.compile()`; no Python-scalar graph-constants that change step to step; tuple/list mutations stay outside compiled regions; every op called from model code exists as a pure-PyTorch reference that runs correctly on CPU, XLA and CUDA; fused kernels sit behind feature flags and are tested for parity against the reference. We added a CI gate that walks the model on CPU under a fake XLA device wrapper and asserts no `.item()` calls and no varying Python-scalar graph constants. The gate has caught two regressions since January.

The other audit is the receipt cross-check. Every CUDA receipt and every TPU receipt records the same step-loss values for the first 100 steps on a small canary preset; if those diverge across stacks beyond the documented determinism budget, the receipt fails and the offending path is investigated. We have used this to catch one Mamba 3 scan-order divergence and one MTP head broadcast bug that only manifested on the CUDA Liger reduction path.

## What we kept and threw away

We kept one model module tree, two sharding implementations, two optimisers with shared math, two collective worldviews (NCCL explicit, XLA compiler-inserted), the rule that any new op ships with a pure-PyTorch reference, the CI gate against `.item()` in compiled regions, the receipt cross-check between paths, and the rule that path-specific code never lives inside `model.__call__`.

We threw away "one big graph" as the TPU compile story (per-micro-step is the only stable shape), `XLA_USE_BF16=1` (incompatible with current Pallas flash kernels), `XLA_USE_SPMD=1` as a user-facing toggle, `torch.compile(model)` whole-model on either path (per-block on CUDA, per-micro-step on TPU), bitwise determinism on the CUDA training path (we pay its cost only in regression gates), and any attempt to share collective code between NCCL and XLA. The throughline is short: pick the duplication carefully, keep the model itself unified, and let the paths be themselves below the call boundary.

## How a new feature lands across both paths

The lifecycle of a new feature is sequenced to keep the two paths from drifting. The feature lands first as a pure-PyTorch reference in the shared modules, with unit tests that run on CPU. Once the reference is correct, the CUDA-specific kernel lands behind a feature flag with parity tests against the reference. Once the CUDA path is stable, the TPU-specific implementation lands with its own parity tests. Only after both paths have parity-tested implementations does the feature get wired into a training preset.

The order matters. Landing the kernel first encourages design choices that are easy on one path and hard on the other; landing the reference first forces the design to be portable from day one. We have caught at least three feature designs at the reference stage that would have required Python-side variable shapes on the TPU path, and rewriting them at that stage was an order of magnitude cheaper than discovering it after the CUDA kernel had landed.

## What the receipt cross-check actually catches

The cross-check compares the first 100 step-loss values on a small canary preset between CUDA and TPU. The determinism budget is documented per feature (FA non-determinism, NCCL order sensitivity, FP8 noise) and the cross-check passes when the differences sit inside that budget. When it fails, the diagnosis path is short: which path changed, which feature flag flipped, and which receipt's stack line differs. Two failures in Q1 came out of this gate. The Mamba 3 scan-order divergence was caught when the canary's loss on TPU drifted from CUDA's by more than the f32 tolerance allowed; the fix was to normalise the scan order on both paths. The MTP head broadcast bug on the Liger reduction path was caught the same way; the fix was a `reduction="mean"` adjustment that only mattered on CUDA.

## What lives below the call boundary on each path

On the CUDA path, below `model.__call__` we have FSDP2 wrappers, tensor-parallel module replacements, the Transformer Engine bridge, attention-kernel selectors, Triton call sites, FP8 helpers, NCCL bucket plumbing, and Inductor regional compile setup. On the TPU path, below the same boundary we have XLA SPMD sharding annotations, sharding audits, runtime-flag loading, the per-micro-step `torch_xla.compile()` boundary, persistent cache hookup, and chip-memory helpers. The boundaries are different shapes; the rule that they live below the call is the same.

The model itself is the same file. That is what makes the duplication tractable. Without that rule, two stacks become two models, two models become two test suites, and two test suites become two products. The rule keeps us in one product.

## What we would change if we could only have one path

If we had to drop a path today we would keep CUDA, because FA4 and FP8 are where the architectural iteration speed is highest. We would lose the cost-per-token advantage of TPU and we would lose the SparseCore upside of v7. We would also lose the discipline that the cross-check enforces: with only one path there is no second source of truth for the canary loss, and a regression that happens to be path-specific becomes much harder to detect.

The fact that we keep both paths is not free, but the cost is bounded by the rules above and the value is real. Two paths is the right number for our workload at this size; one path would be cheaper and weaker, three paths would be too much surface to maintain. We do not see the calculus changing soon.

## References

- https://megacpp.com/blog/torch-xla-pjrt-reality.md
- https://megacpp.com/blog/xla-spmd-sharding-annotations.md
- https://megacpp.com/blog/xla-adamw-and-flags-on-tpu.md
- https://megacpp.com/blog/oom-on-v6e.md
- https://megacpp.com/blog/comms-cost-and-overlap.md
- https://megacpp.com/blog/tensor-parallel-and-sharding.md
- https://megacpp.com/blog/fsdp2-on-xla-tpu.md
- https://docs.pytorch.org/xla/master/runtime.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
