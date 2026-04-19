---
title: "The adapter stack: how LoRA, QLoRA, and hot-swap compose MegaCpp specialists"
description: "The LoRA, QLoRA, DoRA, VeRA, and DyLoRA family behind MegaCpp specialists, the registry and lifecycle that turn adapters into versioned releases, the hot-swap runtime, and the inference-facing API they power."
date: "2026-04-18"
tags: ["lora", "qlora", "adapters", "peft", "inference", "specialists"]
---

MegaCpp ships as one base model plus a fleet of small specialists - a project-specific adapter for the codebase you're editing, a language-specific one for the dialect of C++, an online adapter that learns from compiler feedback. The thing that makes that fleet tractable to build, version, ship, and serve is the adapter stack behind the product. This post is a tour of how that stack actually works: what compose differs from merge, what the registry guarantees, how hot-swap works at serving time, and what crosses into the serving API.

## Why MegaCpp cares about this

Economics. A full fine-tune of the C++ specialist for one customer codebase costs as much as the original training run. A LoRA adapter at rank 4-8 over MLP and Mamba projections is a few tens of MB of trainable weights, fits on a single modern GPU, trains in coffee-break time, and at serving costs zero inference overhead because it merges into the base. QLoRA pushes the base into NF4 4-bit so the same GPU can adapt a much larger checkpoint. DyLoRA gives us one adapter that works across multiple ranks. VeRA shares random projection bases across layers and only learns diagonal scalings, crashing per-adapter parameter count by another order of magnitude. We need every variant because the answer to "which recipe is right" is a matrix of (specialist size, base size, target hardware), not a single column.

## How the stack is structured

The load-bearing adapter module defines the core LoRA variants and the injection walker. It defines `LoRALinear`, `DoRALinear`, `VeRALinear`, `QLoRALinear`, `DyLoRALinear`, and the Megatron-tensor-parallel-aware variant `MegatronLoRALinear`, plus the `inject_lora()` walker that swaps `nn.Linear` modules in place by name suffix. The default target set (`DEFAULT_TARGETS = MLP_TARGETS | MAMBA_TARGETS`) covers `c_gate_fc`, `c_fc`, `c_proj`, `in_proj`, `out_proj` - MLP and Mamba projections only. Q/K/V are deliberately excluded by default because adapting them invalidates the KV cache during online adaptation. `ATTENTION_TARGETS = {"c_qkv"}` exists for full-rank fine-tuning where cache-validity isn't a concern. `inject_lora()` detects whether a layer is already wrapped in a Megatron column- or row-parallel TP wrapper and constructs the right adapter variant: column-parallel adapters shard `B` along the output dim and replicate `A`; row-parallel adapters shard `A` along the input dim and replicate `B`. There are two TP backends: XLA/TPU via `apply_lora_spmd_sharding()`, and CUDA DTensor via `apply_lora_dtensor_tp()`. `merge_lora()` folds adapter deltas into the base weights for zero-overhead inference; `unmerge_lora()` reverses the operation by snapshotting the delta at merge time so subsequent A/B mutations don't corrupt the base.

The 4-bit quantization layer is the canonical home of the key primitives: `NF4_LEVELS` and `FP4_LEVELS` codebooks, the `NF4Tensor` dataclass, `quantize_to_nf4` / `dequantize_nf4`, double-quantization of the absmax scales themselves down to FP8 E4M3, and `QuantizedLinear` for inference-only 4-bit layers. The `QLoRALinear` symbol is re-exported from the core adapter module for backward compat but the primitives live here. NF4 places its 16 levels at quantiles of N(0,1), which minimizes expected quantization error for normally-distributed neural-network weights; FP4 places them uniformly in [-1, 1]. We use NF4 by default because the empirical accuracy gap is real and the implementation cost is the same.

The compose and merge layers are the two halves of the arithmetic. Composition (`compose_adapters`, `add_adapters`, `subtract_adapters`, `interpolate_adapters`) is task-vector arithmetic in the spirit of [Editing Models with Task Arithmetic - Ilharco et al., ICLR 2023]. The non-obvious part is that for paired `lora_A`/`lora_B` keys the composition has to happen in delta space - `(w1*B1 + w2*B2) @ (w1*A1 + w2*A2) != w1*(B1@A1) + w2*(B2@A2)` - so we form the weighted sum of `DeltaW = B @ A`, then do a truncated SVD back to the target rank to recover new `A`, `B` factors. Standalone keys such as DoRA magnitude get a simple weighted average. There are explicit guards against mixing variants (`use_dora`, `use_vera`, `use_qlora`, `use_dylora`) because the forward passes are not interchangeable, and against mixing different `alpha/rank` scalings because the scaling is baked into the forward, not the stored weights. Merging is the higher-level operation that turns multiple adapters into one: `merge_adapters_average()` does exact delta-space averaging, `merge_adapters_ties()` is [TIES-Merging - Yadav et al., NeurIPS 2023], and `merge_adapters_dare()` is [DARE - Yu et al., ICML 2024].

The adapter lifecycle layer handles `create_adapter`, `save_adapter`, `load_adapter`, `merge_adapter`, plus the canonical artifact format. The artifact carries `state_dict`, `lora_meta` (rank, alpha, variant, targets, QLoRA quantization metadata, DyLoRA rank-growth metadata), and a schema version. A fail-closed gate rejects PEFT-style "LoRA over a separately quantized base" artifacts masquerading as native QLoRA: it checks `use_qlora` against the presence of `qlora_quant_metadata` and refuses any artifact that claims native QLoRA without carrying it. This prevents the most common cross-toolchain bug - a generic LoRA artifact that loads cleanly and produces silently wrong outputs.

The adapter registry is the global index. Each version gets an auto-incremented per-project tag, a parent ID for lineage, a status (`draft -> active -> archived -> deprecated`), a metrics blob, tags, training config, and a compatibility checker that verifies recorded targets, rank, and variant match the loaded model. The registry is a single JSON index alongside per-version adapter directories, with a report exporter and an A/B comparison helper.

The runtime hot-swap surface has three layers: `AdapterBank` (low-level, holds named state dicts and copies weights into LoRA params), `MultiAdapterManager` (wraps a base model, handles a CPU cache, lazy-injects LoRA on first register, exposes `activate`, `activate_merged([names], weights=[...])`, `deactivate`), and `AdapterRouter` (per request: parse a routing prefix, activate, generate, cache last-used). The bank validates rank compatibility on every activation, validates compose support against runtime metadata for blended activations, and zeroes LoRA params on deactivate so the base model returns clean. `BatchAdapterRouter` handles the case where batch items want different adapters.

The TIES and DARE merge paths are intentionally not delta-space exact - they sparsify on the raw `A`/`B` factors, which is what their original formulation does. We documented this distinction explicitly because merging `(B1, A1)` and `(B2, A2)` after sparsifying is mathematically a different operation from sparsifying `(B@A)` and re-factoring. For exact delta-space merging the user must pick `merge_adapters_average()`.

The online adaptation layer is the inference-time adaptation loop. `OnlineAdapter` keeps a `ReplayBuffer` of `(input_ids, reward)` pairs from the live serving stream (reward = compiler exit code or unit-test pass rate) and runs periodic LoRA-only steps every N samples. The "MLP and Mamba only" target set is enforced because the KV cache is shared across requests and adapting Q/K/V invalidates it. `DCDAdapter` is the no-compiler fallback: it distills the model's behavior on a long-context "teacher" view into a shorter "student" view, giving project-specific adaptation when the only signal is the project's own source tree. We run both for the C++ specialist because they answer different questions.

## How it lands in MegaCpp

The MegaCpp serving stack consumes adapters through the canonical artifact contract from the adapter-management layer. `save_adapter_artifact` and `load_adapter_artifact` are the only entry points production touches. The format is stable across schema versions (currently `ADAPTER_META_SCHEMA_VERSION = 2`, `ADAPTER_ARTIFACT_SCHEMA_VERSION = 1`), and the production loader runs the same fail-closed validation. This lets us train an adapter in one environment and serve it in MegaCpp without a re-export step.

Production hot-swap is conceptually `MultiAdapterManager` minus the experiment-tracking: a per-replica CPU-pinned adapter cache with LRU eviction and one CUDA-side activation slot that copies weights into LoRA params on swap. Per-request routing carries an explicit adapter id in the envelope (the gateway sets it; we don't parse from the prompt in production). Blended activations are supported but rare - the typical request hits one adapter, served at zero inference overhead because hot adapters are merged into the base and unmerged-and-restored on swap.

QLoRA in production is what serves the "large base on small hardware" lane: the base weights are NF4-packed in CPU memory, materialized to bf16 GPU on demand, with double-quantized FP8 absmax scales. The quantization metadata travels with the adapter artifact so the loader can verify base-model compatibility before activation. The Megatron-TP variants (`MegatronLoRALinear`, `apply_lora_dtensor_tp`) are what handle the sharded-base serving case where the model is split across multiple GPUs.

Deliberately not lifted: VeRA's shared-buffer dedup is training-stack internals; MegaCpp only sees the canonical artifact. DyLoRA rank-growing is training-time. The compose and merge utilities live in the training stack and produce artifacts; production only consumes them.

## Ablations and what we kept

LoRA family ablations were less about loss and more about maintenance cost. The canonical recipe - rank 4-8 on `MLP_TARGETS | MAMBA_TARGETS`, alpha = 2 * rank, NF4 base for QLoRA, AdamW on LoRA params only - survived everything. DoRA's magnitude-vector parameterization gives small literature wins; we kept the implementation but it is not the default because the divergent forward pass and the unmerge logic (which had a P0 inversion bug we fixed) outweighed the consistent-but-small win at our shape.

VeRA is in the tree for the case where adapter footprint matters more than per-task accuracy - shared `A_shared`, `B_shared` mean every per-layer adapter is two diagonals of size `rank` rather than two `rank x dim` matrices. The shared-buffer plumbing settled after a sequence of P1 fixes (dedup on save, rebind on resume, EP-aware atomic preflight); the engineering notes around `_rebind_vera_shared_buffers()` and the `assign=True` load path document the painful version.

DyLoRA serves multiple ranks at inference (rank-truncating A/B at request time). It composes badly with TIES/DARE, which is why those merge functions reject DyLoRA inputs. The unmerge bug was instructive: original `unmerge()` recomputed the delta from current A/B instead of snapshotting at merge time, so any A/B mutation between merge and unmerge corrupted the base weights. Snapshot in `merge()`, use the snapshot in `unmerge()` - now the canonical pattern across variants.

QLoRA is the most-used variant in serving by request count. The compression ratio (roughly 4x for base weights, including the FP8-quantized scales) is what makes single-GPU serving of the large base model possible. We did not see meaningful accuracy degradation at rank 8 or above; below rank 8 the quantization noise starts to bite.

Compose-vs-merge took longest to get right. Composition is the live in-RAM op: `0.8 * A + 0.3 * B - 0.2 * C` to make a new specialist on the fly. Merging is offline batch: take a list of adapters and produce one committed artifact. Variant-compatibility check, alpha/rank scaling check, and delta-space SVD are the three things we got wrong before we got them right. Online adaptation went through one simplification: the original NTP/DCD/HYBRID split collapsed HYBRID to "run both" rather than alternating, because the alternation schedule was a hyperparameter we never picked correctly.

## Production checklist

- The default LoRA target set must remain `MLP_TARGETS | MAMBA_TARGETS`. Any expansion to `ATTENTION_TARGETS` invalidates the KV cache and breaks online adaptation; require an explicit opt-in.
- The adapter artifact contract is the single boundary between research and serving. Both sides must validate `lora_meta` against the runtime model contract before activation.
- The native-QLoRA gate that rejects PEFT-style artifacts without `qlora_quant_metadata` is fail-closed and must stay that way; a silent acceptance produces wrong outputs that no eval will catch.
- Compose validates `(use_dora, use_vera, use_qlora, use_dylora)` signatures and `alpha/rank` scaling. Both validations must run before any tensor arithmetic; do not relax them for a "convenience" path.
- DyLoRA, VeRA, and the merged-with-base case must not be passed to TIES/DARE merging - those algorithms operate on raw factors and the compositions are mathematically incorrect.
- VeRA shared buffers are stored at the model root (`_vera_shared_A`, `_vera_shared_B`) and re-bound on every full-checkpoint load with `assign=True`. The rebinding hook must remain in the load path.
- Hot-swap activation must zero the LoRA params on deactivate so the base model is bit-exact recoverable. Do not skip this on the "deactivate to base" fast path.
- Online adaptation only updates LoRA params on the MLP and Mamba targets. Adapting Q/K/V invalidates the shared KV cache and silently corrupts cross-request inference.
- Per-replica adapter cache is bounded; the eviction policy is LRU; the on-swap path snapshots the merged delta so unmerge is bit-exact.
- The compose path uses delta-space SVD to recover A/B factors at the original rank. Do not "optimize" by averaging the factors directly.

## Adapter family snapshot

| Variant | Where it wins | Trade-off |
|---------|---------------|-----------|
| LoRA | default adapter rank 8-32 | cheapest, baseline quality |
| QLoRA | memory-bound fine-tune | extra dequant on forward |
| DoRA | small-rank tasks needing expressiveness | ~10-15% slower |
| VeRA | many adapters per host | shared random basis, per-task vectors |
| DyLoRA | rank unknown at train time | samples rank per step |

```python
# hot-swap at serve time, adapter IDs resolved by registry
from adapters import registry, hot_swap
adapter = registry.resolve(project="llvm-project", dialect="cpp20")
hot_swap(model, adapter, merge=False)
```

## References

- LoRA and QLoRA implementation patterns
- adapter composition and merge arithmetic
- hot-swap adapter serving patterns
- [LoRA: Low-Rank Adaptation of Large Language Models - Hu et al., ICLR 2022]
- [QLoRA: Efficient Finetuning of Quantized LLMs - Dettmers et al., NeurIPS 2023]
- [DoRA: Weight-Decomposed Low-Rank Adaptation - Liu et al., ICML 2024]
- [VeRA: Vector-based Random Matrix Adaptation - Kopiczko et al., ICLR 2024]
- [DyLoRA: Parameter-Efficient Tuning at Multiple Ranks - Valipour et al., EACL 2023]
- [Editing Models with Task Arithmetic - Ilharco et al., ICLR 2023]
- [TIES-Merging - Yadav et al., NeurIPS 2023]
- [DARE: Drop And REscale - Yu et al., ICML 2024]
- [Deep Context Distillation with LoRA Knowledge Modules - 2024-2025]
