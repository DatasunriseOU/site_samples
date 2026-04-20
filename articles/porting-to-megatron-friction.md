---
title: "Porting To Megatron-Core Is Harder Than It Looks"
description: "Why lifting a hybrid attention/Mamba/MoE stack into Megatron-Core is a multi-adapter exercise: base config mapping, layer specs, mixer protocol, and the bridge layer that makes them line up."
date: "2026-04-18"
tags: ["megatron-core", "transformer-engine", "hybrid", "moe", "mamba", "CUDA"]
---

Porting our hybrid stack into NVIDIA Megatron-Core is the single largest framework integration we have done, and it is genuinely hard in ways that are not obvious until the second week. The reason is not that Megatron is badly written - it is not - but that Megatron is a shape, and our models are a slightly different shape. This post is a concrete walk through the adapters we had to build, what they actually paper over, and which pieces are real gaps we could not close.

## Why MegaCpp cares about this

Transformer Engine is where the best kernels live on Hopper and Blackwell today: `TELayerNormColumnParallelLinear`, `TERowParallelLinear`, `TEDotProductAttention`, fused RoPE, fused masked softmax, userbuffer TP comm-overlap, and a working FP8 recipe via `fp8_autocast`. We want all of it for the NVIDIA training lane. Megatron-Core also already has a production-tested pipeline-parallel scheduler, expert-parallel all-to-all with grouped GEMM, distributed optimizer with overlapped reduce-scatter, and a DDP that buckets aware of TP and EP process groups. Writing our own versions of these is possible - we have done pieces of it - but it is years of work to catch up on kernel fusion alone.

So the question was never "should we use Megatron?" - it was "how much of our architecture survives intact when we do?". The hybrid stack is the hard part: dense attention blocks, MLA blocks, DSA blocks, Mamba/M2RNN blocks, and MoE blocks, composed with a mHC residual stream and MTP heads on top. Megatron's `TransformerConfig` was designed around one reasonably regular layer shape repeated `num_layers` times with optional MoE and optional Mamba. Our layer pattern is irregular on purpose. Everything downstream of that mismatch is friction.

## What we built in MegaCpp

The integration lives in a family of adapter modules in the experimentation stack. They are independent on purpose: each one adapts exactly one Megatron surface so features can be turned on and off without rewriting the rest of the model.

The bridge layer is the spine. It lazy-imports Megatron runtime surfaces, guards every call site behind availability checks so non-Megatron environments never trip over missing CUDA dependencies, and owns the lifecycle for Megatron's process groups. The centerpiece is a config-mapping function that translates our flat training configuration into Megatron's transformer configuration. The mapping is dense and painful. A minimal version looks like this:

```python
# bridge-layer config mapping (abbreviated)
def get_megatron_config(gpt_config, **overrides):
    hidden = gpt_config.n_embd
    head_dim = hidden // gpt_config.n_head
    # GQA: Megatron's num_query_groups is our n_kv_head, with MHA encoded as None
    num_query_groups = None if gpt_config.n_kv_head == gpt_config.n_head else gpt_config.n_kv_head
    if gpt_config.activation == "swiglu":
        ffn = int(8 * hidden / 3)
        ffn = ((ffn + 7) // 8) * 8            # match our SwiGLU round_up_8
    else:
        ffn = 4 * hidden
    cfg = dict(
        num_layers=gpt_config.n_layer,
        hidden_size=hidden,
        num_attention_heads=gpt_config.n_head,
        num_query_groups=num_query_groups,
        kv_channels=head_dim,
        ffn_hidden_size=ffn,
        gated_linear_unit=gpt_config.activation == "swiglu",
        normalization="RMSNorm",
        add_bias_linear=False, add_qkv_bias=False,
        bf16=True, params_dtype=torch.bfloat16,
    )
    cfg.update(overrides)
    return TransformerConfig(**cfg)
```

The comments in the real file are almost longer than the code. Every line encodes a decision: SwiGLU wants `gated_linear_unit=True` plus `activation_func=silu` because Megatron folds the gate into the linear; `relu2` has no Megatron-native equivalent and falls back to `relu` with a warning; `rope_theta` is attached after construction because the base `TransformerConfig` does not accept `rotary_base` as a dataclass field (only the MLA subclass does); `attn_softcap` has no equivalent field at all and has to be applied externally or routed through TE's `attn_logit_softcapping`.

MoE is where the bridge is leakiest. `_validate_supported_moe_bridge_subset` explicitly documents which of our routing variants have no `TransformerConfig` analogue: `null_rho` sink tokens, `expert_choice` dual-top-k, group-topk group caps, and the loss-free load-balancing `max_bias` clamp. We do not fail on these any more - Megatron DDP only reads `TransformerConfig` for gradient bucket sizing, not routing semantics - but we warn, because any future code path that starts interpreting those fields will get the wrong answer. Loss-free LB is supported as a mapped subset: we set `moe_router_load_balancing_type="none"`, zero the aux-loss coefficient, enable `moe_router_enable_expert_bias=True`, and pass our bias update rate through as `moe_router_bias_update_rate`.

The transformer-block adapter makes a single Megatron `TransformerLayer` look like our native block interface. Three annoyances live in this wrapper. First, layout: Megatron is sequence-first `(T, B, D)`, we are batch-first `(B, T, D)`, so the wrapper transposes on entry and exit. Second, RoPE: Megatron's training path does not accept external cosine/sine rotary kwargs on the generic training surface, so we build a rotary-embedding module at initialization time and cache the positional embedding per `(seq_len, device)` on the first forward. Third, features: the wrapper has to ignore or translate several side-channel arguments such as local-window hints, KV-cache state, document IDs, attention metadata, augmented-residual kwargs, and extra experimental controls. That is fine for plain attention layers, but it is exactly why we cannot simply replace the whole block list with generic Megatron block instances.

The Mamba-style mixer adapter is the most honest illustration of what an adapter actually is. Megatron's `MambaLayer` expects a mixer object that follows a specific protocol: construction receives configuration plus pipeline metadata, forward receives sequence-first tensors plus inference context, and it returns `(output, bias)` so the outer layer's fused bias-dropout-add path can consume it. Our recurrent mixer takes a different constructor, expects batch-first tensors, and returns a single tensor. The adapter reconciles all of it:

```python
# recurrent-mixer adapter (abbreviated)
class MegaCppM2RNNMixer(nn.Module):
    def __init__(self, config, *, d_model, submodules=None,
                 layer_number=None, pg_collection=None,
                 pp_layer_offset=0, inner_config=None):
        super().__init__()
        from .m2rnn import M2RNNLayer
        cfg = inner_config or config
        if not hasattr(cfg, "n_embd"):
            object.__setattr__(cfg, "n_embd", d_model)
        self._inner = M2RNNLayer(cfg, layer_idx=layer_number or 0, tp_degree=1)

    def forward(self, hidden_states, *, inference_context=None,
                packed_seq_params=None):
        # [s,b,h] -> [b,s,h] -> inner -> [s,b,h]; bias = None for mamba_bda
        y = self._inner(hidden_states.transpose(0, 1).contiguous())
        return y.transpose(0, 1).contiguous(), None
```

Wrapping the mixer, not the whole layer, buys us the things Megatron is actually good at: participation in Megatron DDP's `overlap_grad_reduce` and `overlap_param_gather`, the fused `bias_dropout_add` / norm / residual plumbing, correct `[s,b,h]` pipelining, and integration with the `MambaLayerSubmodules` spec. It does not buy us an inference cache - we explicitly raise on `mamba_state_shapes_per_request` because our matrix state `(B, N, K, V)` is not the `(conv, ssm)` tuple Megatron's generation path expects. That cache wiring is outstanding work.

the Megatron MoE integration module is a much thinner adapter. It is a compatibility layer over `megatron.core.transformer.moe.moe_utils`: `permute`, `unpermute`, and `group_limited_topk`. We use these primitives inside our own router where it makes sense and keep our routing logic intact. The only real work is normalizing our routing inputs - indices `[num_tokens, top_k]` plus aligned weights vs. a dense `[num_tokens, num_experts]` mask with probabilities - into whichever layout the upstream helper expects. The alternative was porting our entire router onto Megatron's `MoELayer`, which would have meant giving up null_rho and expert_choice; not worth it.

`megatron_optimizer.py` is the biggest of the adapters (roughly 1.8k lines) and the one that does real work rather than shape-shifting. It implements an overlapped distributed optimizer - `GradBucket` plus `OverlappedGradReducer` plus a wrapper that runs Muon on matrix params and our XLA-safe AdamW on everything else on each rank's local shard, then all-gathers updated params back. It deliberately mirrors patterns from `megatron/core/distributed/param_and_grad_buffer.py` and `distributed_data_parallel.py`: the `get_default_bucket_size_mb(dp_world_size)` helper reproduces Megatron's `max(40M, 1M * dp_world_size)` parameter-count rule translated into bytes, `pad_buckets_for_high_nccl_busbw` copies Megatron's NVLink/NVSwitch alignment hint, and `grad_reduce_in_fp32` matches Megatron's `DDPConfig` knob for the up-cast-then-reduce path. We use the private `torch.distributed._coalescing_manager` - with a graceful fall-back when the symbol moves between torch releases - to batch per-bucket reduce-scatters into a single NCCL kernel dispatch.

## How it lands in production

In the current MegaCpp architecture the shape is inverted. Megatron is the framework and the custom pieces slot into Megatron-native specs. The public tensor-parallel Mamba mixer sample, recurrent spec sample, authored Mamba spec sample, and DSA-aware attention spec are all `ModuleSpec`-shaped objects that Megatron's `build_module` instantiates directly. The public linear-CE shim and indexer-fusion shim are module-level monkey-patches that swap Megatron primitives for fused variants. The recurrent path runs through fused chunk and Triton kernels rather than the pure-PyTorch recurrence the early adapter wrapped.

The lift-as-is set is small and boring: the bridge itself (config mapping, parallel-state lifecycle), the MoE primitive wrappers, and the distributed optimizer's bucket sizing and coalescing rules. Rewrites: the block wrapper is replaced by production `ModuleSpec` objects that are fully TE-native from end to end. Drops: the pure Python recurrent mixer becomes the Triton fused chunk kernel; the PyTorch AdamW path gives way to TE's fused optimizers when FP8 is on. Feature flags such as Megatron-block, Megatron-recurrent, Megatron-DDP, and the MoE dispatcher switch (`alltoall` vs `allgather`) survive into production as operator-visible toggles.

## Ablations and what we kept

The bridge surface, with the leaks called out:

| Our field / feature | Megatron `TransformerConfig` analogue | Status |
|---------------------|---------------------------------------|--------|
| `n_kv_head` | `num_query_groups` (None for MHA) | mapped |
| `swiglu` | `gated_linear_unit=True` + `activation_func=silu` | mapped |
| `relu2` | `relu` | warn-and-fallback |
| `rope_theta` | post-construction attribute on RoPE module | mapped |
| `attn_softcap` | TE `attn_logit_softcapping` (when TE owns DPA) | partial |
| Loss-free LB | `moe_router_load_balancing_type="none"` + bias hooks | mapped |
| `null_rho` sink tokens | none | warn-only |
| `expert_choice` dual-top-k | none | warn-only |
| Group-topk group caps | none | warn-only |
| `tp_comm_overlap=True` | requires `te.initialize_ub(...)` | off by default |

The integration change notes record the messy middle. Our first iteration of the transformer-block adapter set `tp_comm_overlap=True` by default and was 17x slower than our own block - we reverted it and documented the reason in-line. The `fp8_autocast` scope audit found a mismatch where Megatron enters the FP8 region at the transformer block level and we were entering it per-linear; we now match the block-level scope, and Mamba is explicitly excluded from FP8 because TE's FP8 GEMM paths do not compose with our SSD recurrence.

The SP norm-grad all-reduce fix in the tensor-parallel adapter was another silent-correctness bug: under `--sequence_parallel --megatron_tp`, norm parameters see only a shard of the sequence on each rank, and without an explicit all-reduce of their grads the training diverges. We now install a hook that mirrors Megatron's own final gradient-reduction path. The loss-free LB global-sync patch in the MoE path was the same kind of bug in a different location.

Things that survived: the bridge itself as a thin CUDA-only module, the Mamba mixer adapter pattern (wrap the mixer, not the layer), the distributed optimizer's bucket math, and the process-group lifecycle. Things we dropped: our first attempt at mapping `null_rho` onto Megatron's capacity-factor fields, because the semantics did not match, and the idea that `TransformerConfig` could carry all of our routing fields directly.

## Production checklist

- Keep the bridge CUDA-only. The TPU/XLA path must never import `megatron.core`.
- Gate every Megatron call site behind `is_megatron_available()` / `is_megatron_enabled()` so CPU unit tests stay green.
- When a field has no `TransformerConfig` analogue, warn loudly and document the workaround - do not drop it silently.
- Drive DP/TP/PP/EP sizes through `init_megatron_parallel_state()`; do not hand-roll process groups in the model code.
- When wrapping a mixer for `MambaLayerSubmodules`, return `(output, None)` so `mamba_bda` does not crash.
- Set `persist_layer_norm=False` whenever `WrappedTorchNorm` is the chosen norm wrapper.
- For MoE, route through `alltoall` when EP > 1 and `allgather` otherwise.
- For FP8, enter `fp8_autocast` at block scope, not per-linear; keep Mamba and M²RNN out of the FP8 region.
- Validate loss-free LB mapping on a canary before every release: `moe_router_load_balancing_type="none"`, `moe_aux_loss_coeff=0`, expert-bias enabled, bias update rate matches our config.

## References

- https://github.com/NVIDIA/Megatron-LM
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://github.com/NVIDIA/TransformerEngine
- https://github.com/state-spaces/mamba
- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/cppmega/megatron/tensor-parallel-and-sharding__mamba3_tp_partition_sizes__v1.py
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md
