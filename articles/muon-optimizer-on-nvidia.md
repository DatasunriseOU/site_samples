---
title: "Muon on Hopper and Blackwell: The NVIDIA Lane of the MegaCpp Optimizer Stack"
description: "How Muon, MuonClip, and the QK-clip family get from a single-file POC step into a production AdamW-coexistent optimizer path for the MegaCpp ensemble on H200 and GB10."
date: "2026-04-18"
tags: ["muon", "optimizer", "newton-schulz", "h200", "blackwell", "qk-clip"]
---

Muon is the optimizer we keep circling back to whenever we want a cheaper training run at the same loss. On the NVIDIA lane of the MegaCpp ensemble, "cheaper" means Hopper H200 for the heavy dense baselines and Blackwell GB10 for the small-cluster development loop. The single-file Muon reference works on paper, but once the model has mixed fused-QKV projections, deep hyper-connection stacks, and rectangular MoE experts, there is a fairly long list of knobs you have to get right before it stops diverging. This post walks through the optimizer path as it currently stands in the POC and what ships to the MegaCpp production codebase.

## Why MegaCpp cares about this

Our scaling recipe is an ensemble of specialist SLMs rather than one big model, so every training dollar we save at specialist-scale compounds. Muon gives orthogonalized updates that, per the Keller Jordan writeup [Muon: An optimizer for the hidden layers of neural networks - Keller Jordan, 2024], empirically beat AdamW on compute-matched loss for the 2D linear layers that dominate our attention and MoE blocks. The catch is that Muon's Newton-Schulz iteration normalizes gradient magnitudes, and at a depth-52 hybrid preset with 4-stream hyper-connections we routinely blew up on the H200 dense baseline until we fixed three things at once: split-QKV orthogonalization, Moonlight-style LR scaling, and a weight-space QK clip after every step [Kimi K2 - Moonshot AI, arXiv:2507.20534]. Everything downstream in this post is us engineering that triangle so it does not need an on-call every time we change a recipe.

## What we built in the POC

The public Muon implementation in MegaCpp deliberately tries to be boring. The core step is `_muon_step_fused_impl`: momentum accumulation, Polar Express orthogonalization, optional variance reduction, cautious weight decay, and the parameter update are all in one compiled graph. The variance reduction uses a factored second-moment buffer (per-row when the matrix is tall, per-column otherwise) so we get AdamW-like variance damping on top of the orthogonalized update without materialising a full second moment. Cautious weight decay is gated on agreement between the update and the current parameter sign, not the pre-orthogonalization raw gradient sign. That is not the convention you will see in most reference code; we tried the raw-gradient gate once, it regressed a deep dense receipt back to immediate NaNs, and we left a comment next to the gate telling the next person to stop trying to "fix" it.

Orthogonalization itself is Polar Express [Polar Express Sign Method for Orthogonalization - Amsel et al., arXiv:2505.16932] rather than the classic quintic Newton-Schulz. The coefficients in `polar_express_coeffs` are precomputed for five iterations with a small safety factor and a cushion of two. The iteration runs in bf16 unconditionally. fp32 Polar Express on TPU triggered compile-time HBM OOM with stacked all-reduce buffers on the compiled-optimizer path, and once we confirmed the bf16 path is numerically fine after a 10-to-20-step warmup on NVIDIA too, we stopped maintaining the fp32 branch. The `1.02 * ||X||` pre-normalization is how we keep Polar Express strictly inside its convergence envelope even when gradients spike.

The one piece that is easy to miss is `qkv_split_sizes`. Muon's Newton-Schulz / Polar Express cross-contaminates Q, K, and V subspaces if we orthogonalize a fused QKV matrix as a whole. The POC mirrors NVIDIA NeMo's SplitQkvOrthogonalizedOptimizer pattern: when a parameter carries a `_qkv_split_sizes` tag, the orthogonalization splits along the output dim, runs Polar Express per Q/K/V slice, and concatenates. That single change turned a depth-52 hybrid preset from "NaN at step 4 universally" into a loss around 4.2 at step 50 in our ablations, and it is the reason split-QKV is not a toggle in production.

Three distributed variants coexist in the public Muon module. The plain `Muon` class is the reference optimizer for single-device debugging. `DistMuon` uses batched stacking plus vectorized `reduce_scatter_tensor`/`all_gather_into_tensor` borrowed from modded-nanogpt so the orthogonalization runs on one rank per shard and broadcasts. `FSDP2Muon` is the one we actually run on H200: it consumes FSDP2 `DTensor` local shards directly, infers `Shard(0)` bounds to re-map grads onto the local shape, and rescales `qkv_split_sizes` to the local row count so the split-QKV logic keeps working after the parameter has been sharded along the head dimension. All three paths share the fused kernel through `muon_step_fused`, which is lazy-compiled; the compile decision is deferred to first call so that a late environment flag set by a test or container entrypoint is respected. The default on NVIDIA containers today is uncompiled - shape-varying stacked grads were triggering dynamo recompiles past the limit and stalling deep-dense runs for minutes per step.

Per-parameter learning-rate scaling lives in `_adjust_lr`. Two modes match the PyTorch `torch.optim.Muon` API: Keller's `sqrt(max(1, A/B))` correction for tall matrices, and Moonlight's `0.2 * sqrt(max(A, B))` correction that equalizes update RMS across all shapes [Moonlight - Moonshot AI, arXiv:2502.16982]. The Moonlight mode is the one our MoE presets need. With Keller's rule, expert `w1` of shape `(D, h)` and expert `w2` of shape `(h, D)` get a `sqrt(D/h)` ratio of effective LR, roughly a 1.7x asymmetry between paired up/down projections at our expert aspect ratio. Moonlight's rule eliminates that asymmetry and wants a lower base LR, which is why our MoE receipts ship the `match_rms_adamw` flag together with the lower `matrix_lr`.

The public AdamW companion module is the coresident reference. It is the same fused-step idiom - a single compiled graph with 0-D CPU scalar tensors for hyperparameters so hyperparameter changes do not trigger recompiles - and it is the fallback whenever a parameter is 1D, an embedding table, the LM head, or otherwise excluded by Muon's shape contract. In practice every training run uses both: Muon for 2D hidden linears, AdamW for embeddings, biases, norms, and the MTP head.

The QK-clip story spans the public clip helper and the optimizer policy. The clip helper is the activation-space clip: before softmax it scales the query tensor by `min(1, threshold / (||q||_max * ||k||_max / sqrt(d)))`. It uses `amax` on tensors instead of scalar reductions to avoid a host sync on XLA. `muon_clip.py` is the weight-space clip: `MuonClipState.record` accumulates a Cauchy-Schwarz upper bound on per-head max logits during the forward pass in `O(T)` rather than the true `O(T^2)` max, `muon_clip_qk` walks attention modules after the optimizer step, and when a head exceeds `tau` it rescales the Q rows for that head by `sqrt(gamma_h)` and the K rows for the serving KV head by `sqrt(min_h(gamma_h))`. The conservative per-KV-head minimum is specifically for GQA; it is the correct thing for a KV head that serves multiple query heads, and it is the only modification that survived every ablation on the H200 dense baseline.

## How it lands in MegaCpp

The MegaCpp production codebase is our stack on top of Megatron-Core. The optimizer story lands there in three pieces.

First, the core Muon step, the split-QKV orthogonalization, and the Moonlight LR mode are lifted as-is. The POC module `muon.py` is the source of truth for the step contract; the MegaCpp recipe layer ingests the same hyperparameters through the public Megatron-args sample and translates them into Megatron's optimizer config. We do not re-derive Polar Express coefficients in the MegaCpp production codebase.

Second, the distributed Muon variant we keep in production is the FSDP2Muon interface, not DistMuon. Megatron's distributed optimizer handles the reduce-scatter side, so the part of `muon.py` we actually run under Megatron-Core is the shard-aware local step. This is the hard integration: Megatron ships gradients on `DTensor`-like parameters, and the local step has to accept the same kind of `Shard(0)` inputs `FSDP2Muon` already handled in the POC. That is why `_local_grad_for_fsdp2_param` and `_scale_qkv_split_sizes_for_local_rows` are written defensively around FSDP2 shard shapes - they are the production-facing surface. `DistMuon` stays in the POC as the single-source-of-truth reference we can diff against when a Megatron integration regresses.

Third, MuonClip is a feature-flagged hook in the MegaCpp production codebase, not a loss on the default path. The clip threshold `tau=100` matches the Kimi K2 paper default; `0` disables the clip and is the setting for presets that already use a smaller depth or for runs we specifically want to burn to see whether logits still explode. The forward-side recording goes into the attention module; the post-step rescale is a post-optimizer hook mirroring the wiring in the POC training loop. The activation-space clip from `qk_clip.py` stays as a fallback for experiments that cannot afford the per-step weight rescale and as the XLA-side variant because it does not need a host sync.

What does not land: raw Nesterov-only Muon without variance reduction, the unsplit-QKV path, and the fp32 Polar Express branch. The first two are strictly worse on our deep-dense receipts; the third has no numerical benefit on NVIDIA once the safety factor in the input normalization is in place.

## Ablations and what we kept

The interesting part of the history is a narrow one. It sits in our CHANGELOG entries for the period around the H200 bring-up and the Kimi K2 follow-through.

The deep dense baseline went NaN at step 1 through step 6 under every Muon configuration we tried that was not "split-QKV plus hybrid architecture". Lowering the matrix LR did not help; a 20x lower LR still NaN'd at step 2. A matched AdamW run at the same base LR completed 20 steps cleanly. That was the evidence that identified Muon's orthogonalized magnitude as the cause of the forward-pass activation explosion in the presence of deep hyper-connections, not any LR or init pathology. The fix that actually worked was split-QKV orthogonalization plus hybrid Transformer + structured-state layer interleaving, which reached loss around 4.2 at step 50 on the stable receipt.

On top of that we added MuonClip. Without MuonClip, long runs with Muon drifted toward rising max attention logits that, left unchecked, pushed us back into the earlier NaN regime. The `muon_clip/max_logit`, `muon_clip/n_clipped`, and `muon_clip/n_total` wandb metrics are the leading indicator; when the number of clipped heads per step creeps up over a training window, something upstream has changed.

Muon weight decay went through its own small ablation. The POC originally decayed weights even when update and parameter disagreed in sign. The aligned-gate mode (the current behavior) cuts the nominal weight decay by about an order of magnitude on some layers, so we had to re-tune `matrix_lr` and `weight_decay` together when we turned it on; the payoff is that 15T-token-class recipes stop drifting in a way that only shows up at the very long tail of training.

On GB10 (Blackwell consumer variant, SM121a) the story is different: we are memory-bandwidth-bound at 273 GB/s, so the optimizer wall-clock fraction barely moves between Muon and AdamW. We still run Muon on GB10 for numerical parity with H200, but the compiled-step kernel is the cheaper decision, not the difference maker. The GB10 changelog has the NVFP4 bring-up notes; for the optimizer the only GB10-specific change was lifting the inductor `is_big_gpu()` check so the 48-SM board would autotune the fused step at all.

## Production checklist

- Muon is the default only for 2D hidden linears. Embeddings, LM head, biases, and norms stay on the public AdamW path. No exceptions.

Muon vs AdamW routing in the MegaCpp stack:

| Parameter group           | Optimizer | State bytes/param | Notes                         |
|---------------------------|-----------|-------------------|-------------------------------|
| 2D hidden linears, experts| Muon      | 2 (bf16 momentum) | Polar Express in bf16         |
| Fused QKV projections     | Muon      | 2                 | split-QKV orthogonalization   |
| Embeddings, LM head       | AdamW     | 8 (fp32 m+v)      | no exceptions                 |
| Biases, RMSNorm scales    | AdamW     | 8                 | excluded from Muon's contract |
| MTP head                  | AdamW     | 8                 | public AdamW path             |

Split-QKV orthogonalization, sketched:

```text
# muon.py: split-QKV orthogonalization (sketch)
def orthogonalize(param, grad):
    sizes = getattr(param, "_qkv_split_sizes", None)
    if sizes is None:
        return polar_express(grad)
    q, k, v = grad.split(sizes, dim=0)
    return torch.cat([polar_express(q),
                      polar_express(k),
                      polar_express(v)], dim=0)
```

- Fused QKV parameters must carry `_qkv_split_sizes`. If a recipe introduces a new attention projection layout, the split metadata is part of the parameter construction, not of the optimizer.
- Moonlight-style `match_rms_adamw` LR scaling is the MoE default, with a correspondingly lower `matrix_lr`. Keller's `original` mode is documented but gated behind an explicit opt-in.
- MuonClip is on for every deep-dense or deep-hybrid preset, with `tau=100`. `muon_clip/max_logit` is a training-health signal, not a debug aid.
- Distributed Muon under Megatron-Core uses the FSDP2-style local-shard contract. Shard row counts flow into `_scale_qkv_split_sizes_for_local_rows` so the split-QKV invariant survives sharding.
- Compile of the Muon step is opt-in on NVIDIA until the shape-varying stacked-grad recompile limit is raised. TPU/XLA keeps the compiled path.
- bf16 Polar Express everywhere. The safety factor in the input normalization is not tunable without a new receipt.
- Post-step hook order is fixed: optimizer step, then MuonClip, then LR scheduler bookkeeping. Reordering breaks the Cauchy-Schwarz recording contract.

## References

- `muon.py`, `muon_clip.py`, `qk_clip.py`, `adamw.py` in the POC.
- `megatron_args.py`, `mla_shared.py` in the MegaCpp production codebase.
- [Muon: An optimizer for the hidden layers of neural networks - Keller Jordan, 2024].
- [Polar Express Sign Method for Orthogonalization - Amsel, Persson, Musco, Gower, arXiv:2505.16932].
- [Moonlight: LR scaling for Muon - Moonshot AI, arXiv:2502.16982].
- [Kimi K2 (MuonClip) - Moonshot AI, arXiv:2507.20534].
