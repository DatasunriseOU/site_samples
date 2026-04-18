---
title: "Mamba 3 + Transformers: Why a Hybrid Stack Beats Pure Attention for C++"
description: "A deep dive into the Mamba 3 / Transformer hybrid MegaCpp trains on: layer interleaving, PsiV caching, the MIMO scan, and the register-split kernel that nearly shipped."
date: "2026-04-18"
tags: ["mamba3", "transformers", "hybrid", "state-space", "cpp", "mimo", "tilelang"]
---

Pure-attention Transformers are a bad fit for what MegaCpp does for a living: read long, deeply-nested C++ translation units, follow chains of `#include`s and template instantiations, and produce patches that are consistent with code the model saw 30k tokens ago. Quadratic attention over 32k-128k tokens of C++ is not a minor tax; it is the tax. At the same time, pure state-space models lose precision on the exact local lookups that matter for C++: matching a `}` to its `namespace`, picking the right overload from a set of seven, naming the type that came out of a CTAD deduction. The architecture we train is a hybrid: a Mamba 3 backbone interleaved with a minority of Transformer blocks, most of the compute an O(N) scan in state space and a handful of attention blocks handling sharp retrieval where they pay for themselves.

## Why this matters

A hybrid is not a compromise; it is a division of labor. A C++ token stream has two statistical regimes, a slow-varying context (type environment, namespace, macro vocabulary, file style) that rewards compression into a running state, and a sharp-lookup regime (exact signatures, overload resolution, cross-file references) that rewards content-addressable attention. Giving each regime the layer it is good at turns 32k-128k context from a quadratic line item into a linear one for most of the stack, with a small attention budget kept for the lookups that actually need it. On our v4 context-graph snippets (up to 64k tokens of Callers -> Target -> Callees, extracted with Tree-sitter), that division is what makes long-context training viable at all.

## 1. Why hybrid, specifically for C++

A pure Mamba stack loses sharp retrieval; a pure Transformer pays the full quadratic price to do both jobs. The frontier hybrid models (Nemotron Nano 3, Jamba, Zamba, Samba) converged on the same shape independently: attention is a minority, SSMs are the majority, and the exact ratio is tuned per target domain. For us, the domain is C++, and the sampling regime makes the hybrid argument sharper. Our training snippets are long, semi-structured, and full of cross-references packed specifically so local context sits near the target. The Mamba half carries scope and type state across the window; the attention half answers "what was the exact signature of `Buffer::append` we declared 12k tokens ago?".

## 2. Layer interleaving

Concretely, the backbone is Mamba-majority with attention sprinkled at specific depths. The exact spec lives in `nam_full_spec.py` on the training tree, but the shape is roughly seven Mamba 3 layers per attention layer, with attention biased toward the middle and later third of the network rather than the first blocks. Early layers embed tokens and accumulate local state; attention is wasted there. By the middle of the network, representations are rich enough that sharp retrieval earns its FLOPs.

The Mamba layers are configured through the author-pure seam in `config.py` in the Mamba-3 feature tree. The `AuthorMamba3Config` dataclass pins the contract: `d_model`, `d_state`, `expand`, `headdim`, `ngroups`, plus Mamba-3-specific `rope_fraction`, `dt_min`, `dt_max`, `dt_init_floor`, `A_floor`, `is_outproj_norm`, `is_mimo`, `mimo_rank`, and `chunk_size`. the authored Mamba3 config builder maps Megatron's config surface onto that dataclass and refuses overrides that do not satisfy `H = hidden_size * expand / head_dim`; silent mismatches on SSM head geometry corrupt gradients in ways that only show up hours into training.

The reference shape we run on the mid-sized hybrid backbone is `B=1, S=8192, H=16, G=1, N=64, P=64, R=4, chunk=16`, per a Mamba phase-one design note. `R=4` is the MIMO rank, and it is the reason the Mamba-3 layer in our stack is not just "Mamba 2 with RoPE".

### 2.1 Interleaving rules

- No attention in the first two blocks.
- One attention block roughly every seven Mamba layers thereafter.
- The final layer is always full-context attention (`L` in the window pattern).
- Half-context (`S`) attention slots are used to keep the KV cache bounded at 64K.

## 3. MIMO: why the rank exists

Mamba 2 already generalizes Mamba 1 by reframing the selective scan as a structured state-space duality with a matrix SSM. Mamba 3 MIMO goes one step further: instead of each head producing a single output channel per state update, each head produces `R` outputs from `R` input projections, sharing the same scan. In our config, `R=4`.

Mechanically, MIMO replaces what would be a rank-1 outer product in the state update with a rank-R one. The PsiV tensor that dominates the kernel is computed as

```python
psi_v[cs, r, p] = v[b, chunk_start + cs, h, p] * psi[h, r, p]
```

per a PSIV cache design note. It has an explicit `R` axis. For our reference shape that means `H=16, R=4, P=64` per head, so each state update carries four up-projections of `V` through the scan simultaneously. Arithmetic intensity goes up without widening the head or adding heads. On H200, `bwd_bwd` runs at ~479 arithmetic intensity against the ridge around 206; this is a compute-bound kernel, not a memory-bound one.

In product terms, MIMO is how we get attention-like representational width out of an O(N) kernel. For C++ tokens where one head needs to track both "what scope am I in" and "what type does this identifier bind to", a single scan with four channels behaves closer to four narrow scans than to one wider one, and the profile stays linear in sequence length.

## 4. The PsiV cache (P2)

The MIMO scan is the hot kernel, and PsiV appears five times in its loop body across `fwd`, `bwd_fwd`, and `bwd_bwd`, recomputed from scratch each time even though its two inputs are stable within a single forward-backward iteration. `psi` is a module parameter that does not change within a step, and `v` is a per-step activation. The P2 design in a PSIV cache design note is about killing two of those three recomputations by hoisting PsiV into an activation-checkpoint-style buffer. Dependency analysis is worth stating plainly:

- PsiV cannot be cached across training steps (`v` changes).
- PsiV cannot be cached across CUDA-graph replays (would hold a stale activation).
- PsiV is a perfectly well-defined intra-step tensor, and the same `v` flows through `fwd -> bwd_fwd -> bwd_bwd` inside one forward+backward iteration.

So the cache is an activation checkpoint, not a hash table. Storage is an extra output tensor on `mamba_mimo_fwd`, saved via `ctx.save_for_backward`, shape `(B, S, H, R, P)`, BF16, chunk-contiguous layout so the backward kernels' per-chunk access pattern stays coherent. For our reference shape at MBS=8 that is roughly 5.6 GiB extra per rank, inside the ~132 GiB H200 peak we already run at. The design refuses buffer pooling in v1 on purpose; a one-line `torch.empty()` is easier to reason about than a pool allocator, and we can always pool later.

Expected win is low single-digit percent TFLOP/s, modeled at roughly 1.5 to 2.3 percent. There is a real failure mode the design is ready to accept: if TileLang's scheduler is already CSE-ing `psi_v = v * psi` across its stages (hoisting the load and keeping the product in a register across back-to-back `ct.mma` calls), the runtime cost is already near zero and the cache saves nothing. That is why Phase A is a Python-level prototype that materializes PsiV before the kernel call; if Python does not move nsys numbers, the whole pursuit is archived.

## 5. Register split: the kernel that did not ship

The double-backward kernel `mamba_mimo_bwd_bwd` is the tall pole of the Mamba backward pass, roughly 2.1 seconds per step versus 1.2s for `mamba_mimo_fwd` and 1.0s for `mamba_mimo_bwd_fwd`. It runs at ~12.5 percent occupancy at 255 registers per thread (right at the H200 compiler ceiling, `65536 / (2 * 128) = 256`) with ~228 KiB shared memory, spilling on top.

The P3 design in a register-split design note proposed splitting `bwd_bwd` into two kernels connected by a gmem tensor. Pass 1 would run the state-reverse scan end-to-end, producing a `dstates_per_chunk` buffer (a few hundred MiB at our reference shape), free of PsiV and `qk_dot` fragments. Pass 2 would consume that buffer and re-derive PsiV and `qk_dot` for the chunk-local gradients. Pitch: both passes fit in ~130 registers, occupancy doubles from ~12.5 to ~25 percent, and a 1.3-1.8x throughput bump on a compute-bound kernel turns into a roughly 1 percent total TFLOP/s improvement.

The design did not survive a careful second read. Three blockers:

1. **The split point is not clean.** The loop-carried `dstates_frag` is updated at the end of each reverse chunk via `T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag, clear_accum=False)` and carried into the next reverse iteration. Pass 1 must therefore still hold `q_shared`, `dPhiO_shared`, and `dstates_frag` live, the exact fragments the split was supposed to drop. Separating them would cost an extra `[B, H, nchunks, chunk_size * R, P]` buffer several times bigger than `dstates_per_chunk`.
2. **The small SM_121 test system is not a viable correctness platform.** At the shapes we would need for a baseline-versus-split diff, even the upstream forward kernel fails to compile on its 99 KiB smem (`TMA desc init error 716` on small shapes, `Auto-tuning failed: No configuration successfully compiled` on the mid-sized reference shape at small). Any correctness validation has to run on H200.
3. **H200 time was not available for a week of kernel work.** On the day the audit ran, our H200 slot was not reachable, which meant no perf measurement, which meant no ROI validation for roughly a week of implementation.

Decision: do not ship P3. Pursue the PsiV cache instead, which removes three fragment tiles from `bwd_bwd`'s inner live set for two to three days of implementation rather than eight to twelve days for the full split, with the same expected 1-2 percent envelope. The cleanest engineering win on the Mamba kernel path came from rejecting a plausible-sounding optimization after reading the kernel line by line, not from implementing it.

## 6. Fork discipline

Running a Mamba 3 hybrid in production means running a lightly forked `mamba_ssm` in production. The reconciliation note `mamba_fork_canonical_2026_04_14.md` captures the state: both training systems are on upstream commit `31f3d7baba`, with three identical working-tree patches on `mamba3.py`, the public Mamba backward kernel sample, and the public Mamba varlen backward kernel sample, plus one real divergence on the public Mamba SISO combined sample where one system carries the PR #909 "cache `ctx.saved_tensors` for checkpoint compat" tweak. Accessing `ctx.saved_tensors` twice on a recomputed node raises under gradient checkpointing, so the one-line cache is a correctness patch, not a perf patch. The canonical working tree is the superset of both; an earlier commit-hash split in a brief turned out to be stale metadata.

The reason tiny-fork discipline matters is the hybrid itself. We apply patches on top of `mamba_ssm` via the public Mamba patch application helper and, in the future, the public PSIV patch application helper: env-gated, idempotent, lock-synchronized so rank-0 patches while the other ranks wait on a sentinel. If the Mamba half drifts silently between hosts, the only symptom is divergent loss curves three days into a run. The patch pipeline, the md5 reconciliation, and the author-pure `AuthorMamba3Config` all exist to make that failure mode loud instead of silent.

## Hybrid components at a glance

| Component | Role | Cost | Enabled on |
|---|---|---|---|
| Mamba 3 MIMO scan | long-context state, O(N) | register-heavy, compute-bound | long-range specialists |
| Mamba 3 SISO | short-context state, O(N), cheap | low | short-range specialists |
| Grouped-query attention | exact retrieval, O(N^2) | KV cache | every specialist |
| DSA sparse attention | retrieval at 64K+ context | sparse tile compute | longest context phases |
| MTP head (training only) | auxiliary prediction for specdec | small | every specialist |

## What we kept and what we threw away

Kept: the hybrid Mamba-3-plus-GQA backbone with attention as a minority, MIMO at `R=4` for long-range specialists, the P1 TMA + warp-specialization flag flips (gated), the TMA 3D-to-2D layout fix (correctness verified on the small SM_121 box), the PsiV cache as the next perf pass, the env-gated idempotent patcher, and fork md5 reconciliation. Thrown away: the P3 register split of `bwd_bwd` (rejected on the three blockers above), a pure-attention backbone at 64K (O(N^2) is the tax), a pure-Mamba backbone (loses sharp retrieval), buffer pooling in v1 of the PsiV cache (defer until measured need), and trying to use the small SM_121 box as a correctness platform for kernel splits (incompatible smem budget). Every kept item has a design doc. Every thrown-away item has a paragraph in one.

## References

- mamba3_mimo_p1_notes.md
- mamba3_mimo_p2_psiv_cache_design.md
- mamba3_mimo_p3_register_split_design.md
- mamba_fork_canonical_2026_04_14.md
- mamba_integration_log.md
- v4_architecture.md
- megacpp_model.md
- CURRENT_STATE.md
