# Kernel Samples

This directory collects public-safe MegaCpp POC kernel examples that mirror the
real CUDA helper contracts used across training and inference.

What is here:
- `attention_validity_prefix_sample.py`: normalize token-prefix and slot-prefix validity metadata.
- `dense_fa4_execute_proof_sample.py`: rollout-side execute-proof builder for dense FA4.
- `dense_fa4_kvcache_decode_sample.py`: bounded KV-cache decode contract for FA4 intent.
- `doc_window_mask_sample.py`: dense document and local-window mask builder shared by attention wrappers.
- `fused_bias_dropout_add_sample.py`: compiled bias + dropout + residual-add block-boundary helper.
- `fused_linear_cross_entropy_chunked_sample.py`: bounded-memory chunked loss path.
- `fused_mla_projection_sample.py`: MLA down-projection + RMSNorm + up-projection recompute helper.
- `fused_relu_squared_sample.py`: relu2 expert activation surface.
- `fused_residual_add_rms_norm_sample.py`: block-boundary residual + norm fusion contract.
- `fused_rope_qk_sample.py`: fused rotary application for Q/K attention ingress.
- `mhc_fused_static_sample.py`: static 4-stream fused mixing surface for mHC fast paths.
- `moba_block_sparse_decode_sample.py`: requested-vs-actual backend receipt for blockized sparse decode.
- `triton_row_gather_sample.py`: single-tensor sparse row gather staging.
- `triton_row_gather_pair_sample.py`: paired K/V sparse row gather staging.

How these fit into the model/runtime:
- Attention ingress: RoPE and KV staging helpers reduce launch count before the
  actual attention backend runs.
- Attention mask contract: validity-prefix and document-window builders keep
  dense, sparse, and local wrappers aligned on which tokens are real.
- Attention backend rollout: dense FA4 helpers keep rollout and decode claims
  tied to a real bounded contract.
- Blockized sparse decode: the MoBA-style resolver records when a FLASH-style
  sparse request had to execute as Triton instead.
- Block boundaries: residual + RMSNorm fusion and bias-dropout-add cut repeated
  elementwise traffic.
- Expert compute: relu2 is one of the cheap expert activation surfaces used in
  specialist-model experiments.
- Multi-stream fusion: mHC static mixing keeps hyper-connection style branch
  merges from degenerating into many tiny kernels.
- Loss path: chunked fused CE prevents logits memory from exploding on large
  vocab runs.
- MLA experiments: the fused projection sample shows how the MLA path trades
  recomputation for lower activation residency.

What is deliberately not here:
- Private paths, hostnames, and machine-specific glue.
- Full vendor kernels copied verbatim when a smaller public contract shows the
  real behavior more clearly.

Primary MegaCpp POC source modules:
- `attention_validity.py`
- `triton_kernels.py`
- `fused_bias_dropout_add.py`
- `fused_relu2.py`
- `fused_residual.py`
- `fused_mla_projection.py`
- `mhc_fused.py`
- `flash_attention.py`
- `kernels.py`
