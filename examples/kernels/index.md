# Kernel Index

## Attention kernels
- `attention_validity_prefix_sample.py`: normalize valid-token and valid-slot prefix metadata.
- `doc_window_mask_sample.py`: build one dense mask that respects causality, document ids, valid prefixes, and local windows.
- `exact_token_sparse_telemetry_sample.py`: record when exact-token sparse stays sparse vs reroutes to full attention.
- `fused_rope_qk_sample.py`: rotates Q and K under one ingress contract before attention.
- `dense_fa4_execute_proof_sample.py`: proves the dense FA4 path with a real CUDA smoke command.
- `dense_fa4_kvcache_decode_sample.py`: bounded append-style decode helper for FA4 intent.

## Sparse staging kernels
- `triton_row_gather_sample.py`: gathers one contiguous `(N, H, D)` tensor by row index.
- `triton_row_gather_pair_sample.py`: gathers matching K/V rows under one contract.
- `moba_block_sparse_decode_sample.py`: resolves requested vs actual backend for blockized sparse decode.
- `clustered_sparse_three_phase_sample.py`: separates importance scoring, union selection, and clustered sparse attention.
- `exact_mask_contract_cache_sample.py`: cache only static exact-mask semantics, not per-batch tensors.

## Block-boundary kernels
- `fused_bias_dropout_add_sample.py`: compiled bias + dropout + residual-add helper.
- `fused_residual_add_rms_norm_sample.py`: fuses residual update and RMSNorm checks.

## Expert and activation kernels
- `fused_relu_squared_sample.py`: relu2 activation with fused forward/backward helpers.

## Multi-stream fusion
- `mhc_fused_static_sample.py`: static 4-stream merge surface for fused mHC paths.

## MLA helpers
- `fused_mla_projection_sample.py`: recompute-heavy MLA projection helper.

## Loss kernels
- `fused_linear_cross_entropy_chunked_sample.py`: chunked linear + CE path that bounds logits memory.

Where they are used:
- Transformer and hybrid blocks call the residual/norm, bias-dropout-add, and RoPE surfaces.
- Sparse and clustered attention packers call the row-gather helpers.
- Clustered TPU sparse kernels use the three-phase split and exact-mask cache key helpers.
- FA4 rollout and decode helpers sit between backend selection and live attention calls.
- Dense and local attention wrappers share the validity and document/window mask builders.
- Expert layers use relu2-style activation helpers.
- mHC layers use the fused static mix helper when the runtime is on the narrow 4-stream fast path.
- MLA experiments use the fused projection helper while the runtime wiring is validated.
- Final-token loss computation can route through chunked CE when full logits are too expensive.
