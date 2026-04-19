# Kernel Index

## Attention kernels
- `fused_rope_qk_sample.py`: rotates Q and K under one ingress contract before attention.
- `dense_fa4_execute_proof_sample.py`: proves the dense FA4 path with a real CUDA smoke command.
- `dense_fa4_kvcache_decode_sample.py`: bounded append-style decode helper for FA4 intent.

## Sparse staging kernels
- `triton_row_gather_sample.py`: gathers one contiguous `(N, H, D)` tensor by row index.
- `triton_row_gather_pair_sample.py`: gathers matching K/V rows under one contract.

## Block-boundary kernels
- `fused_residual_add_rms_norm_sample.py`: fuses residual update and RMSNorm checks.

## Expert and activation kernels
- `fused_relu_squared_sample.py`: relu2 activation with fused forward/backward helpers.

## MLA helpers
- `fused_mla_projection_sample.py`: recompute-heavy MLA projection helper.

## Loss kernels
- `fused_linear_cross_entropy_chunked_sample.py`: chunked linear + CE path that bounds logits memory.

Where they are used:
- Transformer and hybrid blocks call the residual/norm and RoPE surfaces.
- Sparse and clustered attention packers call the row-gather helpers.
- FA4 rollout and decode helpers sit between backend selection and live attention calls.
- Expert layers use relu2-style activation helpers.
- MLA experiments use the fused projection helper while the runtime wiring is validated.
- Final-token loss computation can route through chunked CE when full logits are too expensive.
