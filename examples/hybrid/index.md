# Hybrid Index

Use these samples when you want the model-assembly view rather than one kernel.

- Start with `hybrid_pattern_sample.py` for the block menu.
- Read `attention_backend_variants_sample.py` when you care about sparse-attention backend choices.
- Read `mhc_stream_residual_sample.py` when you care about cross-layer mixing and residual behavior.

Main source surfaces for this directory:
- `unified_superblock.py` from the MegaCpp POC repo
- `gpt.py` from the MegaCpp POC repo
- `adapter.py` from the native-sparse experiment pack
- `README.md` from the native-sparse experiment pack
