# Hybrid Index

Use these samples when you want the model-assembly view rather than one kernel.

- Start with `hybrid_pattern_sample.py` for the block menu.
- Read `attention_backend_variants_sample.py` when you care about sparse-attention backend choices.
- Read `mhc_stream_residual_sample.py` when you care about cross-layer mixing and residual behavior.
- Read `mod_routing_surface_sample.py` for the MoD router contract.
- Read `adapter_lora_runtime_sample.py` for runtime adapter metadata and safe composition.
- Read `deltanet_hyperconnection_sample.py` for DeltaNet and mHC topology choices.
- Read `block_taxonomy_sample.py` for the public block-name glossary.

Main source surfaces for this directory:
- `unified_superblock.py` from the MegaCpp POC repo
- `gpt.py` from the MegaCpp POC repo
- `mod.py` from the MegaCpp POC repo
- `gated_deltanet.py` from the MegaCpp POC repo
- `hyper_connections.py` from the MegaCpp POC repo
- `multi_adapter.py` and `lora.py` from the MegaCpp POC repo
- `adapter.py` from the native-sparse experiment pack
- `README.md` from the native-sparse experiment pack
