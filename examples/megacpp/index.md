# MegaCpp Index

Use this folder for public NAM56R recipe surfaces, Megatron/Nemotron
translation seams, and two runtime-example lanes:
- compact receipts for a smaller teaching-sized view of each issue
- near-copy receipts for a closer MegaCpp POC contract when shape/layout details matter

Recipe and launch surfaces:
- `nam56r_nemo_recipe_sample.py`
- `nam56r_nemo_recipe_contract_sample.py`
- `nam56r_megatron_plan_sample.py`
- `nam56r_launch_contract_sample.py`
- `megatron_args_sample.py`
- `nam56r_cuda_graph_launcher_sample.sh`
- `nam56r_launcher_profile_sample.py`
- `nam56r_runtime_patch_surface_sample.py`
- `nemotron_recipe_to_megatron_sample.py`

Pattern and feature-planning surfaces:
- `nam56r_block_taxonomy_sample.py`
- `nam56r_pattern_composition_sample.py`
- `nam56r_feature_placement_sample.py`
- `fail_closed_pattern_translation_sample.py`
- `mla_integration_pattern_sample.py`
- `mla_shared_adapter_sample.py`
- `m2rnn_mixer_spec_sample.py`
- `structure_embedding_contract_sample.py`

Data and migration bridges:
- `parquet_to_megatron_indexed_dataset_sample.py`
- `prepare_format_megacpp_sample.py`

Runnable runtime receipts:
- `dsa_cuda_graph_safety_sample.py`
- `dsa_indexer_memory_sample.py`
- `mamba_linear_ce_parity_sample.py`
- `mamba3_mimo_3d_to_2d_smem_sample.py`
- `tilelang_tma_bulk_copy_smem_sample.py`

Near-copy runtime receipts:
- `dsa_cuda_graph_safety_nearcopy.py`
- `dsa_indexer_memory_nearcopy.py`
- `mamba_linear_ce_parity_nearcopy.py`
- `mamba3_mimo_3d_to_2d_smem_nearcopy.py`
- `tilelang_tma_bulk_copy_smem_nearcopy.py`
- `sparse_mla_fp8_dispatch_nearcopy.py`
- `sparse_mla_dimension_generalization_nearcopy.py`

In simple words: this directory is the bridge between the public NAM56R model
story and the smaller contracts that make the recipe, runtime, and kernel
surfaces legible.
