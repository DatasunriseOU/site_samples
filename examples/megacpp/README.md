# MegaCpp Model Wiring Examples

This directory contains public-safe MegaCpp model-spec examples centered on
NAM56R and its launch/runtime wiring.

What is here:
- `nam56r_nemo_recipe_sample.py`: authoritative NAM56R recipe values and CLI emission
- `megatron_args_sample.py`: argument shaping for Megatron-style launch flows
- `nam56r_cuda_graph_launcher_sample.sh`: shell-side CUDA graph launcher example
- `nam56r_block_taxonomy_sample.py`: decoded block-letter taxonomy for A/E/M/R
- `nam56r_pattern_composition_sample.py`: expanded pattern counts and layer-rank map
- `nam56r_feature_placement_sample.py`: where the main feature families attach in NAM56R
- `nam56r_launcher_profile_sample.py`: grouped launcher env/profile controls
- `nam56r_runtime_patch_surface_sample.py`: runtime patch surfaces layered on top of the recipe
- `dsa_cuda_graph_safety_sample.py`: branchless DSA index-mask update pattern for CUDA Graph safety
- `mamba3_mimo_3d_to_2d_smem_sample.py`: shared-memory layout legality example for Mamba3-style kernels
- `tilelang_tma_bulk_copy_smem_sample.py`: minimal TileLang TMA/shared-memory lowering example
- `mamba_linear_ce_parity_sample.py`: output-layer and CE-loss parity surface for Mamba-style stacks
- `dsa_indexer_memory_sample.py`: fused top-k score path that avoids a larger attention-score intermediate
- `nemotron_recipe_to_megatron_sample.py`: compact Nemotron-style recipe lowered into Megatron-native args
- `fail_closed_pattern_translation_sample.py`: fail-closed pattern translator for hybrid block strings
- `mla_integration_pattern_sample.py`: narrow adapter seam for MLA integration
- `m2rnn_mixer_spec_sample.py`: recurrent-style mixer spec surface for Megatron integration
- `structure_embedding_contract_sample.py`: validated structure-input normalization before embedding fusion
- `nam56r_nemo_recipe_contract_sample.py`: recipe-object contract for NAM56R-style Nemotron authoring
- `nam56r_megatron_plan_sample.py`: explicit hybrid-plan translation into Megatron-native roles
- `nam56r_launch_contract_sample.py`: split between generated native args and fixed launch policy
- `mla_shared_adapter_sample.py`: shared MLA compatibility adapter contract
- `parquet_to_megatron_indexed_dataset_sample.py`: parquet-token-shard to indexed-dataset bridge
- `prepare_format_megacpp_sample.py`: thin public wrapper for naming and split policy in Megatron-ready data prep
- `dsa_cuda_graph_safety_nearcopy.py`: heavier near-copy reproducer for CUDA-graph-safe DSA mask updates
- `mamba3_mimo_3d_to_2d_smem_nearcopy.py`: heavier near-copy layout refactor for the Mamba3 shared-memory/TMA issue
- `tilelang_tma_bulk_copy_smem_nearcopy.py`: heavier near-copy lowering contract for TileLang TMA bulk-copy layouts
- `mamba_linear_ce_parity_nearcopy.py`: heavier near-copy class-contract reproducer for Mamba linear-CE parity
- `dsa_indexer_memory_nearcopy.py`: heavier near-copy reproducer for the fp32 DSA score-intermediate blow-up

What problem these files solve:
- they keep the public model description tied to real recipe and launcher surfaces
- they explain what each block family means and where features like DSA, Engram,
  mHC, MoE, MoD, MTP, and Mamba belong
- they make the compact pattern and runtime patch points explicit instead of
  hiding them behind one short launcher string
- they turn a long launcher into a readable profile instead of a shell blob
- they preserve small runnable reproducer surfaces for CUDA-graph safety,
  memory fixes, parity checks, and TileLang/Hopper layout issues without
  dragging in the full training tree
- they also keep a second lane of heavier near-copy receipts for the places
  where compact abstractions hide the real contract too aggressively
- they keep Megatron/Nemotron translation, MLA integration, recurrent mixer
  seams, and structure-aware embedding contracts inspectable as separate units

Where this fits in the model:
- these are the top-level recipe, layout, and launch adapters for the training stack
- they sit above the lower-level kernel and distributed examples
- the reproducer-style files sit at the boundary between recipe/launcher policy
  and low-level runtime behavior, where a small standalone sample is often more
  useful than a full model launch
