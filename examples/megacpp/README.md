# MegaCpp Model Wiring Examples

This directory contains public-safe MegaCpp model-spec examples centered on
NAM56R and its launch/runtime wiring.

What is here:
- `nam56r_nemo_recipe_sample.py`: authoritative NAM56R recipe values and CLI emission
- `megatron_args_sample.py`: argument shaping for Megatron-style launch flows
- `nam56r_cuda_graph_launcher_sample.sh`: shell-side CUDA graph launcher example
- `nam56r_block_taxonomy_sample.py`: decoded block-letter taxonomy for A/E/M/R
- `nam56r_feature_placement_sample.py`: where the main feature families attach in NAM56R
- `nam56r_launcher_profile_sample.py`: grouped launcher env/profile controls

What problem these files solve:
- they keep the public model description tied to real recipe and launcher surfaces
- they explain what each block family means and where features like DSA, Engram,
  mHC, MoE, MoD, MTP, and Mamba belong
- they turn a long launcher into a readable profile instead of a shell blob

Where this fits in the model:
- these are the top-level recipe, layout, and launch adapters for the training stack
- they sit above the lower-level kernel and distributed examples
