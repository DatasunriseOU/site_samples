# MegaCpp Index

Use this folder for the public NAM56R recipe, taxonomy, and launcher surfaces.

- `nam56r_nemo_recipe_sample.py`: hidden size, hybrid pattern, MoE, MLA, and launch-arg emission.
- `megatron_args_sample.py`: Megatron-style argument normalization.
- `nam56r_cuda_graph_launcher_sample.sh`: shell-side launcher example for graph capture.
- `nam56r_block_taxonomy_sample.py`: what A-block, E-block, M-block, and R-block mean.
- `nam56r_feature_placement_sample.py`: where Engram, DSA, mHC, MoE, MoD, MTP, and structure features attach.
- `nam56r_launcher_profile_sample.py`: grouped layout/parallelism/runtime launcher profile.

In simple words: this directory is the bridge between the public NAM56R model
story and the actual recipe plus launcher controls that make it run.
