# MegaCpp Model Wiring Examples

This directory contains public-safe MegaCpp model-spec examples centered on
NAM56R.

What is here:
- `nam56r_nemo_recipe_sample.py`: the authoritative NAM56R recipe surface
- `megatron_args_sample.py`: argument shaping for Megatron-style launch flows
- `nam56r_cuda_graph_launcher_sample.sh`: launcher-side CUDA graph wiring

What problem these files solve:
- they keep the public model description tied to a real launcher/config surface
- they show how hidden size, block pattern, MoE, MLA, and runtime flags fit in
  one place instead of scattered shell snippets

Where this fits in the model:
- these are the top-level recipe and launch adapters for the training stack
