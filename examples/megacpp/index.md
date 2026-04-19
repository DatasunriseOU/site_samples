# MegaCpp Index

Use this folder for the public NAM56R recipe and launch surfaces.

- `nam56r_nemo_recipe_sample.py`: model dimensions, hybrid pattern, MoE, MLA,
  and launch-arg emission.
- `megatron_args_sample.py`: argument normalization for Megatron-style runs.
- `nam56r_cuda_graph_launcher_sample.sh`: shell-side launcher example for graph
  capture.

In simple words: this directory is the bridge between the public model spec and
the actual training command line.
