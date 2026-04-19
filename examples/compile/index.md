# Compile Index

Use this folder when you want the MegaCpp POC compile-control surfaces rather
than model math.

- `cuda_graph_block_validation_sample.py`: validate requested graph block names
  against the actual model module set.
- `cuda_graph_env_defaults_sample.py`: normalize the env defaults that keep
  Inductor CUDA graph capture turned on when the runtime asks for it.

In simple words: these files are the guard rails around compile and graph
capture, not the kernels themselves.
