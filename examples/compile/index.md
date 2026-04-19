# Compile Index

Use this folder when you want the MegaCpp POC compile-control surfaces rather
than model math.

- `cuda_graph_block_validation_sample.py`: validate requested graph block names
  against the actual model module set.
- `cuda_graph_env_defaults_sample.py`: normalize the env defaults that keep
  Inductor CUDA graph capture turned on when the runtime asks for it.
- `compile_warmup_policy_sample.py`: explicit warmup skip policy for unstable
  CUDA regional-compile plus MoE lanes.
- `dynamic_batch_compile_policy_sample.py`: how dynamic-batch compile is guarded
  and when warmup dbs is promoted.
- `regional_compile_ordering_sample.py`: the required ordering between TP/SP,
  regional compile, and FSDP2.
- `compiled_adamw_policy_sample.py`: when the fused optimizer step is compiled
  dynamically and when it stays eager.
- `opaque_kernel_compile_wrapper_sample.py`: custom-op wrapper pattern for one
  compile-fragile kernel.
- `compile_runtime_receipt_sample.py`: one compact receipt of the effective
  compile/runtime lane.

In simple words: these files are the guard rails around compile and graph
capture, plus the policy layer that decides which compile path is actually safe
to run.
