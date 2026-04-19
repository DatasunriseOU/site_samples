# Compile And Runtime Capture Examples

This directory covers the MegaCpp POC surfaces around `torch.compile`, regional
compile, and CUDA graph capture.

What is here:
- `cuda_graph_block_validation_sample.py` checks whether requested block types
  actually exist before trying to force CUDA-graph capture.
- `cuda_graph_env_defaults_sample.py` shows the env-default contract used to keep
  Inductor graph settings aligned with runtime intent.
- `compile_warmup_policy_sample.py` records when explicit compile warmup is
  skipped instead of forced.
- `dynamic_batch_compile_policy_sample.py` shows the guarded dynamic-batch
  compile contract.
- `regional_compile_ordering_sample.py` captures the TP/SP -> regional compile
  -> FSDP2 ordering rule.
- `compiled_adamw_policy_sample.py` shows why the fused optimizer compile path
  is lazy and dynamic.
- `opaque_kernel_compile_wrapper_sample.py` describes the custom-op pattern used
  to keep one fragile kernel opaque while the surrounding block stays compiled.
- `compile_runtime_receipt_sample.py` summarizes the effective runtime lane after
  compile flags are combined.

What problem these samples solve:
- they prevent silent "compile flags were set but nothing meaningful happened"
  failures
- they keep graph/capture behavior tied to explicit runtime checks instead of
  magic shell state

Where this fits in the model:
- these helpers sit around the compiled training loop and block registry
- they decide whether graph capture should be enabled for a given runtime shape
- they also document how compile interacts with distributed wrappers, optimizer
  steps, and runtime lane selection
