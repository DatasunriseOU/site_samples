# Compile And Runtime Capture Examples

This directory covers the MegaCpp POC surfaces around `torch.compile`, regional
compile, and CUDA graph capture.

What is here:
- `cuda_graph_block_validation_sample.py` checks whether requested block types
  actually exist before trying to force CUDA-graph capture.
- `cuda_graph_env_defaults_sample.py` shows the env-default contract used to keep
  Inductor graph settings aligned with runtime intent.

What problem these samples solve:
- they prevent silent "compile flags were set but nothing meaningful happened"
  failures
- they keep graph/capture behavior tied to explicit runtime checks instead of
  magic shell state

Where this fits in the model:
- these helpers sit around the compiled training loop and block registry
- they decide whether graph capture should be enabled for a given runtime shape
