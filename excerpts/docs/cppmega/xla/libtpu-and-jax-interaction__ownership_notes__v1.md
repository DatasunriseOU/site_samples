> Public note.
>
> Source repo: MegaCpp public samples
> Source material: https://github.com/DatasunriseOU/site_samples/blob/main/docs/tpu-backend-ownership.md
> Purpose: explain the public ownership split between torch_xla, libtpu, JAX-side helpers, and pallas-like kernel decisions
> Edited for clarity.

When TPU execution is described publicly, it helps to separate ownership layers instead of saying "XLA" as if one component owned everything.

- `torch_xla` is the application frontend for tracing, graph capture, and SPMD-facing APIs.
- `libtpu` belongs to the TPU runtime stack and should be tracked as provenance, versioning, and compatibility context.
- JAX interactions are useful as reference taxonomy or helper context, but they should not be confused with the execution frontend when the model path is PyTorch/XLA.
- Pallas-like kernels are a narrow performance tool for explicit tile control and hot loops, not a blanket replacement for default XLA lowering.
