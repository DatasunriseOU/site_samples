---
title: "DSA CUDA graph safety deep dive"
description: "A deeper reproducer-driven look at why DSA index-mask updates break CUDA graph capture, and how a branchless fix preserves the same eager semantics."
date: "2026-04-19"
tags: ["dsa", "cuda-graphs", "runtime", "deep-dive"]
---

The compact DSA CUDA-graph article explains the rule. The near-copy sample is
useful for something stricter: it preserves the failure pattern itself.

In the unpatched path, two operations are the real problem: a validation check
that forces a Python bool and a branch on a GPU reduction. Both are usually
acceptable in eager mode. Both are hostile to stream capture. In this
reproducer, the primary failure mode is capture legality, not numerical
mismatch.

The patched path does not change the intended mask semantics. It changes the
capture behavior. That is the right lesson to preserve publicly.

The important engineering detail is that both failing operations look innocent
in eager mode. Validation checks that force a host-visible bool, and branches
that depend on a Python bool derived from GPU results, are common patterns.
Under CUDA graph capture they stop being bookkeeping and become forbidden
CPU-GPU synchronization points. That is why the public sample keeps both the
unpatched and patched forms visible side by side.

## Example -> article -> upstream docs

- example: [`dsa_cuda_graph_safety_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_nearcopy.py)
- article: this deep-dive route, [`dsa-cuda-graph-safety-deep-dive`](https://megacpp.com/blog/dsa-cuda-graph-safety-deep-dive/), with the compact companion in [`dsa-cuda-graph-safety`](https://megacpp.com/blog/dsa-cuda-graph-safety/)
- upstream docs: PyTorch CUDA Graph notes, `torch.cuda.graph` and `torch.cuda.CUDAGraph`, and NVIDIA capture-failures guidance

## References

- [DSA CUDA graph safety near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_nearcopy.py)
- [DSA CUDA graph safety compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_sample.py)
- [NVIDIA CUDA Graph constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html)
- [NVIDIA CUDA Graph capture failures](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/capture-failures.html)
- [PyTorch `torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [PyTorch `torch.cuda.CUDAGraph`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
- [PyTorch CUDA graphs docs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [PyTorch CUDA Graph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
