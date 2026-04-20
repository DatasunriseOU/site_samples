---
title: "DSA and CUDA graph safety"
description: "Why DSA index-mask updates need branchless graph-capture-safe logic, and why small host-sync accidents can break an otherwise valid CUDA graph path."
date: "2026-04-19"
tags: ["dsa", "CUDA-graphs", "runtime", "kernels"]
---

CUDA graph capture is unforgiving about hidden host sync points. That makes DSA
index-mask updates a good example of a broader MegaCpp rule: math parity is not
enough if the path still branches on GPU reductions or validation checks that
become Python booleans.

The public sample keeps the fix simple: branchless scatter plus a small fixup.
That is more useful than a broad CUDA-graphs slogan because it shows the exact
runtime pattern that becomes safe.

## References

- [DSA CUDA graph safety sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_sample.py)
- [NAM56R CUDA graph launcher sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_cuda_graph_launcher_sample.sh)
- [CUDA graph block validation sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/compile/cuda_graph_block_validation_sample.py)
- [PyTorch CUDA graphs docs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
