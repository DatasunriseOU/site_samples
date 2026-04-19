---
title: "DSA CUDA graph safety deep dive"
description: "A deeper reproducer-driven look at why DSA index-mask updates break CUDA graph capture, and how a branchless fix preserves the same eager semantics."
date: "2026-04-19"
tags: ["dsa", "cuda-graphs", "runtime", "deep-dive"]
---

The compact DSA CUDA-graph article explains the rule. The near-copy sample is
useful for something stricter: it preserves the failure pattern itself.

In the unpatched path, two operations are the real problem: a validation check
that forces a Python bool and a branch on a GPU reduction. Both are harmless in
eager mode. Both are hostile to stream capture. That is why this is a runtime
contract bug rather than a numerical bug.

The patched path does not change the intended mask semantics. It changes the
capture behavior. That is the right lesson to preserve publicly.

The important engineering detail is that both failing operations look innocent
in eager mode. `torch.equal(...)` and a branch on `torch.any(...)` are common
validation patterns. Under CUDA graph capture they stop being bookkeeping and
become hidden host synchronizations. That is why the public sample keeps both
the unpatched and patched forms visible side by side.

## Example -> article -> upstream docs

- example: [`dsa_cuda_graph_safety_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_nearcopy.py)
- article: [`dsa-cuda-graph-safety.md`](https://megacpp.com/blog/dsa-cuda-graph-safety/)
- upstream docs: PyTorch CUDA Graph notes, compiler troubleshooting, and CUDA Graph API guidance

## References

- [DSA CUDA graph safety near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_nearcopy.py)
- [DSA CUDA graph safety compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_sample.py)
- [PyTorch `torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [PyTorch CUDA graphs docs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
