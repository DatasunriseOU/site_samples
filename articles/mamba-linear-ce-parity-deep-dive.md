---
title: "Mamba linear CE parity deep dive"
description: "Why output-layer swaps in Mamba-style stacks need explicit CE parity checks, not just shape compatibility checks."
date: "2026-04-19"
tags: ["mamba", "cross-entropy", "parity", "deep-dive"]
---

An output path can preserve tensor shapes while still drifting on the loss
contract. That is why a CE parity reproducer is worth publishing. It narrows
the question to the only thing that matters: does the alternate path still mean
the same thing to the loss.

The near-copy version keeps the specific contract visible: one path owns a
`LinearCrossEntropyModule`-style output layer and the other keeps a plain
column-parallel layer until the class contract is restored. That is closer to
the real failure than a generic `cross_entropy(hidden @ W.T, targets)` toy.

## Example -> article -> upstream docs

- example: [`mamba_linear_ce_parity_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_nearcopy.py)
- article: [`mamba-linear-ce-parity-deep-dive.md`](https://megacpp.com/blog/mamba-linear-ce-parity-deep-dive/)
- upstream docs: PyTorch cross-entropy guidance and Megatron-LM model docs for the output-layer/loss split

## References

- [Mamba linear CE parity near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_nearcopy.py)
- [Mamba linear CE parity compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_sample.py)
- [PyTorch cross entropy docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
