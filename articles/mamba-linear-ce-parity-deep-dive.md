---
title: "Mamba linear CE parity deep dive"
description: "Why output-layer swaps in Mamba-style stacks need explicit CE parity checks, not just shape compatibility checks."
date: "2026-04-19"
tags: ["mamba", "cross-entropy", "parity", "deep-dive"]
---

An output path can preserve tensor shapes while still drifting on the logits to
loss contract. That is why a CE parity reproducer is worth publishing. It
narrows the question to the only thing that matters: does the alternate path
preserve the same logits-to-loss contract.

The near-copy version keeps the output-layer contract visible: one path uses a
fused linear-plus-cross-entropy module and the other keeps a plain
column-parallel output layer until the loss-path contract is restored. That is
closer to the real failure than a generic `cross_entropy(hidden @ W.T,
targets)` toy.

## Example -> article -> upstream docs

- example: [`mamba_linear_ce_parity_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_nearcopy.py)
- article: this deep-dive route, [`mamba-linear-ce-parity-deep-dive`](https://megacpp.com/blog/mamba-linear-ce-parity-deep-dive/)
- upstream docs: PyTorch cross-entropy plus Megatron Core and Megatron-LM loss/output-layer surfaces

## References

- [Mamba linear CE parity near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_nearcopy.py)
- [Mamba linear CE parity compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba_linear_ce_parity_sample.py)
- [PyTorch cross entropy docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [Megatron Core language module docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.models.common.language_module.language_module.html)
- [Megatron Core fused cross entropy docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_cross_entropy.html)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
