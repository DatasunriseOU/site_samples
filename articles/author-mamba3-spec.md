---
title: "Author Mamba3 spec inside Megatron"
description: "Why an author-pure Mamba3 path still needs an explicit pre-projection RMSNorm when it is wrapped into a Megatron-local Mamba stack."
date: "2026-04-19"
tags: ["mamba3", "megatron", "rmsnorm", "spec"]
---

This seam is easy to describe badly. It is not just "drop an author model into
Megatron." The real issue is that the surrounding stack has assumptions about
where normalization happens.

In the author Mamba3 path, the projection is a plain linear surface. In the
Megatron-local path, normalization may already be fused elsewhere or replaced by
an identity surface. If that surrounding norm is not actually doing work, the
author path must restore the missing pre-projection RMSNorm explicitly.

That is why this public example keeps the contrast visible: one lane restores
the norm contract, the other leaves the residual stream unconstrained.

## Why this is worth showing publicly

This is exactly the kind of integration bug that disappears in abstract design
diagrams. Everything still "looks like Mamba3." The failure is in the seam
between author assumptions and host-framework assumptions.

The public example is useful because it shows the real rule in a compact form:
if the embedding or mixer path expects a fused norm and the wrapped module does
not supply it, that norm has to be put back explicitly.

## Example -> article -> upstream docs

- example: [`author_mamba3_spec_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/author_mamba3_spec_nearcopy.py)
- article: [`author-mamba3-spec`](https://megacpp.com/blog/author-mamba3-spec/)
- upstream docs: PyTorch RMSNorm docs, Mamba repository context, and Megatron Core model/spec context

## References

- [Author Mamba3 spec near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/author_mamba3_spec_nearcopy.py)
- [PyTorch RMSNorm docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [state-spaces/mamba repository](https://github.com/state-spaces/mamba)
- [Megatron Core user guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
