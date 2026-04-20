---
title: "FIRE, DASH, and ReDo as one plasticity toolkit"
description: "How three separate plasticity ideas fit into one toolkit, what the public samples actually show, and which design choices are worth preserving as the stack evolves."
date: "2026-04-18"
tags: ["fire", "dash", "redo", "plasticity", "nam52", "nam56r", "training"]
---

The plasticity toolkit is interesting because it is not just "we added FIRE." It combines three distinct interventions with different time scales: FIRE for phase boundaries, DASH for periodic directional shrinkage, and ReDo for recycling dormant neurons. The public sample is small enough to inspect directly, and the public-facing writeups are enough to show the division of labor without overclaiming a one-to-one paper implementation.

Many plasticity discussions collapse everything into one magic lever. This toolkit does the opposite. It treats plasticity as a maintenance stack. One tool repairs geometry at a boundary, one tool nudges weights during training, and one tool revives neurons that have gone quiet. That decomposition is the reason the implementation is worth studying. It is also why later integrations should preserve structure and timing, not just names.

The source map is unusually clear once it is presented through public materials. The toolkit sample shows the combined control surface, and the public references for FIRE, DASH, and ReDo are useful as background context. That is enough to ground the engineering story without leaning on private notes or claiming exact external parity where the public sample already says enough.

## The toolkit works because the methods are scheduled differently

The first mistake people make with plasticity work is trying to apply every intervention at the same cadence. This toolkit does not do that.

FIRE is a boundary operation. It projects 2D weight matrices toward orthogonality with Newton-Schulz iterations and is meant for moments when the training regime changes, such as context extension or other curriculum transitions. That is a one-shot structural reset.

DASH is much lighter. It looks at row-wise cosine similarity between weights and gradients and shrinks rows whose updates are tracking their own current direction too closely. That makes it a periodic maintenance action during training rather than a rare phase change tool.

ReDo is different again. It needs activity diagnostics over time. `ReDoDiagnostics` attaches hooks, tracks EMA-style activity, and lets `recycle_dormant_neurons()` reinitialize rows that have effectively dropped out. That means it is not about geometry in the same sense as FIRE, and it is not about directional shrinkage like DASH. It is about waking neurons back up.

| Method | Public code surface | Time scale | Main failure mode it targets |
| --- | --- | --- | --- |
| FIRE | orthogonalization and target-selection helpers | Phase boundary | Loss of isometry and stale geometry between regimes |
| DASH | directional-shrinkage step | Periodic in-training | Rows over-aligning with their own gradients |
| ReDo | dormant-unit diagnostics and recycle helpers | Periodic with accumulated diagnostics | Dormant MLP neurons |

This separation is the key design win. The toolkit is not three variants of the same knob. It is three maintenance layers aimed at different times and different failure modes.

## What the FIRE implementation really adds in practice

The public sample makes a strong point: the theory is nice, but a working training system still has to solve multiple engineering problems that short method summaries do not really discuss. The most important one is optimizer state staleness.

After FIRE rewrites a weight matrix, the optimizer's stored state still describes the old basis. If you keep Adam-style `exp_avg` and `exp_avg_sq`, or Muon momentum buffers, the next updates can partially undo the re-orthogonalization. The public sample addresses that with selective optimizer-state reset, not a global wipe. `reset_optimizer_states_for_fired_params()` clears state only for parameters that were actually touched.

That is not cosmetic. It is the difference between a clean intervention and a self-canceling one.

The second important addition is DTensor safety. The public sample explicitly makes DASH and FIRE safe under FSDP2-style sharding. Helpers such as `_local_tensor_if_dtensor()` and `_match_grad_to_local_shard()` exist because the real parameter seen by the optimizer may be a shard, not a monolithic tensor. A notebook implementation can ignore that. A training system cannot.

The third addition is parameter targeting. The default path in the toolkit is careful about what it touches. Two-dimensional projection weights are in scope. Embeddings, head weights, scalar state parameters, and various bias-like tensors are generally not. That is especially important in hybrid architectures where not every learnable parameter represents the same kind of geometry.

```python
touched = apply_fire(model, targets=get_fire_targets(model, mode="context_extension"))
reset_optimizer_states_for_fired_params(optimizer, touched)
```

That short sequence encodes a lot of engineering judgment: select a topology-aware target set, rewrite only the intended matrices, and invalidate only the optimizer state that became stale.

## Why DASH and ReDo belong in the same module

At first glance, DASH and ReDo seem unrelated. One is about cosine alignment of rows and gradients. The other is about dormant neuron detection. The reason they belong together is that they are both trying to prevent late-training rigidity, just on different observables.

`dash_step()` is the lighter-weight tool. When a row's gradient keeps pointing in the same direction as the row itself, the model can become locally self-reinforcing in an unhelpful way. Shrinking that row is a way to restore room for change without a dramatic reset.

ReDo is much more targeted. It looks for neurons that have effectively stopped firing, based on normalized EMA activity. The reinitialization path then restores incoming weights at a normal scale and damps outgoing weights, which is a sensible compromise between waking the neuron up and avoiding a destabilizing spike.

The subtle but important local insight is that ReDo is activation-family dependent. The integration notes explicitly connect dormant-neuron pressure to `relu2`, while also discussing SwiGLU as a way to reduce the need for ReDo-style maintenance. That matters because it prevents the toolkit from turning into dogma. If the activation choice changes the dormant-neuron problem, the right amount of ReDo also changes.

## Hybrid blocks are why targeting matters

NAM52 and NAM56R-style stacks make all of this harder because the architecture is heterogeneous on purpose. `A`, `M`, `E`, and `R` do not all want the same intervention.

Attention projections are natural FIRE targets because they are 2D linear maps with a clear geometric story. MLP projections are similar. Some Mamba-style projections can also make sense. But one-dimensional state parameters, convolution kernels, and topology-specific auxiliary structures do not all benefit from the same orthogonalization logic.

The integration log is especially useful here because it spells out which parameter classes should be touched and which should be excluded. That is the sort of evidence a production port actually needs. A vague instruction like "apply FIRE to the model" is too blunt for a hybrid system.

The same targeting logic shows up in the context-extension mode. Rather than treating every 2D parameter equally, `get_fire_targets()` can narrow the intervention to the Q/K surfaces that matter most for extending context. That is a better operational story than uniform global treatment because it respects the block topology.

| Block family | Likely toolkit role | Why |
| --- | --- | --- |
| `ablock` / attention projections | Strong FIRE candidates | Geometry matters directly for Q/K/V and output projections |
| `mblock` linear projections | Conditional FIRE candidates | Some 2D maps benefit; state scalars do not |
| `eblock` feed-forward projections | More DASH/ReDo/FIRE depending on activation path | Large 2D MLP surfaces and dormant-neuron risk |
| `rblock` / recurrent-specific state | Usually narrower targeting | Many parameters are not natural FIRE surfaces |

That table is the real operational takeaway. Plasticity is not a global property. It is a block-local maintenance problem.

## What the tests prove, and what they do not

The test coverage around the toolkit is useful because it shows the sample is not relying on hand-wavy claims. The checks cover FIRE's effect on a proxy geometry metric, verify that the model still runs after intervention, and exercise the broader plasticity wiring. That means the toolkit has crossed the line from concept to maintained code.

But the tests also reveal the right humility. A passing unit test does not prove that a late-phase FIRE pass improves convergence in every NAM56R lane. A passing ReDo test does not prove that every dormant-neuron issue is solved. The toolkit should therefore be read as a set of grounded mechanisms with operational constraints, not as a guaranteed universal win.

That is exactly why the public sample is so valuable. It preserves the mismatch between an elegant method-level story and messy training reality: optimizer state has to be reset, sharded tensors have to be handled locally, and activation choice changes whether dormant-neuron recycling is even the right tool.

## What later integrations should preserve from this work

The port should keep the decomposition, not just the names. That means at least four things.

1. Preserve phase-boundary FIRE as a topology-aware targeted intervention.
2. Preserve selective optimizer-state reset for touched parameters.
3. Preserve the distinction between periodic DASH and diagnostic-driven ReDo.
4. Preserve the idea that block family and activation family determine which tool is appropriate.

The main thing worth preserving is the separation of responsibilities. FIRE fits best as a curriculum-boundary utility, DASH as a lightweight periodic maintenance option, and ReDo only where the activation path and hook surfaces make the signal trustworthy. Flattening all three into a single switch would discard most of the design value the public sample makes visible.

The main thing to avoid is flattening the toolkit into a single feature flag. Once that happens, all the useful timing and targeting discipline disappears. The strongest contribution here is showing that plasticity support can be modular, code-grounded, and still operational.

## Why this matters beyond one implementation snapshot

The deepest value of the toolkit is not that it proves one external reference right. It is that it turns plasticity from an abstract training slogan into a set of maintainable engineering surfaces. That is a big difference.

FIRE gives a principled way to repair geometry at transitions. DASH gives a cheap maintenance move for rows that are becoming too self-aligned. ReDo gives a direct response to dead neurons when the activation family makes that a real problem. Put together, they form a credible answer to a common late-training complaint: the model is still updating, but it is learning less than it should.

That is why this work should survive beyond one sample snapshot. Not because every run needs all three methods, but because the public sample already shows the harder part: how to separate cadences, target the right parameter families, and keep the interventions compatible with sharding and optimizer state. That is the part worth keeping.

## Code and notes

- [FIRE: Functional Interpolation for Relative Entropy Minimization](https://arxiv.org/abs/2602.08040)
- [DASH: Dynamic Adaptation via Shrinkage of High-Alignment Directions](https://arxiv.org/abs/2410.23495)
- [ReDo: Rethinking Dead Neurons in Neural Networks](https://arxiv.org/abs/2302.12902)

## Further reading

- [FIRE sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/fire/fire_sample.py)
- [DASH sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/fire/dash_sample.py)
- [ReDo sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/fire/redo_sample.py)
- [Plasticity phase schedule sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/fire/plasticity_phase_schedule_sample.py)
- [FIRE context extension filter sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/fire/fire_context_extension_filter_sample.py)
- [FIRE paper](https://arxiv.org/abs/2602.08040)
- [DASH paper](https://arxiv.org/abs/2410.23495)
- [ReDo paper](https://arxiv.org/abs/2302.12902)
- [PyTorch Distributed Tensor documentation](https://pytorch.org/docs/stable/distributed.tensor.html)
