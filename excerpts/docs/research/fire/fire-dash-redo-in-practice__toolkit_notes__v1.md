> Public note.
>
> Source repo: MegaCpp public samples
> Source material: https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/docs/research/fire/fire-dash-redo-in-practice__toolkit_notes__v1.md
> Purpose: explain why FIRE, DASH, and ReDo were treated as separate tools rather than one vague plasticity claim
> Edited for clarity.

FIRE, DASH, and ReDo were kept separate because they act on different time scales and different failure modes.

- FIRE is a phase-boundary intervention.
- DASH is a periodic shrinkage rule during training.
- ReDo is a dormant-neuron recycling pass driven by activity tracking.

The useful engineering result was not just that all three existed in the same codebase. It was that they could share scheduling and safety rules without pretending to be the same algorithm.
