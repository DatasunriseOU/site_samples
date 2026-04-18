---
title: "The MoE Routing We Actually Shipped"
description: "Token Choice vs Expert Choice, null-expert debugging, gating stability, and the production routing decisions behind the MegaCpp SLM ensemble."
author: "Boris Tamarkin"
date: "2026-04-18"
tags: ["MoE", "Token Choice", "Null Experts", "Routing", "SLM", "C++ Codegen"]
readTime: "12 min read"
---

# The MoE Routing We Actually Shipped

Mixture-of-experts routing has a literature problem. A lot of recent
papers describe routers that are elegant at batch size 1024 and broken at
batch size 1, and a lot of production writeups bury which variant they
actually serve behind. For the MegaCpp SLM ensemble, we had to pick one
router, debug it under load, and keep it stable across training, eval, and
autoregressive inference on a code model that sees long, boilerplate-heavy
C++ tokens. This is the routing we shipped, the variants we ran against
it, and the null-expert work that gave us real compute savings without
breaking causality.

## Token Choice vs Expert Choice, For Real

Every production MoE model we checked ships Token Choice. DeepSeek V3 and
R1, Kimi K2, Qwen 3 and 3.5, Nemotron v3 Nano, MiniMax M1 and 2.5, Mixtral
8x7B and 8x22B, DBRX, Grok-1, Snowflake Arctic, OLMoE, Switch Transformer,
GShard. The scoring function differs (sigmoid in newer models, softmax in
older ones), the expert count ranges from 8 to 384, the top-k ranges from
1 to 8, the load-balance story differs across aux-loss, aux-free bias, and
hierarchical group routing. What is common is the direction of selection:
each token picks its top-k experts, not the other way around.

Expert Choice is the plausible alternative in training. Each expert picks
its top-C tokens from the batch; load balance is perfect by construction;
XLA is happy because C is static. It is also broken at autoregressive
inference: with batch size 1 and sequence length 1, each expert's "top-C
from the batch" is "top-C from a single token," which collapses to zero or
one token per expert, and the train-inference mismatch is the whole ball
game. OLMoE's ablation went in the same direction as the production
tally: Token Choice wins on quality when you measure the serving
distribution, not just training throughput.

We started in Expert Choice on an earlier 3B-class research run because
it was convenient. It is no longer convenient. The ensemble ships Token
Choice.

## Our Baseline Router

The shipped configuration for our 4B-8B specialists, per expert layer,
sits close to the DeepSeek V3 / Kimi K2 family shape:

- 64 routed experts, 1 shared expert.
- Top-k = 6, selected per token.
- Sigmoid scoring (independent per-expert logits, no zero-sum softmax
  competition).
- Aux-free bias correction in the spirit of DeepSeek V3's `noaux_tc`: a
  per-expert bias is added to the router's selection score and dynamically
  pushed toward uniform load, but the bias does not enter the gating
  weights applied to expert outputs.
- One always-on shared SwiGLU expert, outside the routing pool, so every
  token gets a baseline compute path regardless of where the router sends
  it.
- Routed scaling factor ~2.5, to keep the dynamic range of routed outputs
  comparable to the shared expert after top-k summation.

Three routing modes exist in the code (`soft`, `expert_choice`,
`token_choice`). Only `token_choice` is live in the shipped presets.
`soft` is the default softmax Token Choice path we keep for comparability
and for regression tests. `expert_choice` stays as a research-only option
for training-only ablations; our serving configs reject it explicitly.

## Why Sigmoid, Not Softmax

Softmax scoring is the older default and forces experts to compete for
a normalized probability mass per token. That sounds like a feature and
is usually a subtle bug for code: two experts that are both genuinely
well-suited to the same `#include` line end up dragging each other's
scores down, which makes the router more brittle to small routing noise
and more aggressive about concentrating tokens on a single winner.
Sigmoid scoring makes each expert's selection independent. Combined with
an aux-free bias that keeps aggregate load uniform, the signal the router
actually learns is "is this expert a good fit for this token," not "is
this expert the single best fit relative to 63 others."

The measurable consequence for a code model is smoother gating:
`top-k` membership changes less between consecutive tokens in the same
function, which is exactly what we want, because consecutive tokens in
the same function typically belong to the same structural context and
should route to overlapping expert sets.

## Null Experts: How We Actually Used Them

The most useful routing paper we read during this design was the Meta
null-experts work (Token Choice only, by construction). The idea is
simple: expand the router's output from N logits to N+1, duplicate that
single "null" logit M times to reach a candidate pool of N+M slots, pick
top-`k_max` from that pool, and skip compute for any token-expert slot
that landed on a null. The parameter that matters is `rho`, the fraction
of real experts in the candidate pool; `rho = 0.5` with `M = N` was the
paper's sweet spot and is where we started.

The key bookkeeping: if you want the model's expected number of *real*
experts per token to stay at your existing top-k, you size `k_max` to
`ceil(top_k / rho)`. With our shipped `top_k = 6` and `rho = 0.5`,
that means `k_max = 12`: each token selects 12 slots from the
`N+M = 128` candidate pool, and roughly 6 of those are real experts.
When `rho = 1.0`, `k_max = top_k` and the behavior is backward-compatible
with a non-null Token Choice router. Our flag semantics match that:
`--moe_top_k` is the *desired* expected count of real experts per token,
and `k_max` is computed internally from `rho`.

We rolled null experts out in three phases, matching the paper's
implementation priority but on our schedule:

1. Soft routing with masking. Expand the router to N+1, duplicate the
   null logit, mask null selections after top-k, renormalize the gate
   weights over surviving real experts. All experts still run; there is
   no compute saving at this phase. The point is to validate the
   training signal and confirm that the router can learn to choose null
   on tokens where extra compute is genuinely wasted.
2. Gather/scatter Token Choice with variable per-expert token counts.
   This is where the compute saving actually lands. On GPU we used the
   grouped path; on TPU we stay with padded static shapes inside the
   scan to keep XLA happy.
3. Ablation over `rho` in `{0.5, 0.67, 0.75}` on our C++ mix. The
   question was whether code tokens are heterogeneous enough for null
   routing to help. Hypothesis: yes, because roughly 40% of C++ tokens
   are low-entropy boilerplate - includes, closing braces, semicolons,
   namespace lines, forward declarations. Those are exactly the tokens
   where we want the router to pick all-null and let the shared SwiGLU
   expert carry the generation.

Concrete savings on our 24 expert-layer specialist, with N=64, rho=0.5,
E[K]=6: roughly 50% reduction in routed expert FLOPs per token, shared
expert unchanged, net MoE compute savings around 35-40%. That is before
any quality degradation, which we did not measure at `rho=0.5`; it stayed
within noise of the non-null baseline on our internal C++ eval suite.

## The Shared Expert Is the Safety Net

This is worth stating on its own. In the null-experts regime, a token
that routes entirely to null still gets the shared expert's output and
the residual stream. That changes what the router is actually learning.
It is no longer "which six experts should process this token"; it is
"are any of the routed experts useful here, or should the shared expert
carry it alone?" For boilerplate C++, the second question has a clear
answer, and the null-expert path lets the router express it.

Two corollaries. First, the shared expert must stay a first-class
component and never be treated as an optimization to remove. Second,
we do not combine null experts with ablations that disable the shared
expert; that combination collapses the safety net and makes the null
signal look worse than it is.

## Null-Experts Debugging: The Non-Obvious Failures

Three failure modes were not in the paper but showed up in our
implementation:

- Gate renormalization after masking. If you mask null slots after top-k
  and forget to renormalize the gate weights over the surviving real
  experts, a token that lands heavily on null slots has its effective
  contribution scaled down proportionally, which looks like random
  quality loss that correlates with `rho`. The fix is a per-token
  renormalization over surviving slots; the observable signature is that
  pre-fix eval gets *worse* as `rho` decreases, which is the wrong
  direction.
- Aux-loss interaction. A standard load-balance aux loss
  (alpha * N * sum(f_i * P_i)) computed over the expanded N+M pool
  rewards the router for sending everything to null, because null is
  perfectly balanced by construction. The aux loss must be computed
  over real experts only, and it must see the renormalized real-expert
  gate distribution, not the pre-mask N+M distribution.
- Router z-loss stability. We kept the OLMoE-style z-loss at ~1e-3.
  Without it, sigmoid scoring plus sparse null routing occasionally
  drove router logits to saturating magnitudes on tokens that landed
  mostly-null, which produced the gradient equivalent of a dead-expert
  spiral.

The practical guardrails we keep in training telemetry: log real-expert
usage entropy per layer, log null-fraction per layer per step, and log
per-expert bias trajectories. A healthy run shows entropy dropping
smoothly into a plateau, null-fraction settling near `1 - rho`, and
biases oscillating in a narrow band. A sick run shows null-fraction
drifting toward 1 or collapsing toward 0 within a few thousand steps;
either direction is a bug, not a feature.

## Gating Stability Without Aux-Loss Tuning

DeepSeek V3's aux-free bias correction (`noaux_tc`) is the production
mechanism we copied, with a small modification. The bias adjusts
*selection* scores only; gating weights used to combine expert outputs
see only the raw router logits after top-k. That separation matters for
two reasons. First, it decouples load-balance pressure from the learned
gate magnitudes, which keeps the router from drifting into a regime
where aggressive balance correction distorts gate weights and therefore
expert gradients. Second, it makes the bias safe to update with a much
simpler controller than a true auxiliary loss: our update is a clipped
EMA toward uniform load per layer per step, which is boring, stable, and
not entangled with the main optimizer's learning rate schedule.

We kept Qwen 3's dual balance story (global expert balance plus local
token balance) in reserve, behind a flag, and never needed it. The
per-expert bias correction plus z-loss plus the null-experts safety net
were enough.

## Comparisons We Ran

The routers we compared on small-scale training runs before picking the
shipped configuration:

| Router                                           | Top-k  | Quality        | Routed FLOPs | Shipped? |
|--------------------------------------------------|--------|----------------|--------------|----------|
| Softmax Token Choice + standard aux loss         | 6 / 64 | stable, concentrating | baseline  | no       |
| Sigmoid Token Choice + aux-free bias             | 6 / 64 | flatter usage  | baseline     | prior    |
| Sigmoid TC + aux-free bias + null at rho=0.5     | 6 real, k_max=12 | same within noise | ~40% lower | yes |
| Expert Choice (static C per expert)              | n/a    | train ok, serve broken | n/a    | no       |
| Nemotron-style hierarchical top-1 then top-6     | 6      | parity         | baseline     | no       |

The shipped config, as a compact reference:

```python
# MoE routing contract (shipped)
MOE_ROUTING = dict(
    score      = "sigmoid",     # not softmax
    token_choice = True,
    n_experts      = 64,
    n_shared       = 1,         # always on
    top_k          = 6,
    null_rho       = 0.5,       # k_max = 12
    bias_update    = "noaux_tc",# DeepSeek-V3 aux-free
    z_loss         = 1e-3,
    renorm_over    = "real",    # post-null-mask
    aux_loss_over  = "real",    # no null pool
)
```

The short read: sigmoid TC with aux-free bias beats softmax on long-run
expert-usage flatness; adding null experts at `rho=0.5` matches that
quality within noise at ~40% lower routed FLOPs; Expert Choice trains
fine and breaks at decode; hierarchical group routing buys no quality
win for our scale and costs extra routing plumbing.

## What We Did Not Ship

A few routing ideas looked interesting and did not make the cut.

Structure-aware routing - using offline AST or call-graph metadata to
bias the router toward experts specialized on specific structural
contexts - is on the roadmap but outside this post. It belongs with the
structure-aware attention work and the enriched-parquet training
contract.

Large top-k values (`>= 8`) at fine-grained expert counts (128-384).
DeepSeek V3 and Kimi K2 ship there; our specialists are smaller and the
marginal quality at top-k = 8 did not justify the extra all-to-all
communication for our serving layout.

Expert Choice on serving. The train-inference mismatch is not a
hyperparameter; it is a semantics change. There is no serving-time
rescue for Expert Choice on a single-token decode step, and nothing we
tried on the inference side made it worth the training-time perfect
load balance.

## The Short Version

Token Choice, sigmoid scoring, aux-free bias correction, 64 routed
experts plus 1 shared, top-k = 6, null experts at `rho = 0.5` for
`k_max = 12`, z-loss at 1e-3, per-token gate renormalization after null
masking, aux loss computed over real experts only, shared expert always
on, Expert Choice explicitly rejected at serving. The routing that
survives production pressure is the routing that stays causal, stays
cheap on boilerplate, and stays honest about what it is doing on the
token that is actually being decoded.

---

References (filenames only): `MOE_ROUTING_COMPARISON.md`,
`null_experts_analysis.md`.
