# Distributed example index

Use this folder when you want public-safe code samples for the MegaCpp POC
distributed training stack.

Key specialist-model surfaces:
- routing modes: token-choice, group-limited routing, null-slot routing
- expert-parallel dispatch: token exchange, fused permute, overlap options
- shared expert behavior: always-on path plus optional learned gate
- routing losses: load balancing and router z-loss bookkeeping
- expert compute: grouped expert-bank style dense execution
- adaptive compute budgets: GateSkip and FlexiDepth skip surfaces
- multi-token prediction: shared-block recursive future-token heads
- pipeline overlap rules: DualPipe and DualPipeV stage contracts
- optimizer shard handling: FSDP2 local-row contracts for Muon-style updates

How these plug into the model:
- the router scores tokens against routed experts
- the dispatcher moves token slices to expert owners in expert-parallel runs
- local expert banks process those slices
- shared experts run for every token and are merged back at the end
- skip routers decide when a token should take the full path or a cheap path
- MTP keeps a separate future-token loss alive on the last stage without duplicating a full decoder stack
- pipeline helpers decide how stages overlap and when aux losses must be injected
- FSDP2 optimizer helpers map full-shape metadata onto local shards before the step
