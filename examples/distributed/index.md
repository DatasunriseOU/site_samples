# Distributed example index

Use this folder when you want public-safe code samples for the MegaCpp POC
distributed training stack.

Key specialist-model surfaces:
- routing modes: token-choice, group-limited routing, null-slot routing
- expert-parallel dispatch: token exchange, fused permute, overlap options
- shared expert behavior: always-on path plus optional learned gate
- routing losses: load balancing and router z-loss bookkeeping
- expert compute: grouped expert-bank style dense execution

How these plug into the model:
- the router scores tokens against routed experts
- the dispatcher moves token slices to expert owners in expert-parallel runs
- local expert banks process those slices
- shared experts run for every token and are merged back at the end
