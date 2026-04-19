# Distributed examples

These examples show how the MegaCpp POC handles distributed and specialist-model
training surfaces.

What is here:
- routing and expert-parallel dispatch samples for Mixture-of-Experts
- compile, CUDA graphs, and memory-management helpers
- pipeline, FSDP, TP, and other multi-device layout examples

What the new MoE files cover:
- `expert_parallel_routing_sample.py`: simple token-to-expert capacity math
- `moe_dispatch_fast_paths.py`: optional dispatch accelerators and overlap flags
- `moe_group_routing_sample.py`: two-stage group routing before top-k expert pick
- `moe_null_slot_routing_sample.py`: null-slot expansion for data-sparse routing
- `shared_expert_gate_sample.py`: gate on the always-on shared expert path
- `moe_loss_collection_sample.py`: aux-loss and z-loss collection after forward
- `fused_expert_bank_probe_sample.py`: grouped expert-bank compute surface

In simple terms:
- routed experts are the specialists
- shared experts are the always-on generalists
- routing decides which specialists see each token
- dispatch moves those token slices to the right expert shard
- aux-loss and z-loss keep the router stable and balanced
