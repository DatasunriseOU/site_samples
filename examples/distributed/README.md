# Distributed examples

These examples show how the MegaCpp POC handles distributed and specialist-model
training surfaces.

What is here:
- routing and expert-parallel dispatch samples for Mixture-of-Experts
- compile, CUDA graphs, and memory-management helpers
- pipeline, FSDP, TP, and other multi-device layout examples
- adaptive skip and budgeted compute samples for GateSkip and FlexiDepth
- DualPipe and DualPipeV stage-contract notes for overlapped pipeline runs
- FSDP2 local-shard optimizer helpers for Muon-style updates

What the new MoE files cover:
- `expert_parallel_routing_sample.py`: simple token-to-expert capacity math
- `moe_dispatch_fast_paths.py`: optional dispatch accelerators and overlap flags
- `moe_group_routing_sample.py`: two-stage group routing before top-k expert pick
- `moe_null_slot_routing_sample.py`: null-slot expansion for data-sparse routing
- `shared_expert_gate_sample.py`: gate on the always-on shared expert path
- `moe_loss_collection_sample.py`: aux-loss and z-loss collection after forward
- `fused_expert_bank_probe_sample.py`: grouped expert-bank compute surface

Additional distributed control files:
- `gateskip_residual_budget_sample.py`: residual gating budget schedule
- `flexidepth_skip_router_sample.py`: adaptive layer-skip router contract
- `mtp_shared_block_sample.py`: shared-block multi-token prediction contract
- `fsdp2_muon_local_shard_sample.py`: local-row optimizer shard mapping
- `dualpipe_stage_contract_sample.py`: DualPipe output lifetime and aux-loss rules

In simple terms:
- routed experts are the specialists
- shared experts are the always-on generalists
- routing decides which specialists see each token
- dispatch moves those token slices to the right expert shard
- aux-loss and z-loss keep the router stable and balanced
- GateSkip and FlexiDepth reduce wasted compute without changing the global model layout
- MTP reuses one extra block across several depths so future-token prediction stays cheap and compile-friendly
- FSDP2 still needs local-shard-aware optimizer math even when the model topology is right
- DualPipe schedules only work if stage outputs and aux losses survive long enough for overlap
