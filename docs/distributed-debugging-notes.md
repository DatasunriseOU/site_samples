# Distributed Debugging Notes

This note summarizes public debugging patterns for multi-GPU training.

Common failure classes:
- activation growth that looks like a parameter-memory problem
- collective hangs caused by mismatched step boundaries
- optimizer-state growth that appears only after accumulation
- checkpoint formats that resume cleanly on one layout but not another

Practical habits:
- measure parameters, activations, optimizer state, and communication separately
- keep one narrow receipt per failure family
- test split strategies independently before combining them
- prefer small reproducible batches when validating a new sharding plan

Why this matters:
- small specialist models can still go out-of-memory on large GPUs
- split strategy names alone do not explain what is truly sharded
- public notes are easier to cite when they focus on mechanisms, not machines
