# Pallas Kernel Selection Notes

This note records the public rules MegaCpp uses when deciding whether a TPU
kernel should stay in XLA lowering or move into Pallas.

## Reach for Pallas when

- you need explicit tile control
- you need to avoid materializing a dense mask
- segment ids or local-window structure can be handled inside the hot loop
- the default lowering adds an avoidable extra pass over a large tensor

## Leave the path in XLA when

- the case is plain dense attention or another well-supported default lowering
- the sequence is short and compile overhead dominates
- the operation is mostly elementwise or reduction-heavy and XLA already fuses it well
- dynamic shapes would turn every step into a recompile story

## Public takeaway

Pallas is useful, but it is not the default answer to every TPU performance
problem. In MegaCpp it is a narrow tool for narrow cases.
