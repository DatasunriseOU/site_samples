# Mamba-3 Porting Notes for TPU

This note summarizes the public engineering rules MegaCpp uses for its
Mamba-style TPU path.

## Porting rules

- keep the scan body shape-stable
- materialize sequence or segment identifiers explicitly
- prefer plain XLA fusion around the scan by default
- use Pallas only when custom tiling or mask handling clearly pays for itself
- keep document-boundary semantics inside the compiled path

## Why this matters

State-space updates can become launch-heavy or recompile-prone if prologue,
scan, and boundary metadata are treated as separate dynamic steps. TPU-friendly
execution prefers one compile-stable path over clever but fragile runtime glue.

## Public takeaway

The important point is not that one exact kernel is universally best. The
important point is that TPU execution rewards static-shape contracts and visible
boundary metadata.
