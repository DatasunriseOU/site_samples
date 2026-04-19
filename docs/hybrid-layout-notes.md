# Hybrid Layout Notes

This note summarizes the public naming and ownership rules behind
MegaCpp pattern strings such as `AEMEAEMEAEMR`. The ownership column below is a
note-level summary of typical MegaCpp surfaces, not a substitute for the
public glossary or sample code.

## Pattern tokens

The letters are MegaCpp's own architectural shorthand. They are useful because
they keep block ownership explicit across recipe code, scheduling code, and
benchmark notes.

| Token | MegaCpp shorthand | Main ownership surface | Public meaning |
| --- | --- | --- | --- |
| `A` | `ablock` | attention kernels, qkv/proj geometry, positional handling | an attention-heavy block |
| `M` | `mblock` | state-space or Mamba-style mixers, scan kernels, recurrent state updates | a sequence-mixer block |
| `E` | `eblock` | router logic, expert FFNs, MoE scheduling | a conditional-capacity block |
| `R` | `rblock` | recurrent-style or tail consolidation logic | a specialized tail block |
| `C` | `cblock` | connector or cross-stream coordination logic | a lightweight coordination block |

## Why the notation stays useful

- It makes recipe declarations auditable.
- It stops all layers from being described as if they were interchangeable.
- It keeps runtime scheduling honest about which block family owns which cost.
- It gives evaluation and profiling notes a vocabulary that matches the code.

## Naming policy

- Treat `ablock`, `mblock`, `eblock`, `rblock`, and `cblock` as MegaCpp-local names.
- Do not present them as industry standards.
- Explain the meaning of the letters before using them in a public article.

## Related public files

- https://github.com/DatasunriseOU/site_samples/blob/main/examples/hybrid/hybrid_pattern_sample.py
- https://megacpp.com/blog/megacpp-model-glossary/
