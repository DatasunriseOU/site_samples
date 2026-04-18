> Sanitized public note.
>
> Source repo: MegaCpp codebase
> Source file: production and reproducibility notes
> Purpose: summarize the stable H200 x8 training lane in article-safe wording
> Edited for public clarity.

The stable H200 x8 lane depended on keeping the runtime recipe explicit: exact pattern layout, exact precision mode, explicit activation policy, and a launcher surface that made those choices visible. The main operational lesson was that large-GPU bringup failed less from missing raw memory and more from mismatched ownership of activations, optimizer states, expert routing buffers, and compile-time temporary tensors.
