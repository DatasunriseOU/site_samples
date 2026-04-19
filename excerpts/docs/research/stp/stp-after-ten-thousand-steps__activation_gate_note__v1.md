> Public note.
>
> Source repo: MegaCpp public samples
> Reference note: docs/stp-notes.md
> Purpose: explain why STP starts after an explicit warmup boundary
> Edited for clarity.

This sample keeps the auxiliary trajectory term behind an explicit start-step gate so the main loss and the auxiliary loss do not begin at the same boundary. The practical rule is simple: collect hidden states early enough to prove the path is wired correctly, but do not charge the auxiliary loss into the main objective until the run has crossed the chosen activation boundary.

The same note also records a practical failure mode: the training path could keep running even when the auxiliary collector or loss assembly path was not contributing as intended. Delaying the loss made it easier to separate early-training noise from the behavior of the auxiliary path and to detect when that path had silently dropped out.
