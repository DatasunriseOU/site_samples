> Sanitized public note.
>
> Source repo: MegaCpp research repo
> Source file: changelog and runtime notes
> Purpose: explain why STP starts after an explicit warmup boundary
> Edited for public clarity.

The STP lane was kept behind an explicit start-step gate so that the auxiliary geometry term would not be judged during the noisiest warmup window. The practical rule was simple: collect the hidden states early enough to prove the path is wired correctly, but do not charge the STP loss into the main objective until the run has crossed the chosen activation boundary.

The same note also records why this mattered operationally: several earlier wiring mistakes let training continue while STP was effectively inactive. Delaying the loss made it easier to separate warmup noise from real geometry effects and easier to detect when the collector or loss assembly path had silently dropped out.
