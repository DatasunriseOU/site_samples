"""Warm compiled pipeline stages without running the real schedule.

This example shows the MegaCpp POC warmup used for pipeline stages. It solves the
practical problem that compiled blocks still need one forward and backward pass,
but the full pipeline scheduler adds transport logic that is not part of the
compile target.
"""

from __future__ import annotations

import torch


def run_pp_stage_compile_warmup(stage_mods, model_config, *, warmup_dbs: int, seq_len: int, device, dynamic_batch: bool = False):
    """Run one synthetic forward/backward pass per stage module.

    The MegaCpp POC promotes B=1 to B=2 for dynamic batch warmup because size-1 dims
    are specialized as constants and produce a different compile cache key.
    """

    if dynamic_batch and warmup_dbs == 1:
        warmup_dbs = 2

    for stage_mod in stage_mods:
        is_first = stage_mod.embed is not None
        is_last = stage_mod.head is not None
        if is_first:
            x_warmup = torch.randint(0, model_config.vocab_size, (warmup_dbs, seq_len), device=device, dtype=torch.long)
        else:
            x_warmup = torch.randn(warmup_dbs, seq_len, model_config.n_embd, device=device, dtype=torch.bfloat16)

        if dynamic_batch:
            if x_warmup.size(0) <= 1:
                torch._dynamo.maybe_mark_dynamic(x_warmup, 0)
            else:
                torch._dynamo.mark_dynamic(x_warmup, 0)

        stage_mod.train()
        stage_mod.set_structure_meta(None)
        saved_dealloc = stage_mod._deallocate_output
        stage_mod._deallocate_output = False
        try:
            out = stage_mod(x_warmup)
            if is_last:
                local_vocab = out.size(-1)
                targets = torch.randint(0, local_vocab, (warmup_dbs, seq_len), device=device, dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(out.view(-1, local_vocab), targets.view(-1))
            else:
                loss = out.sum().to(out.dtype)

            for param in stage_mod.parameters():
                if param.is_leaf and hasattr(param, "grad_dtype"):
                    try:
                        param.grad_dtype = None
                    except Exception:
                        pass

            loss.backward()
        finally:
            stage_mod._deallocate_output = saved_dealloc
            for param in stage_mod.parameters():
                param.grad = None
            stage_mod._saved_output = None
