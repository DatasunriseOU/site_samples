"""Dynamic-batch compile policy for changing microbatch sizes.

What it is: a public-safe summary of the MegaCpp POC dynamic-batch contract for
`torch.compile` lanes that want database-style batch-size changes without full
recompilation.

Why it exists: changing batch size after compile is one of the fastest ways to
accidentally trigger repeated recompiles or mismatched sparse-mask assumptions.

What problem it solves: it shows the exact guard rails around `dynamic_batch`,
compile warmup, and partial dynamic sparse-mask support.
"""

from __future__ import annotations


def summarize_dynamic_batch_policy(*, dynamic_batch: bool, compile_warmup_dbs: int | None, uses_flex_blockmask: bool, uses_moba_indexer: bool) -> dict[str, object]:
    warnings: list[str] = []
    warmup_dbs = compile_warmup_dbs

    if dynamic_batch and warmup_dbs is None:
        warnings.append("dynamic_batch should be paired with compile_warmup_dbs so the first compiled lane sees a stable starting shape")
    if dynamic_batch and warmup_dbs == 1:
        warmup_dbs = 2
        warnings.append("compile_warmup_dbs=1 is promoted to 2 so the warmup lane exercises a real batched shape")
    if dynamic_batch and uses_flex_blockmask:
        warnings.append("Flex-style block masks only have partial dynamic-shape coverage; treat this as a guarded lane")
    if dynamic_batch and uses_moba_indexer:
        warnings.append("MoBa-style indexers can invalidate the no-recompile promise; keep this lane eager or use a safer indexer")

    return {
        "dynamic_batch": dynamic_batch,
        "effective_compile_warmup_dbs": warmup_dbs,
        "recommended_compile_dynamic": dynamic_batch and not uses_moba_indexer,
        "warnings": warnings,
    }
