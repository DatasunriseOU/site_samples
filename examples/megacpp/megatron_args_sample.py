"""Grounded donor excerpt for Megatron CLI argument emission.

This sample keeps the real flag-building logic, but replaces project-local plan
imports with a tiny local contract so the file stays import-clean.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleMegatronHybridPlan:
    engram: object | None = None
    ngram_hash: object | None = None
    mhc: object | None = None
    mod: object | None = None
    moda: object | None = None
    structure: object | None = None


@dataclass(frozen=True)
class MegatronArgsBundle:
    args: tuple[str, ...]
    custom_notes: tuple[str, ...] = ()

    def to_shell_fragment(self) -> str:
        return " ".join(self.args)


def _bool_flag(enabled: bool, flag: str) -> tuple[str, ...]:
    return (flag,) if enabled else ()


def build_megatron_args_bundle(
    *,
    plan: SampleMegatronHybridPlan,
    use_mla: bool = True,
    q_lora_rank: int = 64,
    kv_lora_rank: int = 64,
    qk_head_dim: int = 64,
    qk_pos_emb_head_dim: int = 64,
    v_head_dim: int = 64,
    use_mtp: bool = False,
    mtp_mode: str = "gpt",
    mtp_num_predictors: int = 1,
    use_fim: bool = False,
    use_moe: bool = False,
    moe_num_experts: int = 16,
    moe_router_topk: int = 4,
    moe_ffn_hidden_size: int = 896,
    moe_shared_expert_intermediate_size: int = 1024,
    moe_grouped_gemm: bool = True,
    use_dsa: bool = False,
    dsa_indexer_n_heads: int = 32,
    dsa_indexer_head_dim: int = 64,
    dsa_indexer_topk: int = 256,
    dsa_indexer_loss_coeff: float = 0.001,
    dsa_indexer_dtype: str = "bf16",
) -> MegatronArgsBundle:
    args: list[str] = []
    notes: list[str] = []

    if use_mla:
        args.extend(
            [
                "--multi-latent-attention",
                "--q-lora-rank",
                str(q_lora_rank),
                "--kv-lora-rank",
                str(kv_lora_rank),
                "--qk-head-dim",
                str(qk_head_dim),
                "--qk-pos-emb-head-dim",
                str(qk_pos_emb_head_dim),
                "--v-head-dim",
                str(v_head_dim),
            ]
        )

    if use_mtp:
        if mtp_mode == "gpt":
            args.extend(["--multi-token-prediction", "--mtp-num-layers", str(mtp_num_predictors)])
        elif mtp_mode == "hybrid":
            args.extend(["--mtp-num-layers", str(mtp_num_predictors)])
        else:
            raise ValueError(f"unsupported mtp_mode: {mtp_mode!r}")

    args.extend(_bool_flag(use_fim, "--fim-rate"))
    if use_fim:
        args.append("0.5")

    if use_moe:
        args.extend(
            [
                "--expert-model-parallel-size",
                "1",
                "--num-experts",
                str(moe_num_experts),
                "--moe-router-topk",
                str(moe_router_topk),
                "--moe-ffn-hidden-size",
                str(moe_ffn_hidden_size),
                "--moe-shared-expert-intermediate-size",
                str(moe_shared_expert_intermediate_size),
                "--moe-token-dispatcher-type",
                "flex",
                "--moe-router-dtype",
                "fp32",
                "--moe-permute-fusion",
            ]
        )
        args.extend(_bool_flag(moe_grouped_gemm, "--moe-grouped-gemm"))

    if use_dsa:
        if dsa_indexer_dtype != "bf16":
            raise ValueError(f"unsupported dsa_indexer_dtype: {dsa_indexer_dtype!r}")
        args.extend(
            [
                "--experimental-attention-variant",
                "dsa",
                "--dsa-indexer-n-heads",
                str(dsa_indexer_n_heads),
                "--dsa-indexer-head-dim",
                str(dsa_indexer_head_dim),
                "--dsa-indexer-topk",
                str(dsa_indexer_topk),
                "--dsa-indexer-loss-coeff",
                str(dsa_indexer_loss_coeff),
            ]
        )

    if plan.engram is not None:
        notes.append("Engram remains custom; no Megatron-native emitter yet")
    if plan.ngram_hash is not None:
        notes.append("ngram hash enrichment remains custom; no Megatron-native emitter yet")
    if plan.mhc is not None:
        notes.append("mHC remains custom; no Megatron-native emitter yet")
    if plan.mod is not None:
        notes.append("MoD remains custom; no Megatron-native emitter yet")
    if plan.moda is not None:
        notes.append("MoDA remains custom; no Megatron-native emitter yet")
    if plan.structure is not None:
        notes.append("structure-aware enrichment remains custom; no Megatron-native emitter yet")

    return MegatronArgsBundle(args=tuple(args), custom_notes=tuple(notes))
