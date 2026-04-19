"""Grounded donor excerpt for the NAM56R NeMo recipe surface.

This keeps the real constants, pattern mapping, and launcher argument logic, but
uses a local pattern parser so the sample stays self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NAM56R_PATTERN = "AEMEAEMEAEMR"
NAM56R_DEPTH = 52
NAM56R_HIDDEN = 3584
NAM56R_FFN_HIDDEN = 18944
NAM56R_ATTN_HEADS = 56
NAM56R_KV_HEADS = 8
NAM56R_VOCAB = 65536
NAM56R_SEQ_LEN = 4096
MOE_NUM_EXPERTS = 16
MOE_TOPK = 4
MOE_FFN_HIDDEN = 896
MOE_SHARED_EXPERT_SIZE = 1024
MLA_Q_LORA_RANK = 64
MLA_KV_LORA_RANK = 64
MLA_QK_HEAD_DIM = 64
MLA_QK_POS_EMB_HEAD_DIM = 64
MLA_V_HEAD_DIM = 64
MAMBA_NUM_HEADS = 56
MAMBA_STATE_DIM = 64
MAMBA_HEAD_DIM = 64
MAMBA_NUM_GROUPS = 8
DEFAULT_MICRO_BATCH = 4
DEFAULT_GLOBAL_BATCH = 64


def parse_pattern(pattern: str, depth: int) -> list[str]:
    upper = pattern.upper()
    if not upper:
        raise ValueError("pattern must be non-empty")
    if "|" in upper:
        flattened = "".join(upper.split("|"))
        if len(flattened) != depth:
            raise ValueError(f"pipe-delimited pattern expands to {len(flattened)} layers, expected {depth}")
        return list(flattened)
    return [upper[index % len(upper)] for index in range(depth)]


def build_nemo_hybrid_pattern(
    *,
    pattern: str = NAM56R_PATTERN,
    depth: int = NAM56R_DEPTH,
    mtp_depths: int = 0,
    use_moe: bool = True,
) -> str:
    symbol_map = {"A": "*", "E": "E" if use_moe else "-", "M": "M", "R": "M", "D": "G", "G": "G"}
    translated = []
    for symbol in parse_pattern(pattern, depth):
        mapped = symbol_map.get(symbol)
        if mapped is None:
            raise ValueError(f"unsupported NAM56R symbol: {symbol!r}")
        translated.append(mapped)
    result = "".join(translated)
    if mtp_depths > 0:
        result += "/" + "/".join("*-" for _ in range(mtp_depths))
    return result


@dataclass(frozen=True)
class NAM56RNeMoRecipe:
    mode: Literal["nemo_native", "author_dp"] = "author_dp"
    hidden_size: int = NAM56R_HIDDEN
    ffn_hidden_size: int = NAM56R_FFN_HIDDEN
    num_attention_heads: int = NAM56R_ATTN_HEADS
    num_query_groups: int = NAM56R_KV_HEADS
    num_layers: int = NAM56R_DEPTH
    seq_length: int = NAM56R_SEQ_LEN
    max_position_embeddings: int = NAM56R_SEQ_LEN
    vocab_size: int = NAM56R_VOCAB
    mamba_num_heads: int = MAMBA_NUM_HEADS
    mamba_state_dim: int = MAMBA_STATE_DIM
    mamba_head_dim: int = MAMBA_HEAD_DIM
    mamba_num_groups: int = MAMBA_NUM_GROUPS
    pattern: str = NAM56R_PATTERN
    mtp_depths: int = 0
    use_moe: bool = True
    moe_num_experts: int = MOE_NUM_EXPERTS
    moe_router_topk: int = MOE_TOPK
    moe_ffn_hidden_size: int = MOE_FFN_HIDDEN
    moe_shared_expert_size: int = MOE_SHARED_EXPERT_SIZE
    use_mla: bool = True
    q_lora_rank: int = MLA_Q_LORA_RANK
    kv_lora_rank: int = MLA_KV_LORA_RANK
    qk_head_dim: int = MLA_QK_HEAD_DIM
    qk_pos_emb_head_dim: int = MLA_QK_POS_EMB_HEAD_DIM
    v_head_dim: int = MLA_V_HEAD_DIM
    micro_batch_size: int = DEFAULT_MICRO_BATCH
    global_batch_size: int = DEFAULT_GLOBAL_BATCH
    precision: Literal["bf16", "fp8"] = "bf16"
    spec_module: str = ""
    spec_name: str = ""
    ngram_hash_enabled: bool = False
    structure_enabled: bool = False
    use_tp_mamba3_mixer: bool = False

    def _tp(self) -> int:
        return 2 if self.mode == "nemo_native" else 1

    def build_hybrid_pattern(self) -> str:
        return build_nemo_hybrid_pattern(
            pattern=self.pattern,
            depth=self.num_layers,
            mtp_depths=self.mtp_depths,
            use_moe=self.use_moe,
        )

    def to_args(self) -> list[str]:
        tp = self._tp()
        args = [
            "--tensor-model-parallel-size",
            str(tp),
            "--pipeline-model-parallel-size",
            "1",
            "--context-parallel-size",
            "1",
            "--hybrid-layer-pattern",
            self.build_hybrid_pattern(),
            "--hidden-size",
            str(self.hidden_size),
            "--ffn-hidden-size",
            str(self.ffn_hidden_size),
            "--num-attention-heads",
            str(self.num_attention_heads),
            "--num-query-groups",
            str(self.num_query_groups),
            "--num-layers",
            str(self.num_layers),
            "--seq-length",
            str(self.seq_length),
            "--max-position-embeddings",
            str(self.max_position_embeddings),
            "--mamba-num-heads",
            str(self.mamba_num_heads),
            "--mamba-state-dim",
            str(self.mamba_state_dim),
            "--mamba-head-dim",
            str(self.mamba_head_dim),
            "--mamba-num-groups",
            str(self.mamba_num_groups),
            "--micro-batch-size",
            str(self.micro_batch_size),
            "--global-batch-size",
            str(self.global_batch_size),
        ]
        if tp > 1:
            args.append("--sequence-parallel")
        if self.use_mla:
            args.extend(
                [
                    "--multi-latent-attention",
                    "--q-lora-rank",
                    str(self.q_lora_rank),
                    "--kv-lora-rank",
                    str(self.kv_lora_rank),
                    "--qk-head-dim",
                    str(self.qk_head_dim),
                    "--qk-pos-emb-head-dim",
                    str(self.qk_pos_emb_head_dim),
                    "--v-head-dim",
                    str(self.v_head_dim),
                ]
            )
        if self.use_moe:
            args.extend(
                [
                    "--num-experts",
                    str(self.moe_num_experts),
                    "--moe-router-topk",
                    str(self.moe_router_topk),
                    "--moe-ffn-hidden-size",
                    str(self.moe_ffn_hidden_size),
                    "--moe-shared-expert-intermediate-size",
                    str(self.moe_shared_expert_size),
                ]
            )
        if self.spec_module:
            args.extend(["--spec", self.spec_module])
        return args
