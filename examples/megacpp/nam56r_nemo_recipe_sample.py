"""This example captures the NAM56R recipe values and CLI emission pattern.

Why it exists: the model needs one authoritative place that fixes hidden size,
head layout, Mamba dimensions, and MoE settings together.

What problem it solves: it turns one model spec into a deterministic Megatron
launcher argument list instead of hand-editing long shell commands.
"""

from __future__ import annotations

from dataclasses import dataclass


NAM56R_PATTERN = "AEMEAEMEAEMR"
NAM56R_DEPTH = 52
NAM56R_HIDDEN = 3584
NAM56R_FFN_HIDDEN = 18944
NAM56R_ATTN_HEADS = 56
NAM56R_KV_HEADS = 8
NAM56R_VOCAB = 65536
NAM56R_SEQ_LEN = 4096
NAM56R_ROPE_THETA = 500_000

MOE_NUM_EXPERTS = 16
MOE_ROUTER_TOPK = 4
MOE_FFN_HIDDEN = 896
MOE_SHARED_EXPERT_SIZE = 1024

MAMBA_NUM_HEADS = 56
MAMBA_STATE_DIM = 64
MAMBA_HEAD_DIM = 64
MAMBA_NUM_GROUPS = 8


def build_nemo_hybrid_pattern(*, pattern: str = NAM56R_PATTERN, depth: int = NAM56R_DEPTH, mtp_depths: int = 0, use_moe: bool = True) -> str:
    """Translate the MegaCpp POC NAM pattern into the NeMo/Megatron hybrid pattern syntax."""

    mapping = {
        "A": "*",
        "E": "E" if use_moe else "*",
        "M": "M",
        "R": "M",
    }

    tiled = [pattern[i % len(pattern)].upper() for i in range(depth)]
    translated: list[str] = []
    for symbol in tiled:
        try:
            translated.append(mapping[symbol])
        except KeyError as exc:
            raise ValueError(f"unsupported NAM56R symbol: {symbol!r}") from exc

    result = "".join(translated)
    if mtp_depths > 0:
        result = result + "/" + "/".join("*-" for _ in range(mtp_depths))
    return result


@dataclass
class NAM56RNeMoRecipe:
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
    mode: str = "nemo_native"

    use_moe: bool = True
    moe_num_experts: int = MOE_NUM_EXPERTS
    moe_router_topk: int = MOE_ROUTER_TOPK
    moe_ffn_hidden_size: int = MOE_FFN_HIDDEN
    moe_shared_expert_size: int = MOE_SHARED_EXPERT_SIZE

    use_mla: bool = True
    q_lora_rank: int = 64
    kv_lora_rank: int = 64
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 64

    attention_backend: str = "flash"
    use_selective_recompute: bool = True
    use_cuda_graphs: bool = False

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
        args: list[str] = []

        args.extend([
            "--tensor-model-parallel-size", str(tp),
            "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1",
        ])
        if tp > 1:
            args.append("--sequence-parallel")

        args.extend([
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        ])

        args.extend([
            "--hybrid-layer-pattern", self.build_hybrid_pattern(),
            "--hidden-size", str(self.hidden_size),
            "--ffn-hidden-size", str(self.ffn_hidden_size),
            "--num-attention-heads", str(self.num_attention_heads),
            "--num-query-groups", str(self.num_query_groups),
            "--num-layers", str(self.num_layers),
            "--seq-length", str(self.seq_length),
            "--max-position-embeddings", str(self.max_position_embeddings),
            "--make-vocab-size-divisible-by", "128",
        ])

        args.extend([
            "--mamba-num-heads", str(self.mamba_num_heads),
            "--mamba-state-dim", str(self.mamba_state_dim),
            "--mamba-head-dim", str(self.mamba_head_dim),
            "--mamba-num-groups", str(self.mamba_num_groups),
        ])

        args.extend([
            "--position-embedding-type", "rope",
            "--rotary-base", str(NAM56R_ROPE_THETA),
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--untie-embeddings-and-output-weights",
            "--no-gradient-accumulation-fusion",
            "--cross-entropy-loss-fusion",
            "--attention-backend", self.attention_backend,
        ])

        if self.use_selective_recompute and not self.use_cuda_graphs:
            args.extend(["--recompute-granularity", "selective"])

        if self.use_mla:
            args.extend([
                "--multi-latent-attention",
                "--q-lora-rank", str(self.q_lora_rank),
                "--kv-lora-rank", str(self.kv_lora_rank),
                "--qk-head-dim", str(self.qk_head_dim),
                "--qk-pos-emb-head-dim", str(self.qk_pos_emb_head_dim),
                "--v-head-dim", str(self.v_head_dim),
            ])

        if self.use_moe:
            args.extend([
                "--expert-model-parallel-size", "1",
                "--num-experts", str(self.moe_num_experts),
                "--moe-router-topk", str(self.moe_router_topk),
                "--moe-ffn-hidden-size", str(self.moe_ffn_hidden_size),
                "--moe-shared-expert-intermediate-size", str(self.moe_shared_expert_size),
                "--moe-grouped-gemm",
                "--moe-aux-loss-coeff", "0.0001",
                "--moe-router-score-function", "sigmoid",
                "--moe-router-enable-expert-bias",
                "--moe-router-dtype", "fp32",
                "--moe-token-dispatcher-type", "alltoall",
            ])

        return args
