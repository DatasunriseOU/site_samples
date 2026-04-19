"""Near-copy MegaCpp POC example: SparseMLA dimension generalization contract.

This keeps the real comparison visible: one path assumes DeepSeek-sized MLA
dimensions, while the generalized path accepts NAM56R-sized dimensions and
threads them through both forward and backward plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SparseMlaShape:
    d_qk: int
    d_v: int
    topk: int
    kv_lora_rank: int
    qk_pos_emb_head_dim: int

    @property
    def d_total(self) -> int:
        return self.kv_lora_rank + self.qk_pos_emb_head_dim


DEEPSEEK_STYLE = SparseMlaShape(
    d_qk=576,
    d_v=512,
    topk=64,
    kv_lora_rank=512,
    qk_pos_emb_head_dim=64,
)

NAM56R_STYLE = SparseMlaShape(
    d_qk=128,
    d_v=64,
    topk=64,
    kv_lora_rank=64,
    qk_pos_emb_head_dim=64,
)


def hardcoded_dimension_path(shape: SparseMlaShape) -> dict[str, object]:
    return {
        "accepted": shape.d_total == 576 and shape.d_v == 512,
        "assumed_d_total": 576,
        "assumed_d_v": 512,
        "actual_d_total": shape.d_total,
        "actual_d_v": shape.d_v,
    }


def generalized_dimension_path(shape: SparseMlaShape) -> dict[str, object]:
    return {
        "accepted": True,
        "d_total": shape.d_total,
        "d_v": shape.d_v,
        "topk": shape.topk,
        "forward_contract": (shape.d_total, shape.d_v),
        "backward_contract": (shape.d_total, shape.d_v),
    }


def compare_dimension_paths() -> dict[str, dict[str, object]]:
    return {
        "deepseek_hardcoded_on_nam56r": hardcoded_dimension_path(NAM56R_STYLE),
        "generalized_on_nam56r": generalized_dimension_path(NAM56R_STYLE),
        "generalized_on_deepseek": generalized_dimension_path(DEEPSEEK_STYLE),
    }
