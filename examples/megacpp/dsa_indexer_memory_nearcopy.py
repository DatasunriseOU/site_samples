"""Near-copy MegaCpp POC example: DSA index-score materialization reproducer.

This keeps the core shape and measurement contract close to the original
reproducer. The point is not just that one variant is smaller. The point is
that the upstream-style path materializes an fp32 intermediate shaped like
``[sq, b, h, sk]`` before the later reduction, while the fused path accumulates
directly into the final ``[b, sq, sk]`` output.

That is why a model that looks small on paper can still hit a much larger HBM
footprint during one helper.
"""

from __future__ import annotations

import gc

import torch


def upstream_compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    # [sq, b, h, d] x [sk, b, d] -> [sq, b, h, sk]
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)
    return index_scores


def fused_compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    sq, b, h, _ = q.shape
    sk = k.shape[0]
    index_scores = torch.zeros((b, sq, sk), dtype=torch.float32, device=q.device)
    k_bds = k.float().permute(1, 2, 0).contiguous()
    for hi in range(h):
        q_h = q[:, :, hi, :].float().permute(1, 0, 2).contiguous()
        logits_h = torch.bmm(q_h, k_bds)
        logits_h = torch.relu(logits_h)
        w_h = weights[:, :, hi].float().transpose(0, 1).unsqueeze(-1)
        index_scores.add_(logits_h * w_h)
    return index_scores


def reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**2)
    return float("nan")


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-8)
    return diff / scale


def measure(
    name: str,
    fn,
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    reset_cuda_memory()
    base_mb = peak_mb()
    out = fn(q, weights, k)
    peak = peak_mb()
    delta = peak - base_mb
    print(
        f"{name:<16} peak_alloc={peak:9.1f} MiB "
        f"(delta vs inputs {delta:+.1f} MiB)"
    )
    return out, delta


SHAPES = {
    "small": dict(b=2, sq=256, sk=256, h=4, d=32),
    "prod": dict(b=4, sq=4096, sk=4096, h=8, d=128),
    "full": dict(b=8, sq=4096, sk=4096, h=32, d=128),
}


def run_shape(shape_name: str, device: torch.device, dtype=torch.bfloat16) -> float:
    cfg = SHAPES[shape_name]
    b, sq, sk, h, d = cfg["b"], cfg["sq"], cfg["sk"], cfg["h"], cfg["d"]
    intermediate_mib = sq * b * h * sk * 4 / (1024**2)
    output_mib = b * sq * sk * 4 / (1024**2)
    print(
        f"shape={shape_name} b={b} sq={sq} sk={sk} h={h} d={d} dtype={dtype}"
    )
    print(f"expected upstream [sq,b,h,sk] fp32 intermediate: {intermediate_mib:.1f} MiB")
    print(f"expected output   [b,sq,sk]   fp32:              {output_mib:.1f} MiB")

    torch.manual_seed(0)
    q = torch.randn(sq, b, h, d, dtype=dtype, device=device) * 0.1
    weights = torch.randn(sq, b, h, dtype=dtype, device=device).abs() * 0.1
    k = torch.randn(sk, b, d, dtype=dtype, device=device) * 0.1

    out_up, peak_up = measure("upstream", upstream_compute_index_scores, q, weights, k)
    out_fu, peak_fu = measure("fused", fused_compute_index_scores, q, weights, k)
    rel = relative_error(out_up, out_fu)
    print(f"correctness: max rel_err = {rel:.3e}")
    print(
        f"memory: upstream {peak_up:.1f} MiB -> fused {peak_fu:.1f} MiB "
        f"(saved {peak_up - peak_fu:.1f} MiB)"
    )
    return rel


def run_gradcheck(device: torch.device) -> tuple[bool, float, float, float, float]:
    b, sq, sk, h, d = 2, 8, 8, 3, 4
    torch.manual_seed(1)
    q = torch.randn(sq, b, h, d, dtype=torch.float64, device=device, requires_grad=True)
    weights = (
        torch.randn(sq, b, h, dtype=torch.float64, device=device)
        .abs()
        .detach()
        .requires_grad_(True)
    )
    k = torch.randn(sk, b, d, dtype=torch.float64, device=device, requires_grad=True)

    def upstream_double(q_: torch.Tensor, w_: torch.Tensor, k_: torch.Tensor) -> torch.Tensor:
        idx = torch.einsum("sbhd,tbd->sbht", q_, k_)
        idx = torch.relu(idx)
        idx = idx * w_.unsqueeze(-1)
        idx = idx.sum(dim=2).transpose(0, 1)
        return idx

    def fused_double(q_: torch.Tensor, w_: torch.Tensor, k_: torch.Tensor) -> torch.Tensor:
        sq_, b_, h_, _ = q_.shape
        sk_ = k_.shape[0]
        out = torch.zeros((b_, sq_, sk_), dtype=q_.dtype, device=q_.device)
        k_bds = k_.permute(1, 2, 0).contiguous()
        for hi in range(h_):
            qh = q_[:, :, hi, :].permute(1, 0, 2).contiguous()
            lh = torch.bmm(qh, k_bds)
            lh = torch.relu(lh)
            wh = w_[:, :, hi].transpose(0, 1).unsqueeze(-1)
            out = out + lh * wh
        return out

    ok_up = torch.autograd.gradcheck(upstream_double, (q, weights, k), eps=1e-6, atol=1e-4, rtol=1e-3)
    ok_fu = torch.autograd.gradcheck(fused_double, (q, weights, k), eps=1e-6, atol=1e-4, rtol=1e-3)

    out_up = upstream_double(q, weights, k)
    out_fu = fused_double(q, weights, k)
    fwd_rel = relative_error(out_up, out_fu)

    for p in (q, weights, k):
        p.grad = None
    out_up.sum().backward()
    gq_up = q.grad.clone()
    gw_up = weights.grad.clone()
    gk_up = k.grad.clone()
    for p in (q, weights, k):
        p.grad = None
    out_fu.sum().backward()
    gq_fu = q.grad.clone()
    gw_fu = weights.grad.clone()
    gk_fu = k.grad.clone()

    return (
        ok_up and ok_fu,
        fwd_rel,
        relative_error(gq_up, gq_fu),
        relative_error(gw_up, gw_fu),
        relative_error(gk_up, gk_fu),
    )
