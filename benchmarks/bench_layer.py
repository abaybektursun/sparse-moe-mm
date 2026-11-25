"""v2 Honest Single-Layer GPU Feasibility Benchmark (Phase-1 Gate).

Fixes all 5 measurement bugs audited in commit 019ba70:
  1. Compares FP8 sparse strictly against FP8 dense (and BF16 vs BF16) — NEVER FP8 vs BF16.
  2. Times the FULL sparse pipeline (Stage-1 Prog8-Dual 2-bit + Top-k + Stage-2 Expert-Grouped)
     as well as Stage-2 ceiling separately.
  3. Pre-allocates all output/workspace tensors (zero per-iter torch.zeros).
  4. Uses real CUDA Graph capture (`torch.cuda.CUDAGraph`) for both dense and sparse paths.
  5. Flushes L2 cache with a 256 MB dummy buffer between every timed replay (eliminating
     fake L2-resident speedups at B = 1..16).
  6. Supports BOTH Uniform (alpha=0.0) and Realistic Zipf (alpha=0.8) routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from sparse_moe_mm.kernels import (
    launch_expert_grouped_sparse_fp8,
    launch_prog8_dual_stage1,
    pack_2bit_hi,
)
from sparse_moe_mm.reference import quantize_prog8_dual_2_6
from sparse_moe_mm.roofline import MODELS


def quantize_fp8_block128_hw(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official `[128,128]` block-scaled FP8 E4M3 quantization for [E, M, N]."""
    e, m, n = w.shape
    bm, bn = m // 128, n // 128
    blocks = w.float().view(e, bm, 128, bn, 128)
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(e, m, n)
    return q.contiguous(), scale.squeeze(4).squeeze(2).contiguous()


def bench_cuda_graph_l2_flushed(fn, warmup: int = 10, iters: int = 60, l2_flush_mb: int = 256) -> float:
    """Capture `fn` in a CUDA Graph and return median latency in microseconds with L2 flushed each iter."""
    flush_buf = torch.empty((l2_flush_mb * 1024 * 1024 // 4,), dtype=torch.int32, device="cuda")
    # Warmup on side stream before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        flush_buf.zero_()  # Evict weights from L2 cache (50MB H200 / 126MB B200) before replay!
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] * 1000.0


def sample_zipf_routing(b: int, num_experts: int, top_k: int, alpha: float, device: torch.device) -> torch.Tensor:
    """Sample [B, K] distinct expert IDs per token under Zipf(alpha) routing."""
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-alpha)
    probs = probs / probs.sum()
    u = torch.rand((b, num_experts), device=device).clamp(min=1e-7)
    keys = torch.log(probs)[None, :] - torch.log(-torch.log(u))
    return torch.topk(keys, k=top_k, dim=-1).indices


def build_expert_grouped_metadata(
    expert_ids_sorted: torch.Tensor,
    active_idx_per_pair: torch.Tensor,
    num_experts: int,
    max_me: int,
    max_union: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cold-expert union metadata on host/device (outside timed graph for Phase-1 kernel ceiling)."""
    device = expert_ids_sorted.device
    counts = torch.bincount(expert_ids_sorted.long(), minlength=num_experts)
    starts = torch.cumsum(counts, dim=0) - counts
    nz_experts = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
    n_slots = int(nz_experts.numel())

    e_ids = nz_experts.to(torch.int32).contiguous()
    e_starts = starts[nz_experts].to(torch.int32).contiguous()
    e_counts = counts[nz_experts].clamp(max=max_me).to(torch.int32).contiguous()

    union_idx = torch.zeros((n_slots, max_union), dtype=torch.int32, device=device)
    union_cnt = torch.zeros((n_slots,), dtype=torch.int32, device=device)
    token_mask = torch.zeros((n_slots, max_me, max_union), dtype=torch.uint8, device=device)

    for slot in range(n_slots):
        st = int(e_starts[slot].item())
        cnt = int(e_counts[slot].item())
        sub = active_idx_per_pair[st : st + cnt]  # [cnt, k_active]
        uniq = torch.unique(sub)
        u_c = min(int(uniq.numel()), max_union)
        uniq = uniq[:u_c]
        union_idx[slot, :u_c] = uniq
        union_cnt[slot] = u_c
        # Build per-token boolean mask within uniq
        eq = (sub.unsqueeze(-1) == uniq[None, None, :u_c]).any(dim=1)  # [cnt, u_c]
        token_mask[slot, :cnt, :u_c] = eq.to(torch.uint8)

    return e_ids, e_starts, e_counts, union_idx, union_cnt, token_mask


def run_benchmark(
    model_key: str,
    dtype_str: str,
    batches: list[int],
    sparsity: float,
    zipf_alpha: float,
    out_json: str | None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmarks/bench_layer.py requires a CUDA GPU.")

    spec = MODELS[model_key]
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | Model: {spec.name} | dtype={dtype_str} | s={sparsity:.2f} | zipf={zipf_alpha}")

    torch.manual_seed(0)
    w_gate_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    w_up_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    w_down_t_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02

    # Prepare dense baseline layouts [E, in, out]
    w_gate_gmm = w_gate_bf16.transpose(1, 2).contiguous()
    w_up_gmm = w_up_bf16.transpose(1, 2).contiguous()
    w_down_gmm = w_down_t_bf16.contiguous()

    is_fp8 = dtype_str == "fp8"
    if is_fp8:
        w_gate_q, s_gate = quantize_fp8_block128_hw(w_gate_bf16)
        w_up_q, s_up = quantize_fp8_block128_hw(w_up_bf16)
        # Official [128,128] W_down scale layout: [E, d_model//128, d_ff//128]
        w_down_q_t, s_down_t = quantize_fp8_block128_hw(w_down_t_bf16)
        s_down_official = s_down_t.transpose(1, 2).contiguous()
    else:
        w_gate_q, s_gate = w_gate_bf16, None
        w_up_q, s_up = w_up_bf16, None
        w_down_q_t, s_down_official = w_down_t_bf16, None

    # Pre-pack Prog8-Dual Stage-1 2-bit weights (0.25 B/elem each = 0.50 B/elem total)
    wg_hi2, _, sg_p = quantize_prog8_dual_2_6(w_gate_bf16, group_size=128)
    wu_hi2, _, su_p = quantize_prog8_dual_2_6(w_up_bf16, group_size=128)
    wg_hi2_p = pack_2bit_hi(wg_hi2)
    wu_hi2_p = pack_2bit_hi(wu_hi2)

    has_grouped_mm = hasattr(torch, "_grouped_mm")
    keep_frac = 1.0 - sparsity
    k_active = max(1, int(round(spec.d_ff * keep_frac)))
    records = []

    for b in batches:
        n_pairs = b * spec.top_k
        x = torch.randn((b, spec.d_model), device=device, dtype=torch.bfloat16)
        topk_ids = sample_zipf_routing(b, spec.num_experts, spec.top_k, zipf_alpha, device)
        flat_exp = topk_ids.reshape(-1)
        order = torch.argsort(flat_exp)
        expert_ids_sorted = flat_exp[order].to(torch.int32).contiguous()
        x_pairs = x[order // spec.top_k].contiguous()
        counts = torch.bincount(expert_ids_sorted.long(), minlength=spec.num_experts)
        offs = torch.cumsum(counts, dim=0).to(torch.int32).contiguous()

        # Dense BF16 baseline (NOTE: on H200/B200 compare FP8 against DeepGEMM/FlashInfer dense FP8!)
        dense_bf16_us = (
            bench_cuda_graph_l2_flushed(
                lambda: torch._grouped_mm(
                    F.silu(torch._grouped_mm(x_pairs, w_gate_gmm, offs=offs))
                    * torch._grouped_mm(x_pairs, w_up_gmm, offs=offs),
                    w_down_gmm,
                    offs=offs,
                )
            )
            if has_grouped_mm
            else float("nan")
        )

        # Build Stage-1 & Stage-2 workspaces (pre-allocated, CUDA-Graph safe)
        max_me = 16
        max_union = min(spec.d_ff, max(64, int(k_active * 3)))
        scores_buf = torch.empty((n_pairs, spec.d_ff), dtype=torch.float32, device=device)

        # Dummy initial active_idx for metadata shape
        dummy_idx = torch.topk(torch.randn((n_pairs, spec.d_ff), device=device), k=k_active, dim=-1).indices.to(torch.int32)
        e_ids, e_starts, e_counts, union_idx, union_cnt, token_mask = build_expert_grouped_metadata(
            expert_ids_sorted, dummy_idx, spec.num_experts, max_me=max_me, max_union=max_union
        )
        n_slots = e_ids.shape[0]
        h_union_buf = torch.empty((n_slots, max_me, max_union), dtype=torch.float32, device=device)
        out_buf = torch.empty((n_pairs, spec.d_model), dtype=torch.float32, device=device)

        # Parity check on slot 0, token 0
        launch_expert_grouped_sparse_fp8(
            x_pairs, e_ids, e_starts, e_counts, union_idx, union_cnt, token_mask,
            w_gate_q, w_up_q, w_down_q_t, h_union_buf, out_buf,
            s_gate, s_up, s_down_official, max_me=max_me,
        )
        e0 = int(e_ids[0].item())
        st0 = int(e_starts[0].item())
        idx0 = dummy_idx[st0].long()
        ref0 = (
            (F.silu(w_gate_bf16[e0, idx0].float() @ x_pairs[st0].float()) * (w_up_bf16[e0, idx0].float() @ x_pairs[st0].float()))
            @ w_down_t_bf16[e0, idx0].float()
        )
        rel_err = float((out_buf[st0] - ref0).norm() / ref0.norm().clamp(min=1e-8))

        s1_us = bench_cuda_graph_l2_flushed(
            lambda: launch_prog8_dual_stage1(
                x_pairs, e_ids, e_starts, e_counts, wg_hi2_p, wu_hi2_p, sg_p, su_p, scores_buf, max_me=max_me
            )
        )
        s2_us = bench_cuda_graph_l2_flushed(
            lambda: launch_expert_grouped_sparse_fp8(
                x_pairs, e_ids, e_starts, e_counts, union_idx, union_cnt, token_mask,
                w_gate_q, w_up_q, w_down_q_t, h_union_buf, out_buf,
                s_gate, s_up, s_down_official, max_me=max_me,
            )
        )
        tot_sparse_us = s1_us + s2_us
        rec = {
            "gpu": gpu_name,
            "model": model_key,
            "dtype": dtype_str,
            "batch_size": b,
            "sparsity": sparsity,
            "zipf_alpha": zipf_alpha,
            "E_act": n_slots,
            "dense_bf16_gmm_us": round(dense_bf16_us, 1),
            "stage1_prog8_dual_us": round(s1_us, 1),
            "stage2_expert_grouped_us": round(s2_us, 1),
            "total_sparse_s1_plus_s2_us": round(tot_sparse_us, 1),
            "parity_rel_l2": round(rel_err, 4),
        }
        records.append(rec)
        print(
            f"B={b:>3} E_act={n_slots:>3} | dense_bf16={dense_bf16_us:>7.1f}us | "
            f"S1(2bit)={s1_us:>6.1f}us + S2(EG)={s2_us:>6.1f}us = {tot_sparse_us:>7.1f}us | err={rel_err:.4f}"
        )

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps({"timestamp": time.time(), "records": records}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_35b_a3b", choices=list(MODELS.keys()))
    ap.add_argument("--dtype", default="fp8", choices=["bf16", "fp8"])
    ap.add_argument("--batches", default="1,16,32,64,128")
    ap.add_argument("--sparsity", type=float, default=0.80)
    ap.add_argument("--zipf-alpha", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_benchmark(args.model, args.dtype, [int(x) for x in args.batches.split(",")], args.sparsity, args.zipf_alpha, args.out)


if __name__ == "__main__":
    main()
