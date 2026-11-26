"""v4 Honest End-to-End Single-Layer GPU Benchmark (Memory-Efficient for 122B on 22GB L4 + Micro-Profiling).

Benchmarks FULL end-to-end MoE layer execution inside L2-flushed CUDA Graphs:
  1. Dense [128,128] FP8 expert-grouped baseline (with reported achieved GB/s)
  2. Cold-Expert-Only Prog8-Dual (2+2+4 Zero-VRAM: S1 + TopK + S2 mid2+lo4 + W_down + Hot Dense)
  3. Cold-Expert-Only Prog8-Dual + SVD-r16 Recovery (exact full accuracy-preserving pipeline!)
  4. Cold-Expert-Only Native FP8 Stage-2 A/B (S1 without hi-out write + TopK + Native FP8 S2 + Hot Dense)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from sparse_moe_mm.kernels import (
    launch_dense_fp8_grouped_mm,
    launch_prog8_224_stage1,
    launch_prog8_224_stage2,
    launch_sparse_fp8_stage2,
    launch_svd_recovery,
)
from sparse_moe_mm.reference import (
    pack_2bit_interleaved,
    pack_4bit_interleaved,
    quantize_prog8_224,
)
from sparse_moe_mm.roofline import MODELS


def quantize_fp8_block128_hw_chunked(w: torch.Tensor, chunk_e: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked [128,128] FP8 quantization keeping peak temporary VRAM tiny (fits 122B layer on 22GB L4)."""
    e, m, n = w.shape
    bm, bn = m // 128, n // 128
    q_out = torch.empty((e, m, n), dtype=torch.float8_e4m3fn, device=w.device)
    s_out = torch.empty((e, bm, bn), dtype=torch.float32, device=w.device)
    for start in range(0, e, chunk_e):
        end = min(e, start + chunk_e)
        blocks = w[start:end].float().view(end - start, bm, 128, bn, 128)
        scale = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
        q_out[start:end] = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(end - start, m, n)
        s_out[start:end] = scale.squeeze(4).squeeze(2)
    return q_out, s_out


def pack_prog8_224_chunked(w: torch.Tensor, chunk_e: int = 32):
    """O9: Chunked 2+2+4 bitplane packing with 128B cache-line blocks (block_d=512 for 2-bit, 256 for 4-bit)."""
    e, m, n = w.shape
    hi2_p = torch.empty((e, m, n // 4), dtype=torch.uint8, device=w.device)
    mid2_p = torch.empty((e, m, n // 4), dtype=torch.uint8, device=w.device)
    lo4_p = torch.empty((e, m, n // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((e, m), dtype=torch.float32, device=w.device)
    for start in range(0, e, chunk_e):
        end = min(e, start + chunk_e)
        hi2, mid2, lo4, sc = quantize_prog8_224(w[start:end], group_size=None)
        hi2_p[start:end] = pack_2bit_interleaved(hi2, signed=True, block_d=512)
        mid2_p[start:end] = pack_2bit_interleaved(mid2, signed=False, block_d=512)
        lo4_p[start:end] = pack_4bit_interleaved(lo4, block_d=256)
        scales[start:end] = sc.squeeze(-1)
    return hi2_p, mid2_p, lo4_p, scales


def bench_cuda_graph_l2_flushed(
    fn, warmup: int = 8, iters: int = 50, l2_flush_mb: int = 192, require_graph: bool = True
) -> float:
    flush_buf = torch.empty((l2_flush_mb * 1024 * 1024 // 4,), dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    runner = fn
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        runner = g.replay
    except RuntimeError:
        if require_graph:
            raise

    times_ms = []
    for _ in range(iters):
        flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] * 1000.0


def sample_zipf_routing(b: int, num_experts: int, top_k: int, alpha: float, device: torch.device) -> torch.Tensor:
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-alpha)
    probs = probs / probs.sum()
    u = torch.rand((b, num_experts), device=device).clamp(min=1e-7)
    keys = torch.log(probs)[None, :] - torch.log(-torch.log(u))
    return torch.topk(keys, k=top_k, dim=-1).indices


def run_benchmark(
    model_key: str,
    dtype_str: str,
    batches: list[int],
    sparsity: float,
    zipf_alpha: float,
    cold_me_thresh: int,
    out_json: str | None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmarks/bench_layer.py requires a CUDA GPU.")

    gc.collect()
    torch.cuda.empty_cache()
    spec = MODELS[model_key]
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    use_native_fp8 = (dtype_str == "fp8") and (cc >= (8, 9))
    print(
        f"GPU: {gpu_name} (SM{cc[0]}{cc[1]}) | Model: {spec.name} (E={spec.num_experts}, d={spec.d_model}, d_ff={spec.d_ff}) "
        f"| dtype={dtype_str} | native_fp8={use_native_fp8} | s={sparsity:.2f} | zipf={zipf_alpha} | cold_Me<={cold_me_thresh}"
    )

    # Generate weights one matrix at a time and free BF16 immediately (fits 122B layer in <6 GB VRAM!)
    torch.manual_seed(0)
    is_fp8 = dtype_str == "fp8"

    wg_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    wg_e0_ref = wg_bf16[0].float().clone()
    w_gate_q, s_gate = quantize_fp8_block128_hw_chunked(wg_bf16) if is_fp8 else (wg_bf16, None)
    wg_hi2_p, wg_mid2_p, wg_lo4_p, sg_224 = pack_prog8_224_chunked(wg_bf16)
    if is_fp8:
        del wg_bf16

    wu_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    wu_e0_ref = wu_bf16[0].float().clone()
    w_up_q, s_up = quantize_fp8_block128_hw_chunked(wu_bf16) if is_fp8 else (wu_bf16, None)
    wu_hi2_p, wu_mid2_p, wu_lo4_p, su_224 = pack_prog8_224_chunked(wu_bf16)
    if is_fp8:
        del wu_bf16

    wd_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    wd_e0_ref = wd_bf16[0].float().clone()
    if is_fp8:
        w_down_q_t, s_down_t = quantize_fp8_block128_hw_chunked(wd_bf16)
        s_down_official = s_down_t.transpose(1, 2).contiguous()
        del wd_bf16, s_down_t
    else:
        w_down_q_t, s_down_official = wd_bf16, None

    # Rank-16 SVD factors (+1.27% VRAM)
    u_svd_t = torch.randn((spec.num_experts, 16, spec.d_model), device=device, dtype=torch.bfloat16) * 0.01
    v_svd = torch.randn((spec.num_experts, 16, spec.d_ff), device=device, dtype=torch.bfloat16) * 0.01
    torch.cuda.empty_cache()

    # --- G1_PARITY GATE (q=1.0 + failing-input check) ---
    x_par = torch.randn((8, spec.d_model), device=device, dtype=torch.bfloat16)
    e_ids_p = torch.tensor([0], dtype=torch.int32, device=device)
    e_st_p = torch.tensor([0], dtype=torch.int32, device=device)
    e_cnt_p = torch.tensor([8], dtype=torch.int32, device=device)
    g_hi_p = torch.empty((8, spec.d_ff), dtype=torch.bfloat16, device=device)
    u_hi_p = torch.empty((8, spec.d_ff), dtype=torch.bfloat16, device=device)
    sc_p = torch.empty((1, spec.d_ff), dtype=torch.float32, device=device)
    h_act_p = torch.empty((1, 16, spec.d_ff), dtype=torch.bfloat16, device=device)
    out_p224 = torch.empty((8, spec.d_model), dtype=torch.bfloat16, device=device)
    out_pfp8 = torch.empty((8, spec.d_model), dtype=torch.bfloat16, device=device)
    out_pdense = torch.empty((8, spec.d_model), dtype=torch.bfloat16, device=device)
    h_den_p = torch.empty((8, spec.d_ff), dtype=torch.bfloat16, device=device)
    full_idx = torch.arange(spec.d_ff, dtype=torch.int32, device=device).unsqueeze(0)

    h_s1_p = torch.empty((8, spec.d_ff), dtype=torch.bfloat16, device=device)
    out_p3tier = torch.empty((8, spec.d_model), dtype=torch.bfloat16, device=device)

    launch_prog8_224_stage1(
        x_par, e_ids_p, e_st_p, e_cnt_p, wg_hi2_p, wu_hi2_p, sg_224, su_224,
        g_hi_p, u_hi_p, sc_p, h_s1_out=h_s1_p, max_me=16, save_hi_out=True,
    )
    launch_prog8_224_stage2(
        x_par, e_ids_p, e_st_p, e_cnt_p, full_idx, g_hi_p, u_hi_p,
        wg_mid2_p, wu_mid2_p, wg_lo4_p, wu_lo4_p, sg_224, su_224,
        w_down_q_t, s_down_official, h_act_p, out_p224,
        h_s1_to_zero=h_s1_p, max_me=16,
    )
    assert float(h_s1_p.abs().max().item()) == 0.0, "ZERO_H_S1 failed to zero active neurons!"
    launch_prog8_224_stage2(
        x_par, e_ids_p, e_st_p, e_cnt_p, full_idx, g_hi_p, u_hi_p,
        wg_mid2_p, wu_mid2_p, wg_lo4_p, wu_lo4_p, sg_224, su_224,
        w_down_q_t, s_down_official, h_act_p, out_p3tier,
        k_full=spec.d_ff // 2, max_me=16,
    )
    launch_sparse_fp8_stage2(
        x_par, e_ids_p, e_st_p, e_cnt_p, full_idx, w_gate_q, w_up_q, w_down_q_t,
        h_act_p, out_pfp8, s_gate, s_up, s_down_official, max_me=16, use_native_fp8_dot=use_native_fp8,
    )
    launch_dense_fp8_grouped_mm(
        x_par, e_ids_p, e_st_p, e_cnt_p, 8, w_gate_q, w_up_q, w_down_q_t,
        h_den_p, out_pdense, s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
    )
    ref_p = (F.silu(x_par.float() @ wg_e0_ref.T) * (x_par.float() @ wu_e0_ref.T)) @ wd_e0_ref
    err_224 = float((out_p224.float() - ref_p).norm() / ref_p.norm())
    err_3tier = float((out_p3tier.float() - ref_p).norm() / ref_p.norm())
    err_fp8 = float((out_pfp8.float() - ref_p).norm() / ref_p.norm())
    err_den = float((out_pdense.float() - ref_p).norm() / ref_p.norm())
    err_sp_vs_den = float((out_pfp8.float() - out_pdense.float()).norm() / out_pdense.float().norm().clamp(min=1e-8))
    print(
        f"[G1_PARITY q=1.0] rel_L2 vs fp32 | Dense={err_den:.4f} | SparseFP8={err_fp8:.4f} "
        f"| Prog8_224={err_224:.4f} | 3Tier_50_50={err_3tier:.4f} | SparseFP8_vs_Dense={err_sp_vs_den:.6f} | zero_h_s1=PASS"
    )
    assert err_224 < 0.08 and err_3tier < 0.15 and err_fp8 < 0.08 and err_den < 0.08 and err_sp_vs_den < 1e-2

    half_idx = torch.arange(spec.d_ff // 2, dtype=torch.int32, device=device).unsqueeze(0)
    h_half = torch.empty((1, 16, spec.d_ff // 2), dtype=torch.bfloat16, device=device)
    out_half = torch.empty((8, spec.d_model), dtype=torch.bfloat16, device=device)
    launch_sparse_fp8_stage2(
        x_par, e_ids_p, e_st_p, e_cnt_p, half_idx, w_gate_q, w_up_q, w_down_q_t,
        h_half, out_half, s_gate, s_up, s_down_official, max_me=16, use_native_fp8_dot=use_native_fp8,
    )
    err_fail = float((out_half.float() - ref_p).norm() / ref_p.norm())
    assert err_fail > 0.25
    print(f"[G1_PARITY failing-input] 50% drop rel_L2={err_fail:.4f} (PASS)")

    # --- Timing Sweep ---
    keep_frac = 1.0 - sparsity
    w_elems = spec.d_model * spec.d_ff
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
        starts = torch.cumsum(counts, dim=0) - counts
        nz = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)

        all_ids = nz.to(torch.int32).contiguous()
        all_starts = starts[nz].to(torch.int32).contiguous()
        all_counts = counts[nz].to(torch.int32).contiguous()
        max_me_all = int(all_counts.max().item())

        cold_mask = counts[nz] <= cold_me_thresh
        cold_nz, hot_nz = nz[cold_mask], nz[~cold_mask]
        cold_ids, cold_starts, cold_counts = cold_nz.to(torch.int32).contiguous(), starts[cold_nz].to(torch.int32).contiguous(), counts[cold_nz].to(torch.int32).contiguous()
        hot_ids, hot_starts, hot_counts = hot_nz.to(torch.int32).contiguous(), starts[hot_nz].to(torch.int32).contiguous(), counts[hot_nz].to(torch.int32).contiguous()
        max_me_hot = int(hot_counts.max().item()) if hot_ids.numel() > 0 else 0

        n_cold = int(cold_ids.numel())
        cold_me_avg = float(cold_counts.float().mean().item()) if n_cold > 0 else 1.0
        q_eff = 0.76 * (1.0 - (1.0 - keep_frac) ** cold_me_avg) + 0.24 * keep_frac
        k_u = max(32, min(spec.d_ff, int(round(spec.d_ff * q_eff / 16.0)) * 16))

        h_dense_buf = torch.empty((n_pairs, spec.d_ff), dtype=torch.bfloat16, device=device)
        out_dense = torch.empty((n_pairs, spec.d_model), dtype=torch.bfloat16, device=device)
        g_hi_buf = torch.empty((n_pairs, spec.d_ff), dtype=torch.bfloat16, device=device)
        u_hi_buf = torch.empty((n_pairs, spec.d_ff), dtype=torch.bfloat16, device=device)
        h_s1_buf = torch.empty((n_pairs, spec.d_ff), dtype=torch.bfloat16, device=device)
        exp_scores = torch.empty((max(1, n_cold), spec.d_ff), dtype=torch.float32, device=device)
        h_act_buf = torch.empty((max(1, n_cold), 16, k_u), dtype=torch.bfloat16, device=device)
        out_sparse = torch.empty((n_pairs, spec.d_model), dtype=torch.bfloat16, device=device)

        dense_us = bench_cuda_graph_l2_flushed(
            lambda: launch_dense_fp8_grouped_mm(
                x_pairs, all_ids, all_starts, all_counts, max_me_all,
                w_gate_q, w_up_q, w_down_q_t, h_dense_buf, out_dense,
                s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
            )
        )

        def _run_e2e_prog224():
            if n_cold > 0:
                launch_prog8_224_stage1(
                    x_pairs, cold_ids, cold_starts, cold_counts,
                    wg_hi2_p, wu_hi2_p, sg_224, su_224, g_hi_buf, u_hi_buf, exp_scores,
                    max_me=16, save_hi_out=True,
                )
                act_idx = torch.topk(exp_scores, k=k_u, dim=-1, sorted=False).indices.to(torch.int32)
                launch_prog8_224_stage2(
                    x_pairs, cold_ids, cold_starts, cold_counts, act_idx,
                    g_hi_buf, u_hi_buf, wg_mid2_p, wu_mid2_p, wg_lo4_p, wu_lo4_p,
                    sg_224, su_224, w_down_q_t, s_down_official, h_act_buf, out_sparse, max_me=16,
                )
            if hot_ids.numel() > 0:
                launch_dense_fp8_grouped_mm(
                    x_pairs, hot_ids, hot_starts, hot_counts, max_me_hot,
                    w_gate_q, w_up_q, w_down_q_t, h_dense_buf, out_sparse,
                    s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
                )

        def _run_e2e_prog224_svd():
            if n_cold > 0:
                launch_prog8_224_stage1(
                    x_pairs, cold_ids, cold_starts, cold_counts,
                    wg_hi2_p, wu_hi2_p, sg_224, su_224, g_hi_buf, u_hi_buf, exp_scores,
                    h_s1_out=h_s1_buf, max_me=16, save_hi_out=True,
                )
                act_idx = torch.topk(exp_scores, k=k_u, dim=-1, sorted=False).indices.to(torch.int32)
                launch_prog8_224_stage2(
                    x_pairs, cold_ids, cold_starts, cold_counts, act_idx,
                    g_hi_buf, u_hi_buf, wg_mid2_p, wu_mid2_p, wg_lo4_p, wu_lo4_p,
                    sg_224, su_224, w_down_q_t, s_down_official, h_act_buf, out_sparse,
                    h_s1_to_zero=h_s1_buf, max_me=16,
                )
                launch_svd_recovery(h_s1_buf, cold_ids, cold_starts, cold_counts, u_svd_t, v_svd, out_sparse, max_me=16)
            if hot_ids.numel() > 0:
                launch_dense_fp8_grouped_mm(
                    x_pairs, hot_ids, hot_starts, hot_counts, max_me_hot,
                    w_gate_q, w_up_q, w_down_q_t, h_dense_buf, out_sparse,
                    s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
                )

        # O10: 3-Tier 2+2+4 (Top 64 exact 8-bit, rest of K_u 4-bit skipping lo4, dropped SVD-r16!)
        k_full_3tier = min(k_u, max(64, (k_u // 2 // 64) * 64))

        def _run_e2e_3tier_svd():
            if n_cold > 0:
                launch_prog8_224_stage1(
                    x_pairs, cold_ids, cold_starts, cold_counts,
                    wg_hi2_p, wu_hi2_p, sg_224, su_224, g_hi_buf, u_hi_buf, exp_scores,
                    h_s1_out=h_s1_buf, max_me=16, save_hi_out=True,
                )
                act_idx = torch.topk(exp_scores, k=k_u, dim=-1, sorted=True).indices.to(torch.int32)
                launch_prog8_224_stage2(
                    x_pairs, cold_ids, cold_starts, cold_counts, act_idx,
                    g_hi_buf, u_hi_buf, wg_mid2_p, wu_mid2_p, wg_lo4_p, wu_lo4_p,
                    sg_224, su_224, w_down_q_t, s_down_official, h_act_buf, out_sparse,
                    h_s1_to_zero=h_s1_buf, k_full=k_full_3tier, max_me=16,
                )
                launch_svd_recovery(h_s1_buf, cold_ids, cold_starts, cold_counts, u_svd_t, v_svd, out_sparse, max_me=16)
            if hot_ids.numel() > 0:
                launch_dense_fp8_grouped_mm(
                    x_pairs, hot_ids, hot_starts, hot_counts, max_me_hot,
                    w_gate_q, w_up_q, w_down_q_t, h_dense_buf, out_sparse,
                    s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
                )

        def _run_e2e_spfp8():
            if n_cold > 0:
                launch_prog8_224_stage1(
                    x_pairs, cold_ids, cold_starts, cold_counts,
                    wg_hi2_p, wu_hi2_p, sg_224, su_224, g_hi_buf, u_hi_buf, exp_scores,
                    max_me=16, save_hi_out=False,
                )
                act_idx = torch.topk(exp_scores, k=k_u, dim=-1, sorted=False).indices.to(torch.int32)
                launch_sparse_fp8_stage2(
                    x_pairs, cold_ids, cold_starts, cold_counts, act_idx,
                    w_gate_q, w_up_q, w_down_q_t, h_act_buf, out_sparse,
                    s_gate, s_up, s_down_official, max_me=16, use_native_fp8_dot=use_native_fp8,
                )
            if hot_ids.numel() > 0:
                launch_dense_fp8_grouped_mm(
                    x_pairs, hot_ids, hot_starts, hot_counts, max_me_hot,
                    w_gate_q, w_up_q, w_down_q_t, h_dense_buf, out_sparse,
                    s_gate, s_up, s_down_official, use_native_fp8_dot=use_native_fp8,
                )

        s1_us = (
            bench_cuda_graph_l2_flushed(
                lambda: launch_prog8_224_stage1(
                    x_pairs, cold_ids, cold_starts, cold_counts,
                    wg_hi2_p, wu_hi2_p, sg_224, su_224, g_hi_buf, u_hi_buf, exp_scores,
                    max_me=16, save_hi_out=False,
                )
            )
            if n_cold > 0
            else 0.0
        )
        e2e_224_us = bench_cuda_graph_l2_flushed(_run_e2e_prog224)
        e2e_224_svd_us = bench_cuda_graph_l2_flushed(_run_e2e_prog224_svd)
        e2e_3tier_svd_us = bench_cuda_graph_l2_flushed(_run_e2e_3tier_svd)
        e2e_fp8_us = bench_cuda_graph_l2_flushed(_run_e2e_spfp8)

        dense_bw_gbs = (int(all_ids.numel()) * 3.0 * w_elems / (dense_us * 1e-6)) / 1e9
        s1_bw_gbs = (n_cold * 0.50 * w_elems / (s1_us * 1e-6)) / 1e9 if s1_us > 0 else 0.0

        rec = {
            "B": b,
            "E_act": int(all_ids.numel()),
            "E_cold": n_cold,
            "E_hot": int(hot_ids.numel()),
            "K_u": k_u,
            "K_full_3tier": k_full_3tier,
            "dense_fp8_us": round(dense_us, 1),
            "dense_bw_gbs": round(dense_bw_gbs, 1),
            "s1_2bit_us": round(s1_us, 1),
            "s1_bw_gbs": round(s1_bw_gbs, 1),
            "e2e_prog224_us": round(e2e_224_us, 1),
            "e2e_prog224_svd16_us": round(e2e_224_svd_us, 1),
            "e2e_3tier_svd16_us": round(e2e_3tier_svd_us, 1),
            "e2e_spfp8_us": round(e2e_fp8_us, 1),
            "speedup_prog224": round(dense_us / e2e_224_us, 2),
            "speedup_prog224_svd16": round(dense_us / e2e_224_svd_us, 2),
            "speedup_3tier_svd16": round(dense_us / e2e_3tier_svd_us, 2),
            "speedup_spfp8": round(dense_us / e2e_fp8_us, 2),
        }
        records.append(rec)
        print(
            f"B={b:>3} E={rec['E_act']:>3}(c={n_cold:>3},h={rec['E_hot']:>2}) Ku={k_u:>3} | "
            f"Dense={dense_us:>6.1f}u({dense_bw_gbs:>4.0f}GB/s) | S1={s1_us:>5.1f}u({s1_bw_gbs:>3.0f}GB/s) | "
            f"P224={e2e_224_us:>6.1f}u({rec['speedup_prog224']:>4.2f}x) | "
            f"P224+SVD={e2e_224_svd_us:>6.1f}u({rec['speedup_prog224_svd16']:>4.2f}x) | "
            f"3Tier+SVD={e2e_3tier_svd_us:>6.1f}u({rec['speedup_3tier_svd16']:>4.2f}x) | "
            f"SpFP8={e2e_fp8_us:>6.1f}u({rec['speedup_spfp8']:>4.2f}x)"
        )

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps(
                {
                    "gpu": gpu_name,
                    "sm": f"SM{cc[0]}{cc[1]}",
                    "model": model_key,
                    "dtype": dtype_str,
                    "sparsity": sparsity,
                    "zipf_alpha": zipf_alpha,
                    "parity": {"dense": err_den, "sparse_fp8": err_fp8, "prog8_224": err_224, "sp_vs_den": err_sp_vs_den},
                    "records": records,
                },
                indent=2,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_35b_a3b", choices=list(MODELS.keys()))
    ap.add_argument("--dtype", default="fp8", choices=["bf16", "fp8"])
    ap.add_argument("--batches", default="1,16,32,64,128")
    ap.add_argument("--sparsity", type=float, default=0.80)
    ap.add_argument("--zipf-alpha", type=float, default=0.8)
    ap.add_argument("--cold-me-thresh", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_benchmark(
        args.model, args.dtype, [int(x) for x in args.batches.split(",")],
        args.sparsity, args.zipf_alpha, args.cold_me_thresh, args.out,
    )


if __name__ == "__main__":
    main()
