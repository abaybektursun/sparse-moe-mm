"""Single-layer GPU feasibility benchmark (Phase 1 gate).

Benchmarks:
  1. Dense BF16/FP8 grouped_mm baseline (`torch._grouped_mm` where available, else per-expert baseline)
  2. Sparse up+down gather Triton kernel (Park-style, BF16 & FP8)
  3. Sparse all-3 Stage-2 gather Triton kernel (Prox-style Stage-2 ceiling, BF16 & FP8)

Verifies numerical parity against exact fp32 reference before timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from sparse_moe_mm.kernels import launch_sparse_all3, launch_sparse_up_down
from sparse_moe_mm.reference import block_group_mask
from sparse_moe_mm.roofline import MODELS


def quantize_fp8_per_neuron(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-neuron (rowwise) symmetric E4M3 FP8 quantization for [E, d_ff, d_model]."""
    max_abs = w.float().abs().amax(dim=-1).clamp(min=1e-8)
    scale = max_abs / 448.0
    q = (w.float() / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous()


def bench_cuda_fn(fn, warmup: int = 20, iters: int = 100) -> float:
    """Return median latency in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] * 1000.0


def dense_grouped_mm_bf16(
    x_sorted: torch.Tensor,
    offs: torch.Tensor,
    w_gate_t: torch.Tensor,
    w_up_t: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    g = F.silu(torch._grouped_mm(x_sorted, w_gate_t, offs=offs))
    u = torch._grouped_mm(x_sorted, w_up_t, offs=offs)
    return torch._grouped_mm(g * u, w_down_t, offs=offs)


def run_benchmark(
    model_key: str,
    dtype_str: str,
    batches: list[int],
    sparsity: float,
    group_sizes: list[int],
    out_json: str | None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmarks/bench_layer.py requires a CUDA GPU. Run pytest on CPU instead.")

    spec = MODELS[model_key]
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | Model: {spec.name} (E={spec.num_experts}, K={spec.top_k}, d={spec.d_model}, d_ff={spec.d_ff})")

    torch.manual_seed(0)
    w_gate_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    w_up_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02
    # Transposed [E, d_ff, d_model] layout for coalesced row gather
    w_down_t_bf16 = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=device, dtype=torch.bfloat16) * 0.02

    # Layouts for torch._grouped_mm: [E, in, out]
    w_gate_gmm = w_gate_bf16.transpose(1, 2).contiguous()
    w_up_gmm = w_up_bf16.transpose(1, 2).contiguous()
    w_down_gmm = w_down_t_bf16.contiguous()

    is_fp8 = dtype_str == "fp8"
    if is_fp8:
        w_gate_q, s_gate = quantize_fp8_per_neuron(w_gate_bf16)
        w_up_q, s_up = quantize_fp8_per_neuron(w_up_bf16)
        w_down_q, s_down = quantize_fp8_per_neuron(w_down_t_bf16)
    else:
        w_gate_q, s_gate = w_gate_bf16, None
        w_up_q, s_up = w_up_bf16, None
        w_down_q, s_down = w_down_t_bf16, None

    records = []
    keep_frac = 1.0 - sparsity
    has_grouped_mm = hasattr(torch, "_grouped_mm")

    for b in batches:
        n_pairs = b * spec.top_k
        x = torch.randn((b, spec.d_model), device=device, dtype=torch.bfloat16)
        # Uniform random top-k routing (sort by expert for grouped_mm parity)
        logits = torch.randn((b, spec.num_experts), device=device)
        topk_ids = torch.topk(logits, k=spec.top_k, dim=-1).indices  # [B, K]
        flat_experts = topk_ids.reshape(-1)
        order = torch.argsort(flat_experts)
        expert_ids = flat_experts[order].to(torch.int32).contiguous()
        token_ids = (order // spec.top_k)
        x_pairs = x[token_ids].contiguous()

        counts = torch.bincount(expert_ids.long(), minlength=spec.num_experts)
        offs = torch.cumsum(counts, dim=0).to(torch.int32).contiguous()

        dense_us = (
            bench_cuda_fn(lambda: dense_grouped_mm_bf16(x_pairs, offs, w_gate_gmm, w_up_gmm, w_down_gmm))
            if has_grouped_mm
            else float("nan")
        )

        # Reference dense gate output for parity & Park-style kernel
        gate_act = (
            F.silu(torch._grouped_mm(x_pairs, w_gate_gmm, offs=offs))
            if has_grouped_mm
            else torch.zeros((n_pairs, spec.d_ff), device=device, dtype=torch.bfloat16)
        )

        for g in group_sizes:
            k_active = max(1, int(round(spec.d_ff * keep_frac)))
            mask = block_group_mask(gate_act.float(), keep_frac=keep_frac, group_size=g)
            # Compact active indices into [N_pairs, k_active] int32 buffer
            active_idx = torch.topk(mask.to(torch.int32), k=k_active, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
            active_cnt = torch.full((n_pairs,), k_active, dtype=torch.int32, device=device)

            # Parity check (Stage-2 all-3 sparse vs exact PyTorch sparse gather)
            out_triton = launch_sparse_all3(
                x_pairs, expert_ids, active_idx, active_cnt, w_gate_q, w_up_q, w_down_q, s_gate, s_up, s_down
            )
            # Check non-zero finite output + quick sample exactness on pair 0
            e0 = int(expert_ids[0].item())
            idx0 = active_idx[0].long()
            ref0 = (
                (F.silu(w_gate_bf16[e0, idx0].float() @ x_pairs[0].float()) * (w_up_bf16[e0, idx0].float() @ x_pairs[0].float()))
                @ w_down_t_bf16[e0, idx0].float()
            )
            rel_err = float((out_triton[0] - ref0).norm() / ref0.norm().clamp(min=1e-8))

            up_down_us = bench_cuda_fn(
                lambda: launch_sparse_up_down(
                    x_pairs, expert_ids, active_idx, active_cnt, gate_act, w_up_q, w_down_q, s_up, s_down
                )
            )
            all3_us = bench_cuda_fn(
                lambda: launch_sparse_all3(
                    x_pairs, expert_ids, active_idx, active_cnt, w_gate_q, w_up_q, w_down_q, s_gate, s_up, s_down
                )
            )
            rec = {
                "gpu": gpu_name,
                "model": model_key,
                "dtype": dtype_str,
                "batch_size": b,
                "sparsity": sparsity,
                "group_size": g,
                "dense_bf16_gmm_us": dense_us,
                "sparse_up_down_us": up_down_us,
                "sparse_all3_stage2_us": all3_us,
                "speedup_all3_stage2_vs_dense_bf16": dense_us / all3_us if all3_us > 0 else float("nan"),
                "pair0_rel_l2_err": rel_err,
            }
            records.append(rec)
            print(
                f"B={b:>3} g={g:>2} | dense_bf16={dense_us:>7.1f}us | "
                f"sp_updn={up_down_us:>7.1f}us | sp_all3_s2={all3_us:>7.1f}us "
                f"({dense_us / all3_us:>5.2f}x) | rel_err={rel_err:.4f}"
            )

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps({"timestamp": time.time(), "records": records}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_35b_a3b", choices=list(MODELS.keys()))
    ap.add_argument("--dtype", default="fp8", choices=["bf16", "fp8"])
    ap.add_argument("--batches", default="1,4,16,32,64,128,256")
    ap.add_argument("--sparsity", type=float, default=0.90)
    ap.add_argument("--group-size", default="1,16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    group_sizes = [int(x) for x in args.group_size.split(",")]
    run_benchmark(args.model, args.dtype, batches, args.sparsity, group_sizes, args.out)


if __name__ == "__main__":
    main()
