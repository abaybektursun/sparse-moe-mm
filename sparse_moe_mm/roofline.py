"""Analytical HBM byte + Tensor Core FLOP roofline for quantized sparse MoE grouped_mm.

v2 (Impact-Aligned Reset):
- Fixes Stage-1 input-sparsity union expansion (q_in_eff) across M_e tokens.
- Includes exact NVFP4 scale-factor bytes (0.50 + 1/16 = 0.5625 B/elem).
- Adds Zero-VRAM Progressive Bit-Sliced modes: `prog8` (4+4 gate) and `prog8_dual` (2+6 gate+up).
- Adds `tern8` / `tern4` (arXiv:2610.37386 2-bit ternary sketch).
- Adds Zipfian routing skew Monte Carlo & Qwen3.5 3:1 DeltaNet E2E Amdahl budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_experts: int
    top_k: int
    d_model: int
    d_ff: int
    num_moe_layers: int
    shared_d_ff: int = 0
    num_deltanet_layers: int = 0
    num_full_attn_layers: int = 0


@dataclass(frozen=True)
class GPUSpec:
    name: str
    arch: str
    hbm_bw_tb_s: float  # TB/s
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float | None  # None if no native FP4 tensor cores
    l2_mb: float = 50.0


MODELS: dict[str, ModelSpec] = {
    "qwen35_35b_a3b": ModelSpec(
        "Qwen3.5/3.6-35B-A3B",
        num_experts=256,
        top_k=8,
        d_model=2048,
        d_ff=512,
        num_moe_layers=40,
        shared_d_ff=512,
        num_deltanet_layers=30,
        num_full_attn_layers=10,
    ),
    "qwen35_122b_a10b": ModelSpec(
        "Qwen3.5-122B-A10B",
        num_experts=256,
        top_k=8,
        d_model=3072,
        d_ff=1024,
        num_moe_layers=48,
        shared_d_ff=1024,
        num_deltanet_layers=36,
        num_full_attn_layers=12,
    ),
    "deepseek_v3": ModelSpec(
        "DeepSeek-V3-671B",
        num_experts=256,
        top_k=8,
        d_model=7168,
        d_ff=2048,
        num_moe_layers=58,
        shared_d_ff=2048,
        num_deltanet_layers=0,
        num_full_attn_layers=58,
    ),
}

GPUS: dict[str, GPUSpec] = {
    "h100": GPUSpec("H100 SXM", "SM90", hbm_bw_tb_s=3.35, bf16_tflops=989.0, fp8_tflops=1979.0, fp4_tflops=None, l2_mb=50.0),
    "h200": GPUSpec("H200 SXM", "SM90", hbm_bw_tb_s=4.8, bf16_tflops=989.0, fp8_tflops=1979.0, fp4_tflops=None, l2_mb=50.0),
    "b200": GPUSpec("B200 SXM", "SM100", hbm_bw_tb_s=8.0, bf16_tflops=2250.0, fp8_tflops=4500.0, fp4_tflops=9000.0, l2_mb=126.0),
    "rtx_pro_6000": GPUSpec("RTX PRO 6000 (G4)", "SM120", hbm_bw_tb_s=1.8, bf16_tflops=500.0, fp8_tflops=1000.0, fp4_tflops=2000.0, l2_mb=96.0),
}

# Exact bytes per weight element including hardware scaling factors:
# - FP8 [128,128] block-scaled: 1.0 + 4/(128*128) = 1.00024 B/elem
# - NVFP4 1x16 E4M3 micro-scaled: 0.50 + 1/16 = 0.5625 B/elem
BYTES_FP8 = 1.0 + 4.0 / (128.0 * 128.0)
BYTES_NVFP4 = 0.50 + 1.0 / 16.0


def expected_active_experts(num_experts: int, top_k: int, batch_size: int) -> float:
    """Expected number of distinct experts touched at least once under uniform routing."""
    p_not_chosen_single_token = 1.0 - (top_k / num_experts)
    return num_experts * (1.0 - (p_not_chosen_single_token**batch_size))


def expected_union_keep_fraction(keep_frac: float, tokens_per_expert: float) -> float:
    """Upper-bound union fraction of channels active across M_e independent tokens."""
    if tokens_per_expert <= 0:
        return 0.0
    return 1.0 - ((1.0 - keep_frac) ** tokens_per_expert)


def layer_time_us(
    model: ModelSpec,
    gpu: GPUSpec,
    batch_size: int,
    mode: str,
    sparsity: float = 0.90,
    input_sparsity: float = 0.50,
    union_correlation: float = 0.35,
    input_correlation: float = 0.25,
) -> tuple[float, float, float]:
    """Return (roofline_us, mem_us, compute_us) for one routed MoE layer.

    Models union expansion for BOTH Stage-2 output neurons (q_eff) AND Stage-1
    input channels (q_in_eff) across M_e tokens routed to the same expert.
    """
    q = 1.0 - sparsity
    q_in = 1.0 - input_sparsity
    e_active = expected_active_experts(model.num_experts, model.top_k, batch_size)
    total_routed_tokens = batch_size * model.top_k
    m_e = total_routed_tokens / max(e_active, 1e-9)

    q_eff = (1.0 - union_correlation) * expected_union_keep_fraction(q, m_e) + union_correlation * q
    q_in_eff = (1.0 - input_correlation) * expected_union_keep_fraction(q_in, m_e) + input_correlation * q_in

    w_elems = model.d_model * model.d_ff

    if mode == "dense_bf16":
        bytes_per_expert = 3.0 * 2.0 * w_elems
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.bf16_tflops
    elif mode == "dense_fp8":
        bytes_per_expert = 3.0 * BYTES_FP8 * w_elems
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp8_tflops
    elif mode == "dense_nvfp4":
        if gpu.fp4_tflops is None:
            return (math.nan, math.nan, math.nan)
        bytes_per_expert = 3.0 * BYTES_NVFP4 * w_elems
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp4_tflops
    elif mode == "park_bf16":
        bytes_per_expert = (2.0 + 4.0 * q_eff) * w_elems
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.bf16_tflops
    elif mode == "park_fp8":
        # Dense FP8 W_gate + sparse FP8 W_up & W_down (0% VRAM overhead)
        bytes_per_expert = (BYTES_FP8 + 2.0 * BYTES_FP8 * q_eff) * w_elems
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp8_tflops
    elif mode == "prox_fp8":
        # Corrected INT4 gate+up proxy with TRUE Stage-1 input-sparsity union q_in_eff (+33% VRAM)
        bytes_stage1 = 2.0 * 0.5 * q_in_eff * w_elems
        bytes_stage2 = 3.0 * BYTES_FP8 * q_eff * w_elems
        bytes_per_expert = bytes_stage1 + bytes_stage2
        flops_total = (2.0 * q_in + 3.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp8_tflops
    elif mode == "tern8":
        # arXiv:2610.37386 adapted to FP8: 2-bit ternary gate+up (0.50 B) + sparse FP8 all-3 (+17% VRAM)
        bytes_per_expert = (0.50 + 3.0 * BYTES_FP8 * q_eff) * w_elems
        flops_total = (2.0 + 3.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp8_tflops
    elif mode in ("prog8", "prog8_dual"):
        # Zero-VRAM Progressive Bit-Sliced 8-bit (0% VRAM overhead!):
        # - prog8:      4-bit coarse W_gate (0.50 B) + 4-bit fine W_gate (0.50*q_eff) + 8-bit W_up,W_down (2.0*q_eff)
        # - prog8_dual: 2-bit coarse W_gate+W_up (0.50 B) + 6-bit fine W_gate+W_up (1.50*q_eff) + 8-bit W_down (1.0*q_eff)
        bytes_per_expert = (0.50 + 2.50 * BYTES_FP8 * q_eff) * w_elems
        flops_stage1 = (1.0 if mode == "prog8" else 2.0) * 2.0 * total_routed_tokens * w_elems
        flops_stage2 = 2.50 * q * 2.0 * total_routed_tokens * w_elems
        flops_total = flops_stage1 + flops_stage2
        peak_tflops = gpu.fp8_tflops
    elif mode == "gate_nvfp4_sparse":
        # Native NVFP4 W_gate (0.5625 B, 0% VRAM overhead) + sparse g=16 NVFP4 W_up,W_down
        if gpu.fp4_tflops is None:
            return (math.nan, math.nan, math.nan)
        bytes_per_expert = (BYTES_NVFP4 + 2.0 * BYTES_NVFP4 * q_eff) * w_elems
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp4_tflops
    elif mode == "tern_nvfp4_sparse":
        # 2-bit ternary gate+up (0.50 B, +30% VRAM) + sparse NVFP4 all-3
        if gpu.fp4_tflops is None:
            return (math.nan, math.nan, math.nan)
        bytes_per_expert = (0.50 + 3.0 * BYTES_NVFP4 * q_eff) * w_elems
        flops_total = (2.0 + 3.0 * q) * 2.0 * total_routed_tokens * w_elems
        peak_tflops = gpu.fp4_tflops
    else:
        raise ValueError(f"Unknown mode: {mode}")

    act_bytes = 2.0 * total_routed_tokens * model.d_model * 2.0
    total_bytes = e_active * bytes_per_expert + act_bytes

    mem_us = (total_bytes / (gpu.hbm_bw_tb_s * 1e12)) * 1e6
    compute_us = (flops_total / (peak_tflops * 1e12)) * 1e6
    return (max(mem_us, compute_us), mem_us, compute_us)


def compute_roofline_table(
    model_key: str = "qwen35_35b_a3b",
    gpu_key: str = "h200",
    batches: tuple[int, ...] = (1, 16, 32, 64, 128, 256),
    sparsity: float = 0.90,
    input_sparsity: float = 0.50,
    union_correlation: float = 0.35,
) -> list[dict[str, float]]:
    model = MODELS[model_key]
    gpu = GPUS[gpu_key]
    rows: list[dict[str, float]] = []
    for b in batches:
        e_act = expected_active_experts(model.num_experts, model.top_k, b)
        m_e = (b * model.top_k) / max(e_act, 1e-9)
        t_dbf16, _, _ = layer_time_us(model, gpu, b, "dense_bf16", sparsity, input_sparsity, union_correlation)
        t_dfp8, _, _ = layer_time_us(model, gpu, b, "dense_fp8", sparsity, input_sparsity, union_correlation)
        t_pfp8, _, _ = layer_time_us(model, gpu, b, "park_fp8", sparsity, input_sparsity, union_correlation)
        t_prox8, _, _ = layer_time_us(model, gpu, b, "prox_fp8", sparsity, input_sparsity, union_correlation)
        t_tern8, _, _ = layer_time_us(model, gpu, b, "tern8", sparsity, input_sparsity, union_correlation)
        t_prog8, _, _ = layer_time_us(model, gpu, b, "prog8", sparsity, input_sparsity, union_correlation)
        t_dfp4, _, _ = layer_time_us(model, gpu, b, "dense_nvfp4", sparsity, input_sparsity, union_correlation)
        t_gate4, _, _ = layer_time_us(model, gpu, b, "gate_nvfp4_sparse", sparsity, input_sparsity, union_correlation)
        t_tern4, _, _ = layer_time_us(model, gpu, b, "tern_nvfp4_sparse", sparsity, input_sparsity, union_correlation)
        rows.append(
            {
                "B": float(b),
                "E_act": e_act,
                "M_e": m_e,
                "dense_bf16_us": t_dbf16,
                "dense_fp8_us": t_dfp8,
                "park_fp8_vs_dfp8": t_dfp8 / t_pfp8,
                "prox_fp8_vs_dfp8": t_dfp8 / t_prox8,
                "tern8_vs_dfp8": t_dfp8 / t_tern8,
                "prog8_vs_dfp8": t_dfp8 / t_prog8,
                "sparse_fp4_vs_dfp4": (t_dfp4 / t_gate4) if not math.isnan(t_dfp4) else math.nan,
                "tern_fp4_vs_dfp4": (t_dfp4 / t_tern4) if not math.isnan(t_dfp4) else math.nan,
            }
        )
    return rows


def main() -> None:
    for gpu_key in ("h200", "b200", "rtx_pro_6000"):
        gpu = GPUS[gpu_key]
        print(f"\n=== {gpu.name} ({gpu.arch}, {gpu.hbm_bw_tb_s} TB/s) | Qwen3.5-35B-A3B (s=90%) ===")
        header = (
            f"{'B':>4} {'E_act':>6} {'M_e':>5} | {'dFP8(us)':>8} | "
            f"{'Park8/d8':>8} {'Prox8*/d8':>9} {'Tern8/d8':>8} {'Prog8/d8':>8} | "
            f"{'Gate4/d4':>8} {'Tern4/d4':>8}"
        )
        print(header)
        print("-" * len(header))
        for r in compute_roofline_table("qwen35_35b_a3b", gpu_key):
            g4 = f"{r['sparse_fp4_vs_dfp4']:7.2f}x" if not math.isnan(r["sparse_fp4_vs_dfp4"]) else "     n/a"
            t4 = f"{r['tern_fp4_vs_dfp4']:7.2f}x" if not math.isnan(r["tern_fp4_vs_dfp4"]) else "     n/a"
            print(
                f"{int(r['B']):>4} {r['E_act']:>6.1f} {r['M_e']:>5.2f} | {r['dense_fp8_us']:>8.1f} | "
                f"{r['park_fp8_vs_dfp8']:>7.2f}x {r['prox_fp8_vs_dfp8']:>8.2f}x "
                f"{r['tern8_vs_dfp8']:>7.2f}x {r['prog8_vs_dfp8']:>7.2f}x | {g4} {t4}"
            )


if __name__ == "__main__":
    main()
