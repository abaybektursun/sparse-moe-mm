"""Analytical HBM byte + Tensor Core FLOP roofline for sparse MoE grouped_mm.

Runs locally on CPU with zero GPU or model weights. Use this to quantify exact
speedup ceilings and memory-to-compute crossover batch sizes before writing kernels.
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


@dataclass(frozen=True)
class GPUSpec:
    name: str
    arch: str
    hbm_bw_tb_s: float  # TB/s
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float | None  # None if no native FP4 tensor cores


MODELS: dict[str, ModelSpec] = {
    "qwen35_35b_a3b": ModelSpec("Qwen3.5/3.6-35B-A3B", 256, 8, 2048, 512, 40),
    "qwen35_122b_a10b": ModelSpec("Qwen3.5-122B-A10B", 256, 8, 3072, 1024, 48),
    "deepseek_v3": ModelSpec("DeepSeek-V3-671B", 256, 8, 7168, 2048, 58),
}

GPUS: dict[str, GPUSpec] = {
    "h200": GPUSpec("H200 SXM", "SM90", hbm_bw_tb_s=4.8, bf16_tflops=989.0, fp8_tflops=1979.0, fp4_tflops=None),
    "b200": GPUSpec("B200 SXM", "SM100", hbm_bw_tb_s=8.0, bf16_tflops=2250.0, fp8_tflops=4500.0, fp4_tflops=9000.0),
    "rtx_pro_6000": GPUSpec("RTX PRO 6000 WS", "SM120", hbm_bw_tb_s=1.8, bf16_tflops=500.0, fp8_tflops=1000.0, fp4_tflops=2000.0),
}


def expected_active_experts(num_experts: int, top_k: int, batch_size: int) -> float:
    """Expected number of distinct experts touched at least once under uniform routing."""
    p_not_chosen_single_token = 1.0 - (top_k / num_experts)
    return num_experts * (1.0 - (p_not_chosen_single_token**batch_size))


def expected_union_keep_fraction(keep_frac: float, tokens_per_expert: float) -> float:
    """Upper-bound union fraction of neurons active across M_e independent tokens."""
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
) -> tuple[float, float, float]:
    """Return (roofline_us, mem_us, compute_us) for one routed MoE layer.

    union_correlation in [0, 1] interpolates between independent tokens (0.0)
    and perfectly correlated/flocked tokens (1.0) when computing the HBM weight
    union across M_e tokens hitting the same expert.
    """
    q = 1.0 - sparsity
    q_in = 1.0 - input_sparsity
    e_active = expected_active_experts(model.num_experts, model.top_k, batch_size)
    total_routed_tokens = batch_size * model.top_k
    m_e = total_routed_tokens / max(e_active, 1e-9)

    q_union_indep = expected_union_keep_fraction(q, m_e)
    q_eff_mem = (1.0 - union_correlation) * q_union_indep + union_correlation * q

    w_elems_per_proj = model.d_model * model.d_ff

    if mode == "dense_bf16":
        bytes_per_expert = 3.0 * 2.0 * w_elems_per_proj
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.bf16_tflops
    elif mode == "dense_fp8":
        bytes_per_expert = 3.0 * 1.0 * w_elems_per_proj
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.fp8_tflops
    elif mode == "dense_nvfp4":
        if gpu.fp4_tflops is None:
            return (math.nan, math.nan, math.nan)
        bytes_per_expert = 3.0 * 0.5 * w_elems_per_proj
        flops_total = 3.0 * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.fp4_tflops
    elif mode == "park_bf16":
        # Dense BF16 gate + sparse BF16 up & down
        bytes_per_expert = (2.0 + 2.0 * 2.0 * q_eff_mem) * w_elems_per_proj
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.bf16_tflops
    elif mode == "park_fp8":
        # Dense FP8 gate + sparse FP8 up & down
        bytes_per_expert = (1.0 + 2.0 * 1.0 * q_eff_mem) * w_elems_per_proj
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.fp8_tflops
    elif mode == "prox_fp8":
        # Stage-1 INT4 (0.5B) gate+up proxy with input keep q_in + Stage-2 exact FP8 all-3 at q
        bytes_stage1 = 2.0 * 0.5 * q_in * w_elems_per_proj
        bytes_stage2 = 3.0 * 1.0 * q_eff_mem * w_elems_per_proj
        bytes_per_expert = bytes_stage1 + bytes_stage2
        flops_stage1 = 2.0 * q_in * 2.0 * total_routed_tokens * w_elems_per_proj
        flops_stage2 = 3.0 * q * 2.0 * total_routed_tokens * w_elems_per_proj
        flops_total = flops_stage1 + flops_stage2
        peak_tflops = gpu.fp8_tflops
    elif mode == "gate_nvfp4_sparse":
        # Dense NVFP4 gate (0.5B, zero extra proxy copy) + sparse NVFP4 up & down
        if gpu.fp4_tflops is None:
            return (math.nan, math.nan, math.nan)
        bytes_per_expert = (0.5 + 2.0 * 0.5 * q_eff_mem) * w_elems_per_proj
        flops_total = (1.0 + 2.0 * q) * 2.0 * total_routed_tokens * w_elems_per_proj
        peak_tflops = gpu.fp4_tflops
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Add activation read/write traffic (small vs weights at low B, non-trivial at high B)
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
        t_pbf16, _, _ = layer_time_us(model, gpu, b, "park_bf16", sparsity, input_sparsity, union_correlation)
        t_pfp8, _, _ = layer_time_us(model, gpu, b, "park_fp8", sparsity, input_sparsity, union_correlation)
        t_prox8, _, _ = layer_time_us(model, gpu, b, "prox_fp8", sparsity, input_sparsity, union_correlation)
        t_dfp4, _, _ = layer_time_us(model, gpu, b, "dense_nvfp4", sparsity, input_sparsity, union_correlation)
        t_sfp4, _, _ = layer_time_us(model, gpu, b, "gate_nvfp4_sparse", sparsity, input_sparsity, union_correlation)
        rows.append(
            {
                "B": float(b),
                "E_act": e_act,
                "M_e": m_e,
                "dense_bf16_us": t_dbf16,
                "dense_fp8_us": t_dfp8,
                "park_bf16_vs_dbf16": t_dbf16 / t_pbf16,
                "park_fp8_vs_dfp8": t_dfp8 / t_pfp8,
                "prox_fp8_vs_dfp8": t_dfp8 / t_prox8,
                "sparse_fp4_vs_dfp4": (t_dfp4 / t_sfp4) if not math.isnan(t_dfp4) else math.nan,
            }
        )
    return rows


def main() -> None:
    for gpu_key in ("h200", "b200", "rtx_pro_6000"):
        gpu = GPUS[gpu_key]
        print(f"\n=== {gpu.name} ({gpu.arch}, {gpu.hbm_bw_tb_s} TB/s) | Qwen3.5-35B-A3B (s=90%, s_in=50%) ===")
        header = (
            f"{'B':>4} {'E_act':>6} {'M_e':>5} | "
            f"{'dBF16(us)':>9} {'dFP8(us)':>9} | "
            f"{'ParkBF16/d16':>12} {'ParkFP8/d8':>10} {'ProxFP8/d8':>10} {'SpFP4/d4':>9}"
        )
        print(header)
        print("-" * len(header))
        for r in compute_roofline_table("qwen35_35b_a3b", gpu_key):
            sp4 = f"{r['sparse_fp4_vs_dfp4']:8.2f}x" if not math.isnan(r["sparse_fp4_vs_dfp4"]) else "      n/a"
            print(
                f"{int(r['B']):>4} {r['E_act']:>6.1f} {r['M_e']:>5.2f} | "
                f"{r['dense_bf16_us']:>9.1f} {r['dense_fp8_us']:>9.1f} | "
                f"{r['park_bf16_vs_dbf16']:>11.2f}x {r['park_fp8_vs_dfp8']:>9.2f}x "
                f"{r['prox_fp8_vs_dfp8']:>9.2f}x {sp4}"
            )


if __name__ == "__main__":
    main()
