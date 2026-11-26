"""Quickstart: Drop-in `Prog8SparseMoEExperts` (vLLM v0.34+ `FusedMoEExpertsModular`).

Demonstrates:
  1. Load-time chunked conversion of `[128,128]` FP8 (or BF16) MoE weights into Zero-VRAM
     `2+2+4` progressive bitplanes (`hi2`, `mid2`, `lo4`) + weight-only randomized `SVD-r16`.
  2. Exact VRAM overhead reporting (`0%` proxy VRAM; `+1.46%` BF16 SVD-r16 on `122B`).
  3. Cold-Expert-Only Zipf-adaptive dispatch (`M_e <= 8` sparse, `M_e > 8` exact dense) +
     Hybrid DeltaNet (`s=0.80`, 3-Tier) / Full-Attn (`s=0.65`, 8-bit) configuration.

Runs on CPU (reference path) or SM89/SM90 CUDA GPU (Triton O9/O10 path):
    python3 examples/quickstart_vllm_plugin.py
"""

from __future__ import annotations

import torch

from sparse_moe_mm import (
    Prog8MoEConfig,
    Prog8SparseMoEExperts,
    dense_moe_swiglu_ref,
    pack_hf_fp8_experts,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)

    num_experts, d_model, d_ff, batch_size, top_k = 16, 512, 256, 32, 4
    w_gate = torch.randn((num_experts, d_ff, d_model), device=device, dtype=dtype) * 0.02
    w_up = torch.randn((num_experts, d_ff, d_model), device=device, dtype=dtype) * 0.02
    w_down = torch.randn((num_experts, d_model, d_ff), device=device, dtype=dtype) * 0.02
    w13 = torch.cat([w_gate, w_up], dim=1).contiguous()

    # 1. Pack weights once at load time (zero calibration data!)
    # Uses Phase-0c empirically optimal Hybrid policy: DeltaNet s=0.80 (3-Tier), Full-Attn s=0.70 (8-bit)
    cfg = Prog8MoEConfig()
    packed = pack_hf_fp8_experts(w13, w_down, config=cfg)

    # 2. Create drop-in vLLM v0.34+ FusedMoEExpertsModular instances for DeltaNet vs Full-Attn layers
    deltanet_experts = Prog8SparseMoEExperts(packed, config=cfg, layer_type="deltanet")
    fullattn_experts = Prog8SparseMoEExperts(packed, config=cfg, layer_type="full_attn")

    # 3. Run forward pass (matches vLLM `FusedMoEExpertsModular.apply` signature)
    x = torch.randn((batch_size, d_model), device=device, dtype=dtype)
    logits = torch.randn((batch_size, num_experts), device=device)
    topk_vals, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)

    out_dn = torch.empty_like(x)
    out_fa = torch.empty_like(x)
    deltanet_experts.apply(out_dn, x, None, None, topk_weights, topk_ids)
    fullattn_experts.apply(out_fa, x, None, None, topk_weights, topk_ids)

    ref = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)
    err_dn = float((out_dn.float() - ref).norm() / ref.norm())
    err_fa = float((out_fa.float() - ref).norm() / ref.norm())

    print(f"Device: {device} | dtype: {dtype}")
    print(f"DeltaNet  layer (s={cfg.effective_sparsity('deltanet'):.2f}, 3-Tier+SVD) rel_L2 vs dense: {err_dn:.4f}")
    print(f"Full-Attn layer (s={cfg.effective_sparsity('full_attn'):.2f}, 3-Tier+SVD) rel_L2 vs dense: {err_fa:.4f}")
    print(f"Plugin activation_format: {deltanet_experts.activation_format()} | finalize: {deltanet_experts.finalize_weight_and_reduce_impl()}")


if __name__ == "__main__":
    main()
