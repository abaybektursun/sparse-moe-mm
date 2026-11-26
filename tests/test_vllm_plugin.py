"""CPU unit tests for the drop-in vLLM `FusedMoEExpertsModular` plugin (`Prog8SparseMoEExperts`)."""

import torch

from sparse_moe_mm.reference import dense_moe_swiglu_ref
from sparse_moe_mm.vllm_plugin import (
    PLUGIN_REGISTRY,
    Prog8MoEConfig,
    Prog8SparseMoEExperts,
    pack_hf_fp8_experts,
    register,
)


def _make_vllm_moe_weights(e: int = 6, d_model: int = 128, d_ff: int = 128, seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    w_gate = torch.randn((e, d_ff, d_model), generator=g) * 0.05
    w_up = torch.randn((e, d_ff, d_model), generator=g) * 0.05
    w_down = torch.randn((e, d_model, d_ff), generator=g) * 0.05
    w13 = torch.cat([w_gate, w_up], dim=1).contiguous()
    return w13, w_gate, w_up, w_down


def test_vram_overhead_gate_qwen35_and_122b_shapes():
    """Verify 0%-proxy VRAM (<=0.15% without SVD, <=2.75% BF16 SVD-r16 on 35B, <=1.50% on 122B)."""
    # E=1 gives exact same percentage overhead as E=256 while running in <0.1s on CPU
    for d_model, d_ff, max_svd_pct in [(2048, 512, 2.75), (3072, 1024, 1.50)]:
        w13 = torch.randn((1, 2 * d_ff, d_model), dtype=torch.bfloat16) * 0.02
        w2 = torch.randn((1, d_model, d_ff), dtype=torch.bfloat16) * 0.02
        # Simulate FP8 w2 storage size (1B/elem)
        w2_fp8 = w2.to(torch.float8_e4m3fn)
        s2 = torch.ones((1, d_model // 128, d_ff // 128), dtype=torch.float32)

        packed_nosvd = pack_hf_fp8_experts(w13, w2_fp8, w2_scale=s2, config=Prog8MoEConfig(svd_rank=0))
        assert packed_nosvd.vram_overhead_pct() < 0.15, f"Expected ~0% VRAM, got {packed_nosvd.vram_overhead_pct():.2f}%"

        packed_svd16 = pack_hf_fp8_experts(w13, w2_fp8, w2_scale=s2, config=Prog8MoEConfig(svd_rank=16))
        assert packed_svd16.vram_overhead_pct() < max_svd_pct, (
            f"Expected <{max_svd_pct}% VRAM with BF16 SVD-r16, got {packed_svd16.vram_overhead_pct():.2f}%"
        )


def test_plugin_q1_exact_parity_and_router_weight_on_input():
    w13, w_gate, w_up, w_down = _make_vllm_moe_weights(e=6, d_model=128, d_ff=128)
    g = torch.Generator().manual_seed(7)
    b, top_k = 10, 2
    x = torch.randn((b, 128), generator=g)
    logits = torch.randn((b, 6), generator=g)
    topk_vals, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)

    ref = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)

    cfg = Prog8MoEConfig(sparsity=0.0, svd_rank=0, cold_me_thresh=8)
    packed = pack_hf_fp8_experts(w13, w_down, config=cfg)
    experts = Prog8SparseMoEExperts(packed, config=cfg)

    out = torch.empty_like(x)
    experts.apply(out, x, None, None, topk_weights, topk_ids)
    rel_err = float((out - ref).norm() / ref.norm())
    assert rel_err < 0.02, f"Expected q=1.0 rel_err < 0.02, got {rel_err:.4f}"

    # Verify apply_router_weight_on_input=True matches mathematically on SiLU-gated linear output when weights=1
    ones_w = torch.ones_like(topk_weights)
    ref_ones = dense_moe_swiglu_ref(x, topk_ids, ones_w, w_gate, w_up, w_down)
    out_ones = torch.empty_like(x)
    experts.apply(out_ones, x, None, None, ones_w, topk_ids, apply_router_weight_on_input=True)
    assert float((out_ones - ref_ones).norm() / ref_ones.norm()) < 0.02


def test_cold_hot_zipf_split_keeps_hot_experts_exact():
    """Verify hot experts (M_e > cold_me_thresh) stay 100% dense even at 75% sparsity."""
    w13, w_gate, w_up, w_down = _make_vllm_moe_weights(e=4, d_model=128, d_ff=128)
    g = torch.Generator().manual_seed(19)
    # 6 tokens all route to expert 0 (hot: M_e=6 > thresh=2), 2 tokens route to expert 1 (cold: M_e=2 <= 2)
    x = torch.randn((8, 128), generator=g)
    topk_ids = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [1]], dtype=torch.long)
    topk_weights = torch.ones((8, 1), dtype=torch.float32)

    ref = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)
    cfg = Prog8MoEConfig(sparsity=0.75, cold_me_thresh=2, svd_rank=0)
    packed = pack_hf_fp8_experts(w13, w_down, config=cfg)
    experts = Prog8SparseMoEExperts(packed, config=cfg)

    out = torch.empty_like(x)
    experts.apply(out, x, None, None, topk_weights, topk_ids)

    hot_err = float((out[:6] - ref[:6]).norm() / ref[:6].norm())
    cold_err = float((out[6:] - ref[6:]).norm() / ref[6:].norm())
    assert hot_err < 0.02, f"Hot expert should be exact dense, got err={hot_err:.4f}"
    assert cold_err > 0.15, f"Cold expert should be genuinely sparsified, got err={cold_err:.4f}"


def test_svd_r16_recovery_improves_dropped_neuron_error_and_3tier_runs():
    """Verify SVD-r16 recovers low-rank W_down energy on dropped neurons & 3-Tier runs cleanly."""
    g = torch.Generator().manual_seed(55)
    e, d, d_ff = 2, 128, 128
    w_gate = torch.randn((e, d_ff, d), generator=g) * 0.1
    w_up = torch.randn((e, d_ff, d), generator=g) * 0.1
    # Construct structured low-rank + small noise W_down so rank-16 SVD captures real signal
    u_true = torch.randn((e, d, 8), generator=g) * 0.2
    v_true = torch.randn((e, 8, d_ff), generator=g) * 0.2
    w_down = torch.bmm(u_true, v_true) + torch.randn((e, d, d_ff), generator=g) * 0.01
    w13 = torch.cat([w_gate, w_up], dim=1).contiguous()

    x = torch.randn((4, d), generator=g)
    topk_ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)
    topk_weights = torch.ones((4, 1))
    ref = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)

    cfg_nosvd = Prog8MoEConfig(sparsity=0.75, cold_me_thresh=8, svd_rank=0)
    cfg_svd = Prog8MoEConfig(sparsity=0.75, cold_me_thresh=8, svd_rank=16)
    cfg_3tier = Prog8MoEConfig(sparsity=0.75, cold_me_thresh=8, svd_rank=16, use_3tier=True)

    out_nosvd = torch.empty_like(x)
    out_svd = torch.empty_like(x)
    out_3tier = torch.empty_like(x)

    Prog8SparseMoEExperts(pack_hf_fp8_experts(w13, w_down, config=cfg_nosvd), cfg_nosvd).apply(
        out_nosvd, x, None, None, topk_weights, topk_ids
    )
    Prog8SparseMoEExperts(pack_hf_fp8_experts(w13, w_down, config=cfg_svd), cfg_svd).apply(
        out_svd, x, None, None, topk_weights, topk_ids
    )
    Prog8SparseMoEExperts(pack_hf_fp8_experts(w13, w_down, config=cfg_3tier), cfg_3tier).apply(
        out_3tier, x, None, None, topk_weights, topk_ids
    )

    err_nosvd = float((out_nosvd - ref).norm() / ref.norm())
    err_svd = float((out_svd - ref).norm() / ref.norm())
    err_3tier = float((out_3tier - ref).norm() / ref.norm())
    assert err_svd < err_nosvd, f"Expected SVD-r16 ({err_svd:.4f}) < no-SVD ({err_nosvd:.4f})"
    assert err_3tier < err_nosvd * 1.05, f"Expected 3Tier+SVD ({err_3tier:.4f}) close to no-SVD ({err_nosvd:.4f})"


def test_hybrid_layer_policy_and_plugin_registration():
    cfg = Prog8MoEConfig(sparsity=0.80, full_attn_sparsity=0.70)
    assert cfg.compute_k_u(512, cold_me_avg=2.0, layer_type="full_attn") > cfg.compute_k_u(
        512, cold_me_avg=2.0, layer_type="deltanet"
    )
    register()
    assert PLUGIN_REGISTRY["prog8_sparse_fp8"] is Prog8SparseMoEExperts
