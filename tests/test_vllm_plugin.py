"""CPU unit tests for the drop-in vLLM `FusedMoEExpertsModular` plugin (`Prog8SparseMoEExperts`)."""

import torch

from sparse_moe_mm.reference import dense_moe_swiglu_ref
from sparse_moe_mm.vllm_plugin import (
    PLUGIN_REGISTRY,
    Prog8MoEConfig,
    Prog8SparseMoEExperts,
    pack_hf_fp8_experts,
    prepare_fp8_moe_layer_for_prog8,
    register,
    should_enable_prog8_sparse,
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
    assert cfg.effective_use_3tier("deltanet") is True
    assert cfg.effective_use_3tier("full_attn") is False
    register()
    assert PLUGIN_REGISTRY["prog8_sparse_fp8"] is Prog8SparseMoEExperts

    # Verify vLLM v0.34+ FusedMoEExpertsModular abstract contract methods
    w13, _, _, w_down = _make_vllm_moe_weights(e=4, d_model=128, d_ff=128)
    packed = pack_hf_fp8_experts(w13, w_down, config=cfg)
    experts = Prog8SparseMoEExperts(packed, config=cfg)
    ws13, ws2, out_s = experts.workspace_shapes(
        M=16, N=128, K=128, topk=4, global_num_experts=4, local_num_experts=4
    )
    assert ws13 == (64, 128) and ws2 == (64, 128) and out_s == (16, 128)
    fin = experts.finalize_weight_and_reduce_impl()
    from sparse_moe_mm.vllm_plugin import HAS_VLLM

    if HAS_VLLM:
        assert type(fin).__name__ == "TopKWeightAndReduceNoOP"
    else:
        assert fin == "noop"


def test_expert_parallel_expert_map_and_layer_prepare_helper():
    """Verify vLLM EP expert_map (-1 non-local experts) & prepare_fp8_moe_layer_for_prog8."""
    # Global E=6, local rank owns experts {0, 2, 4} -> local IDs {0, 1, 2}; {1, 3, 5} -> -1
    w13_loc, wg_loc, wu_loc, wd_loc = _make_vllm_moe_weights(e=3, d_model=128, d_ff=128, seed=77)
    expert_map = torch.tensor([0, -1, 1, -1, 2, -1], dtype=torch.int32)

    class DummyMoELayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(w13_loc.clone(), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(wd_loc.clone(), requires_grad=False)

    layer = DummyMoELayer()
    cfg = Prog8MoEConfig(sparsity=0.0, svd_rank=0, cold_me_thresh=8)
    experts = prepare_fp8_moe_layer_for_prog8(layer, config=cfg, free_dense_w13=True)
    assert layer.w13_weight is None  # Strict 0% duplicate w13 VRAM!
    assert experts.weights.w_gate_fp8 is None and experts.weights.w_up_fp8 is None

    g = torch.Generator().manual_seed(99)
    b = 7  # Odd batch size edge case
    x = torch.randn((b, 128), generator=g)
    topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 3], [0, 4], [5, 2], [3, 1]], dtype=torch.long)
    topk_weights = torch.full((b, 2), 0.5)

    out = torch.empty_like(x)
    experts.apply(out, x, None, None, topk_weights, topk_ids, global_num_experts=6, expert_map=expert_map)

    # Exact reference: only local experts {0->0, 2->1, 4->2} contribute; tokens [3] and [6] have 0 local hits!
    ref = torch.zeros_like(x)
    for t in range(b):
        for s in range(2):
            loc_e = int(expert_map[topk_ids[t, s]].item())
            if loc_e >= 0:
                h = torch.nn.functional.silu(wg_loc[loc_e] @ x[t]) * (wu_loc[loc_e] @ x[t])
                ref[t] += topk_weights[t, s] * (wd_loc[loc_e] @ h)

    assert float(out[3].abs().max().item()) == 0.0
    assert float(out[6].abs().max().item()) == 0.0
    rel_err = float((out - ref).norm() / ref.norm())
    assert rel_err < 0.02, f"EP expert_map rel_err={rel_err:.4f}"


def test_oracle_dispatch_predicate_across_gpus():
    """Verify hardware-calibrated vLLM oracle predicate activates only in winning regimes."""
    # L40S (614 GB/s): 35B active at B=16..128, disabled at B=256 (dense regime)
    assert should_enable_prog8_sparse(64, 256, 2048, 512, hw_mem_bw_gbs=614.0) is True
    assert should_enable_prog8_sparse(256, 256, 2048, 512, hw_mem_bw_gbs=614.0) is False
    # H200 (3,894 GB/s): 35B excluded (<180us dense!), 122B & DeepSeek-V3 enabled at B=16..128!
    assert should_enable_prog8_sparse(64, 256, 2048, 512, hw_mem_bw_gbs=3894.0) is False
    assert should_enable_prog8_sparse(32, 256, 3072, 1024, hw_mem_bw_gbs=3894.0) is True
    assert should_enable_prog8_sparse(32, 256, 7168, 2048, hw_mem_bw_gbs=3894.0) is True
    assert should_enable_prog8_sparse(1, 256, 7168, 2048, hw_mem_bw_gbs=3894.0) is False  # 127us -> opt-in or >=180us
    # Non-512-aligned d_model cleanly rejected
    assert should_enable_prog8_sparse(64, 256, 768, 512, hw_mem_bw_gbs=614.0) is False
