"""CPU unit tests for exact parity, failing-input gates, bit-slicing, quantization, permutation, and roofline."""

import math
import torch
import torch.nn.functional as F

from sparse_moe_mm.reference import (
    apply_expert_permutation,
    block_group_mask,
    compute_coactivation_permutation,
    dense_moe_swiglu_ref,
    dequantize_int4_rowwise,
    dequantize_prog8_4_4,
    dequantize_prog8_dual_2_6,
    prog8_sparse_moe_ref,
    proxy_gated_sparse_moe_ref,
    quantize_fp8_block128,
    quantize_int4_rowwise,
    quantize_nvfp4_block16,
    quantize_prog8_4_4,
    quantize_prog8_dual_2_6,
    quantize_ternary_2bit,
    sparse_moe_swiglu_ref,
)
from sparse_moe_mm.roofline import compute_roofline_table


def _make_toy_moe(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    e, k, t, d, d_ff = 8, 2, 6, 32, 64
    x = torch.randn((t, d), generator=g)
    logits = torch.randn((t, e), generator=g)
    topk_vals, topk_ids = torch.topk(logits, k=k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    w_gate = torch.randn((e, d_ff, d), generator=g) * 0.1
    w_up = torch.randn((e, d_ff, d), generator=g) * 0.1
    w_down = torch.randn((e, d, d_ff), generator=g) * 0.1
    return x, topk_ids, topk_weights, w_gate, w_up, w_down


def test_identity_parity_and_failing_input_gate():
    x, topk_ids, topk_weights, w_gate, w_up, w_down = _make_toy_moe()
    y_dense = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)

    y_full, masks_full = sparse_moe_swiglu_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, group_size=1
    )
    assert masks_full.all()
    assert torch.allclose(y_full, y_dense, atol=1e-6, rtol=1e-6)

    # Registered failing input
    y_half, _ = sparse_moe_swiglu_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=0.5, group_size=1
    )
    rel_err = float((y_half - y_dense).norm() / y_dense.norm())
    assert rel_err > 0.05


def test_prog8_4_4_and_2_6_bit_exact_reconstruction():
    g = torch.Generator().manual_seed(11)
    w = torch.randn((4, 64, 128), generator=g)

    # 4+4 exact split: 16*hi4 + lo4 == symmetric INT8
    hi4, lo4, s4 = quantize_prog8_4_4(w, group_size=128)
    assert int(hi4.min()) >= -8 and int(hi4.max()) <= 7
    assert int(lo4.min()) >= 0 and int(lo4.max()) <= 15
    w_full4 = dequantize_prog8_4_4(hi4, lo4, s4, group_size=128)
    w_coarse4 = dequantize_prog8_4_4(hi4, None, s4, group_size=128)
    rel_full4 = float((w_full4 - w).norm() / w.norm())
    rel_coarse4 = float((w_coarse4 - w).norm() / w.norm())
    assert rel_full4 < 0.015, f"Expected INT8 4+4 rel_l2 < 0.015, got {rel_full4}"
    assert rel_full4 < rel_coarse4 < 0.20

    # 2+6 exact split: 64*hi2 + lo6 == symmetric INT8
    hi2, lo6, s2 = quantize_prog8_dual_2_6(w, group_size=128)
    assert int(hi2.min()) >= -2 and int(hi2.max()) <= 1
    assert int(lo6.min()) >= 0 and int(lo6.max()) <= 63
    w_full2 = dequantize_prog8_dual_2_6(hi2, lo6, s2, group_size=128)
    # Must match 4+4 bit-exactly because both encode the exact same INT8 grid!
    assert torch.allclose(w_full2, w_full4, atol=1e-6, rtol=1e-6)

    # 2-bit packed uint8 roundtrip (4 weights per byte = 0.25 B/elem)
    from sparse_moe_mm.kernels import pack_2bit_hi, unpack_2bit_hi

    packed2 = pack_2bit_hi(hi2)
    assert packed2.dtype == torch.uint8 and packed2.shape == (4, 64, 32)
    assert torch.equal(unpack_2bit_hi(packed2), hi2)


def test_fp8_block128_nvfp4_block16_and_ternary_bounds():
    g = torch.Generator().manual_seed(7)
    w = torch.randn((2, 128, 256), generator=g)

    w_fp8, s_fp8 = quantize_fp8_block128(w, block_size=128)
    assert s_fp8.shape == (2, 1, 2)
    rel_fp8 = float((w_fp8 - w).norm() / w.norm())
    assert 0.005 < rel_fp8 < 0.05

    w_fp4, s_fp4 = quantize_nvfp4_block16(w, block_size=16)
    assert s_fp4.shape == (2, 128, 16)
    rel_fp4 = float((w_fp4 - w).norm() / w.norm())
    assert 0.05 < rel_fp4 < 0.25

    w_tern = quantize_ternary_2bit(w)
    rel_tern = float((w_tern - w).norm() / w.norm())
    assert 0.30 < rel_tern < 0.65

    packed, scale = quantize_int4_rowwise(w)
    w_i4 = dequantize_int4_rowwise(packed, scale)
    rel_i4 = float((w_i4 - w).norm() / w.norm())
    assert 0.05 < rel_i4 < 0.20


def test_block_group_mask_alignment():
    scores = torch.arange(64, dtype=torch.float32)
    m = block_group_mask(scores, keep_frac=0.25, group_size=8)
    assert int(m.sum().item()) == 16
    blocks = m.view(8, 8)
    assert torch.all((blocks.sum(dim=-1) == 0) | (blocks.sum(dim=-1) == 8))
    assert m[-16:].all() and not m[:-16].any()


def test_prog8_and_proxy_references_parity():
    x, topk_ids, topk_weights, w_gate, w_up, w_down = _make_toy_moe()
    y_dense = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)

    y_p4, _ = prog8_sparse_moe_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, quant_group=32, dual=False
    )
    y_p2, _ = prog8_sparse_moe_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, quant_group=32, dual=True
    )
    assert float((y_p4 - y_dense).norm() / y_dense.norm()) < 0.02
    assert float((y_p2 - y_dense).norm() / y_dense.norm()) < 0.02

    y_prox, _ = proxy_gated_sparse_moe_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, input_keep_frac=0.5, group_size=8
    )
    assert torch.allclose(y_prox, y_dense, atol=1e-6, rtol=1e-6)


def test_lossless_coactivation_permutation_g16():
    torch.manual_seed(99)
    d, d_ff, g = 128, 128, 16
    nc = d_ff // g
    latents = F.normalize(torch.randn(nc, d), dim=-1)
    wg = torch.cat([latents[c] + 0.02 * torch.randn(g, d) for c in range(nc)], dim=0)
    wu = torch.randn(d_ff, d) * 0.1
    wd = torch.randn(d, d_ff) * 0.1
    # Scramble neuron order
    scr = torch.randperm(d_ff)
    wg, wu, wd = apply_expert_permutation(wg, wu, wd, scr)

    x = torch.randn(256, d) * 0.2
    for i in range(256):
        x[i] += latents[torch.randperm(nc)[:2]].sum(dim=0) * 3.0

    h = F.silu(x @ wg.T) * (x @ wu.T)
    y_dense = h @ wd.T

    perm = compute_coactivation_permutation(h.abs(), group_size=g)
    wg_p, wu_p, wd_p = apply_expert_permutation(wg, wu, wd, perm)
    h_p = F.silu(x @ wg_p.T) * (x @ wu_p.T)
    y_perm_dense = h_p @ wd_p.T

    # Exact lossless identity at q=1.0
    assert float((y_perm_dense - y_dense).norm() / y_dense.norm()) < 1e-5

    # Substantially beats naive unpermuted g=16 at 75% sparsity
    m_naive = block_group_mask(h, keep_frac=0.25, group_size=g)
    m_perm = block_group_mask(h_p, keep_frac=0.25, group_size=g)
    err_naive = float(((h * m_naive) @ wd.T - y_dense).norm() / y_dense.norm())
    err_perm = float(((h_p * m_perm) @ wd_p.T - y_dense).norm() / y_dense.norm())
    assert err_perm < 0.65 * err_naive, f"Expected permuted g=16 ({err_perm:.3f}) << naive ({err_naive:.3f})"


def test_corrected_roofline_table_sanity():
    rows_h200 = compute_roofline_table("qwen35_35b_a3b", "h200")
    assert len(rows_h200) == 6
    r128 = [r for r in rows_h200 if int(r["B"]) == 128][0]
    # Corrected roofline: Prog8 > Tern8 > Park8 > Prox8* at B=128!
    assert r128["prog8_vs_dfp8"] > r128["tern8_vs_dfp8"] > r128["park_fp8_vs_dfp8"] > r128["prox_fp8_vs_dfp8"]
    assert math.isnan(r128["sparse_fp4_vs_dfp4"])

    rows_b200 = compute_roofline_table("qwen35_35b_a3b", "b200")
    r64_b200 = [r for r in rows_b200 if int(r["B"]) == 64][0]
    # On Blackwell NVFP4, native 0%-VRAM Gate4 beats Tern4 at B >= 16!
    assert r64_b200["sparse_fp4_vs_dfp4"] > r64_b200["tern_fp4_vs_dfp4"] > 2.0
