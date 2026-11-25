"""CPU unit tests for exact parity, failing-input gates, INT4 packing, and roofline."""

import math
import torch

from sparse_moe_mm.reference import (
    block_group_mask,
    dequantize_int4_rowwise,
    dense_moe_swiglu_ref,
    proxy_gated_sparse_moe_ref,
    quantize_int4_rowwise,
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

    # Pass condition: keep_frac=1.0 matches dense reference to fp32 machine precision
    y_full, masks_full = sparse_moe_swiglu_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, group_size=1
    )
    assert masks_full.all()
    assert torch.allclose(y_full, y_dense, atol=1e-6, rtol=1e-6)

    # Registered failing input: dropping 50% of neurons must fail tight parity gate
    y_half, _ = sparse_moe_swiglu_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=0.5, group_size=1
    )
    rel_err = float((y_half - y_dense).norm() / y_dense.norm())
    assert rel_err > 0.05, f"Expected failing-input rel_err > 0.05, got {rel_err}"


def test_int4_quantize_roundtrip_and_bounds():
    g = torch.Generator().manual_seed(7)
    w = torch.randn((4, 32, 64), generator=g)
    packed, scale = quantize_int4_rowwise(w)
    assert packed.dtype == torch.uint8
    assert packed.shape == (4, 32, 32)
    w_hat = dequantize_int4_rowwise(packed, scale)
    assert w_hat.shape == w.shape
    rel_l2 = float((w_hat - w).norm() / w.norm())
    # Symmetric 4-bit rowwise quantization error on Gaussian weights is ~0.10-0.14
    assert 0.05 < rel_l2 < 0.20, f"Unexpected INT4 quantization rel_l2={rel_l2}"


def test_block_group_mask_alignment():
    scores = torch.arange(64, dtype=torch.float32)
    m = block_group_mask(scores, keep_frac=0.25, group_size=8)
    assert int(m.sum().item()) == 16
    # Every 8-element block is either all True or all False
    blocks = m.view(8, 8)
    assert torch.all((blocks.sum(dim=-1) == 0) | (blocks.sum(dim=-1) == 8))
    # Top 2 blocks should be the last 16 elements
    assert m[-16:].all() and not m[:-16].any()


def test_proxy_gated_matches_oracle_at_full_keep_and_recovers_high_energy():
    x, topk_ids, topk_weights, w_gate, w_up, w_down = _make_toy_moe()
    y_dense = dense_moe_swiglu_ref(x, topk_ids, topk_weights, w_gate, w_up, w_down)

    y_prox_full, _ = proxy_gated_sparse_moe_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=1.0, input_keep_frac=0.5, group_size=8
    )
    assert torch.allclose(y_prox_full, y_dense, atol=1e-6, rtol=1e-6)

    _, m_oracle = sparse_moe_swiglu_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=0.25, group_size=1
    )
    _, m_proxy = proxy_gated_sparse_moe_ref(
        x, topk_ids, topk_weights, w_gate, w_up, w_down, keep_frac=0.25, input_keep_frac=1.0, group_size=1
    )
    recall = float((m_oracle & m_proxy).sum().item() / m_oracle.sum().item())
    assert recall > 0.70, f"Expected INT4 proxy mask recall > 0.70 at q_in=1.0, got {recall:.3f}"


def test_roofline_table_sanity():
    rows_h200 = compute_roofline_table("qwen35_35b_a3b", "h200")
    assert len(rows_h200) == 6
    # On H200 at B=16, Prox FP8 (all-3) must beat Park FP8 (dense gate) in roofline ceiling
    r16 = [r for r in rows_h200 if int(r["B"]) == 16][0]
    assert r16["prox_fp8_vs_dfp8"] > r16["park_fp8_vs_dfp8"] > 1.5
    assert math.isnan(r16["sparse_fp4_vs_dfp4"])

    rows_b200 = compute_roofline_table("qwen35_35b_a3b", "b200")
    assert rows_b200[0]["sparse_fp4_vs_dfp4"] > 2.0
