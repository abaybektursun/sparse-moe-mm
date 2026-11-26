"""Structural, zero-copy VRAM, O11 0x50 FP8 bitcast exactness, and artifact contract tests."""

import json
from pathlib import Path
import torch

import sparse_moe_mm
import sparse_moe_mm.kernels as kernels_pkg
from sparse_moe_mm.kernels import _common, sm89, sm90
from sparse_moe_mm.packing import Prog8MoEConfig, pack_hf_fp8_experts


def test_zero_copy_w13_storage_sharing():
    """Verify wg_*/wu_* share exact underlying storage with unified w13_* (strict 0% duplicate memory)."""
    w13 = torch.randn((2, 256, 128), dtype=torch.bfloat16) * 0.02
    w2 = torch.randn((2, 128, 128), dtype=torch.bfloat16) * 0.02
    packed = pack_hf_fp8_experts(w13, w2, config=Prog8MoEConfig(svd_rank=0))

    assert packed.w13_hi2 is not None and packed.w13_mid2 is not None
    assert packed.w13_lo4 is not None and packed.s13 is not None

    assert packed.wg_hi2.untyped_storage().data_ptr() == packed.w13_hi2.untyped_storage().data_ptr()
    assert packed.wu_hi2.untyped_storage().data_ptr() == packed.w13_hi2.untyped_storage().data_ptr()
    assert packed.wg_mid2.untyped_storage().data_ptr() == packed.w13_mid2.untyped_storage().data_ptr()
    assert packed.wu_mid2.untyped_storage().data_ptr() == packed.w13_mid2.untyped_storage().data_ptr()
    assert packed.wg_lo4.untyped_storage().data_ptr() == packed.w13_lo4.untyped_storage().data_ptr()
    assert packed.wu_lo4.untyped_storage().data_ptr() == packed.w13_lo4.untyped_storage().data_ptr()
    assert packed.sg.untyped_storage().data_ptr() == packed.s13.untyped_storage().data_ptr()
    assert packed.su.untyped_storage().data_ptr() == packed.s13.untyped_storage().data_ptr()


def test_sm90_o11_0x50_fp8_bitcast_bit_exact_all_int8():
    """Prove O11 0x50 IEEE float8_e4m3fn identity is bit-exact (0.0 error) for all 256 signed INT8 values."""
    q_all = torch.arange(-128, 128, dtype=torch.int32)
    hi2 = q_all >> 6  # {-2, -1, 0, 1}
    rem6 = q_all - hi2 * 64  # {0..63}
    mid2 = (rem6 >> 4) & 3
    lo4 = rem6 & 15

    # Stage-1 2-bit: (hi2 + 2) | 0x50 -> {8, 9, 10, 11} == (hi2 + 0.5) + 9.5
    u_hi = (hi2 + 2).to(torch.uint8)
    fp8_hi = (u_hi | 0x50).view(torch.float8_e4m3fn).float()
    assert torch.allclose(fp8_hi - 9.5, hi2.float() + 0.5, atol=0.0)

    # Full 8-bit via two 3-bit FP8 fields: mid3 = (mid2 << 1) | (lo4 >> 3), lo3 = lo4 & 7
    mid3 = ((mid2 << 1) | ((lo4 >> 3) & 1)).to(torch.uint8)
    lo3 = (lo4 & 7).to(torch.uint8)
    fp8_mid3 = (mid3 | 0x50).view(torch.float8_e4m3fn).float()
    fp8_lo3 = (lo3 | 0x50).view(torch.float8_e4m3fn).float()

    q_rec = 64.0 * fp8_hi + 8.0 * fp8_mid3 + fp8_lo3 - 712.0
    max_err = float((q_rec - q_all.float()).abs().max().item())
    assert max_err == 0.0, f"Expected 0.0 reconstruction error across all 256 INT8 values, got {max_err}"


def test_modular_kernels_subpackage_exports():
    """Verify split kernels/{_common,sm89,sm90} re-export every launcher through sparse_moe_mm.kernels."""
    expected_common = [
        "launch_weighted_unpermute_combine",
        "launch_fast_cold_topk",
        "launch_svd_recovery",
    ]
    expected_sm89 = [
        "launch_prog8_224_stage1",
        "launch_prog8_224_stage2",
        "launch_sparse_fp8_stage2",
        "launch_dense_224_grouped_mm",
        "launch_dense_fp8_grouped_mm",
    ]
    expected_sm90 = [
        "launch_quantize_x_fp8_sum",
        "launch_prog8_sm90_stage1",
        "launch_prog8_sm90_stage2",
        "launch_dense_sm90_224_grouped_mm",
    ]
    for name in expected_common:
        assert hasattr(_common, name) and getattr(kernels_pkg, name) is getattr(_common, name)
    for name in expected_sm89:
        assert hasattr(sm89, name) and getattr(kernels_pkg, name) is getattr(sm89, name)
    for name in expected_sm90:
        assert hasattr(sm90, name) and getattr(kernels_pkg, name) is getattr(sm90, name)
    assert hasattr(sparse_moe_mm, "should_enable_prog8_sparse")
    assert hasattr(sparse_moe_mm, "prepare_fp8_moe_layer_for_prog8")


def test_verified_result_artifacts_integrity():
    """Verify all tracked benchmark/accuracy JSON artifacts parse cleanly and satisfy ship invariants."""
    root = Path(__file__).resolve().parent.parent / "results"
    h200 = json.loads((root / "h200_vs_vllm034_sota.json").read_text())
    assert h200["full_apply_cuda_graph_captured"] is True
    assert h200["strict_zero_duplicate_w13"] is True
    assert h200["live_122b_vram_fit_proof"]["fits_single_h200"] is True

    gsm8k = json.loads((root / "gsm8k_qwen35_35b_l40s.json").read_text())
    assert gsm8k["live_triton_kernel"] is True
    assert gsm8k["strict_zero_duplicate_w13"] is True
    assert gsm8k["hybrid_dn80_fa70_svd16_live_triton"]["repetition_loops"] == 0
    assert gsm8k["relative_retention_pct"] >= 98.5
