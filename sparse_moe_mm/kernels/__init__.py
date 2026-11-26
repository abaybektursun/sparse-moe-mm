"""Architecture-dispatched Triton kernels for Zero-VRAM Prog8-Dual (2+2+4) sparse MoE.

Submodules:
  - `_common`: Shared Down projection, SVD-r16 recovery, fused weighted combine, register bitonic TopK.
  - `sm89`:    SM89 (L4/L40S) Marlin 0x6000/0x6400 FP16 bitcast + 128B cache-line kernels (O9/O10).
  - `sm90`:    SM90 Hopper (H100/H200) 0x50505050 LOP3-pack4 FP8 bitcast + W@X^T WGMMA kernels (O11).
"""

from __future__ import annotations

from sparse_moe_mm.kernels._common import (
    HAS_TRITON,
    _require_triton_cuda,
    launch_fast_cold_topk,
    launch_svd_recovery,
    launch_weighted_unpermute_combine,
    pack_2bit_hi,
    pack_2bit_interleaved,
    pack_4bit_interleaved,
    unpack_2bit_hi,
    unpack_2bit_interleaved,
    unpack_4bit_interleaved,
)
from sparse_moe_mm.kernels.sm89 import (
    launch_dense_224_grouped_mm,
    launch_dense_fp8_grouped_mm,
    launch_prog8_224_stage1,
    launch_prog8_224_stage2,
    launch_sparse_fp8_stage2,
)
from sparse_moe_mm.kernels.sm90 import (
    launch_dense_sm90_224_grouped_mm,
    launch_prog8_sm90_stage1,
    launch_prog8_sm90_stage2,
    launch_quantize_x_fp8_sum,
)

if HAS_TRITON:
    from sparse_moe_mm.kernels._common import (
        _dense_eg_down_fp8_b128_kernel,
        _fast_cold_topk_kernel,
        _sparse_eg_down_fp8_b128_v3_kernel,
        _svd_recovery_eg_kernel,
        _weighted_unpermute_combine_kernel,
    )
    from sparse_moe_mm.kernels.sm89 import (
        _dense_eg_gate_up_224_kernel,
        _dense_eg_gate_up_fp8_b128_kernel,
        _prog8_224_stage1_kernel,
        _prog8_224_stage2_gate_up_kernel,
        _sparse_eg_gate_up_fp8_b128_v3_kernel,
    )
    from sparse_moe_mm.kernels.sm90 import (
        _dense_sm90_224_gu_kernel,
        _dense_swiglu_kernel,
        _prog8_sm90_stage1_kernel,
        _prog8_sm90_stage2_gu_kernel,
        _quantize_x_fp8_sum_kernel,
        _stage1_swiglu_score_kernel,
        _stage2_swiglu_zero_hs1_kernel,
    )

__all__ = [
    "HAS_TRITON",
    "launch_dense_224_grouped_mm",
    "launch_dense_fp8_grouped_mm",
    "launch_dense_sm90_224_grouped_mm",
    "launch_fast_cold_topk",
    "launch_prog8_224_stage1",
    "launch_prog8_224_stage2",
    "launch_prog8_sm90_stage1",
    "launch_prog8_sm90_stage2",
    "launch_quantize_x_fp8_sum",
    "launch_sparse_fp8_stage2",
    "launch_svd_recovery",
    "launch_weighted_unpermute_combine",
    "pack_2bit_hi",
    "unpack_2bit_hi",
]
