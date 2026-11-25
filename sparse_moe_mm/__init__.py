"""sparse_moe_mm: Quantized proxy-gated sparse grouped_mm for fine-grained MoEs."""

from sparse_moe_mm.reference import (
    MoEShape,
    block_group_mask,
    dequantize_int4_rowwise,
    dense_moe_swiglu_ref,
    proxy_gated_sparse_moe_ref,
    quantize_int4_rowwise,
    sparse_moe_swiglu_ref,
)
from sparse_moe_mm.roofline import GPUS, MODELS, compute_roofline_table

__all__ = [
    "GPUS",
    "MODELS",
    "MoEShape",
    "block_group_mask",
    "compute_roofline_table",
    "dequantize_int4_rowwise",
    "dense_moe_swiglu_ref",
    "proxy_gated_sparse_moe_ref",
    "quantize_int4_rowwise",
    "sparse_moe_swiglu_ref",
]
