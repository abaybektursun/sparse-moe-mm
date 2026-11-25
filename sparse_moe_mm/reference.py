"""Exact fp32 ground-truth references for dense, sparse, and proxy-gated MoE layers.

All arithmetic runs in fp32 so every Triton/CUDA kernel and quantization scheme
can be verified against a clean analytical reference on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEShape:
    num_experts: int
    top_k: int
    d_model: int
    d_ff: int


def quantize_int4_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row signed INT4 quantization packed as uint8 (2 nibbles/byte).

    weight: [..., out_features, in_features], in_features must be even.
    Returns (packed_u8 [..., out_features, in_features // 2], scales fp32 [..., out_features, 1]).
    """
    if weight.shape[-1] % 2 != 0:
        raise ValueError(f"in_features must be even for INT4 packing, got {weight.shape[-1]}")
    w = weight.float()
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    # Two's-complement 4-bit nibble packing: low nibble = even col, high nibble = odd col
    q_u8 = (q & 0x0F).to(torch.uint8)
    low = q_u8[..., 0::2]
    high = q_u8[..., 1::2]
    packed = low | (high << 4)
    return packed.contiguous(), scale.contiguous()


def dequantize_int4_rowwise(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int4_rowwise -> fp32 tensor [..., out_features, in_features]."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign-extend 4-bit signed values in [0..15] -> [-8..7]
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = low.float()
    out[..., 1::2] = high.float()
    return out * scale.float()


def block_group_mask(
    scores: torch.Tensor,
    keep_frac: float,
    group_size: int = 1,
) -> torch.Tensor:
    """Create a boolean keep mask [..., d_ff] selecting top-k groups of `group_size` neurons.

    For group_size == 1, selects top-k individual neurons by |scores|.
    For group_size > 1, scores each contiguous block of `group_size` neurons by L2 norm
    so active neurons stay tile-aligned for Tensor Core MMA instructions.
    """
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    d_ff = scores.shape[-1]
    if d_ff % group_size != 0:
        raise ValueError(f"d_ff={d_ff} must be divisible by group_size={group_size}")
    if keep_frac == 1.0:
        return torch.ones_like(scores, dtype=torch.bool)

    num_groups = d_ff // group_size
    k_groups = max(1, int(round(num_groups * keep_frac)))
    if group_size == 1:
        group_scores = scores.float().abs()
    else:
        grouped = scores.float().view(*scores.shape[:-1], num_groups, group_size)
        group_scores = grouped.pow(2).sum(dim=-1)

    topk_idx = torch.topk(group_scores, k=k_groups, dim=-1, sorted=False).indices
    group_keep = torch.zeros_like(group_scores, dtype=torch.bool)
    group_keep.scatter_(-1, topk_idx, True)
    if group_size == 1:
        return group_keep
    return group_keep.unsqueeze(-1).expand(*scores.shape[:-1], num_groups, group_size).reshape_as(scores)


def dense_moe_swiglu_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Exact fp32 dense SwiGLU MoE reference.

    x: [T, d_model]
    topk_ids: [T, K] int64 in [0, E)
    topk_weights: [T, K]
    w_gate, w_up: [E, d_ff, d_model]
    w_down: [E, d_model, d_ff]
    """
    x_f = x.float()
    tw = topk_weights.float()
    wg = w_gate.float()
    wu = w_up.float()
    wd = w_down.float()

    t_len, d_model = x_f.shape
    k = topk_ids.shape[1]
    out = torch.zeros((t_len, d_model), dtype=torch.float32, device=x.device)

    for t in range(t_len):
        xt = x_f[t]
        for slot in range(k):
            e = int(topk_ids[t, slot].item())
            g = F.silu(wg[e] @ xt)
            u = wu[e] @ xt
            h = g * u
            out[t] += tw[t, slot] * (wd[e] @ h)
    return out


def sparse_moe_swiglu_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    keep_frac: float = 0.10,
    group_size: int = 1,
    rank_by: str = "swiglu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact fp32 masked SwiGLU MoE reference returning (output [T, d], masks [T, K, d_ff]).

    rank_by:
      - 'swiglu': oracle ranking by |silu(gate) * up|
      - 'gate':   Park/CATS ranking by |silu(gate)| only (skips up & down)
    """
    if rank_by not in ("swiglu", "gate"):
        raise ValueError(f"rank_by must be 'swiglu' or 'gate', got {rank_by}")

    x_f = x.float()
    tw = topk_weights.float()
    wg = w_gate.float()
    wu = w_up.float()
    wd = w_down.float()

    t_len, d_model = x_f.shape
    k = topk_ids.shape[1]
    d_ff = wg.shape[1]
    out = torch.zeros((t_len, d_model), dtype=torch.float32, device=x.device)
    masks = torch.zeros((t_len, k, d_ff), dtype=torch.bool, device=x.device)

    for t in range(t_len):
        xt = x_f[t]
        for slot in range(k):
            e = int(topk_ids[t, slot].item())
            g = F.silu(wg[e] @ xt)
            u = wu[e] @ xt
            h = g * u
            scores = h if rank_by == "swiglu" else g
            m = block_group_mask(scores, keep_frac=keep_frac, group_size=group_size)
            masks[t, slot] = m
            out[t] += tw[t, slot] * (wd[e] @ (h * m.float()))
    return out, masks


def proxy_gated_sparse_moe_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    keep_frac: float = 0.10,
    input_keep_frac: float = 0.50,
    group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage Prox-style MoE reference:

    Stage 1: Symmetric INT4 proxy weights (w_gate_q, w_up_q) + input magnitude mask
             (input_keep_frac) estimate SwiGLU magnitude and build boolean mask `m`.
    Stage 2: Exact fp32 computation of ONLY active channels across all three matrices
             (w_gate, w_up, w_down).
    """
    wg_q, wg_s = quantize_int4_rowwise(w_gate)
    wu_q, wu_s = quantize_int4_rowwise(w_up)
    wg_proxy = dequantize_int4_rowwise(wg_q, wg_s)
    wu_proxy = dequantize_int4_rowwise(wu_q, wu_s)

    x_f = x.float()
    tw = topk_weights.float()
    wg = w_gate.float()
    wu = w_up.float()
    wd = w_down.float()

    t_len, d_model = x_f.shape
    k = topk_ids.shape[1]
    d_ff = wg.shape[1]
    out = torch.zeros((t_len, d_model), dtype=torch.float32, device=x.device)
    masks = torch.zeros((t_len, k, d_ff), dtype=torch.bool, device=x.device)

    for t in range(t_len):
        xt = x_f[t]
        x_in_mask = block_group_mask(xt, keep_frac=input_keep_frac, group_size=1)
        xt_sparse = xt * x_in_mask.float()
        for slot in range(k):
            e = int(topk_ids[t, slot].item())
            # Stage 1: INT4 proxy salience estimate
            h_proxy = F.silu(wg_proxy[e] @ xt_sparse) * (wu_proxy[e] @ xt_sparse)
            m = block_group_mask(h_proxy, keep_frac=keep_frac, group_size=group_size)
            masks[t, slot] = m
            # Stage 2: Exact sparse computation on active channels only
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            g_act = F.silu(wg[e, idx] @ xt)
            u_act = wu[e, idx] @ xt
            h_act = g_act * u_act
            out[t] += tw[t, slot] * (wd[e, :, idx] @ h_act)
    return out, masks
