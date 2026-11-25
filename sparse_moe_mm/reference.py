"""Exact fp32 ground-truth references for quantized sparse MoE architectures.

v2 (Impact-Aligned Reset) includes:
- Official `[128,128]` block-scaled FP8 (`Qwen3.5-FP8` / `DeepSeek-V3`)
- Blackwell `1x16` micro-block NVFP4 (`E2M1` + `E4M3` block scale)
- Zero-VRAM Progressive Bit-Sliced `Prog8` (4+4 exact lossless INT8 split)
- Zero-VRAM Progressive Bit-Sliced `Prog8-Dual` (2+6 exact lossless INT8 split on gate+up)
- 2-bit Ternary Sketch (`arXiv:2610.37386` baseline)
- Offline Lossless Co-Activation Neuron Permutation (`g = 16` NVFP4 micro-block alignment)
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


# --- NVFP4 E2M1 Codebook (16 representable signed values) ---
_NVFP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def quantize_int4_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row signed INT4 quantization packed as uint8 (2 nibbles/byte)."""
    if weight.shape[-1] % 2 != 0:
        raise ValueError(f"in_features must be even for INT4 packing, got {weight.shape[-1]}")
    w = weight.float()
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    q_u8 = (q & 0x0F).to(torch.uint8)
    packed = q_u8[..., 0::2] | (q_u8[..., 1::2] << 4)
    return packed.contiguous(), scale.contiguous()


def dequantize_int4_rowwise(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int4_rowwise -> fp32 tensor [..., out_features, in_features]."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = low.float()
    out[..., 1::2] = high.float()
    return out * scale.float()


def quantize_fp8_block128(weight: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate official `[128,128]` block-scaled FP8 E4M3 quantization (Qwen3.5-FP8 / DeepSeek-V3).

    Returns (w_fp8_as_fp32 [..., M, N], scales [..., ceil(M/B), ceil(N/B)]).
    """
    w = weight.float()
    *prefix, m, n = w.shape
    bm = (m + block_size - 1) // block_size
    bn = (n + block_size - 1) // block_size
    pad_m = bm * block_size - m
    pad_n = bn * block_size - n
    w_pad = F.pad(w, (0, pad_n, 0, pad_m)) if (pad_m or pad_n) else w
    blocks = w_pad.view(*prefix, bm, block_size, bn, block_size)
    max_abs = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-8)
    scale = max_abs / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    w_hat = (q.float() * scale).reshape(*prefix, bm * block_size, bn * block_size)[..., :m, :n]
    return w_hat.contiguous(), scale.squeeze(-1).squeeze(-2).contiguous()


def quantize_nvfp4_block16(weight: torch.Tensor, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate Blackwell NVFP4 (`E2M1` values + `1x16` `E4M3` micro-block scale along last dim).

    Returns (dequantized_fp32, block_scales_e4m3_fp32).
    """
    w = weight.float()
    if w.shape[-1] % block_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} must be divisible by NVFP4 block_size={block_size}")
    *prefix, out_f, in_f = w.shape
    num_blocks = in_f // block_size
    blocks = w.view(*prefix, out_f, num_blocks, block_size)
    # Global tensor scale + per-16 E4M3 block scale
    tensor_max = w.abs().amax().clamp(min=1e-8)
    tensor_scale = tensor_max / (6.0 * 448.0)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    raw_block_scale = (block_max / (6.0 * tensor_scale)).clamp(max=448.0)
    # Quantize block scale to FP8 E4M3
    q_block_scale = raw_block_scale.to(torch.float8_e4m3fn).float().clamp(min=1e-8)
    eff_scale = q_block_scale * tensor_scale
    # Snap normalized values to E2M1 codebook
    normed = (blocks / eff_scale).clamp(-6.0, 6.0)
    levels = _NVFP4_LEVELS.to(w.device)
    idx = (normed.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q_vals = levels[idx]
    w_hat = (q_vals * eff_scale).reshape(*prefix, out_f, in_f)
    return w_hat.contiguous(), eff_scale.squeeze(-1).contiguous()


def quantize_prog8_4_4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-VRAM Progressive `4+4` exact lossless INT8 bit-slice.

    Quantizes weight to symmetric INT8 in [-127, 127] with 1x`group_size` scales,
    then splits each int8 value `q` into bit-exact `(hi4, lo4)` such that:
        q == 16 * hi4 + lo4
    where hi4 in [-8..7] (signed 4-bit) and lo4 in [0..15] (unsigned 4-bit).
    Total storage = 4 + 4 = 8 bits/param (0% VRAM overhead vs FP8/INT8!).
    """
    w = weight.float()
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} must be divisible by group_size={group_size}")
    *prefix, out_f, in_f = w.shape
    nb = in_f // group_size
    blocks = w.view(*prefix, out_f, nb, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(blocks / scale).clamp(-128, 127).to(torch.int16)
    # Exact floor division by 16: q = 16 * hi4 + lo4 with lo4 in [0..15], hi4 in [-8..7]
    hi4 = torch.div(q, 16, rounding_mode="floor").to(torch.int8)
    lo4 = (q - 16 * hi4.to(torch.int16)).to(torch.uint8)
    return hi4.reshape(*prefix, out_f, in_f), lo4.reshape(*prefix, out_f, in_f), scale.squeeze(-1)


def dequantize_prog8_4_4(
    hi4: torch.Tensor,
    lo4: torch.Tensor | None,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct Stage-1 coarse (`lo4=None`) or bit-exact Stage-2 full (`lo4!=None`) weights.

    When `lo4=None`, uses the unbiased midpoint `16 * hi4 + 7.5` (centering unsigned `lo4 in [0..15]`).
    """
    *prefix, out_f, in_f = hi4.shape
    nb = in_f // group_size
    if lo4 is None:
        q = 16.0 * hi4.float() + 7.5
    else:
        q = 16.0 * hi4.float() + lo4.float()
    return (q.view(*prefix, out_f, nb, group_size) * scale.unsqueeze(-1)).reshape(*prefix, out_f, in_f)


def quantize_prog8_dual_2_6(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-VRAM Progressive `2+6` exact lossless INT8 bit-slice for `Prog8-Dual` (gate+up).

    Splits each signed INT8 value `q in [-128..127]` into:
        q == 64 * hi2 + lo6
    where hi2 in [-2..1] (2-bit signed, 0.25 B/elem) and lo6 in [0..63] (6-bit unsigned, 0.75 B/elem).
    Total storage = 2 + 6 = 8 bits/param (0% VRAM overhead!).
    """
    w = weight.float()
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} must be divisible by group_size={group_size}")
    *prefix, out_f, in_f = w.shape
    nb = in_f // group_size
    blocks = w.view(*prefix, out_f, nb, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(blocks / scale).clamp(-128, 127).to(torch.int16)
    hi2 = torch.div(q, 64, rounding_mode="floor").to(torch.int8)
    lo6 = (q - 64 * hi2.to(torch.int16)).to(torch.uint8)
    return hi2.reshape(*prefix, out_f, in_f), lo6.reshape(*prefix, out_f, in_f), scale.squeeze(-1)


def dequantize_prog8_dual_2_6(
    hi2: torch.Tensor,
    lo6: torch.Tensor | None,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct Stage-1 2-bit coarse (`lo6=None`) or bit-exact Stage-2 full (`lo6!=None`) weights."""
    *prefix, out_f, in_f = hi2.shape
    nb = in_f // group_size
    if lo6 is None:
        q = 64.0 * hi2.float() + 31.5
    else:
        q = 64.0 * hi2.float() + lo6.float()
    return (q.view(*prefix, out_f, nb, group_size) * scale.unsqueeze(-1)).reshape(*prefix, out_f, in_f)


def quantize_ternary_2bit(weight: torch.Tensor) -> torch.Tensor:
    """2-bit TWN ternary sketch `{-alpha, 0, +alpha}` per output channel (arXiv:2610.37386 style)."""
    w = weight.float()
    delta = 0.7 * w.abs().mean(dim=-1, keepdim=True)
    active = w.abs() > delta
    sign = torch.where(active, w.sign(), torch.zeros_like(w))
    alpha = (w.abs() * active).sum(dim=-1, keepdim=True) / active.sum(dim=-1, keepdim=True).clamp(min=1)
    return (sign * alpha).contiguous()


def block_group_mask(
    scores: torch.Tensor,
    keep_frac: float,
    group_size: int = 1,
) -> torch.Tensor:
    """Calibration-free top-k boolean keep mask [..., d_ff] selecting top-k groups of `group_size` neurons."""
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


def compute_coactivation_permutation(
    act_magnitudes: torch.Tensor,
    group_size: int = 16,
    top_frac: float = 0.15,
) -> torch.Tensor:
    """Offline greedy balanced co-activation clustering returning a permutation `perm [d_ff]`.

    Given calibration activation magnitudes `[N_tok, d_ff]` for a single expert,
    places frequently co-activating neurons into contiguous blocks of size `group_size`
    (e.g., `g = 16` for Blackwell NVFP4 `1x16` micro-block alignment).
    """
    n_tok, d_ff = act_magnitudes.shape
    if d_ff % group_size != 0:
        raise ValueError(f"d_ff={d_ff} must be divisible by group_size={group_size}")
    num_groups = d_ff // group_size
    if n_tok < 2:
        return torch.arange(d_ff, device=act_magnitudes.device, dtype=torch.long)

    k_top = max(1, int(round(d_ff * top_frac)))
    thr = torch.topk(act_magnitudes.float(), k=k_top, dim=-1).values[:, -1:]
    act_bin = (act_magnitudes.float() >= thr).float()  # [N_tok, d_ff]
    freq = act_bin.sum(dim=0)
    act_norm = F.normalize(act_bin.T, dim=-1)  # [d_ff, N_tok]
    sim = act_norm @ act_norm.T  # [d_ff, d_ff]

    used = torch.zeros(d_ff, dtype=torch.bool, device=act_magnitudes.device)
    blocks = []
    for _ in range(num_groups):
        seed = int(freq.masked_fill(used, -1.0).argmax().item())
        s_row = sim[seed].masked_fill(used, -1e9)
        blk = torch.topk(s_row, k=group_size).indices
        used[blk] = True
        blocks.append(blk)
    return torch.cat(blocks, dim=0)


def apply_expert_permutation(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply lossless neuron permutation `perm [d_ff]` to a single expert's weights.

    w_gate, w_up: [d_ff, d_model]; w_down: [d_model, d_ff].
    """
    return w_gate[perm].contiguous(), w_up[perm].contiguous(), w_down[:, perm].contiguous()


def dense_moe_swiglu_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Exact fp32 dense SwiGLU MoE reference."""
    x_f = x.float()
    tw = topk_weights.float()
    wg, wu, wd = w_gate.float(), w_up.float(), w_down.float()

    t_len, d_model = x_f.shape
    k = topk_ids.shape[1]
    out = torch.zeros((t_len, d_model), dtype=torch.float32, device=x.device)
    for t in range(t_len):
        xt = x_f[t]
        for slot in range(k):
            e = int(topk_ids[t, slot].item())
            h = F.silu(wg[e] @ xt) * (wu[e] @ xt)
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
    """Exact fp32 masked SwiGLU MoE reference returning (output [T, d], masks [T, K, d_ff])."""
    if rank_by not in ("swiglu", "gate"):
        raise ValueError(f"rank_by must be 'swiglu' or 'gate', got {rank_by}")

    x_f = x.float()
    tw = topk_weights.float()
    wg, wu, wd = w_gate.float(), w_up.float(), w_down.float()

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


def prog8_sparse_moe_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    keep_frac: float = 0.10,
    group_size: int = 1,
    quant_group: int = 32,
    dual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-VRAM Progressive Bit-Sliced MoE reference (`Prog8` 4+4 gate or `Prog8-Dual` 2+6 gate+up).

    Stage 1: Reads ONLY the coarse slice (`hi4` of `w_gate` if `dual=False`,
             or `hi2` of `w_gate` + `w_up` if `dual=True`) -> 0.50 B/elem total.
    Stage 2: Reconstructs exact 8-bit values ONLY on the selected active channels
             using the remaining fine residual (`lo4` or `lo6`), plus `w_up`/`w_down`.
    """
    x_f = x.float()
    tw = topk_weights.float()
    if dual:
        wg_hi, wg_lo, wg_s = quantize_prog8_dual_2_6(w_gate, group_size=quant_group)
        wu_hi, wu_lo, wu_s = quantize_prog8_dual_2_6(w_up, group_size=quant_group)
        wg_s1 = dequantize_prog8_dual_2_6(wg_hi, None, wg_s, group_size=quant_group)
        wu_s1 = dequantize_prog8_dual_2_6(wu_hi, None, wu_s, group_size=quant_group)
        wg_s2 = dequantize_prog8_dual_2_6(wg_hi, wg_lo, wg_s, group_size=quant_group)
        wu_s2 = dequantize_prog8_dual_2_6(wu_hi, wu_lo, wu_s, group_size=quant_group)
    else:
        wg_hi, wg_lo, wg_s = quantize_prog8_4_4(w_gate, group_size=quant_group)
        wg_s1 = dequantize_prog8_4_4(wg_hi, None, wg_s, group_size=quant_group)
        wg_s2 = dequantize_prog8_4_4(wg_hi, wg_lo, wg_s, group_size=quant_group)
        wu_s1 = None
        wu_s2 = w_up.float()
    wd_s2 = w_down.float()

    t_len, d_model = x_f.shape
    k = topk_ids.shape[1]
    d_ff = w_gate.shape[1]
    out = torch.zeros((t_len, d_model), dtype=torch.float32, device=x.device)
    masks = torch.zeros((t_len, k, d_ff), dtype=torch.bool, device=x.device)

    for t in range(t_len):
        xt = x_f[t]
        for slot in range(k):
            e = int(topk_ids[t, slot].item())
            if dual:
                scores = F.silu(wg_s1[e] @ xt) * (wu_s1[e] @ xt)
            else:
                scores = F.silu(wg_s1[e] @ xt)
            m = block_group_mask(scores, keep_frac=keep_frac, group_size=group_size)
            masks[t, slot] = m
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            h_act = F.silu(wg_s2[e, idx] @ xt) * (wu_s2[e, idx] @ xt)
            out[t] += tw[t, slot] * (wd_s2[e, :, idx] @ h_act)
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
    """Legacy Prox-style INT4 proxy reference kept for head-to-head comparison."""
    wg_q, wg_s = quantize_int4_rowwise(w_gate)
    wu_q, wu_s = quantize_int4_rowwise(w_up)
    wg_proxy = dequantize_int4_rowwise(wg_q, wg_s)
    wu_proxy = dequantize_int4_rowwise(wu_q, wu_s)

    x_f = x.float()
    tw = topk_weights.float()
    wg, wu, wd = w_gate.float(), w_up.float(), w_down.float()

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
            h_proxy = F.silu(wg_proxy[e] @ xt_sparse) * (wu_proxy[e] @ xt_sparse)
            m = block_group_mask(h_proxy, keep_frac=keep_frac, group_size=group_size)
            masks[t, slot] = m
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            h_act = F.silu(wg[e, idx] @ xt) * (wu[e, idx] @ xt)
            out[t] += tw[t, slot] * (wd[e, :, idx] @ h_act)
    return out, masks
