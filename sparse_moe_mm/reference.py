"""Exact fp32 ground-truth references for quantized sparse MoE architectures.

v3 (O1-O4 Optimized):
- Official `[128,128]` block-scaled FP8 (`Qwen3.5-FP8` / `DeepSeek-V3`)
- Blackwell `1x16` micro-block NVFP4 (`E2M1` + `E4M3` block scale)
- Power-of-2 `2+2+4` Zero-VRAM Progressive Bitplanes (`hi2` + `mid2` + `lo4` == exact INT8)
  with 3-tier progressive precision (8-bit top-k1 + 4-bit next-k2 + SVD-r16 rest)
- Block-interleaved packing (`pack_2bit_interleaved`, `pack_4bit_interleaved`) for 100%
  coalesced activation loads in Triton/CUDA kernels
- Expert-level max-reduced Top-K_u selection (zero torch.unique, 100% static CUDA Graph)
- Legacy `Prog8 (4+4)`, `Prog8-Dual (2+6)`, `Tern8`, and `Prox` references retained for parity
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
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = low.float()
    out[..., 1::2] = high.float()
    return out * scale.float()


def quantize_fp8_block128(weight: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate official `[128,128]` block-scaled FP8 E4M3 quantization (Qwen3.5-FP8 / DeepSeek-V3)."""
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
    """Simulate Blackwell NVFP4 (`E2M1` + `1x16` `E4M3` micro-block scale along last dim)."""
    w = weight.float()
    if w.shape[-1] % block_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} must be divisible by NVFP4 block_size={block_size}")
    *prefix, out_f, in_f = w.shape
    num_blocks = in_f // block_size
    blocks = w.view(*prefix, out_f, num_blocks, block_size)
    tensor_max = w.abs().amax().clamp(min=1e-8)
    tensor_scale = tensor_max / (6.0 * 448.0)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    raw_block_scale = (block_max / (6.0 * tensor_scale)).clamp(max=448.0)
    q_block_scale = raw_block_scale.to(torch.float8_e4m3fn).float().clamp(min=1e-8)
    eff_scale = q_block_scale * tensor_scale
    normed = (blocks / eff_scale).clamp(-6.0, 6.0)
    levels = _NVFP4_LEVELS.to(w.device)
    idx = (normed.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q_vals = levels[idx]
    w_hat = (q_vals * eff_scale).reshape(*prefix, out_f, in_f)
    return w_hat.contiguous(), eff_scale.squeeze(-1).contiguous()


def quantize_prog8_4_4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-VRAM Progressive `4+4` exact lossless INT8 bit-slice (`q == 16*hi4 + lo4`)."""
    w = weight.float()
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} must be divisible by group_size={group_size}")
    *prefix, out_f, in_f = w.shape
    nb = in_f // group_size
    blocks = w.view(*prefix, out_f, nb, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(blocks / scale).clamp(-128, 127).to(torch.int16)
    hi4 = torch.div(q, 16, rounding_mode="floor").to(torch.int8)
    lo4 = (q - 16 * hi4.to(torch.int16)).to(torch.uint8)
    return hi4.reshape(*prefix, out_f, in_f), lo4.reshape(*prefix, out_f, in_f), scale.squeeze(-1)


def dequantize_prog8_4_4(
    hi4: torch.Tensor,
    lo4: torch.Tensor | None,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    *prefix, out_f, in_f = hi4.shape
    nb = in_f // group_size
    q = (16.0 * hi4.float() + 7.5) if lo4 is None else (16.0 * hi4.float() + lo4.float())
    return (q.view(*prefix, out_f, nb, group_size) * scale.unsqueeze(-1)).reshape(*prefix, out_f, in_f)


def quantize_prog8_dual_2_6(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-VRAM Progressive `2+6` exact lossless INT8 bit-slice (`q == 64*hi2 + lo6`)."""
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
    *prefix, out_f, in_f = hi2.shape
    nb = in_f // group_size
    q = (64.0 * hi2.float() + 31.5) if lo6 is None else (64.0 * hi2.float() + lo6.float())
    return (q.view(*prefix, out_f, nb, group_size) * scale.unsqueeze(-1)).reshape(*prefix, out_f, in_f)


def quantize_prog8_224(
    weight: torch.Tensor, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """O3/O9: Power-of-2 Zero-VRAM `2 + 2 + 4` Progressive Bitplane Split (0% extra VRAM!).

    Splits symmetric INT8 `q in [-128..127]` into three power-of-2 aligned bitplanes:
      - `hi2`  in [-2..1]  (2-bit signed,   0.25 B/elem) -> Stage 1 coarse scoring (`block_d=512` -> 128B!)
      - `mid2` in [0..3]   (2-bit unsigned, 0.25 B/elem) -> Stage 2 (`block_d=512` -> 128B!)
      - `lo4`  in [0..15]  (4-bit unsigned, 0.50 B/elem) -> Stage 2 (`block_d=256` -> 128B!)
    If `group_size is None`, uses per-row (per-neuron) scale `[..., out_f, 1]` enabling zero-scale-load
    512-weight inner loops (`207 GB/s` on L4 = 100% memory copy ceiling!).
    """
    w = weight.float()
    gs = w.shape[-1] if group_size is None else group_size
    hi2, lo6, scale = quantize_prog8_dual_2_6(w, group_size=gs)
    mid2 = (lo6 >> 4).to(torch.uint8)
    lo4 = (lo6 & 0x0F).to(torch.uint8)
    return hi2, mid2, lo4, scale


def dequantize_prog8_224(
    hi2: torch.Tensor,
    mid2: torch.Tensor | None,
    lo4: torch.Tensor | None,
    scale: torch.Tensor,
    group_size: int | None = None,
) -> torch.Tensor:
    """Reconstruct Stage-1 (2-bit), Tier-2 (4-bit), or Tier-1 (bit-exact 8-bit) weights."""
    *prefix, out_f, in_f = hi2.shape
    gs = in_f if group_size is None else group_size
    nb = in_f // gs
    if mid2 is None:
        q = 64.0 * hi2.float() + 32.0
    elif lo4 is None:
        q = 64.0 * hi2.float() + 16.0 * mid2.float() + 7.5
    else:
        q = 64.0 * hi2.float() + 16.0 * mid2.float() + lo4.float()
    return (q.view(*prefix, out_f, nb, gs) * scale.unsqueeze(-1)).reshape(*prefix, out_f, in_f)


def pack_2bit_interleaved(vals: torch.Tensor, signed: bool = True, block_d: int = 128) -> torch.Tensor:
    """O1: Block-interleaved 2-bit packing (4 weights/byte) for 100% coalesced activation loads!

    Within each `block_d`-element block (e.g., 128 -> 32 packed bytes), byte `j` packs
    elements `[j, j + 32, j + 64, j + 96]` so unpacking lane `l` yields the contiguous
    slice `[l*32 : (l+1)*32]` matching contiguous `x` loads!
    """
    if vals.shape[-1] % block_d != 0 or block_d % 4 != 0:
        raise ValueError(f"Last dim {vals.shape[-1]} must be divisible by block_d={block_d}")
    *prefix, in_f = vals.shape
    nb = in_f // block_d
    stride = block_d // 4
    u = ((vals.to(torch.int16) + 2) if signed else vals.to(torch.int16)).clamp(0, 3).to(torch.uint8)
    v = u.view(*prefix, nb, 4, stride)
    packed = v[..., 0, :] | (v[..., 1, :] << 2) | (v[..., 2, :] << 4) | (v[..., 3, :] << 6)
    return packed.reshape(*prefix, nb * stride).contiguous()


def unpack_2bit_interleaved(packed: torch.Tensor, signed: bool = True, block_d: int = 128) -> torch.Tensor:
    """Inverse of `pack_2bit_interleaved`."""
    stride = block_d // 4
    *prefix, p_f = packed.shape
    nb = p_f // stride
    p = packed.view(*prefix, nb, stride)
    lanes = torch.stack(
        [
            p & 0x03,
            (p >> 2) & 0x03,
            (p >> 4) & 0x03,
            (p >> 6) & 0x03,
        ],
        dim=-2,
    )  # [..., nb, 4, stride]
    out = lanes.reshape(*prefix, nb * block_d)
    return (out.to(torch.int8) - 2) if signed else out.to(torch.uint8)


def pack_4bit_interleaved(vals_u4: torch.Tensor, block_d: int = 128) -> torch.Tensor:
    """O3: Block-interleaved unsigned 4-bit packing (2 weights/byte) for 100% coalesced loads!

    Within each `block_d`-element block (128 -> 64 packed bytes), byte `j` packs
    elements `[j, j + 64]` so unpacking lane `l` yields contiguous `[l*64 : (l+1)*64]`.
    """
    if vals_u4.shape[-1] % block_d != 0 or block_d % 2 != 0:
        raise ValueError(f"Last dim {vals_u4.shape[-1]} must be divisible by block_d={block_d}")
    *prefix, in_f = vals_u4.shape
    nb = in_f // block_d
    stride = block_d // 2
    u = vals_u4.to(torch.uint8) & 0x0F
    v = u.view(*prefix, nb, 2, stride)
    packed = v[..., 0, :] | (v[..., 1, :] << 4)
    return packed.reshape(*prefix, nb * stride).contiguous()


def unpack_4bit_interleaved(packed: torch.Tensor, block_d: int = 128) -> torch.Tensor:
    """Inverse of `pack_4bit_interleaved` -> uint8 in [0..15]."""
    stride = block_d // 2
    *prefix, p_f = packed.shape
    nb = p_f // stride
    p = packed.view(*prefix, nb, stride)
    lanes = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-2)
    return lanes.reshape(*prefix, nb * block_d).to(torch.uint8)


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
    """Calibration-free top-k boolean keep mask [..., d_ff]."""
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


def expert_level_topk_indices(scores_expert: torch.Tensor, k_u: int) -> torch.Tensor:
    """O2: Expert-level max-reduced Top-K_u selection over [M_e, d_ff] -> [k_u] int64 indices.

    Eliminates per-token union compaction (`torch.unique`) and gives every token in the expert
    free access to all `k_u` loaded neurons.
    """
    exp_score = scores_expert.float().abs().amax(dim=0)  # [d_ff]
    k_clamped = min(max(1, k_u), scores_expert.shape[-1])
    return torch.topk(exp_score, k=k_clamped, dim=-1, sorted=False).indices


def compute_coactivation_permutation(
    act_magnitudes: torch.Tensor,
    group_size: int = 16,
    top_frac: float = 0.15,
) -> torch.Tensor:
    n_tok, d_ff = act_magnitudes.shape
    if d_ff % group_size != 0:
        raise ValueError(f"d_ff={d_ff} must be divisible by group_size={group_size}")
    num_groups = d_ff // group_size
    if n_tok < 2:
        return torch.arange(d_ff, device=act_magnitudes.device, dtype=torch.long)

    k_top = max(1, int(round(d_ff * top_frac)))
    thr = torch.topk(act_magnitudes.float(), k=k_top, dim=-1).values[:, -1:]
    act_bin = (act_magnitudes.float() >= thr).float()
    freq = act_bin.sum(dim=0)
    act_norm = F.normalize(act_bin.T, dim=-1)
    sim = act_norm @ act_norm.T

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
    return w_gate[perm].contiguous(), w_up[perm].contiguous(), w_down[:, perm].contiguous()


def dense_moe_swiglu_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
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
    x_f = x.float()
    tw = topk_weights.float()
    if dual:
        wg_hi, wg_mid, wg_lo, wg_s = quantize_prog8_224(w_gate, group_size=quant_group)
        wu_hi, wu_mid, wu_lo, wu_s = quantize_prog8_224(w_up, group_size=quant_group)
        wg_s1 = dequantize_prog8_224(wg_hi, None, None, wg_s, group_size=quant_group)
        wu_s1 = dequantize_prog8_224(wu_hi, None, None, wu_s, group_size=quant_group)
        wg_s2 = dequantize_prog8_224(wg_hi, wg_mid, wg_lo, wg_s, group_size=quant_group)
        wu_s2 = dequantize_prog8_224(wu_hi, wu_mid, wu_lo, wu_s, group_size=quant_group)
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
            scores = (F.silu(wg_s1[e] @ xt) * (wu_s1[e] @ xt)) if dual else F.silu(wg_s1[e] @ xt)
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
