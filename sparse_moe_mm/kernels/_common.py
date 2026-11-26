"""Shared Triton primitives: Down projection, SVD-r16 recovery, Fused Combine, and Register Bitonic TopK."""

from __future__ import annotations

import torch

from sparse_moe_mm.reference import (
    pack_2bit_interleaved,
    pack_4bit_interleaved,
    unpack_2bit_interleaved,
    unpack_4bit_interleaved,
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton_cuda() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Run on a CUDA GPU with triton>=3.2.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run Triton kernels.")


pack_2bit_hi = pack_2bit_interleaved
unpack_2bit_hi = unpack_2bit_interleaved


if HAS_TRITON:

    @triton.jit
    def _sparse_eg_down_fp8_b128_v3_kernel(
        h_act_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        active_idx_ptr,
        w_down_t_ptr,
        w_down_scale_ptr,
        out_ptr,
        stride_h_e,
        stride_h_m,
        stride_h_k,
        stride_idx_e,
        stride_idx_k,
        stride_w_e,
        stride_w_f,
        stride_w_d,
        stride_s_e,
        stride_s_db,
        stride_s_fb,
        stride_out_n,
        stride_out_d,
        K_U: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP8: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        d_tile = tl.program_id(1)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        m_offs = tl.arange(0, MAX_ME)
        m_mask = m_offs < m_e
        d_offs = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)

        out_acc = tl.zeros((MAX_ME, BLOCK_D), dtype=tl.float32)

        for k_start in range(0, K_U, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K_U
            n_ids = tl.load(active_idx_ptr + exp_slot * stride_idx_e + k_offs * stride_idx_k, mask=k_mask, other=0).to(tl.int64)
            h_tile = tl.load(
                h_act_ptr + exp_slot * stride_h_e + m_offs[:, None] * stride_h_m + k_offs[None, :] * stride_h_k,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            if IS_FP8:
                fb_ids = n_ids // 128
                sd = tl.load(
                    w_down_scale_ptr + eid * stride_s_e + d_tile * stride_s_db + fb_ids * stride_s_fb,
                    mask=k_mask,
                    other=1.0,
                ).to(tl.float32)
                h_tile = h_tile * sd[None, :]

            wd_ptrs = (
                w_down_t_ptr
                + eid * stride_w_e
                + n_ids[:, None] * stride_w_f
                + d_offs[None, :] * stride_w_d
            )
            wd_tile = tl.load(wd_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
            out_acc += tl.dot(h_tile.to(tl.bfloat16), wd_tile)

        tl.store(
            out_ptr + (x_start + m_offs[:, None]) * stride_out_n + d_offs[None, :] * stride_out_d,
            out_acc.to(tl.bfloat16),
            mask=m_mask[:, None],
        )

    @triton.jit
    def _dense_eg_down_fp8_b128_kernel(
        h_dense_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        w_down_t_ptr,
        w_down_scale_ptr,
        out_ptr,
        stride_h_n,
        stride_h_f,
        stride_w_e,
        stride_w_f,
        stride_w_d,
        stride_s_e,
        stride_s_db,
        stride_s_fb,
        stride_out_n,
        stride_out_d,
        D_FF: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP8: tl.constexpr,
        ME_MIN: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        d_tile = tl.program_id(1)
        m_tile = tl.program_id(2)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        m_start = m_tile * TILE_M
        if m_e < ME_MIN or m_start >= m_e:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot) + m_start
        m_offs = tl.arange(0, TILE_M)
        m_mask = (m_start + m_offs) < m_e
        d_offs = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)

        out_acc = tl.zeros((TILE_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, D_FF, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            h_t = tl.load(
                h_dense_ptr + (x_start + m_offs[:, None]) * stride_h_n + k_offs[None, :] * stride_h_f,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if IS_FP8:
                fb_id = k_start // 128
                sd = tl.load(w_down_scale_ptr + eid * stride_s_e + d_tile * stride_s_db + fb_id * stride_s_fb).to(tl.float32)
                h_t = h_t * sd
            wd_t = tl.load(
                w_down_t_ptr + eid * stride_w_e + k_offs[:, None] * stride_w_f + d_offs[None, :] * stride_w_d
            ).to(tl.bfloat16)
            out_acc += tl.dot(h_t.to(tl.bfloat16), wd_t)

        tl.store(
            out_ptr + (x_start + m_offs[:, None]) * stride_out_n + d_offs[None, :] * stride_out_d,
            out_acc.to(tl.bfloat16),
            mask=m_mask[:, None],
        )

    @triton.jit
    def _svd_recovery_eg_kernel(
        h_s1_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        u_svd_ptr,
        v_svd_ptr,
        out_ptr,
        stride_h_n,
        stride_h_f,
        stride_u_e,
        stride_u_r,
        stride_u_d,
        stride_v_e,
        stride_v_r,
        stride_v_f,
        stride_out_n,
        stride_out_d,
        D_FF: tl.constexpr,
        RANK: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        d_tile = tl.program_id(1)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        m_offs = tl.arange(0, MAX_ME)
        m_mask = m_offs < m_e
        r_offs = tl.arange(0, RANK)
        d_offs = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)

        z_acc = tl.zeros((MAX_ME, RANK), dtype=tl.float32)
        for f_start in range(0, D_FF, 128):
            f_offs = f_start + tl.arange(0, 128)
            h_t = tl.load(
                h_s1_ptr + (x_start + m_offs[:, None]) * stride_h_n + f_offs[None, :] * stride_h_f,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            v_t = tl.load(
                v_svd_ptr + eid * stride_v_e + r_offs[:, None] * stride_v_r + f_offs[None, :] * stride_v_f
            ).to(tl.bfloat16)
            z_acc += tl.dot(h_t, tl.trans(v_t))

        u_t = tl.load(
            u_svd_ptr + eid * stride_u_e + r_offs[:, None] * stride_u_r + d_offs[None, :] * stride_u_d
        ).to(tl.bfloat16)
        rec = tl.dot(z_acc.to(tl.bfloat16), u_t)

        out_ptrs = out_ptr + (x_start + m_offs[:, None]) * stride_out_n + d_offs[None, :] * stride_out_d
        prev = tl.load(out_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptrs, (prev + rec).to(tl.bfloat16), mask=m_mask[:, None])

    @triton.jit
    def _weighted_unpermute_combine_kernel(
        pair_out_ptr,  # [B * K, d_model] bf16
        inv_ptr,  # [B * K] int64 (maps flat slot t*K + s -> sorted position)
        weights_ptr,  # [B, K] fp32 (already zeroed for non-local EP slots)
        out_ptr,  # [B, d_model] bf16/fp16
        stride_p_n,
        stride_p_d,
        stride_w_b,
        stride_w_k,
        stride_o_b,
        stride_o_d,
        TOP_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b_id = tl.program_id(0)
        d_tile = tl.program_id(1)
        d_offs = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for s in tl.static_range(0, TOP_K):
            w_s = tl.load(weights_ptr + b_id * stride_w_b + s * stride_w_k).to(tl.float32)
            pos = tl.load(inv_ptr + b_id * TOP_K + s)
            v = tl.load(pair_out_ptr + pos * stride_p_n + d_offs * stride_p_d).to(tl.float32)
            acc += w_s * v
        tl.store(out_ptr + b_id * stride_o_b + d_offs * stride_o_d, acc.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _fast_cold_topk_kernel(
        # 1-kernel register bitonic TopK (early-exits empty/hot experts, 3.1x-4.4x faster than torch.topk!)
        scores_ptr,  # [N_slots, D_FF] non-negative fp32
        counts_ptr,  # [N_slots] int32
        out_idx_ptr,  # [N_slots, K_U] int32
        stride_s_e,
        stride_s_f,
        stride_o_e,
        stride_o_k,
        D_FF: tl.constexpr,
        K_U: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        slot = tl.program_id(0)
        m_e = tl.load(counts_ptr + slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        offs = tl.arange(0, D_FF)
        vals = tl.load(scores_ptr + slot * stride_s_e + offs * stride_s_f)
        bits = vals.to(tl.uint32, bitcast=True)
        # Non-negative fp32 -> top 20 bits preserve exact monotonic order; XOR flips to descending!
        key = ((bits & 0xFFFFF000) ^ 0xFFFFF000) | offs.to(tl.uint32)
        sorted_keys = tl.sort(key, dim=0)
        out_idx = (sorted_keys & 0xFFF).to(tl.int32)
        tl.store(out_idx_ptr + slot * stride_o_e + offs * stride_o_k, out_idx, mask=offs < K_U)


def launch_weighted_unpermute_combine(
    pair_out: torch.Tensor,
    inv: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    block_d: int = 256,
) -> torch.Tensor:
    """Fused 1-kernel weighted unpermute + top-k reduce (~3 us, zero intermediate fp32 tensor!)."""
    _require_triton_cuda()
    b, top_k = weights.shape
    d_model = output.shape[1]
    bd = block_d if (d_model % block_d == 0) else 128
    _weighted_unpermute_combine_kernel[(b, d_model // bd)](
        pair_out, inv, weights, output,
        pair_out.stride(0), pair_out.stride(1),
        weights.stride(0), weights.stride(1),
        output.stride(0), output.stride(1),
        TOP_K=top_k, BLOCK_D=bd,
        num_warps=4,
    )
    return output


def launch_fast_cold_topk(
    exp_scores: torch.Tensor,
    expert_counts: torch.Tensor,
    out_idx: torch.Tensor,
    me_min: int = 1,
    me_max: int = 8,
) -> torch.Tensor:
    """Fused 1-kernel register bitonic TopK (early-exits empty/hot experts, 3.1x-4.4x faster than torch.topk!)."""
    _require_triton_cuda()
    n_slots, d_ff = exp_scores.shape
    k_u = out_idx.shape[1]
    if d_ff <= 4096 and (d_ff & (d_ff - 1)) == 0:
        _fast_cold_topk_kernel[(n_slots,)](
            exp_scores, expert_counts, out_idx,
            exp_scores.stride(0), exp_scores.stride(1),
            out_idx.stride(0), out_idx.stride(1),
            D_FF=d_ff, K_U=k_u, ME_MIN=me_min, ME_MAX=me_max,
            num_warps=4,
        )
    else:
        out_idx.copy_(torch.topk(exp_scores, k=k_u, dim=-1, sorted=True).indices.to(torch.int32))
    return out_idx


def launch_svd_recovery(
    h_s1: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    u_svd_t: torch.Tensor,
    v_svd: torch.Tensor,
    out_buf: torch.Tensor,
    max_me: int = 16,
    block_d: int = 256,
    me_min: int = 1,
    me_max: int = 1048576,
) -> torch.Tensor:
    _require_triton_cuda()
    d_model = out_buf.shape[1]
    d_ff = h_s1.shape[1]
    rank = v_svd.shape[1]
    n_slots = expert_ids.shape[0]
    bd = block_d if (d_model % block_d == 0) else 128
    _svd_recovery_eg_kernel[(n_slots, d_model // bd)](
        h_s1, expert_ids, expert_starts, expert_counts, u_svd_t, v_svd, out_buf,
        h_s1.stride(0), h_s1.stride(1),
        u_svd_t.stride(0), u_svd_t.stride(1), u_svd_t.stride(2),
        v_svd.stride(0), v_svd.stride(1), v_svd.stride(2),
        out_buf.stride(0), out_buf.stride(1),
        D_FF=d_ff, RANK=rank, MAX_ME=max_me, BLOCK_D=bd,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4, num_stages=2,
    )
    return out_buf

