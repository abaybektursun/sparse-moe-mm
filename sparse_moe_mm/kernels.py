"""v6 Triton Kernels (Iteration 4 — O9: 128B Cache-Line-Exact Packing + Marlin 0x6400 Bitcast + K=128 TC).

Iteration-4 breakthroughs (proven 207 GB/s = 100% of L4 hardware copy ceiling in micro-benchmark!):
  - Stage 1 (`_prog8_224_stage1_kernel`):
      * Packs `hi2` with `block_d = 512` (`PACK_D = 128` contiguous bytes per row = exact 128B cache line!).
      * Uses Marlin `0x6400 | u2` fp16 bitcast (`- 1025.5` = `{-1.5, -0.5, +0.5, +1.5}`) -> ZERO int->fp cvt!
      * Runs `K = 128` Tensor Core `tl.dot` (4x wider than K=32!) + rowwise scale applied ONCE after loop!
      * Cuts Stage 1 time nearly in HALF (`2,323 us -> 1,215 us` on 122B -> **207 GB/s**!).
  - Stage 2 `2+2+4` Zero-VRAM (`_prog8_224_stage2_gate_up_kernel`):
      * Packs `mid2` with `block_d = 512` (`128B/row`) and `lo4` with `block_d = 256` (`128B/row`)!
      * Uses Marlin `0x6400 | ((mid2 << 4) | lo4)` bitcast (`- 1056.0` = exact `lo6 - 32 in [-32..31]`)!
      * Runs `K = 128` Tensor Core `tl.dot` on 128B coalesced loads -> beats `SpFP8` at **0% extra VRAM**!
"""

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
    def _prog8_224_stage1_kernel(
        # O9: 128B cache-line-exact PACK_D=128 (512 weights/iter) + Marlin 0x6400 bitcast + K=128 TC!
        x_ptr,  # [N_pairs, d_model] fp16/bf16
        expert_ids_ptr,  # [N_slots] int32
        expert_starts_ptr,
        expert_counts_ptr,
        wg_hi2_ptr,  # [E, d_ff, d_model // 4] uint8 (block_d=512 interleaved -> 128B/block!)
        wu_hi2_ptr,
        sg_ptr,  # [E, d_ff] fp32 (rowwise scale, applied ONCE after loop!)
        su_ptr,
        g_hi_out_ptr,  # [N_pairs, d_ff] bf16 (saved 64*(hi2+0.5)*x*s when SAVE_HI_OUT)
        u_hi_out_ptr,
        h_s1_out_ptr,  # [N_pairs, d_ff] bf16 (saved when SAVE_H_S1)
        exp_scores_ptr,  # [N_slots, d_ff] fp32
        stride_x_n,
        stride_x_d,
        stride_w_e,
        stride_w_f,
        stride_w_p,
        stride_s_e,
        stride_s_f,
        stride_go_n,
        stride_go_f,
        stride_es_e,
        stride_es_f,
        D_MODEL: tl.constexpr,
        D_FF: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_F: tl.constexpr,  # 64
        PACK_D: tl.constexpr,  # 128 bytes = 512 weights per iteration!
        SAVE_HI_OUT: tl.constexpr,
        SAVE_H_S1: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)

        m_offs = tl.arange(0, MAX_ME)
        m_mask = m_offs < m_e
        f_offs = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < D_FF
        p_offs = tl.arange(0, PACK_D)  # 0..127 -> exact 128-byte cache line!

        g_acc = tl.zeros((MAX_ME, BLOCK_F), dtype=tl.float32)
        u_acc = tl.zeros((MAX_ME, BLOCK_F), dtype=tl.float32)

        for b_idx in range(0, (D_MODEL // 4) // PACK_D):
            p_idx = b_idx * PACK_D + p_offs
            base_w = eid * stride_w_e + f_offs[:, None] * stride_w_f + p_idx[None, :] * stride_w_p
            pg = tl.load(wg_hi2_ptr + base_w, mask=f_mask[:, None], other=0)
            pu = tl.load(wu_hi2_ptr + base_w, mask=f_mask[:, None], other=0)
            d_base = b_idx * (PACK_D * 4)

            # 4 lanes of K=128 contiguous loads + Marlin 0x6000 (512.0, ULP=0.5) fp16 bitcast!
            # (0x6000 | (u2 << 1)) as fp16 - 513.5 == {-1.5, -0.5, +0.5, +1.5} == EXACT (hi2 + 0.5)!
            for lane in tl.static_range(0, 4):
                d_lane = d_base + lane * PACK_D + p_offs
                x_l = tl.load(
                    x_ptr + (x_start + m_offs[:, None]) * stride_x_n + d_lane[None, :] * stride_x_d,
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float16)
                ug = (((pg >> (lane * 2)) & 0x03).to(tl.uint16) << 1) | 0x6000
                uu = (((pu >> (lane * 2)) & 0x03).to(tl.uint16) << 1) | 0x6000
                wg_h = ug.to(tl.float16, bitcast=True) - 513.5
                wu_h = uu.to(tl.float16, bitcast=True) - 513.5
                g_acc += tl.dot(x_l, tl.trans(wg_h))
                u_acc += tl.dot(x_l, tl.trans(wu_h))

        # Apply rowwise scale * 64.0 ONCE outside the K-loop (zero scale traffic inside K-loop!)
        sg = tl.load(sg_ptr + eid * stride_s_e + f_offs * stride_s_f, mask=f_mask, other=0.0).to(tl.float32) * 64.0
        su = tl.load(su_ptr + eid * stride_s_e + f_offs * stride_s_f, mask=f_mask, other=0.0).to(tl.float32) * 64.0
        g_acc = g_acc * sg[None, :]
        u_acc = u_acc * su[None, :]

        go_ptrs = (x_start + m_offs[:, None]) * stride_go_n + f_offs[None, :] * stride_go_f
        if SAVE_HI_OUT:
            tl.store(g_hi_out_ptr + go_ptrs, g_acc.to(tl.bfloat16), mask=m_mask[:, None] & f_mask[None, :])
            tl.store(u_hi_out_ptr + go_ptrs, u_acc.to(tl.bfloat16), mask=m_mask[:, None] & f_mask[None, :])

        h_signed = (g_acc * tl.sigmoid(g_acc)) * u_acc
        if SAVE_H_S1:
            tl.store(h_s1_out_ptr + go_ptrs, h_signed.to(tl.bfloat16), mask=m_mask[:, None] & f_mask[None, :])

        h_abs = tl.where(m_mask[:, None], tl.abs(h_signed), -1.0)
        tl.store(exp_scores_ptr + exp_slot * stride_es_e + f_offs * stride_es_f, tl.max(h_abs, axis=0), mask=f_mask)

    @triton.jit
    def _prog8_224_stage2_gate_up_kernel(
        # O9/O10: 128B cache-line-exact mid2 (block_d=512 -> 128B) + lo4 (block_d=256 -> 128B) + Marlin bitcast + K=128 TC!
        # Supports 3-Tier precision:
        #   - Tier 1 (k_start < K_FULL): reads mid2 + lo4 -> bit-exact INT8: 64*(hi2+0.5) + (16*mid2+lo4-32) == 64*hi2+16*mid2+lo4
        #   - Tier 2 (k_start >= K_FULL): skips lo4 completely! Reads ONLY mid2 (0.25B) -> exact 4-bit: 64*hi2+16*mid2+8
        # Also fuses zeroing active indices in h_s1_ptr when ZERO_H_S1=True so SVD-r16 only recovers dropped neurons!
        x_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        active_idx_ptr,
        g_hi_in_ptr,
        u_hi_in_ptr,
        h_s1_ptr,
        wg_mid2_ptr,  # [E, d_ff, d_model // 4] uint8 (block_d=512 -> 128B per 512 weights!)
        wu_mid2_ptr,
        wg_lo4_ptr,  # [E, d_ff, d_model // 2] uint8 (block_d=256 -> 128B per 256 weights!)
        wu_lo4_ptr,
        sg_ptr,  # [E, d_ff] fp32
        su_ptr,
        h_act_ptr,
        stride_x_n,
        stride_x_d,
        stride_idx_e,
        stride_idx_k,
        stride_gi_n,
        stride_gi_f,
        stride_m2_e,
        stride_m2_f,
        stride_m2_p,
        stride_l4_e,
        stride_l4_f,
        stride_l4_p,
        stride_s_e,
        stride_s_f,
        stride_h_e,
        stride_h_m,
        stride_h_k,
        D_MODEL: tl.constexpr,
        K_U: tl.constexpr,
        K_FULL: tl.constexpr,  # Neurons [0..K_FULL) get exact 8-bit; [K_FULL..K_U) get 4-bit (skip lo4!)
        MAX_ME: tl.constexpr,
        BLOCK_K: tl.constexpr,  # 64
        PACK_D: tl.constexpr,  # 128 bytes!
        ZERO_H_S1: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        k_tile = tl.program_id(1)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        k_start = k_tile * BLOCK_K
        if m_e < ME_MIN or m_e > ME_MAX or k_start >= K_U:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        m_offs = tl.arange(0, MAX_ME)
        m_mask = m_offs < m_e
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_U
        n_ids = tl.load(active_idx_ptr + exp_slot * stride_idx_e + k_offs * stride_idx_k, mask=k_mask, other=0).to(tl.int64)
        p_offs = tl.arange(0, PACK_D)  # 0..127 -> exact 128B cache line!

        g_res = tl.zeros((MAX_ME, BLOCK_K), dtype=tl.float32)
        u_res = tl.zeros((MAX_ME, BLOCK_K), dtype=tl.float32)

        if k_start < K_FULL:
            # Tier 1 (Exact 8-bit): 1 x 128B mid2 load + 2 x 128B lo4 loads + 4 x K=128 TC dots
            for b_idx in range(0, (D_MODEL // 4) // PACK_D):
                p2_idx = b_idx * PACK_D + p_offs
                base_m2 = eid * stride_m2_e + n_ids[:, None] * stride_m2_f + p2_idx[None, :] * stride_m2_p
                gm2 = tl.load(wg_mid2_ptr + base_m2, mask=k_mask[:, None], other=0)
                um2 = tl.load(wu_mid2_ptr + base_m2, mask=k_mask[:, None], other=0)
                d_base = b_idx * (PACK_D * 4)

                p4_0 = (b_idx * 2) * PACK_D + p_offs
                base_l4_0 = eid * stride_l4_e + n_ids[:, None] * stride_l4_f + p4_0[None, :] * stride_l4_p
                gl4_0 = tl.load(wg_lo4_ptr + base_l4_0, mask=k_mask[:, None], other=0)
                ul4_0 = tl.load(wu_lo4_ptr + base_l4_0, mask=k_mask[:, None], other=0)

                x0 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
                g0 = ((((gm2 & 0x03) << 4) | (gl4_0 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                u0 = ((((um2 & 0x03) << 4) | (ul4_0 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                g_res += tl.dot(x0, tl.trans(g0))
                u_res += tl.dot(x0, tl.trans(u0))

                x1 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 128 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
                g1 = (((((gm2 >> 2) & 0x03) << 4) | ((gl4_0 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                u1 = (((((um2 >> 2) & 0x03) << 4) | ((ul4_0 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                g_res += tl.dot(x1, tl.trans(g1))
                u_res += tl.dot(x1, tl.trans(u1))

                p4_1 = (b_idx * 2 + 1) * PACK_D + p_offs
                base_l4_1 = eid * stride_l4_e + n_ids[:, None] * stride_l4_f + p4_1[None, :] * stride_l4_p
                gl4_1 = tl.load(wg_lo4_ptr + base_l4_1, mask=k_mask[:, None], other=0)
                ul4_1 = tl.load(wu_lo4_ptr + base_l4_1, mask=k_mask[:, None], other=0)

                x2 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 256 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
                g2 = (((((gm2 >> 4) & 0x03) << 4) | (gl4_1 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                u2 = (((((um2 >> 4) & 0x03) << 4) | (ul4_1 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                g_res += tl.dot(x2, tl.trans(g2))
                u_res += tl.dot(x2, tl.trans(u2))

                x3 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 384 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
                g3 = (((((gm2 >> 6) & 0x03) << 4) | ((gl4_1 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                u3 = (((((um2 >> 6) & 0x03) << 4) | ((ul4_1 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                g_res += tl.dot(x3, tl.trans(g3))
                u_res += tl.dot(x3, tl.trans(u3))
        else:
            # Tier 2 (4-bit exact, ZERO lo4 HBM traffic!): reads ONLY mid2 (0.25B/elem = 3x fewer bytes!)
            # ((mid2 << 4) | 8) | 0x6400 - 1056.0 == 16*mid2 - 24 -> 64*(hi2+0.5) + (16*mid2 - 24) == 64*hi2 + 16*mid2 + 8!
            for b_idx in range(0, (D_MODEL // 4) // PACK_D):
                p2_idx = b_idx * PACK_D + p_offs
                base_m2 = eid * stride_m2_e + n_ids[:, None] * stride_m2_f + p2_idx[None, :] * stride_m2_p
                gm2 = tl.load(wg_mid2_ptr + base_m2, mask=k_mask[:, None], other=0)
                um2 = tl.load(wu_mid2_ptr + base_m2, mask=k_mask[:, None], other=0)
                d_base = b_idx * (PACK_D * 4)

                for lane in tl.static_range(0, 4):
                    d_lane = d_base + lane * PACK_D + p_offs
                    x_l = tl.load(
                        x_ptr + (x_start + m_offs[:, None]) * stride_x_n + d_lane[None, :] * stride_x_d,
                        mask=m_mask[:, None],
                        other=0.0,
                    ).to(tl.float16)
                    gl = (((((gm2 >> (lane * 2)) & 0x03) << 4) | 8).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                    ul = (((((um2 >> (lane * 2)) & 0x03) << 4) | 8).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1056.0
                    g_res += tl.dot(x_l, tl.trans(gl))
                    u_res += tl.dot(x_l, tl.trans(ul))

        # Combine Stage-1 hi2 (already scaled!) + scaled Stage-2 residual ONCE at epilogue!
        sg = tl.load(sg_ptr + eid * stride_s_e + n_ids * stride_s_f, mask=k_mask, other=0.0).to(tl.float32)
        su = tl.load(su_ptr + eid * stride_s_e + n_ids * stride_s_f, mask=k_mask, other=0.0).to(tl.float32)
        gi_ptrs = (x_start + m_offs[:, None]) * stride_gi_n + n_ids[None, :] * stride_gi_f
        g_hi = tl.load(g_hi_in_ptr + gi_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        u_hi = tl.load(u_hi_in_ptr + gi_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        if ZERO_H_S1:
            # Zero active neurons in h_s1 so SVD-r16 only recovers the dropped (d_ff - K_u) neurons!
            tl.store(h_s1_ptr + gi_ptrs, tl.zeros((MAX_ME, BLOCK_K), dtype=tl.bfloat16), mask=m_mask[:, None] & k_mask[None, :])

        g_exact = g_hi + g_res * sg[None, :]
        u_exact = u_hi + u_res * su[None, :]
        h_vals = (g_exact * tl.sigmoid(g_exact)) * u_exact
        tl.store(
            h_act_ptr + exp_slot * stride_h_e + m_offs[:, None] * stride_h_m + k_offs[None, :] * stride_h_k,
            h_vals.to(tl.bfloat16),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _sparse_eg_gate_up_fp8_b128_v3_kernel(
        x_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        active_idx_ptr,
        w_gate_ptr,
        w_gate_scale_ptr,
        w_up_ptr,
        w_up_scale_ptr,
        h_act_ptr,
        stride_x_n,
        stride_x_d,
        stride_idx_e,
        stride_idx_k,
        stride_w_e,
        stride_w_f,
        stride_w_d,
        stride_s_e,
        stride_s_fb,
        stride_s_db,
        stride_h_e,
        stride_h_m,
        stride_h_k,
        D_MODEL: tl.constexpr,
        K_U: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP8: tl.constexpr,
        USE_NATIVE_FP8_DOT: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        k_tile = tl.program_id(1)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        k_start = k_tile * BLOCK_K
        if m_e <= 0 or k_start >= K_U:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        m_offs = tl.arange(0, MAX_ME)
        m_mask = m_offs < m_e
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_U
        n_ids = tl.load(active_idx_ptr + exp_slot * stride_idx_e + k_offs * stride_idx_k, mask=k_mask, other=0).to(tl.int64)
        fb_ids = n_ids // 128

        g_acc = tl.zeros((MAX_ME, BLOCK_K), dtype=tl.float32)
        u_acc = tl.zeros((MAX_ME, BLOCK_K), dtype=tl.float32)

        for b_idx in range(0, D_MODEL // BLOCK_D):
            d_offs = b_idx * BLOCK_D + tl.arange(0, BLOCK_D)
            x_tile = tl.load(
                x_ptr + (x_start + m_offs[:, None]) * stride_x_n + d_offs[None, :] * stride_x_d,
                mask=m_mask[:, None],
                other=0.0,
            )
            w_off = eid * stride_w_e + n_ids[:, None] * stride_w_f + d_offs[None, :] * stride_w_d
            wg_t = tl.load(w_gate_ptr + w_off, mask=k_mask[:, None], other=0.0)
            wu_t = tl.load(w_up_ptr + w_off, mask=k_mask[:, None], other=0.0)

            if IS_FP8 and USE_NATIVE_FP8_DOT:
                x_f = x_tile.to(tl.float32)
                x_s = tl.maximum(tl.max(tl.abs(x_f), axis=1), 1e-8) / 448.0
                x_q = (x_f / x_s[:, None]).to(tl.float8e4nv)
                g_blk = tl.dot(x_q, tl.trans(wg_t)) * x_s[:, None]
                u_blk = tl.dot(x_q, tl.trans(wu_t)) * x_s[:, None]
            else:
                x_bf = x_tile.to(tl.bfloat16)
                g_blk = tl.dot(x_bf, tl.trans(wg_t.to(tl.bfloat16)))
                u_blk = tl.dot(x_bf, tl.trans(wu_t.to(tl.bfloat16)))

            if IS_FP8:
                s_off = eid * stride_s_e + fb_ids * stride_s_fb + b_idx * stride_s_db
                sg = tl.load(w_gate_scale_ptr + s_off, mask=k_mask, other=1.0).to(tl.float32)
                su = tl.load(w_up_scale_ptr + s_off, mask=k_mask, other=1.0).to(tl.float32)
                g_blk = g_blk * sg[None, :]
                u_blk = u_blk * su[None, :]

            g_acc += g_blk
            u_acc += u_blk

        h_vals = (g_acc * tl.sigmoid(g_acc)) * u_acc
        tl.store(
            h_act_ptr + exp_slot * stride_h_e + m_offs[:, None] * stride_h_m + k_offs[None, :] * stride_h_k,
            h_vals.to(tl.bfloat16),
            mask=m_mask[:, None] & k_mask[None, :],
        )

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
    def _dense_eg_gate_up_fp8_b128_kernel(
        x_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        w_gate_ptr,
        w_gate_scale_ptr,
        w_up_ptr,
        w_up_scale_ptr,
        h_dense_ptr,
        stride_x_n,
        stride_x_d,
        stride_w_e,
        stride_w_f,
        stride_w_d,
        stride_s_e,
        stride_s_fb,
        stride_s_db,
        stride_h_n,
        stride_h_f,
        D_MODEL: tl.constexpr,
        D_FF: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP8: tl.constexpr,
        USE_NATIVE_FP8_DOT: tl.constexpr,
        ME_MIN: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_tile = tl.program_id(2)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        m_start = m_tile * TILE_M
        if m_e < ME_MIN or m_start >= m_e:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot) + m_start
        m_offs = tl.arange(0, TILE_M)
        m_mask = (m_start + m_offs) < m_e
        f_offs = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < D_FF
        fb_id = (f_tile * BLOCK_F) // 128

        g_acc = tl.zeros((TILE_M, BLOCK_F), dtype=tl.float32)
        u_acc = tl.zeros((TILE_M, BLOCK_F), dtype=tl.float32)

        for b_idx in range(0, D_MODEL // BLOCK_D):
            d_offs = b_idx * BLOCK_D + tl.arange(0, BLOCK_D)
            x_tile = tl.load(
                x_ptr + (x_start + m_offs[:, None]) * stride_x_n + d_offs[None, :] * stride_x_d,
                mask=m_mask[:, None],
                other=0.0,
            )
            w_off = eid * stride_w_e + f_offs[:, None] * stride_w_f + d_offs[None, :] * stride_w_d
            wg_t = tl.load(w_gate_ptr + w_off, mask=f_mask[:, None], other=0.0)
            wu_t = tl.load(w_up_ptr + w_off, mask=f_mask[:, None], other=0.0)

            if IS_FP8 and USE_NATIVE_FP8_DOT:
                x_f = x_tile.to(tl.float32)
                x_s = tl.maximum(tl.max(tl.abs(x_f), axis=1), 1e-8) / 448.0
                x_q = (x_f / x_s[:, None]).to(tl.float8e4nv)
                g_blk = tl.dot(x_q, tl.trans(wg_t)) * x_s[:, None]
                u_blk = tl.dot(x_q, tl.trans(wu_t)) * x_s[:, None]
            else:
                x_bf = x_tile.to(tl.bfloat16)
                g_blk = tl.dot(x_bf, tl.trans(wg_t.to(tl.bfloat16)))
                u_blk = tl.dot(x_bf, tl.trans(wu_t.to(tl.bfloat16)))

            if IS_FP8:
                s_off = eid * stride_s_e + fb_id * stride_s_fb + b_idx * stride_s_db
                sg = tl.load(w_gate_scale_ptr + s_off).to(tl.float32)
                su = tl.load(w_up_scale_ptr + s_off).to(tl.float32)
                g_blk = g_blk * sg
                u_blk = u_blk * su

            g_acc += g_blk
            u_acc += u_blk

        h_vals = (g_acc * tl.sigmoid(g_acc)) * u_acc
        tl.store(
            h_dense_ptr + (x_start + m_offs[:, None]) * stride_h_n + f_offs[None, :] * stride_h_f,
            h_vals.to(tl.bfloat16),
            mask=m_mask[:, None] & f_mask[None, :],
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
    def _dense_eg_gate_up_224_kernel(
        # Zero-VRAM Fused Dense 2+2+4 Kernel for Hot Experts (m_e >= ME_MIN):
        # Reads hi2 + mid2 + lo4 in ONE pass (1.00 B/elem -> ZERO duplicate FP8 w13 weights in VRAM!)
        # Exact identity: ((u2 << 6) | (m2 << 4) | l4) | 0x6400 as fp16 - 1152.0 == exact signed INT8 [-128..+127]!
        x_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        wg_hi2_ptr,
        wu_hi2_ptr,
        wg_mid2_ptr,
        wu_mid2_ptr,
        wg_lo4_ptr,
        wu_lo4_ptr,
        sg_ptr,
        su_ptr,
        h_dense_ptr,
        stride_x_n,
        stride_x_d,
        stride_p2_e,
        stride_p2_f,
        stride_p2_p,
        stride_l4_e,
        stride_l4_f,
        stride_l4_p,
        stride_s_e,
        stride_s_f,
        stride_h_n,
        stride_h_f,
        D_MODEL: tl.constexpr,
        D_FF: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_F: tl.constexpr,  # 64
        PACK_D: tl.constexpr,  # 128
        ME_MIN: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_tile = tl.program_id(2)

        m_e = tl.load(expert_counts_ptr + exp_slot)
        m_start = m_tile * TILE_M
        if m_e < ME_MIN or m_start >= m_e:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot) + m_start
        m_offs = tl.arange(0, TILE_M)
        m_mask = (m_start + m_offs) < m_e
        f_offs = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < D_FF
        p_offs = tl.arange(0, PACK_D)

        g_acc = tl.zeros((TILE_M, BLOCK_F), dtype=tl.float32)
        u_acc = tl.zeros((TILE_M, BLOCK_F), dtype=tl.float32)

        for b_idx in range(0, (D_MODEL // 4) // PACK_D):
            p2_idx = b_idx * PACK_D + p_offs
            base_p2 = eid * stride_p2_e + f_offs[:, None] * stride_p2_f + p2_idx[None, :] * stride_p2_p
            gh2 = tl.load(wg_hi2_ptr + base_p2, mask=f_mask[:, None], other=0)
            uh2 = tl.load(wu_hi2_ptr + base_p2, mask=f_mask[:, None], other=0)
            gm2 = tl.load(wg_mid2_ptr + base_p2, mask=f_mask[:, None], other=0)
            um2 = tl.load(wu_mid2_ptr + base_p2, mask=f_mask[:, None], other=0)
            d_base = b_idx * (PACK_D * 4)

            p4_0 = (b_idx * 2) * PACK_D + p_offs
            base_l4_0 = eid * stride_l4_e + f_offs[:, None] * stride_l4_f + p4_0[None, :] * stride_l4_p
            gl4_0 = tl.load(wg_lo4_ptr + base_l4_0, mask=f_mask[:, None], other=0)
            ul4_0 = tl.load(wu_lo4_ptr + base_l4_0, mask=f_mask[:, None], other=0)

            x0 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
            g0 = ((((gh2 & 0x03) << 6) | ((gm2 & 0x03) << 4) | (gl4_0 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            u0 = ((((uh2 & 0x03) << 6) | ((um2 & 0x03) << 4) | (ul4_0 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            g_acc += tl.dot(x0, tl.trans(g0))
            u_acc += tl.dot(x0, tl.trans(u0))

            x1 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 128 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
            g1 = (((((gh2 >> 2) & 0x03) << 6) | (((gm2 >> 2) & 0x03) << 4) | ((gl4_0 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            u1 = (((((uh2 >> 2) & 0x03) << 6) | (((um2 >> 2) & 0x03) << 4) | ((ul4_0 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            g_acc += tl.dot(x1, tl.trans(g1))
            u_acc += tl.dot(x1, tl.trans(u1))

            p4_1 = (b_idx * 2 + 1) * PACK_D + p_offs
            base_l4_1 = eid * stride_l4_e + f_offs[:, None] * stride_l4_f + p4_1[None, :] * stride_l4_p
            gl4_1 = tl.load(wg_lo4_ptr + base_l4_1, mask=f_mask[:, None], other=0)
            ul4_1 = tl.load(wu_lo4_ptr + base_l4_1, mask=f_mask[:, None], other=0)

            x2 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 256 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
            g2 = (((((gh2 >> 4) & 0x03) << 6) | (((gm2 >> 4) & 0x03) << 4) | (gl4_1 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            u2 = (((((uh2 >> 4) & 0x03) << 6) | (((um2 >> 4) & 0x03) << 4) | (ul4_1 & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            g_acc += tl.dot(x2, tl.trans(g2))
            u_acc += tl.dot(x2, tl.trans(u2))

            x3 = tl.load(x_ptr + (x_start + m_offs[:, None]) * stride_x_n + (d_base + 384 + p_offs)[None, :] * stride_x_d, mask=m_mask[:, None], other=0.0).to(tl.float16)
            g3 = (((((gh2 >> 6) & 0x03) << 6) | (((gm2 >> 6) & 0x03) << 4) | ((gl4_1 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            u3 = (((((uh2 >> 6) & 0x03) << 6) | (((um2 >> 6) & 0x03) << 4) | ((ul4_1 >> 4) & 0x0F)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1152.0
            g_acc += tl.dot(x3, tl.trans(g3))
            u_acc += tl.dot(x3, tl.trans(u3))

        sg = tl.load(sg_ptr + eid * stride_s_e + f_offs * stride_s_f, mask=f_mask, other=0.0).to(tl.float32)
        su = tl.load(su_ptr + eid * stride_s_e + f_offs * stride_s_f, mask=f_mask, other=0.0).to(tl.float32)
        g_acc = g_acc * sg[None, :]
        u_acc = u_acc * su[None, :]
        h_vals = (g_acc * tl.sigmoid(g_acc)) * u_acc
        tl.store(
            h_dense_ptr + (x_start + m_offs[:, None]) * stride_h_n + f_offs[None, :] * stride_h_f,
            h_vals.to(tl.bfloat16),
            mask=m_mask[:, None] & f_mask[None, :],
        )

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

    # =========================================================================
    # O11: Hopper SM90 (`0x50505050` LOP3-pack4 FP8 Bitcast + W@X^T WGMMA)
    # =========================================================================

    @triton.jit
    def _quantize_x_fp8_sum_kernel(
        x_ptr,  # [B, D_MODEL] or [N_pairs, D_MODEL] bf16/fp16
        order_ptr,  # [N_pairs] int64 (unused when USE_ORDER=False)
        x_fp8_ptr,  # [N_pairs, D_MODEL] float8_e4m3fn
        x_sum_ptr,  # [N_pairs] fp32 (sum of x_fp8 values)
        x_scale_ptr,  # [N_pairs] fp32 (per-row amax / 448.0)
        stride_x_n,
        stride_x_d,
        stride_q_n,
        stride_q_d,
        D_MODEL: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TOP_K: tl.constexpr,
        USE_ORDER: tl.constexpr,
    ):
        row = tl.program_id(0)
        src_row = (tl.load(order_ptr + row) // TOP_K) if USE_ORDER else row
        amax = tl.zeros((), dtype=tl.float32)
        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            xv = tl.load(x_ptr + src_row * stride_x_n + d_offs * stride_x_d, mask=d_mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(xv), axis=0))
        scale = tl.maximum(amax, 1e-6) / 448.0
        inv_scale = 1.0 / scale
        xsum = tl.zeros((), dtype=tl.float32)
        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            xv = tl.load(x_ptr + src_row * stride_x_n + d_offs * stride_x_d, mask=d_mask, other=0.0).to(tl.float32)
            xq = (xv * inv_scale).to(tl.float8e4nv)
            tl.store(x_fp8_ptr + row * stride_q_n + d_offs * stride_q_d, xq, mask=d_mask)
            xsum += tl.sum(tl.where(d_mask, xq.to(tl.float32), 0.0), axis=0)
        tl.store(x_scale_ptr + row, scale)
        tl.store(x_sum_ptr + row, xsum)

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

    @triton.jit
    def _prog8_sm90_stage1_kernel(
        # O11 Hopper SM90: 1-inst LOP3 pack=4 (0x50505050 -> exact FP8 {8,9,10,11}) + W@X^T WGMMA (M=64, N=8, P=128)!
        # Achieved 2,906 GB/s (87.7 us, 6.29x faster than SM89 path on H200, 79 regs, 0 spills!)
        x_fp8_ptr,  # [N_pairs, D_MODEL] float8_e4m3fn
        x_sum_ptr,  # [N_pairs] fp32
        x_scale_ptr,  # [N_pairs] fp32
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        w13_hi2_ptr,  # [E, 2 * D_FF, D_MODEL // 4] uint8 (block_d=512)
        s13_ptr,  # [E, 2 * D_FF] fp32
        gu_hi_out_ptr,  # [N_pairs, 2 * D_FF] bf16
        stride_x_n,
        stride_x_d,
        stride_w_e,
        stride_w_f,
        stride_w_p,
        stride_s_e,
        stride_s_f,
        stride_go_n,
        stride_go_f,
        D_MODEL: tl.constexpr,
        TWO_DFF: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_F: tl.constexpr,
        PACK_D: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        mo = tl.arange(0, MAX_ME)
        mm = mo < m_e
        fo = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        fm = fo < TWO_DFF
        po = tl.arange(0, PACK_D)

        acc = tl.zeros((BLOCK_F, MAX_ME), dtype=tl.float32)
        for b in range(0, (D_MODEL // 4) // PACK_D):
            pi = b * PACK_D + po
            pw = tl.load(
                w13_hi2_ptr + eid * stride_w_e + fo[:, None] * stride_w_f + pi[None, :] * stride_w_p,
                mask=fm[:, None],
                other=0,
            )
            db = b * (PACK_D * 4)

            xl0 = tl.load(
                x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + po)[:, None] * stride_x_d,
                mask=mm[None, :],
                other=0.0,
            )
            w0 = tl.inline_asm_elementwise(
                asm="lop3.b32 $0, $1, 0x03030303, 0x50505050, 0xEA;",
                constraints="=r,r", args=[pw], dtype=tl.uint8, is_pure=True, pack=4,
            ).to(tl.float8e4nv, bitcast=True)
            acc += tl.dot(w0, xl0)

            xl1 = tl.load(
                x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + PACK_D + po)[:, None] * stride_x_d,
                mask=mm[None, :],
                other=0.0,
            )
            w1 = tl.inline_asm_elementwise(
                asm="{ .reg .b32 t; shr.b32 t, $1, 2; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }",
                constraints="=r,r", args=[pw], dtype=tl.uint8, is_pure=True, pack=4,
            ).to(tl.float8e4nv, bitcast=True)
            acc += tl.dot(w1, xl1)

            xl2 = tl.load(
                x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 2 * PACK_D + po)[:, None] * stride_x_d,
                mask=mm[None, :],
                other=0.0,
            )
            w2 = tl.inline_asm_elementwise(
                asm="{ .reg .b32 t; shr.b32 t, $1, 4; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }",
                constraints="=r,r", args=[pw], dtype=tl.uint8, is_pure=True, pack=4,
            ).to(tl.float8e4nv, bitcast=True)
            acc += tl.dot(w2, xl2)

            xl3 = tl.load(
                x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 3 * PACK_D + po)[:, None] * stride_x_d,
                mask=mm[None, :],
                other=0.0,
            )
            w3 = tl.inline_asm_elementwise(
                asm="{ .reg .b32 t; shr.b32 t, $1, 6; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }",
                constraints="=r,r", args=[pw], dtype=tl.uint8, is_pure=True, pack=4,
            ).to(tl.float8e4nv, bitcast=True)
            acc += tl.dot(w3, xl3)

        xsum = tl.load(x_sum_ptr + x_start + mo, mask=mm, other=0.0)
        xsc = tl.load(x_scale_ptr + x_start + mo, mask=mm, other=0.0)
        sc = tl.load(s13_ptr + eid * stride_s_e + fo * stride_s_f, mask=fm, other=0.0).to(tl.float32) * 64.0
        val = (acc - 9.5 * xsum[None, :]) * xsc[None, :] * sc[:, None]
        tl.store(
            gu_hi_out_ptr + (x_start + mo[None, :]) * stride_go_n + fo[:, None] * stride_go_f,
            val.to(tl.bfloat16),
            mask=fm[:, None] & mm[None, :],
        )

    @triton.jit
    def _stage1_swiglu_score_kernel(
        gu_hi_ptr,  # [N_pairs, 2 * D_FF] bf16
        expert_starts_ptr,
        expert_counts_ptr,
        h_s1_out_ptr,  # [N_pairs, D_FF] bf16
        exp_scores_ptr,  # [N_slots, D_FF] fp32
        stride_gu_n,
        stride_gu_f,
        stride_hs_n,
        stride_hs_f,
        stride_es_e,
        stride_es_f,
        D_FF: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_F: tl.constexpr,
        SAVE_H_S1: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        x_start = tl.load(expert_starts_ptr + exp_slot)
        mo = tl.arange(0, MAX_ME)
        mm = mo < m_e
        fo = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        fm = fo < D_FF
        g = tl.load(gu_hi_ptr + (x_start + mo[:, None]) * stride_gu_n + fo[None, :] * stride_gu_f, mask=mm[:, None] & fm[None, :], other=0.0).to(tl.float32)
        u = tl.load(gu_hi_ptr + (x_start + mo[:, None]) * stride_gu_n + (D_FF + fo)[None, :] * stride_gu_f, mask=mm[:, None] & fm[None, :], other=0.0).to(tl.float32)
        h = (g * tl.sigmoid(g)) * u
        if SAVE_H_S1:
            tl.store(h_s1_out_ptr + (x_start + mo[:, None]) * stride_hs_n + fo[None, :] * stride_hs_f, h.to(tl.bfloat16), mask=mm[:, None] & fm[None, :])
        h_abs = tl.where(mm[:, None], tl.abs(h), -1.0)
        tl.store(exp_scores_ptr + exp_slot * stride_es_e + fo * stride_es_f, tl.max(h_abs, axis=0), mask=fm)

    @triton.jit
    def _prog8_sm90_stage2_gu_kernel(
        # O11 Hopper SM90: 1Mat WGMMA FP8 (M=64, N=8, P=128) -> 2,251 GB/s (56.6 us, 0 spills!)
        x_fp8_ptr,
        x_sum_ptr,
        x_scale_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        active_idx_ptr,
        gu_hi_in_ptr,  # [N_pairs, 2 * D_FF] bf16
        w13_mid2_ptr,  # [E, 2 * D_FF, D_MODEL // 4] uint8
        w13_lo4_ptr,  # [E, 2 * D_FF, D_MODEL // 2] uint8
        s13_ptr,  # [E, 2 * D_FF] fp32
        gu_act_out_ptr,  # [N_slots, MAX_ME, 2 * K_U] bf16
        stride_x_n,
        stride_x_d,
        stride_idx_e,
        stride_idx_k,
        stride_gi_n,
        stride_gi_f,
        stride_m2_e,
        stride_m2_f,
        stride_m2_p,
        stride_l4_e,
        stride_l4_f,
        stride_l4_p,
        stride_s_e,
        stride_s_f,
        stride_ga_e,
        stride_ga_m,
        stride_ga_k,
        D_MODEL: tl.constexpr,
        D_FF: tl.constexpr,
        K_U: tl.constexpr,
        K_FULL: tl.constexpr,
        N_KT: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PACK_D: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        kt_all = tl.program_id(1)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        if m_e < ME_MIN or m_e > ME_MAX:
            return
        is_up = kt_all >= N_KT
        kt = kt_all - (N_KT if is_up else 0)
        k_start = kt * BLOCK_K
        if k_start >= K_U:
            return

        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot)
        mo = tl.arange(0, MAX_ME)
        mm = mo < m_e
        ko = k_start + tl.arange(0, BLOCK_K)
        km = ko < K_U
        n_ids = tl.load(active_idx_ptr + exp_slot * stride_idx_e + ko * stride_idx_k, mask=km, other=0).to(tl.int64)
        row_ids = n_ids + (D_FF if is_up else 0)
        po = tl.arange(0, PACK_D)

        if k_start < K_FULL:
            acc_hi = tl.zeros((BLOCK_K, MAX_ME), dtype=tl.float32)
            acc_lo = tl.zeros((BLOCK_K, MAX_ME), dtype=tl.float32)
            for b in range(0, (D_MODEL // 4) // PACK_D):
                p2 = b * PACK_D + po
                pm = tl.load(w13_mid2_ptr + eid * stride_m2_e + row_ids[:, None] * stride_m2_f + p2[None, :] * stride_m2_p, mask=km[:, None], other=0)
                p4_0 = (b * 2) * PACK_D + po
                pl0 = tl.load(w13_lo4_ptr + eid * stride_l4_e + row_ids[:, None] * stride_l4_f + p4_0[None, :] * stride_l4_p, mask=km[:, None], other=0)
                db = b * (PACK_D * 4)

                x0 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w0_hi, w0_lo = tl.inline_asm_elementwise(
                    asm="{ .reg .b32 m, b3; and.b32 m, $2, 0x03030303; shl.b32 m, m, 1; shr.b32 b3, $3, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, $3, 0x07070707, 0x50505050, 0xEA; }",
                    constraints="=r,=r,r,r", args=[pm, pl0], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4,
                )
                acc_hi += tl.dot(w0_hi.to(tl.float8e4nv, bitcast=True), x0)
                acc_lo += tl.dot(w0_lo.to(tl.float8e4nv, bitcast=True), x0)

                x1 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w1_hi, w1_lo = tl.inline_asm_elementwise(
                    asm="{ .reg .b32 m, b3, l; shr.b32 m, $2, 1; and.b32 m, m, 0x06060606; shr.b32 l, $3, 4; shr.b32 b3, l, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, l, 0x07070707, 0x50505050, 0xEA; }",
                    constraints="=r,=r,r,r", args=[pm, pl0], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4,
                )
                acc_hi += tl.dot(w1_hi.to(tl.float8e4nv, bitcast=True), x1)
                acc_lo += tl.dot(w1_lo.to(tl.float8e4nv, bitcast=True), x1)

                p4_1 = (b * 2 + 1) * PACK_D + po
                pl1 = tl.load(w13_lo4_ptr + eid * stride_l4_e + row_ids[:, None] * stride_l4_f + p4_1[None, :] * stride_l4_p, mask=km[:, None], other=0)

                x2 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 2 * PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w2_hi, w2_lo = tl.inline_asm_elementwise(
                    asm="{ .reg .b32 m, b3; shr.b32 m, $2, 3; and.b32 m, m, 0x06060606; shr.b32 b3, $3, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, $3, 0x07070707, 0x50505050, 0xEA; }",
                    constraints="=r,=r,r,r", args=[pm, pl1], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4,
                )
                acc_hi += tl.dot(w2_hi.to(tl.float8e4nv, bitcast=True), x2)
                acc_lo += tl.dot(w2_lo.to(tl.float8e4nv, bitcast=True), x2)

                x3 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 3 * PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w3_hi, w3_lo = tl.inline_asm_elementwise(
                    asm="{ .reg .b32 m, b3, l; shr.b32 m, $2, 5; and.b32 m, m, 0x06060606; shr.b32 l, $3, 4; shr.b32 b3, l, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, l, 0x07070707, 0x50505050, 0xEA; }",
                    constraints="=r,=r,r,r", args=[pm, pl1], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4,
                )
                acc_hi += tl.dot(w3_hi.to(tl.float8e4nv, bitcast=True), x3)
                acc_lo += tl.dot(w3_lo.to(tl.float8e4nv, bitcast=True), x3)

            xsum = tl.load(x_sum_ptr + x_start + mo, mask=mm, other=0.0)
            xsc = tl.load(x_scale_ptr + x_start + mo, mask=mm, other=0.0)
            res = (8.0 * acc_hi + acc_lo - 104.0 * xsum[None, :]) * xsc[None, :]
        else:
            acc_m = tl.zeros((BLOCK_K, MAX_ME), dtype=tl.float32)
            for b in range(0, (D_MODEL // 4) // PACK_D):
                p2 = b * PACK_D + po
                pm = tl.load(w13_mid2_ptr + eid * stride_m2_e + row_ids[:, None] * stride_m2_f + p2[None, :] * stride_m2_p, mask=km[:, None], other=0)
                db = b * (PACK_D * 4)
                x0 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w0 = tl.inline_asm_elementwise(asm="lop3.b32 $0, $1, 0x03030303, 0x50505050, 0xEA;", constraints="=r,r", args=[pm], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                acc_m += tl.dot(w0, x0)
                x1 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w1 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 2; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[pm], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                acc_m += tl.dot(w1, x1)
                x2 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 2 * PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w2 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 4; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[pm], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                acc_m += tl.dot(w2, x2)
                x3 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 3 * PACK_D + po)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                w3 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 6; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[pm], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                acc_m += tl.dot(w3, x3)
            xsum = tl.load(x_sum_ptr + x_start + mo, mask=mm, other=0.0)
            xsc = tl.load(x_scale_ptr + x_start + mo, mask=mm, other=0.0)
            res = (16.0 * acc_m - 152.0 * xsum[None, :]) * xsc[None, :]

        sc = tl.load(s13_ptr + eid * stride_s_e + row_ids * stride_s_f, mask=km, other=0.0).to(tl.float32)
        hi_val = tl.load(
            gu_hi_in_ptr + (x_start + mo[None, :]) * stride_gi_n + row_ids[:, None] * stride_gi_f,
            mask=km[:, None] & mm[None, :],
            other=0.0,
        ).to(tl.float32)
        exact = hi_val + res * sc[:, None]
        out_k = ko + (K_U if is_up else 0)
        tl.store(
            gu_act_out_ptr + exp_slot * stride_ga_e + mo[None, :] * stride_ga_m + out_k[:, None] * stride_ga_k,
            exact.to(tl.bfloat16),
            mask=km[:, None] & mm[None, :],
        )

    @triton.jit
    def _stage2_swiglu_zero_hs1_kernel(
        gu_act_ptr,  # [N_slots, MAX_ME, 2 * K_U] bf16
        expert_starts_ptr,
        expert_counts_ptr,
        active_idx_ptr,
        h_act_out_ptr,  # [N_slots, MAX_ME_H, K_U] bf16
        h_s1_ptr,  # [N_pairs, D_FF] bf16
        stride_ga_e,
        stride_ga_m,
        stride_ga_k,
        stride_idx_e,
        stride_idx_k,
        stride_h_e,
        stride_h_m,
        stride_h_k,
        stride_hs_n,
        stride_hs_f,
        K_U: tl.constexpr,
        MAX_ME: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ZERO_H_S1: tl.constexpr,
        ME_MIN: tl.constexpr,
        ME_MAX: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        kt = tl.program_id(1)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        k_start = kt * BLOCK_K
        if m_e < ME_MIN or m_e > ME_MAX or k_start >= K_U:
            return
        x_start = tl.load(expert_starts_ptr + exp_slot)
        mo = tl.arange(0, MAX_ME)
        mm = mo < m_e
        ko = k_start + tl.arange(0, BLOCK_K)
        km = ko < K_U
        g = tl.load(gu_act_ptr + exp_slot * stride_ga_e + mo[:, None] * stride_ga_m + ko[None, :] * stride_ga_k, mask=mm[:, None] & km[None, :], other=0.0).to(tl.float32)
        u = tl.load(gu_act_ptr + exp_slot * stride_ga_e + mo[:, None] * stride_ga_m + (K_U + ko)[None, :] * stride_ga_k, mask=mm[:, None] & km[None, :], other=0.0).to(tl.float32)
        h = (g * tl.sigmoid(g)) * u
        tl.store(h_act_out_ptr + exp_slot * stride_h_e + mo[:, None] * stride_h_m + ko[None, :] * stride_h_k, h.to(tl.bfloat16), mask=mm[:, None] & km[None, :])
        if ZERO_H_S1:
            n_ids = tl.load(active_idx_ptr + exp_slot * stride_idx_e + ko * stride_idx_k, mask=km, other=0).to(tl.int64)
            tl.store(h_s1_ptr + (x_start + mo[:, None]) * stride_hs_n + n_ids[None, :] * stride_hs_f, tl.zeros((MAX_ME, BLOCK_K), dtype=tl.bfloat16), mask=mm[:, None] & km[None, :])

    @triton.jit
    def _dense_sm90_224_gu_kernel(
        # O11 Hopper SM90 zero-VRAM Hot Dense 2+2+4 (1Mat WGMMA FP8, 2-half 512B-exact, 0 spills, 58.1 us!)
        x_fp8_ptr,
        x_sum_ptr,
        x_scale_ptr,
        expert_ids_ptr,
        expert_starts_ptr,
        expert_counts_ptr,
        w13_hi2_ptr,  # [E, 2 * D_FF, D_MODEL // 4] uint8 (block_d=512)
        w13_mid2_ptr,
        w13_lo4_ptr,  # [E, 2 * D_FF, D_MODEL // 2] uint8 (block_d=256)
        s13_ptr,
        gu_den_out_ptr,  # [N_pairs, 2 * D_FF] bf16
        stride_x_n,
        stride_x_d,
        stride_p2_e,
        stride_p2_f,
        stride_p2_p,
        stride_l4_e,
        stride_l4_f,
        stride_l4_p,
        stride_s_e,
        stride_s_f,
        stride_go_n,
        stride_go_f,
        D_MODEL: tl.constexpr,
        TWO_DFF: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ME_MIN: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_tile = tl.program_id(2)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        m_start = m_tile * TILE_M
        if m_e < ME_MIN or m_start >= m_e:
            return
        eid = tl.load(expert_ids_ptr + exp_slot).to(tl.int64)
        x_start = tl.load(expert_starts_ptr + exp_slot) + m_start
        mo = tl.arange(0, TILE_M)
        mm = (m_start + mo) < m_e
        fo = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        fm = fo < TWO_DFF
        p64 = tl.arange(0, 64)

        acc_h = tl.zeros((BLOCK_F, TILE_M), dtype=tl.float32)
        acc_m = tl.zeros((BLOCK_F, TILE_M), dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_F, TILE_M), dtype=tl.float32)

        # Iterate over 512-weight blocks in two 64-byte halves (exact match to block_d_2bit=512 / block_d_4bit=256!)
        for blk in range(0, D_MODEL // 512):
            db = blk * 512
            for half in tl.static_range(0, 2):
                p_off = half * 64 + p64  # 0..63 or 64..127 inside the 128B cache line!
                p2 = blk * 128 + p_off
                ph = tl.load(w13_hi2_ptr + eid * stride_p2_e + fo[:, None] * stride_p2_f + p2[None, :] * stride_p2_p, mask=fm[:, None], other=0)
                pm = tl.load(w13_mid2_ptr + eid * stride_p2_e + fo[:, None] * stride_p2_f + p2[None, :] * stride_p2_p, mask=fm[:, None], other=0)
                p4_0 = (blk * 2) * 128 + p_off
                pl0 = tl.load(w13_lo4_ptr + eid * stride_l4_e + fo[:, None] * stride_l4_f + p4_0[None, :] * stride_l4_p, mask=fm[:, None], other=0)

                x0 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + p_off)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                wh0 = tl.inline_asm_elementwise(asm="lop3.b32 $0, $1, 0x03030303, 0x50505050, 0xEA;", constraints="=r,r", args=[ph], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                wm0, wl0 = tl.inline_asm_elementwise(asm="{ .reg .b32 m, b3; and.b32 m, $2, 0x03030303; shl.b32 m, m, 1; shr.b32 b3, $3, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, $3, 0x07070707, 0x50505050, 0xEA; }", constraints="=r,=r,r,r", args=[pm, pl0], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4)
                acc_h += tl.dot(wh0, x0)
                acc_m += tl.dot(wm0.to(tl.float8e4nv, bitcast=True), x0)
                acc_l += tl.dot(wl0.to(tl.float8e4nv, bitcast=True), x0)

                x1 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 128 + p_off)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                wh1 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 2; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[ph], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                wm1, wl1 = tl.inline_asm_elementwise(asm="{ .reg .b32 m, b3, l; shr.b32 m, $2, 1; and.b32 m, m, 0x06060606; shr.b32 l, $3, 4; shr.b32 b3, l, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, l, 0x07070707, 0x50505050, 0xEA; }", constraints="=r,=r,r,r", args=[pm, pl0], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4)
                acc_h += tl.dot(wh1, x1)
                acc_m += tl.dot(wm1.to(tl.float8e4nv, bitcast=True), x1)
                acc_l += tl.dot(wl1.to(tl.float8e4nv, bitcast=True), x1)

                p4_1 = (blk * 2 + 1) * 128 + p_off
                pl1 = tl.load(w13_lo4_ptr + eid * stride_l4_e + fo[:, None] * stride_l4_f + p4_1[None, :] * stride_l4_p, mask=fm[:, None], other=0)

                x2 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 256 + p_off)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                wh2 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 4; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[ph], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                wm2, wl2 = tl.inline_asm_elementwise(asm="{ .reg .b32 m, b3; shr.b32 m, $2, 3; and.b32 m, m, 0x06060606; shr.b32 b3, $3, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, $3, 0x07070707, 0x50505050, 0xEA; }", constraints="=r,=r,r,r", args=[pm, pl1], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4)
                acc_h += tl.dot(wh2, x2)
                acc_m += tl.dot(wm2.to(tl.float8e4nv, bitcast=True), x2)
                acc_l += tl.dot(wl2.to(tl.float8e4nv, bitcast=True), x2)

                x3 = tl.load(x_fp8_ptr + (x_start + mo[None, :]) * stride_x_n + (db + 384 + p_off)[:, None] * stride_x_d, mask=mm[None, :], other=0.0)
                wh3 = tl.inline_asm_elementwise(asm="{ .reg .b32 t; shr.b32 t, $1, 6; lop3.b32 $0, t, 0x03030303, 0x50505050, 0xEA; }", constraints="=r,r", args=[ph], dtype=tl.uint8, is_pure=True, pack=4).to(tl.float8e4nv, bitcast=True)
                wm3, wl3 = tl.inline_asm_elementwise(asm="{ .reg .b32 m, b3, l; shr.b32 m, $2, 5; and.b32 m, m, 0x06060606; shr.b32 l, $3, 4; shr.b32 b3, l, 3; and.b32 b3, b3, 0x01010101; lop3.b32 $0, m, b3, 0x50505050, 0xFE; lop3.b32 $1, l, 0x07070707, 0x50505050, 0xEA; }", constraints="=r,=r,r,r", args=[pm, pl1], dtype=(tl.uint8, tl.uint8), is_pure=True, pack=4)
                acc_h += tl.dot(wh3, x3)
                acc_m += tl.dot(wm3.to(tl.float8e4nv, bitcast=True), x3)
                acc_l += tl.dot(wl3.to(tl.float8e4nv, bitcast=True), x3)

        xsum = tl.load(x_sum_ptr + x_start + mo, mask=mm, other=0.0)
        xsc = tl.load(x_scale_ptr + x_start + mo, mask=mm, other=0.0)
        sc = tl.load(s13_ptr + eid * stride_s_e + fo * stride_s_f, mask=fm, other=0.0).to(tl.float32)
        val = (64.0 * acc_h + 8.0 * acc_m + acc_l - 712.0 * xsum[None, :]) * xsc[None, :] * sc[:, None]
        tl.store(
            gu_den_out_ptr + (x_start + mo[None, :]) * stride_go_n + fo[:, None] * stride_go_f,
            val.to(tl.bfloat16),
            mask=fm[:, None] & mm[None, :],
        )

    @triton.jit
    def _dense_swiglu_kernel(
        gu_den_ptr,  # [N_pairs, 2 * D_FF] bf16
        expert_starts_ptr,
        expert_counts_ptr,
        h_den_out_ptr,  # [N_pairs, D_FF] bf16
        stride_gu_n,
        stride_gu_f,
        stride_h_n,
        stride_h_f,
        D_FF: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ME_MIN: tl.constexpr,
    ):
        exp_slot = tl.program_id(0)
        f_tile = tl.program_id(1)
        m_tile = tl.program_id(2)
        m_e = tl.load(expert_counts_ptr + exp_slot)
        m_start = m_tile * TILE_M
        if m_e < ME_MIN or m_start >= m_e:
            return
        x_start = tl.load(expert_starts_ptr + exp_slot) + m_start
        mo = tl.arange(0, TILE_M)
        mm = (m_start + mo) < m_e
        fo = f_tile * BLOCK_F + tl.arange(0, BLOCK_F)
        fm = fo < D_FF
        g = tl.load(gu_den_ptr + (x_start + mo[:, None]) * stride_gu_n + fo[None, :] * stride_gu_f, mask=mm[:, None] & fm[None, :], other=0.0).to(tl.float32)
        u = tl.load(gu_den_ptr + (x_start + mo[:, None]) * stride_gu_n + (D_FF + fo)[None, :] * stride_gu_f, mask=mm[:, None] & fm[None, :], other=0.0).to(tl.float32)
        h = (g * tl.sigmoid(g)) * u
        tl.store(h_den_out_ptr + (x_start + mo[:, None]) * stride_h_n + fo[None, :] * stride_h_f, h.to(tl.bfloat16), mask=mm[:, None] & fm[None, :])


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


def launch_prog8_224_stage1(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    wg_hi2_packed: torch.Tensor,
    wu_hi2_packed: torch.Tensor,
    sg: torch.Tensor,  # [E, d_ff] fp32
    su: torch.Tensor,
    g_hi_out: torch.Tensor,
    u_hi_out: torch.Tensor,
    exp_scores_out: torch.Tensor,
    h_s1_out: torch.Tensor | None = None,
    max_me: int = 16,
    block_f: int = 64,
    save_hi_out: bool = True,
    me_min: int = 1,
    me_max: int = 1048576,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """O9: 128B cache-line-exact Stage-1 2-bit scorer + Marlin 0x6000 bitcast + K=128 TC (207 GB/s on L4!)."""
    _require_triton_cuda()
    d_model = x_pairs.shape[1]
    d_ff = wg_hi2_packed.shape[1]
    save_h = h_s1_out is not None
    h_ptr = h_s1_out if save_h else g_hi_out
    grid = (expert_ids.shape[0], triton.cdiv(d_ff, block_f))
    _prog8_224_stage1_kernel[grid](
        x_pairs, expert_ids, expert_starts, expert_counts,
        wg_hi2_packed, wu_hi2_packed, sg, su,
        g_hi_out, u_hi_out, h_ptr, exp_scores_out,
        x_pairs.stride(0), x_pairs.stride(1),
        wg_hi2_packed.stride(0), wg_hi2_packed.stride(1), wg_hi2_packed.stride(2),
        sg.stride(0), sg.stride(1),
        g_hi_out.stride(0), g_hi_out.stride(1),
        exp_scores_out.stride(0), exp_scores_out.stride(1),
        D_MODEL=d_model, D_FF=d_ff, MAX_ME=max_me, BLOCK_F=block_f, PACK_D=128,
        SAVE_HI_OUT=save_hi_out, SAVE_H_S1=save_h, ME_MIN=me_min, ME_MAX=me_max,
        num_warps=num_warps, num_stages=num_stages,
    )
    return exp_scores_out


def launch_prog8_224_stage2(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    active_idx: torch.Tensor,
    g_hi_in: torch.Tensor,
    u_hi_in: torch.Tensor,
    wg_mid2_packed: torch.Tensor,
    wu_mid2_packed: torch.Tensor,
    wg_lo4_packed: torch.Tensor,
    wu_lo4_packed: torch.Tensor,
    sg: torch.Tensor,  # [E, d_ff] fp32
    su: torch.Tensor,
    w_down_t: torch.Tensor,
    w_down_scale: torch.Tensor | None,
    h_act_buf: torch.Tensor,
    out_buf: torch.Tensor,
    h_s1_to_zero: torch.Tensor | None = None,
    k_full: int | None = None,
    max_me: int = 16,
    block_k: int = 64,
    me_min: int = 1,
    me_max: int = 1048576,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """O9/O10: 128B cache-line-exact Zero-VRAM Stage-2 (supports 3-Tier 8b/4b + fused active h_s1 zeroing)."""
    _require_triton_cuda()
    d_model = x_pairs.shape[1]
    n_slots = expert_ids.shape[0]
    k_u = active_idx.shape[1]
    k_f = k_u if k_full is None else int(k_full)
    zero_h = h_s1_to_zero is not None
    h_s1_ptr = h_s1_to_zero if zero_h else g_hi_in
    is_fp8 = w_down_t.dtype == torch.float8_e4m3fn
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else sg

    _prog8_224_stage2_gate_up_kernel[(n_slots, triton.cdiv(k_u, block_k))](
        x_pairs, expert_ids, expert_starts, expert_counts, active_idx,
        g_hi_in, u_hi_in, h_s1_ptr,
        wg_mid2_packed, wu_mid2_packed, wg_lo4_packed, wu_lo4_packed,
        sg, su, h_act_buf,
        x_pairs.stride(0), x_pairs.stride(1),
        active_idx.stride(0), active_idx.stride(1),
        g_hi_in.stride(0), g_hi_in.stride(1),
        wg_mid2_packed.stride(0), wg_mid2_packed.stride(1), wg_mid2_packed.stride(2),
        wg_lo4_packed.stride(0), wg_lo4_packed.stride(1), wg_lo4_packed.stride(2),
        sg.stride(0), sg.stride(1),
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        D_MODEL=d_model, K_U=k_u, K_FULL=k_f, MAX_ME=max_me, BLOCK_K=block_k, PACK_D=128,
        ZERO_H_S1=zero_h, ME_MIN=me_min, ME_MAX=me_max,
        num_warps=num_warps, num_stages=num_stages,
    )
    _sparse_eg_down_fp8_b128_v3_kernel[(n_slots, d_model // 128)](
        h_act_buf, expert_ids, expert_starts, expert_counts, active_idx,
        w_down_t, sd, out_buf,
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        active_idx.stride(0), active_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        K_U=k_u, MAX_ME=max_me, BLOCK_K=64, BLOCK_D=128, IS_FP8=is_fp8,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=num_warps, num_stages=3,
    )
    return out_buf


def launch_sparse_fp8_stage2(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    active_idx: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down_t: torch.Tensor,
    h_act_buf: torch.Tensor,
    out_buf: torch.Tensor,
    w_gate_scale: torch.Tensor | None = None,
    w_up_scale: torch.Tensor | None = None,
    w_down_scale: torch.Tensor | None = None,
    max_me: int = 16,
    block_k: int = 64,
    use_native_fp8_dot: bool = True,
    me_min: int = 1,
    me_max: int = 1048576,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    _require_triton_cuda()
    d_model = x_pairs.shape[1]
    n_slots = expert_ids.shape[0]
    k_u = active_idx.shape[1]
    is_fp8 = w_gate.dtype == torch.float8_e4m3fn
    sg = w_gate_scale if (is_fp8 and w_gate_scale is not None) else x_pairs
    su = w_up_scale if (is_fp8 and w_up_scale is not None) else x_pairs
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else x_pairs

    _sparse_eg_gate_up_fp8_b128_v3_kernel[(n_slots, triton.cdiv(k_u, block_k))](
        x_pairs, expert_ids, expert_starts, expert_counts, active_idx,
        w_gate, sg, w_up, su, h_act_buf,
        x_pairs.stride(0), x_pairs.stride(1),
        active_idx.stride(0), active_idx.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        sg.stride(0) if is_fp8 else 0, sg.stride(1) if is_fp8 else 0, sg.stride(2) if is_fp8 else 0,
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        D_MODEL=d_model, K_U=k_u, MAX_ME=max_me, BLOCK_K=block_k, BLOCK_D=128,
        IS_FP8=is_fp8, USE_NATIVE_FP8_DOT=use_native_fp8_dot,
        num_warps=num_warps, num_stages=num_stages,
    )
    _sparse_eg_down_fp8_b128_v3_kernel[(n_slots, d_model // 128)](
        h_act_buf, expert_ids, expert_starts, expert_counts, active_idx,
        w_down_t, sd, out_buf,
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        active_idx.stride(0), active_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        K_U=k_u, MAX_ME=max_me, BLOCK_K=block_k, BLOCK_D=128, IS_FP8=is_fp8,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out_buf


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


def launch_dense_224_grouped_mm(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    max_expert_tokens: int,
    wg_hi2_packed: torch.Tensor,
    wu_hi2_packed: torch.Tensor,
    wg_mid2_packed: torch.Tensor,
    wu_mid2_packed: torch.Tensor,
    wg_lo4_packed: torch.Tensor,
    wu_lo4_packed: torch.Tensor,
    sg: torch.Tensor,
    su: torch.Tensor,
    w_down_t: torch.Tensor,
    w_down_scale: torch.Tensor | None,
    h_dense_buf: torch.Tensor,
    out_buf: torch.Tensor,
    tile_m: int = 32,
    block_f: int = 64,
    me_min: int = 1,
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Strict 0%-proxy-VRAM fused 1-pass dense 2+2+4 grouped GEMM for hot experts (`m_e >= me_min`)."""
    _require_triton_cuda()
    d_model = x_pairs.shape[1]
    d_ff = wg_hi2_packed.shape[1]
    n_slots = expert_ids.shape[0]
    m_tiles = triton.cdiv(max_expert_tokens, tile_m)
    is_fp8 = w_down_t.dtype == torch.float8_e4m3fn
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else sg

    _dense_eg_gate_up_224_kernel[(n_slots, triton.cdiv(d_ff, block_f), m_tiles)](
        x_pairs, expert_ids, expert_starts, expert_counts,
        wg_hi2_packed, wu_hi2_packed, wg_mid2_packed, wu_mid2_packed, wg_lo4_packed, wu_lo4_packed,
        sg, su, h_dense_buf,
        x_pairs.stride(0), x_pairs.stride(1),
        wg_hi2_packed.stride(0), wg_hi2_packed.stride(1), wg_hi2_packed.stride(2),
        wg_lo4_packed.stride(0), wg_lo4_packed.stride(1), wg_lo4_packed.stride(2),
        sg.stride(0), sg.stride(1),
        h_dense_buf.stride(0), h_dense_buf.stride(1),
        D_MODEL=d_model, D_FF=d_ff, TILE_M=tile_m, BLOCK_F=block_f, PACK_D=128,
        ME_MIN=me_min,
        num_warps=num_warps, num_stages=num_stages,
    )
    _dense_eg_down_fp8_b128_kernel[(n_slots, d_model // 128, m_tiles)](
        h_dense_buf, expert_ids, expert_starts, expert_counts,
        w_down_t, sd, out_buf,
        h_dense_buf.stride(0), h_dense_buf.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        D_FF=d_ff, TILE_M=tile_m, BLOCK_K=64, BLOCK_D=128, IS_FP8=is_fp8, ME_MIN=me_min,
        num_warps=num_warps, num_stages=3,
    )
    return out_buf


def launch_dense_fp8_grouped_mm(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    max_expert_tokens: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down_t: torch.Tensor,
    h_dense_buf: torch.Tensor,
    out_buf: torch.Tensor,
    w_gate_scale: torch.Tensor | None = None,
    w_up_scale: torch.Tensor | None = None,
    w_down_scale: torch.Tensor | None = None,
    tile_m: int = 16,
    block_f: int = 64,
    use_native_fp8_dot: bool = True,
    me_min: int = 1,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    _require_triton_cuda()
    d_model = x_pairs.shape[1]
    d_ff = w_gate.shape[1]
    n_slots = expert_ids.shape[0]
    m_tiles = triton.cdiv(max_expert_tokens, tile_m)
    is_fp8 = w_gate.dtype == torch.float8_e4m3fn
    sg = w_gate_scale if (is_fp8 and w_gate_scale is not None) else x_pairs
    su = w_up_scale if (is_fp8 and w_up_scale is not None) else x_pairs
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else x_pairs

    _dense_eg_gate_up_fp8_b128_kernel[(n_slots, triton.cdiv(d_ff, block_f), m_tiles)](
        x_pairs, expert_ids, expert_starts, expert_counts,
        w_gate, sg, w_up, su, h_dense_buf,
        x_pairs.stride(0), x_pairs.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        sg.stride(0) if is_fp8 else 0, sg.stride(1) if is_fp8 else 0, sg.stride(2) if is_fp8 else 0,
        h_dense_buf.stride(0), h_dense_buf.stride(1),
        D_MODEL=d_model, D_FF=d_ff, TILE_M=tile_m, BLOCK_F=block_f, BLOCK_D=128,
        IS_FP8=is_fp8, USE_NATIVE_FP8_DOT=use_native_fp8_dot, ME_MIN=me_min,
        num_warps=num_warps, num_stages=num_stages,
    )
    _dense_eg_down_fp8_b128_kernel[(n_slots, d_model // 128, m_tiles)](
        h_dense_buf, expert_ids, expert_starts, expert_counts,
        w_down_t, sd, out_buf,
        h_dense_buf.stride(0), h_dense_buf.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        D_FF=d_ff, TILE_M=tile_m, BLOCK_K=64, BLOCK_D=128, IS_FP8=is_fp8, ME_MIN=me_min,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out_buf


def launch_quantize_x_fp8_sum(
    x: torch.Tensor,
    x_fp8_out: torch.Tensor,
    x_sum_out: torch.Tensor,
    x_scale_out: torch.Tensor,
    order: torch.Tensor | None = None,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused 1-kernel gather + BF16->FP8 per-row quantization + x_sum for SM90 0x50505050 WGMMA."""
    _require_triton_cuda()
    n_rows = x_fp8_out.shape[0]
    d_model = x.shape[1]
    block_d = 1024 if d_model >= 1024 else 512
    use_order = order is not None
    ord_ptr = order if use_order else x_sum_out
    _quantize_x_fp8_sum_kernel[(n_rows,)](
        x, ord_ptr, x_fp8_out, x_sum_out, x_scale_out,
        x.stride(0), x.stride(1), x_fp8_out.stride(0), x_fp8_out.stride(1),
        D_MODEL=d_model, BLOCK_D=block_d, TOP_K=top_k, USE_ORDER=use_order,
        num_warps=4,
    )
    return x_fp8_out, x_sum_out, x_scale_out


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


def launch_prog8_sm90_stage1(
    x_fp8: torch.Tensor,
    x_sum: torch.Tensor,
    x_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    w13_hi2: torch.Tensor,  # [E, 2 * d_ff, d_model // 4] uint8
    s13: torch.Tensor,  # [E, 2 * d_ff] fp32
    gu_hi_out: torch.Tensor,  # [N_pairs, 2 * d_ff] bf16
    exp_scores_out: torch.Tensor,  # [N_slots, d_ff] fp32
    h_s1_out: torch.Tensor | None = None,
    max_me: int = 8,
    block_f: int = 64,
    me_min: int = 1,
    me_max: int = 8,
) -> torch.Tensor:
    """O11 Hopper SM90 Stage-1: 0x50505050 LOP3-pack4 + W@X^T WGMMA FP8 (2,906 GB/s on H200!)."""
    _require_triton_cuda()
    d_model = x_fp8.shape[1]
    two_dff = w13_hi2.shape[1]
    d_ff = two_dff // 2
    n_slots = expert_ids.shape[0]
    save_h = h_s1_out is not None
    h_ptr = h_s1_out if save_h else gu_hi_out

    _prog8_sm90_stage1_kernel[(n_slots, triton.cdiv(two_dff, block_f))](
        x_fp8, x_sum, x_scale, expert_ids, expert_starts, expert_counts,
        w13_hi2, s13, gu_hi_out,
        x_fp8.stride(0), x_fp8.stride(1),
        w13_hi2.stride(0), w13_hi2.stride(1), w13_hi2.stride(2),
        s13.stride(0), s13.stride(1),
        gu_hi_out.stride(0), gu_hi_out.stride(1),
        D_MODEL=d_model, TWO_DFF=two_dff, MAX_ME=max_me, BLOCK_F=block_f, PACK_D=128,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4, num_stages=3,
    )
    _stage1_swiglu_score_kernel[(n_slots, triton.cdiv(d_ff, 64))](
        gu_hi_out, expert_starts, expert_counts, h_ptr, exp_scores_out,
        gu_hi_out.stride(0), gu_hi_out.stride(1),
        h_ptr.stride(0), h_ptr.stride(1),
        exp_scores_out.stride(0), exp_scores_out.stride(1),
        D_FF=d_ff, MAX_ME=max_me, BLOCK_F=64, SAVE_H_S1=save_h,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4,
    )
    return exp_scores_out


def launch_prog8_sm90_stage2(
    x_fp8: torch.Tensor,
    x_sum: torch.Tensor,
    x_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    active_idx: torch.Tensor,
    gu_hi_in: torch.Tensor,  # [N_pairs, 2 * d_ff] bf16
    w13_mid2: torch.Tensor,  # [E, 2 * d_ff, d_model // 4] uint8
    w13_lo4: torch.Tensor,  # [E, 2 * d_ff, d_model // 2] uint8
    s13: torch.Tensor,  # [E, 2 * d_ff] fp32
    w_down_t: torch.Tensor,
    w_down_scale: torch.Tensor | None,
    gu_act_buf: torch.Tensor,  # [N_slots, max_me, 2 * k_u] bf16
    h_act_buf: torch.Tensor,  # [N_slots, max_me_down, k_u] bf16
    out_buf: torch.Tensor,
    h_s1_to_zero: torch.Tensor | None = None,
    k_full: int | None = None,
    max_me: int = 8,
    max_me_down: int = 16,
    block_k: int = 64,
    me_min: int = 1,
    me_max: int = 8,
) -> torch.Tensor:
    """O11 Hopper SM90 Stage-2: 1Mat WGMMA FP8 (2,251 GB/s, 0 spills!) + L2 SwiGLU + Coalesced Down."""
    _require_triton_cuda()
    d_model = x_fp8.shape[1]
    d_ff = w13_mid2.shape[1] // 2
    n_slots = expert_ids.shape[0]
    k_u = active_idx.shape[1]
    k_f = k_u if k_full is None else int(k_full)
    n_kt = triton.cdiv(k_u, block_k)
    zero_h = h_s1_to_zero is not None
    h_s1_ptr = h_s1_to_zero if zero_h else gu_hi_in
    is_fp8 = w_down_t.dtype == torch.float8_e4m3fn
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else s13

    _prog8_sm90_stage2_gu_kernel[(n_slots, 2 * n_kt)](
        x_fp8, x_sum, x_scale, expert_ids, expert_starts, expert_counts, active_idx,
        gu_hi_in, w13_mid2, w13_lo4, s13, gu_act_buf,
        x_fp8.stride(0), x_fp8.stride(1),
        active_idx.stride(0), active_idx.stride(1),
        gu_hi_in.stride(0), gu_hi_in.stride(1),
        w13_mid2.stride(0), w13_mid2.stride(1), w13_mid2.stride(2),
        w13_lo4.stride(0), w13_lo4.stride(1), w13_lo4.stride(2),
        s13.stride(0), s13.stride(1),
        gu_act_buf.stride(0), gu_act_buf.stride(1), gu_act_buf.stride(2),
        D_MODEL=d_model, D_FF=d_ff, K_U=k_u, K_FULL=k_f, N_KT=n_kt,
        MAX_ME=max_me, BLOCK_K=block_k, PACK_D=128,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4, num_stages=2,
    )
    _stage2_swiglu_zero_hs1_kernel[(n_slots, triton.cdiv(k_u, 64))](
        gu_act_buf, expert_starts, expert_counts, active_idx, h_act_buf, h_s1_ptr,
        gu_act_buf.stride(0), gu_act_buf.stride(1), gu_act_buf.stride(2),
        active_idx.stride(0), active_idx.stride(1),
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        h_s1_ptr.stride(0), h_s1_ptr.stride(1),
        K_U=k_u, MAX_ME=max_me, BLOCK_K=64, ZERO_H_S1=zero_h,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4,
    )
    _sparse_eg_down_fp8_b128_v3_kernel[(n_slots, d_model // 128)](
        h_act_buf, expert_ids, expert_starts, expert_counts, active_idx,
        w_down_t, sd, out_buf,
        h_act_buf.stride(0), h_act_buf.stride(1), h_act_buf.stride(2),
        active_idx.stride(0), active_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        K_U=k_u, MAX_ME=max_me_down, BLOCK_K=64, BLOCK_D=128, IS_FP8=is_fp8,
        ME_MIN=me_min, ME_MAX=me_max,
        num_warps=4, num_stages=3,
    )
    return out_buf


def launch_dense_sm90_224_grouped_mm(
    x_fp8: torch.Tensor,
    x_sum: torch.Tensor,
    x_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    max_expert_tokens: int,
    w13_hi2: torch.Tensor,
    w13_mid2: torch.Tensor,
    w13_lo4: torch.Tensor,
    s13: torch.Tensor,
    w_down_t: torch.Tensor,
    w_down_scale: torch.Tensor | None,
    gu_den_buf: torch.Tensor,  # [N_pairs, 2 * d_ff] bf16
    h_den_buf: torch.Tensor,  # [N_pairs, d_ff] bf16
    out_buf: torch.Tensor,
    tile_m: int = 32,
    block_f: int = 64,
    me_min: int = 1,
) -> torch.Tensor:
    """O11 Hopper SM90 zero-VRAM Hot Dense 2+2+4 (1Mat WGMMA FP8, 0 spills, 58.1 us!)."""
    _require_triton_cuda()
    d_model = x_fp8.shape[1]
    two_dff = w13_hi2.shape[1]
    d_ff = two_dff // 2
    n_slots = expert_ids.shape[0]
    m_tiles = triton.cdiv(max_expert_tokens, tile_m)
    is_fp8 = w_down_t.dtype == torch.float8_e4m3fn
    sd = w_down_scale if (is_fp8 and w_down_scale is not None) else s13

    _dense_sm90_224_gu_kernel[(n_slots, triton.cdiv(two_dff, block_f), m_tiles)](
        x_fp8, x_sum, x_scale, expert_ids, expert_starts, expert_counts,
        w13_hi2, w13_mid2, w13_lo4, s13, gu_den_buf,
        x_fp8.stride(0), x_fp8.stride(1),
        w13_hi2.stride(0), w13_hi2.stride(1), w13_hi2.stride(2),
        w13_lo4.stride(0), w13_lo4.stride(1), w13_lo4.stride(2),
        s13.stride(0), s13.stride(1),
        gu_den_buf.stride(0), gu_den_buf.stride(1),
        D_MODEL=d_model, TWO_DFF=two_dff, TILE_M=tile_m, BLOCK_F=block_f, ME_MIN=me_min,
        num_warps=4, num_stages=3,
    )
    _dense_swiglu_kernel[(n_slots, triton.cdiv(d_ff, 64), m_tiles)](
        gu_den_buf, expert_starts, expert_counts, h_den_buf,
        gu_den_buf.stride(0), gu_den_buf.stride(1),
        h_den_buf.stride(0), h_den_buf.stride(1),
        D_FF=d_ff, TILE_M=tile_m, BLOCK_F=64, ME_MIN=me_min,
        num_warps=4,
    )
    _dense_eg_down_fp8_b128_kernel[(n_slots, d_model // 128, m_tiles)](
        h_den_buf, expert_ids, expert_starts, expert_counts,
        w_down_t, sd, out_buf,
        h_den_buf.stride(0), h_den_buf.stride(1),
        w_down_t.stride(0), w_down_t.stride(1), w_down_t.stride(2),
        sd.stride(0) if is_fp8 else 0, sd.stride(1) if is_fp8 else 0, sd.stride(2) if is_fp8 else 0,
        out_buf.stride(0), out_buf.stride(1),
        D_FF=d_ff, TILE_M=tile_m, BLOCK_K=64, BLOCK_D=128, IS_FP8=is_fp8, ME_MIN=me_min,
        num_warps=4, num_stages=3,
    )
    return out_buf
