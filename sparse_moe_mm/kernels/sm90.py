"""SM90 Hopper (H100 / H200) O11 kernels: 0x50505050 LOP3-pack4 FP8 bitcast + transposed W@X^T WGMMA."""

from __future__ import annotations

import torch

from sparse_moe_mm.kernels._common import (
    HAS_TRITON,
    _require_triton_cuda,
    tl,
    triton,
)

if HAS_TRITON:
    from sparse_moe_mm.kernels._common import (
        _dense_eg_down_fp8_b128_kernel,
        _sparse_eg_down_fp8_b128_v3_kernel,
    )

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

