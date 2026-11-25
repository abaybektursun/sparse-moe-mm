"""Phase-1 Triton kernels for CUDA-Graph-safe sparse quantized MoE grouped execution.

Designed for SM90 (H100/H200 FP8) and SM100/SM120 (B200 / RTX PRO 6000).
Safe to import on CPU-only machines (raises RuntimeError only if a CUDA kernel is invoked).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU dev laptop without triton
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton_cuda() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Run on a CUDA GPU with triton>=3.2.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run Triton kernels.")


if HAS_TRITON:

    @triton.jit
    def _sparse_up_down_gather_kernel(
        # Inputs
        x_ptr,  # [N_pairs, d_model] bf16/fp16 (replicated/sorted by expert)
        expert_ids_ptr,  # [N_pairs] int32
        active_idx_ptr,  # [N_pairs, max_active] int32 (-1 padded for CUDA Graph safety)
        active_cnt_ptr,  # [N_pairs] int32
        gate_act_ptr,  # [N_pairs, d_ff] bf16/fp16 (or pre-gathered if Stage-2 all-3)
        w_up_ptr,  # [E, d_ff, d_model] fp8 or bf16
        w_up_scale_ptr,  # [E, d_ff] fp32 (or dummy if BF16)
        w_down_t_ptr,  # [E, d_ff, d_model] fp8 or bf16 (transposed layout for coalesced row gather!)
        w_down_scale_ptr,  # [E, d_ff] fp32 (or dummy if BF16)
        out_ptr,  # [N_pairs, d_model] fp32 accumulator (or bf16)
        # Strides
        stride_x_n,
        stride_x_d,
        stride_idx_n,
        stride_idx_k,
        stride_ga_n,
        stride_ga_f,
        stride_wu_e,
        stride_wu_f,
        stride_wu_d,
        stride_ws_e,
        stride_ws_f,
        stride_wd_e,
        stride_wd_f,
        stride_wd_d,
        stride_out_n,
        stride_out_d,
        # Dims
        D_MODEL: tl.constexpr,
        BLOCK_K: tl.constexpr,  # Active-neuron tile (e.g., 32 or 64)
        BLOCK_D: tl.constexpr,  # Hidden-dim tile (e.g., 128 or 256)
        IS_FP8: tl.constexpr,
    ):
        """Fused gather-packed sparse up + SwiGLU + down projection for one (token, expert) pair.

        Key memory layout optimization:
          w_down is stored transposed as [E, d_ff, d_model] so gathering neuron row `f`
          reads contiguous `d_model` elements (fully coalesced across threads).
        CUDA Graph compatibility:
          Grid launches fixed ceil(max_active / BLOCK_K) tiles; tiles past active_cnt exit early.
        """
        pair_id = tl.program_id(0)
        tile_k = tl.program_id(1)

        cnt = tl.load(active_cnt_ptr + pair_id)
        k_start = tile_k * BLOCK_K
        if k_start >= cnt:
            return

        expert_id = tl.load(expert_ids_ptr + pair_id)
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < cnt
        neuron_ids = tl.load(active_idx_ptr + pair_id * stride_idx_n + k_offs * stride_idx_k, mask=k_mask, other=0)

        # Step 1: Compute up_proj[k] = dot(w_up[e, neuron_ids[k], :], x[pair_id, :])
        up_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            x_vec = tl.load(x_ptr + pair_id * stride_x_n + d_offs * stride_x_d, mask=d_mask, other=0.0).to(tl.float32)
            wu_ptrs = (
                w_up_ptr
                + expert_id * stride_wu_e
                + neuron_ids[:, None] * stride_wu_f
                + d_offs[None, :] * stride_wu_d
            )
            wu_tile = tl.load(wu_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            up_acc += tl.sum(wu_tile * x_vec[None, :], axis=1)

        if IS_FP8:
            wu_scale = tl.load(
                w_up_scale_ptr + expert_id * stride_ws_e + neuron_ids * stride_ws_f, mask=k_mask, other=1.0
            ).to(tl.float32)
            up_acc = up_acc * wu_scale

        # Step 2: Multiply by precomputed silu(gate)[neuron_ids]
        g_vals = tl.load(
            gate_act_ptr + pair_id * stride_ga_n + neuron_ids * stride_ga_f, mask=k_mask, other=0.0
        ).to(tl.float32)
        h_vals = g_vals * up_acc

        if IS_FP8:
            wd_scale = tl.load(
                w_down_scale_ptr + expert_id * stride_ws_e + neuron_ids * stride_ws_f, mask=k_mask, other=1.0
            ).to(tl.float32)
            h_vals = h_vals * wd_scale

        # Step 3: Accumulate h_vals @ w_down_t[e, neuron_ids, :] into out[pair_id, :]
        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            wd_ptrs = (
                w_down_t_ptr
                + expert_id * stride_wd_e
                + neuron_ids[:, None] * stride_wd_f
                + d_offs[None, :] * stride_wd_d
            )
            wd_tile = tl.load(wd_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            partial = tl.sum(wd_tile * h_vals[:, None], axis=0)
            tl.atomic_add(out_ptr + pair_id * stride_out_n + d_offs * stride_out_d, partial, mask=d_mask)

    @triton.jit
    def _sparse_all3_gather_kernel(
        # Stage-2 exact all-3 sparse kernel (gate + up + down on active_idx only)
        x_ptr,
        expert_ids_ptr,
        active_idx_ptr,
        active_cnt_ptr,
        w_gate_ptr,
        w_gate_scale_ptr,
        w_up_ptr,
        w_up_scale_ptr,
        w_down_t_ptr,
        w_down_scale_ptr,
        out_ptr,
        stride_x_n,
        stride_x_d,
        stride_idx_n,
        stride_idx_k,
        stride_w_e,
        stride_w_f,
        stride_w_d,
        stride_s_e,
        stride_s_f,
        stride_out_n,
        stride_out_d,
        D_MODEL: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        """Fused Stage-2 sparse gate + up + SiLU + down on pre-selected active neuron indices."""
        pair_id = tl.program_id(0)
        tile_k = tl.program_id(1)

        cnt = tl.load(active_cnt_ptr + pair_id)
        k_start = tile_k * BLOCK_K
        if k_start >= cnt:
            return

        expert_id = tl.load(expert_ids_ptr + pair_id)
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < cnt
        neuron_ids = tl.load(active_idx_ptr + pair_id * stride_idx_n + k_offs * stride_idx_k, mask=k_mask, other=0)

        gate_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            x_vec = tl.load(x_ptr + pair_id * stride_x_n + d_offs * stride_x_d, mask=d_mask, other=0.0).to(tl.float32)
            base_off = expert_id * stride_w_e + neuron_ids[:, None] * stride_w_f + d_offs[None, :] * stride_w_d
            wg_tile = tl.load(w_gate_ptr + base_off, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            wu_tile = tl.load(w_up_ptr + base_off, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            gate_acc += tl.sum(wg_tile * x_vec[None, :], axis=1)
            up_acc += tl.sum(wu_tile * x_vec[None, :], axis=1)

        if IS_FP8:
            s_off = expert_id * stride_s_e + neuron_ids * stride_s_f
            gate_acc = gate_acc * tl.load(w_gate_scale_ptr + s_off, mask=k_mask, other=1.0).to(tl.float32)
            up_acc = up_acc * tl.load(w_up_scale_ptr + s_off, mask=k_mask, other=1.0).to(tl.float32)

        # SiLU(gate) * up
        h_vals = (gate_acc * tl.sigmoid(gate_acc)) * up_acc

        if IS_FP8:
            s_off = expert_id * stride_s_e + neuron_ids * stride_s_f
            h_vals = h_vals * tl.load(w_down_scale_ptr + s_off, mask=k_mask, other=1.0).to(tl.float32)

        for d_start in range(0, D_MODEL, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_MODEL
            wd_ptrs = (
                w_down_t_ptr
                + expert_id * stride_w_e
                + neuron_ids[:, None] * stride_w_f
                + d_offs[None, :] * stride_w_d
            )
            wd_tile = tl.load(wd_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            partial = tl.sum(wd_tile * h_vals[:, None], axis=0)
            tl.atomic_add(out_ptr + pair_id * stride_out_n + d_offs * stride_out_d, partial, mask=d_mask)


def launch_sparse_up_down(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    active_idx: torch.Tensor,
    active_cnt: torch.Tensor,
    gate_act: torch.Tensor,
    w_up: torch.Tensor,
    w_down_t: torch.Tensor,
    w_up_scale: torch.Tensor | None = None,
    w_down_scale: torch.Tensor | None = None,
    block_k: int = 32,
    block_d: int = 256,
) -> torch.Tensor:
    """Launch fused gather-packed sparse up + SwiGLU + down Triton kernel."""
    _require_triton_cuda()
    n_pairs, d_model = x_pairs.shape
    max_active = active_idx.shape[1]
    is_fp8 = w_up.dtype == torch.float8_e4m3fn
    if is_fp8 and (w_up_scale is None or w_down_scale is None):
        raise ValueError("FP8 weights require per-neuron scales (w_up_scale, w_down_scale).")

    dummy_scale = torch.empty(1, device=x_pairs.device, dtype=torch.float32)
    s_up = w_up_scale if is_fp8 else dummy_scale
    s_dn = w_down_scale if is_fp8 else dummy_scale
    out = torch.zeros((n_pairs, d_model), dtype=torch.float32, device=x_pairs.device)
    grid = (n_pairs, triton.cdiv(max_active, block_k))
    _sparse_up_down_gather_kernel[grid](
        x_pairs,
        expert_ids,
        active_idx,
        active_cnt,
        gate_act,
        w_up,
        s_up,
        w_down_t,
        s_dn,
        out,
        x_pairs.stride(0),
        x_pairs.stride(1),
        active_idx.stride(0),
        active_idx.stride(1),
        gate_act.stride(0),
        gate_act.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        w_up.stride(2),
        s_up.stride(0) if is_fp8 else 0,
        s_up.stride(1) if is_fp8 else 0,
        w_down_t.stride(0),
        w_down_t.stride(1),
        w_down_t.stride(2),
        out.stride(0),
        out.stride(1),
        D_MODEL=d_model,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        IS_FP8=is_fp8,
    )
    return out


def launch_sparse_all3(
    x_pairs: torch.Tensor,
    expert_ids: torch.Tensor,
    active_idx: torch.Tensor,
    active_cnt: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down_t: torch.Tensor,
    w_gate_scale: torch.Tensor | None = None,
    w_up_scale: torch.Tensor | None = None,
    w_down_scale: torch.Tensor | None = None,
    block_k: int = 32,
    block_d: int = 256,
) -> torch.Tensor:
    """Launch fused Stage-2 all-3 sparse gate + up + SwiGLU + down Triton kernel."""
    _require_triton_cuda()
    n_pairs, d_model = x_pairs.shape
    max_active = active_idx.shape[1]
    is_fp8 = w_gate.dtype == torch.float8_e4m3fn
    if is_fp8 and (w_gate_scale is None or w_up_scale is None or w_down_scale is None):
        raise ValueError("FP8 weights require per-neuron scales for gate, up, and down.")

    dummy_scale = torch.empty(1, device=x_pairs.device, dtype=torch.float32)
    s_g = w_gate_scale if is_fp8 else dummy_scale
    s_u = w_up_scale if is_fp8 else dummy_scale
    s_d = w_down_scale if is_fp8 else dummy_scale
    out = torch.zeros((n_pairs, d_model), dtype=torch.float32, device=x_pairs.device)
    grid = (n_pairs, triton.cdiv(max_active, block_k))
    _sparse_all3_gather_kernel[grid](
        x_pairs,
        expert_ids,
        active_idx,
        active_cnt,
        w_gate,
        s_g,
        w_up,
        s_u,
        w_down_t,
        s_d,
        out,
        x_pairs.stride(0),
        x_pairs.stride(1),
        active_idx.stride(0),
        active_idx.stride(1),
        w_gate.stride(0),
        w_gate.stride(1),
        w_gate.stride(2),
        s_g.stride(0) if is_fp8 else 0,
        s_g.stride(1) if is_fp8 else 0,
        out.stride(0),
        out.stride(1),
        D_MODEL=d_model,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        IS_FP8=is_fp8,
    )
    return out
