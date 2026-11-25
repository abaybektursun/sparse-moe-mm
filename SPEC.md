# SPEC — Quantized Proxy-Gated Sparse `grouped_mm`

## 1. Core Question

Can a **quantized (FP8 on SM90, NVFP4 on SM100/SM120) all-3-matrix sparse `grouped_mm`** kernel beat production dense FP8/NVFP4 `grouped_mm` by **`>= 1.5×` on the MoE layer** across serving batch sizes `B = 16..128` while keeping benchmark accuracy within **`1.0%` relative** of dense?

## 2. Target MoE Layer Shapes

| Model | `E` (routed) | `K` (active) | `d_model` | `d_ff` (per expert) | `M_e` at `B=64` | `M_e` at `B=128` | Single-layer FP8 size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.5/3.6-35B-A3B` | 256 | 8 | 2048 | 512 | 2.0 | 4.0 | 0.81 GB |
| `Qwen3.5-122B-A10B` | 256 | 8 | 3072 | 1024 | 2.0 | 4.0 | 2.42 GB |
| `DeepSeek-V3 (671B-A37B)` | 256 | 8 | 7168 | 2048 | 2.0 | 4.0 | 11.27 GB |

## 3. Analytical HBM Byte Ceiling (Memory-Bound Decode Regime, `M_e <= 4`)

Per active expert (`d × d_ff` weights per projection, keep fraction `q = 1 - s`, input-sparsity keep fraction `q_in`):

- **Dense BF16:** `6.00 · d · d_ff` bytes (`3 × 2B`).
- **Dense FP8:** `3.00 · d · d_ff` bytes (`3 × 1B`).
- **Dense NVFP4:** `1.50 · d · d_ff` bytes (`3 × 0.5B`).
- **Park-style (`W_gate` dense FP8 + `W_up, W_down` sparse FP8):** `(1 + 2q) · d · d_ff` bytes -> at `q = 0.10`, `1.20` bytes (**`2.50×`** ceiling vs dense FP8).
- **Proxy-gated FP8 (Stage-1 INT4 `W_gate+W_up` with input keep `q_in` + Stage-2 exact FP8 all-3 at `q`):** `(q_in · 1.0 + 3q) · d · d_ff` bytes -> at `q_in = 0.50, q = 0.10`, `0.80` bytes (**`3.75×`** ceiling vs dense FP8).
- **INT4-gate-only proxy + FP8 all-3 (`q_in = 1.0`):** `(0.5 + 3q) · d · d_ff` bytes -> at `q = 0.10`, `0.80` bytes (**`3.75×`** ceiling vs dense FP8).
- **Native NVFP4 (`W_gate` dense NVFP4, `W_up, W_down` sparse NVFP4, zero extra proxy copy):** `(0.5 + 1.0·q) · d · d_ff` bytes -> at `q = 0.10`, `0.60` bytes (**`2.50×`** ceiling vs dense NVFP4, zero VRAM overhead).

Run `python3 -m sparse_moe_mm.roofline` for exact numbers per GPU (`H200`, `B200`, `RTX PRO 6000`).

## 4. Pre-Registered Feasibility Gates (Phase 0 & Phase 1)

Every gate has a lower bound, an upper bound (sanity), and a registered failing input.

1. **`G_PARITY` (Numerical Correctness):**
   - At `q = 1.0` (0% sparsity), sparse kernel output matches exact fp32 reference within `rel_l2 < 1e-2` (BF16) / `< 3e-2` (FP8).
   - *Failing input check:* Zeroing 10% of active indices must produce `rel_l2 > 0.05` and fail the gate.
2. **`G_PROXY_RECALL` (Phase 0 — Real-Weight Calibration on `Qwen3.5/3.6-35B-A3B`):**
   - Stage-1 INT4 proxy (`W_gate+W_up`, `q_in = 0.5..0.7`) recovers `>= 85%` of exact SwiGLU top-`k` energy at `90%` routed sparsity, and block-grouped `g ∈ {8, 16}` keeps WikiText-2 `ΔNLL <= 1.25×` of ungrouped `g = 1` at matched active neuron count.
   - *Refutation:* If `g >= 8` degrades `ΔNLL` by `> 2×` over `g = 1` on real weights, drop block grouping and pursue `g = 1` gather + warp-coalesced layout only.
3. **`G_LAYER_SPEEDUP` (Phase 1 — 5–8 GPU-hour Gate on `1× H200`):**
   - Single-layer sparse FP8 `grouped_mm` achieves **`>= 1.50×`** median wall-clock speedup over dense FP8 `grouped_mm` (PyTorch/CUTLASS baseline, CUDA Graph captured) at `B ∈ {16, 64, 128}` and `90%` routed sparsity on `Qwen3.5-35B-A3B` (`d=2048, d_ff=512`) and `Qwen3.5-122B-A10B` (`d=3072, d_ff=1024`).
   - *Kill switch:* If Phase 1 stays `< 1.25×` over dense FP8 after Nsight Compute (`ncu`) memory-coalescing tuning, stop before full vLLM integration.
4. **`G_E2E` (Phase 2 — Full vLLM Serving Gate):**
   - End-to-end `Qwen3.5-35B-A3B` FP8 serving speedup `>= 1.25×` over dense FP8 vLLM on `1× H200` (1024 in / 1024 out, `B = 1..128`) with average benchmark retention `>= 98%` of dense FP8.
