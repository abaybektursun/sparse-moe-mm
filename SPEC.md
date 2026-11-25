# SPEC — Impact-Aligned Quantized Sparse `grouped_mm` (v2 — Fresh Research Reset)

> **North Star:** See [IMPACT.md](./IMPACT.md) for the live Impact Pillars (`I1`–`I6`) and non-goals governing this spec.

## 1. Core Question (Post-`2610.37386` Reset)

Can a **zero-VRAM, calibration-free quantized intra-expert sparse `grouped_mm`** backend — supporting official **`[128,128]` block-scaled FP8** (Hopper SM90) and **`1×16` micro-scaled NVFP4** (Blackwell SM100) — beat production dense baselines (**FlashInfer CUTLASS / DeepGEMM**, CUDA-Graph captured, L2-flushed) by **`>= 1.50×` on the MoE layer** (`>= 1.25×` end-to-end decode) across serving batch sizes `B = 16..128` while retaining **`>= 98.5%`** relative accuracy?

## 2. Target Production Architectures & Verified Shapes

| Model | Attention Stack | `E` (routed) | `K` | `d_model` | `d_ff` (routed) | `d_ff` (shared) | Layers | Official Quant Formats | Routed MoE % of Decode (`B=16..128`, 2K ctx) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **`Qwen3.5/3.6-35B-A3B`** | 3:1 Gated DeltaNet + Full Attn | 256 | 8 | 2048 | 512 | 512 | 40 | FP8 `[128,128]`, NVFP4 `1×16` | **69–83%** |
| **`Qwen3.5-122B-A10B`** | 3:1 Gated DeltaNet + Full Attn | 256 | 8 | 3072 | 1024 | 1024 | 48 | FP8 `[128,128]`, NVFP4 `1×16` | **~75–85%** |
| **`DeepSeek-V3 (671B-A37B)`** | MLA | 256 | 8 | 7168 | 2048 | 2048 | 58 | FP8 `[128,128]`, NVFP4 `1×16` | ~65–80% |

## 3. Exact Analytical HBM Byte Ceilings (Including Scale Factors & Union Expansion)

Per active expert (`d_model × d_ff` elements per projection), with active-neuron keep fraction $q = 1 - s$ and effective per-expert union keep fraction across $M_e$ tokens $q_{\text{eff}}(M_e) = (1 - \rho)(1 - (1 - q)^{M_e}) + \rho q$:

### A. 8-Bit Serving Regime (Hopper H200 SM90 & Blackwell B200 SM100)
- **Dense FP8 `[128,128]`:** `3.00 B/elem` (scale overhead `< 0.03%`).
- **Park8 (Dense FP8 `W_gate` + Sparse FP8 `W_up, W_down`):** `(1.00 + 2.00 · q_eff) B/elem` — **0% VRAM overhead**.
- **Tern8 (`2610.37386` adapted to FP8: 2-bit ternary `gate+up` + Sparse FP8 all-3):** `(0.50 + 3.00 · q_eff) B/elem` — **+16.7% VRAM overhead**.
- **`Prog8` (`4+4` gate-only) & `Prog8-Dual` (`2+6` gate+up) — Zero-VRAM Progressive Bit-Slicing:**
  - **`Prog8` (`4+4` gate-only):** Store `W_gate` as exact `4-bit hi + 4-bit lo` (`1.00 B/elem`, **0% VRAM overhead**). Stage 1 reads `W_gate_hi4` (`0.50 B`); Stage 2 reads `W_gate_lo4` (`0.50·q_eff`) + `W_up, W_down^T` (`2.00·q_eff`).
  - **`Prog8-Dual` (`2+6` gate+up):** Store BOTH `W_gate` and `W_up` as `2-bit ternary/coarse + 6-bit residual` (`1.00 B/elem` each, **0% VRAM overhead**). Stage 1 reads 2-bit coarse `W_gate + W_up` (`0.25 + 0.25 = 0.50 B`, solving training-free `W_gate`-only accuracy loss noted in *SGA-Sparse*, Nov 2026); Stage 2 reads 6-bit residuals of `W_gate + W_up` (`1.50·q_eff`, 16×6-bit = 12 bytes, aligned to `g = 16`) + 8-bit `W_down^T` (`1.00·q_eff`).
  - **Identical HBM byte budget for both: `(0.50 + 2.50 · q_eff) B/elem` at 0% extra VRAM!**
  - Ceiling on H200 (`Qwen3.5-35B-A3B`, `s = 90%`, `rho = 0.35`): **`3.77×` (`B=16`), `3.16×` (`B=64`), `2.56×` (`B=128`)**.

### B. 4-Bit Serving Regime (Blackwell B200 SM100 — Native NVFP4 `E2M1` + `1×16` `E4M3` Scales)
- **Dense NVFP4 (`0.50 + 1/16 = 0.5625 B/elem` per proj):** `1.6875 B/elem`.
- **`Gate4` (Native Dense NVFP4 `W_gate` + Sparse `g=16` Permuted NVFP4 `W_up, W_down`):**
  - `(0.5625 + 1.125 · q_eff) B/elem` — **0% VRAM overhead**!
  - Strictly beats 2-bit Ternary NVFP4 (`0.50 + 1.6875 · q_eff`, `+30%` VRAM) at every `B >= 16`!
  - Ceiling on B200 (`Qwen3.5-35B-A3B`, `s = 90%`, `rho = 0.35`): **`2.42×` (`B=16`), `2.19×` (`B=64`), `1.93×` (`B=128`)**.
  - **Key Hardware Co-Design Requirement:** `W_down` (`[d_model, d_ff]`) has `1×16` NVFP4 micro-blocks along `d_ff`. Ungrouped `g = 1` breaks native `tl.dot_scaled` / `tcgen05.mma`. We apply **offline lossless co-activation neuron permutation `P_e`** (`rel_L2 = 0` at `q = 1.0`) so `g = 16` sparse blocks remain intact NVFP4 micro-blocks.

## 4. Target Kernel & Library Architecture (Impact Pillars I1, I2, I5, I6)

1. **Calibration-Free Top-`k` Selection:** Static `k_active` per expert/token -> zero calibration drift, zero repetition loops, exact static CUDA Graph grids.
2. **Per-Expert Adaptive Sparse/Dense Dispatch:**
   - Cold/medium experts (`1 <= M_e <= M_thresh`, typically `M_thresh = 8`): **Expert-Grouped Union-Gather Tensor Core Kernel** (loads each active neuron block from HBM **once** per expert; no per-pair redundant loads, no `atomic_add`).
   - Hot Zipfian experts (`M_e > M_thresh`, e.g., `M_e = 16..120`): dispatched directly to dense FP8/NVFP4 `grouped_mm` (DeepGEMM / FlashInfer CUTLASS).
3. **Blackwell SM100 Hardware Acceleration:**
   - **TMA `gather4`** (`cp.async.bulk.tensor.2d.tile::gather4` / Triton `async_gather`) for asynchronous indexed row/block loads directly into SMEM/TMEM.
   - **PDL (Programmatic Dependent Launch)** to overlap Stage-1 epilogue with Stage-2 sparse weight prefetching.
4. **Native Serving Engine Integration:**
   - Implemented as a drop-in **vLLM `FusedMoEExpertsModular`** backend plugin (`--moe-backend sparse_fp8` / `sparse_nvfp4`) compatible with CUDA Graphs, `torch.compile`, and official HF checkpoints.

## 5. Pre-Registered Phased Go/No-Go Gates (Kill-Switch First!)

### Phase 0 — $5–$10 Real-Weight Kill-Switch (1× H100/L40S/A100-80GB, ~2 GPU-hours, Pure PyTorch)
Run on official **`Qwen/Qwen3.5-35B-A3B-FP8`** (and NVFP4-simulated weights) **BEFORE** writing complex Phase-1 kernels:
- **`G0_ACCURACY` (`I3, I4`):** At `85–90%` sparsity, `Prog8` (0% VRAM) retains `>= 98.5%` relative accuracy (WikiText-2 ΔPPL `< 2%`, GSM8K drop `< 1.0%` abs) vs dense FP8, with zero repetition loops.
  - *Pivot trigger:* If `Prog8` (`W_gate`-only Stage 1) fails `< 1.0%` accuracy drop while `Tern8` (`gate+up` Stage 1) passes, immediately pivot Stage 1 to `Tern8` or bit-sliced `2+6 gate+up` (still 0% VRAM!).
- **`G0_PERM_G16` (`I5`):** Offline lossless co-activation permutation `P_e` reduces `g = 16` ΔNLL to `<= 1.35×` of ungrouped `g = 1` on real `Qwen3.5-35B-A3B` experts.
  - *Kill trigger:* If real experts have diffuse co-activation (`g = 16` permuted ΔNLL `> 2.0×` of `g = 1`), drop `g = 16` NVFP4 and focus on `g = 1` FP8 (`Prog8` / `Tern8`).
- **`G0_REAL_ROUTING` (`I1, I6`):** Measure real intra-expert union correlation `corr_out` and real routing Zipf histogram (`M_e` distribution) at `B = 16, 64, 128` on real prompts.
  - *Kill trigger:* If real `corr_out < 0.05` AND `Prog8` cannot exceed `80%` sparsity at `< 1%` accuracy loss, re-evaluate roofline before investing in Phase 1.

### Phase 1 — Single-Layer Hardware Feasibility Gate (1× H200 SM90 + 1× B200 SM100, ~6–10 GPU-hours)
- **`G1_PARITY`:** At `q = 1.0`, kernel matches exact fp32 reference within `rel_l2 < 3e-2` (FP8) / `< 6e-2` (NVFP4); registered failing input (zeroing 10% of active neurons) must fail with `rel_l2 > 0.05`.
- **`G1_LAYER_SPEEDUP` (`I1, I6`):** Full end-to-end MoE layer (Stage 1 + Top-`k` + Stage 2 + adaptive dispatch, **CUDA-Graph captured**, **L2 cache flushed (`>= 256 MB`)**, tested under BOTH uniform and Zipf `alpha = 0.8` routing) beats SOTA dense FP8/NVFP4 (**FlashInfer / DeepGEMM**) by **`>= 1.50×`** median latency at `B ∈ {16, 64, 128}`.
  - *Kill switch:* Stop before vLLM integration if speedup stays `< 1.25×` after `ncu` profiling.

### Phase 2 — vLLM `FusedMoEExpertsModular` Plugin & E2E Serving Gate (`I1, I2, I3`)
- **`G2_E2E`:** Drop-in vLLM serving of `Qwen/Qwen3.5-35B-A3B-FP8` on `1× H200` (and `-NVFP4` on `1× B200`) achieves **`>= 1.25×` E2E decode tok/s** at `B = 16..128` (1024 in / 1024 out + CoT reasoning trace) with **`>= 98.5%`** benchmark retention and **0% extra VRAM**.
