### Motivation.

On fine-grained `[128,128]` FP8 MoE models (`DeepSeek-V3`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`; `E = 256, k = 8`), interactive decode (`B = 1..128`) is strictly HBM-bandwidth-bound. Upstream `vLLM 0.34.0` `fused_experts` already reaches `3,788–4,129 GB/s` on H200 (`3,894 GB/s` D2D copy ceiling) and `589 GB/s` on L40S (`614 GB/s` ceiling). Once a dense FP8 kernel sits at 95–100% of HBM bandwidth, the only way to cut MoE decode latency further is to read fewer weight bytes per layer.

Intra-expert activation sparsity is a natural fit—especially on 3:1 Gated DeltaNet hybrids where routed MoEs dominate decode time—but three practical constraints have kept prior methods out of production FP8 serving in vLLM:

1. **Duplicate proxy VRAM.** Storing a separate 2-bit (`Tern8`, `+16.7%`) or 4-bit (`ProxSparse`, `+33.3%`) gating proxy alongside FP8 `w13` either OOMs `1× H200` on `Qwen3.5-122B-A10B` (`149.8 GiB` vs `139.8 GiB`) or shrinks KV-cache headroom by more than 3×.
2. **CUDA Graph breaks and multi-user quality loss.** Taking a per-token union of active neurons requires `torch.unique` (incompatible with static CUDA Graphs) and degrades perplexity at serving batch sizes (`+3.63%` ΔPPL at `B = 64` on `Qwen3.5-35B-A3B`). Offline activation thresholds also risk repetition loops on long reasoning traces.
3. **Hopper SM90 small-M roofline inversion.** Sub-byte Triton kernels that saturate SM89 (`218–614 GB/s`) hit an ALU and `mma.sync` wall on `4.8 TB/s` H200. When `M_e <= 8` cold experts are padded to `M = 16` on the LHS (`X @ W^T`), Hopper falls back from `WGMMA` to `mma.sync`, and scalar bit-unpacking caps Stage-1 around `470 GB/s` (`0.35×` vs dense vLLM).

This RFC proposes a zero-proxy-VRAM, calibration-free sparse FP8 MoE backend (`Prog8SparseMoEExperts`) targeting `FusedMoEExpertsModular`, along with a clean opt-in integration path via `fused_moe/oracle/fp8.py` (or an OOT oracle registration hook, consistent with #37753 and #39583).

### Proposed Change.

Reference implementation, tests, and raw benchmark JSONs: https://github.com/abaybektursun/sparse-moe-mm (`v0.2.0`, tested live against `vLLM 0.34.0`, `PyTorch 2.14.0+cu130`, `Triton 3.8.0`).

#### 1. Zero-VRAM `2 + 2 + 4` Progressive Bitplanes + Cold/Hot Zipf Dispatch

At load time, `w13` (`W_gate`, `W_up`) is converted in-place from `[128,128]` FP8 into rowwise INT8 split across three 128B-cache-line-aligned bitplanes (`block_d = 512` for 2-bit, `256` for 4-bit), and the original dense `w13` is freed (`0.00%` proxy overhead):

$$q = \underbrace{64(\text{hi2} + 0.5)}_{\text{Stage 1: } 0.25\text{ B}} + \underbrace{16\,\text{mid2}}_{\text{Tier 2: }+0.25\text{ B}} + \underbrace{\text{lo4} - 32}_{\text{Tier 1: }+0.50\text{ B}} \in [-128, 127]$$

- **Cold-expert-only dispatch (`M_e <= 8` sparse, `M_e > 8` exact dense):** On 128-user `Qwen3.5-35B-A3B` routing (`B = 64`), `88.6%` of active weight bytes sit in cold experts (`M_e <= 8`) while hot experts (`M_e > 8`) carry `45.0%` of tokens. Hot experts run a 1-pass zero-VRAM `2+2+4` dense kernel rather than keeping a duplicate FP8 copy.
- **Static expert-level Top-`K_u`:** Stage 1 reads `hi2` (`0.25 B/elem`), computes coarse SwiGLU activations `h_s1`, and selects `K_u` neurons per expert via `max_m |h_s1|`. This removes `torch.unique`, keeps shapes static for CUDA Graph capture, and cuts multi-user ΔPPL by 66% (`+3.63% -> +1.25%`) because all tokens in an expert reuse the same loaded rows.
- **3-Tier Stage 2 + calibration-free `SVD-r16`:** Top `K_full` neurons read `mid2 + lo4` (bit-exact 8-bit); remaining active neurons `[K_full..K_u)` skip `lo4` (exact 4-bit); dropped neurons are approximated by a weight-only randomized `SVD-r16` of `W_down` applied to `h_s1` (`18.6s` offline for all 40 layers of 35B, no calibration data; `+0.69%` VRAM on DeepSeek-V3, `+1.46%` on 122B, `+2.72%` on 35B, or `0%` with `svd_rank=0`).
- **Hybrid layer policy:** For 3:1 DeltaNet/Full-Attention models (`Qwen3.5`), DeltaNet MoE layers default to `s = 0.80` (3-Tier) and Full-Attention MoE layers to `s = 0.70` (exact 8-bit).

#### 2. Solving the Hopper SM90 Roofline Inversion (`O11`)

On SM89 (L4, L40S), Marlin `0x6000`/`0x6400` FP16 bitcast reaches 85–90% of hardware copy bandwidth. On SM90 (H200), we dispatch to a Hopper-specific Triton path:

- **Single-instruction 4-weight FP8 unpack (`0x50505050` `lop3.b32`):** Because IEEE `float8_e4m3fn` has linear unit steps across `0x50..0x57` (`8.0..15.0`), `lop3.b32` with `pack=4` maps four 2-bit or 3-bit bitplanes straight into `float8_e4m3fn` in 1–2 instructions with zero subtraction inside the K-loop (the constant offset is subtracted once using precomputed per-token `x_sum`). Splitting `mid2` + `lo4` into two 3-bit fields (`mid3`, `lo3`) gives `max_abs_err = 0.0` across all 256 signed INT8 values: $q = 64(\text{hi2}+10) + 8(\text{mid3}+8) + (\text{lo3}+8) - 712$.
- **Transposed `W @ X^T` Hopper WGMMA (`M = 64, N = 8, P = 128`):** Moving `BLOCK_F = 64` to LHS `M` unlocks `wgmma.mma_async.m64n8k32` with `N = 8` matching `cold_me_thresh = 8`. On H200 (`122B`, `B = 64`), Stage-1 dropped from `551.9 µs` (`470 GB/s`) to **`87.7 µs` (`2,906 GB/s`, 0 register spills)**—95% of the raw HBM read ceiling (`83.4 µs`, `3,055 GB/s`). Stage-2 gate/up hits `56.6 µs` (`2,251 GB/s`, 0 spills).
- **Overhead removal:** A 1-kernel register bitonic TopK (`_fast_cold_topk_kernel`) early-exits empty/hot experts (`3.1×–4.4×` faster than `torch.topk`, 20/20 exact match). Disjoint hot experts execute on a child stream forked/joined inside the same `torch.cuda.CUDAGraph()`. All expert/neuron pointer offsets use `tl.int64` (verified on `3.76 GiB` DeepSeek-V3 tensors).

#### 3. Empirical Results vs Upstream `vLLM 0.34.0`

All GPU latencies below measure **full `Prog8SparseMoEExperts.apply()` inside `torch.cuda.CUDAGraph()` (`require_graph=True`)**, with L2 cache flushing (`192 MB` on L4/L40S, `256 MB` on H200), Zipf `α = 0.8` routing, and `0%` duplicate `w13` VRAM against `vLLM 0.34.0` `fused_experts` (`[128,128]` FP8).

**NVIDIA H200 NVL 141 GB (`3,894.3 GB/s` D2D ceiling; `results/h200_vs_vllm034_sota.json`):**

| Model (`0%` dup `w13`) | Sparsity | Extra VRAM | `B=1` | `B=16` | `B=32` | `B=64` | `B=128` | `vLLM 0.34.0` BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `DeepSeek-V3-671B` | `0.80` | `+0.69%` | `1.05×` | `1.52×` | `1.68×` | `1.44×` | `1.10×` | `4,129 GB/s` |
| `DeepSeek-V3-671B` | `0.85` | `+0.69%` | `1.11×` | `1.71×` | `1.73×` | `1.53×` | `1.23×` | `4,129 GB/s` |
| `Qwen3.5-122B-A10B` | `0.80` | `+1.46%` | `0.78×` | `1.11×` | `1.12×` | `1.04×` | `0.96×` | `3,788 GB/s` |
| `Qwen3.5-122B-A10B` | `0.85` | `+1.46%` | `0.81×` | `1.22×` | `1.22×` | `1.13×` | `1.01×` | `3,788 GB/s` |
| `Qwen3.5-35B-A3B` *(excluded by oracle)* | `0.80` | `+2.72%` | `0.71×` | `0.85×` | `0.85×` | `0.82×` | `0.69×` | `3,354 GB/s` (`31–175 µs`) |

**NVIDIA L40S 48 GB (`614.1 GB/s`) & NVIDIA L4 24 GB (`218.5 GB/s`):**

| GPU | Model & Sparsity | `B=1` | `B=16` | `B=32` | `B=64` | `B=128` | `vLLM 0.34.0` BW |
|---|---|---:|---:|---:|---:|---:|---:|
| L40S 48 GB | `Qwen3.5-35B-A3B` (`s=0.75`, full `apply()`) | `0.99×` | `1.59×` | `1.58×` | `1.52×` | `1.36×` | `589 GB/s` |
| L40S 48 GB | `Qwen3.5-35B-A3B` (`s=0.80`, full `apply()`) | `1.01×` | `1.67×` | `1.72×` | `1.69×` | `1.42×` | `589 GB/s` |
| L40S 48 GB | `Qwen3.5-122B-A10B` (`s=0.85`, kernel-level) | `1.71×` | `2.28×` | `2.44×` | `2.25×` | `1.98×` | `624 GB/s` (dense FP8) |
| L4 24 GB | `Qwen3.5-35B-A3B` (`s=0.80`, vs `vLLM 0.34.0`) | `1.56×` | `2.01×` | `2.01×` | `1.92×` | `1.67×` | `201 GB/s` |

**Where it should not run (oracle gate):** On H200, `35B` dense FP8 already completes in `31–175 µs`, so multi-stage dispatch overhead outweighs byte savings. Our oracle predicate `should_enable_prog8_sparse(...)` enables the backend only when estimated dense layer latency `>= 180 µs` and `B <= 128`—avoiding regressions on small H200 layers and compute-bound batches (`B > 128`) while activating on `35B+` on L4/L40S/A100 and `122B+`/`DeepSeek-V3` on H200.

**Live 48-layer `Qwen3.5-122B-A10B` VRAM fit on `1× H200` (`139.81 GiB`):** All 48 packed layers (`2+2+4` + `SVD-r16`) plus `4.3 GiB` non-MoE weights allocate **`113.83 GiB`** (`25.98 GiB` KV headroom), compared to `131.83 GiB` for a `+2-bit` proxy (`7.98 GiB` left) and `149.83 GiB` (OOM) for a `+4-bit` proxy.

**Real 40-layer accuracy on `Qwen/Qwen3.5-35B-A3B-FP8`:**
- *Exact per-decode-step `B = 64` WikiText-2 (128 users, `results/phase0c_ship_ablation.json`):* Dense FP8 = `53.4130`; rowwise `2+2+4` (`q = 1.0`) = `52.9350` (`-0.89%` ΔPPL); `3Tier+SVD (s=0.75)` = `53.3307` (`-0.15%`); default Hybrid `DN80-3Tier / FA70-8b + SVD` = `53.7737` (`+0.68%`); `3Tier+SVD (s=0.80)` = `54.2542` (`+1.57%`).
- *Live production Triton GSM8K (`N = 64, B = 16, max_new_tokens = 384, enable_thinking = False`, L40S 48 GB, `results/gsm8k_qwen35_35b_l40s.json`):* All 40 layers packed in-place in `18.6s` (deleting dense `gate_up_proj`/`down_proj` from VRAM; live VRAM `34.09 -> 35.66 GiB`) and run directly through `Prog8SparseMoEExperts.apply()`: **Dense FP8 = `75.00%` (`48/64`, `0` loops)** vs **Live Triton Hybrid = `79.69%` (`51/64`, `106.25%` retention, `0` loops)**.

#### 4. Proposed Integration Surface & Questions for Maintainers

`Prog8SparseMoEExperts` already implements the full `vLLM 0.34.0` `FusedMoEExpertsModular` contract (`__abstractmethods__ == frozenset()`, `TopKWeightAndReduceNoOP`, static CUDA Graphs, EP `expert_map` with `-1` non-local slots verified at `rel_L2 = 0.0123` on SM89 / `0.0390` on SM90) and ships today as an out-of-tree `vllm.general_plugins` package.

Before writing an upstream PR, we would like maintainer guidance on two design questions:

1. **OOT plugin hook vs. opt-in in-tree backend:** Given the direction in #39583 and #37753, would you prefer (a) keeping the Triton kernels out-of-tree in `sparse-moe-mm` and adding only a small custom-backend registration hook in `fused_moe/oracle/fp8.py` (opt-in via `--moe-backend prog8_sparse_fp8`), or (b) landing a self-contained opt-in `experts/prog8_sparse_fp8.py` + `oracle/fp8.py` entry directly in vLLM? Either option leaves all default dense MoE paths untouched.
2. **Pre-PR benchmark bar:** In addition to the single-layer CUDA-Graph benchmarks vs `fused_experts` and the 40-layer live Triton GSM8K check above, what specific E2E `vllm serve` (tok/s, TPOT, `DeepGEMM`/`FlashInfer` SM90 baselines, and larger GSM8K/MMLU-Pro evals) would you want in the PR description?

### Feedback Period.

Two weeks (open through 2026-12-10).

### CC List.

@mgoin @robertgshaw2-redhat

### Any Other Things.

- All single-layer H200/L40S/L4 numbers above compare against `vLLM 0.34.0` `fused_experts` (`deep_gemm` was not installed in the stock container; full SM90 `DeepGEMM`/`FlashInfer` and end-to-end `vllm serve` tok/s comparisons are planned for Milestone 1 once the oracle integration direction is agreed).
- Pure PyTorch + Triton (`triton >= 3.2`), zero external C++ build dependencies, 14 unit tests passing locally and inside `vLLM 0.34.0`.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.
