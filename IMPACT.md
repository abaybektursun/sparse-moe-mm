# IMPACT.md — North Star Impact Ledger

> **Rule of the repo:** Every research hypothesis, kernel, benchmark, and PR in `sparse-moe-mm` must map directly to one or more Impact Pillars below. If an idea stops advancing measurable real-world or scientific impact, we pivot or kill it immediately.

## North Star Mission

Deliver the first **zero-VRAM, calibration-free, quantized (`[128,128]` FP8 & `1×16` NVFP4) intra-expert sparse `grouped_mm`** backend for fine-grained MoE serving (`Qwen3.5/3.6`, `DeepSeek-V3`, `Gemma 4 MoE`) — shipping as a drop-in **vLLM `FusedMoEExpertsModular` / SGLang plugin** that unlocks **`1.5×–2.5×` faster MoE layers** (`1.25×–1.75×` faster E2E decode) at serving batch sizes `B = 1..128`, while introducing genuinely publishable hardware-algorithm co-design for Hopper (SM90) and Blackwell (SM100).

---

## Live Impact Scorecard

| ID | Impact Pillar | Track | Concrete Target Metric | Current Status | Blocking Gate |
|---|---|---|---|---|---|
| **I1** | **Real-World Serving Speedup** | Engineering + Research | `>= 1.50×` MoE layer speedup & `>= 1.25×` E2E decode tok/s over SOTA dense FP8/NVFP4 (FlashInfer/DeepGEMM, CUDA-Graph captured, L2-flushed) at `B = 16..128` on `Qwen3.5-35B-A3B` & `122B-A10B` | `PLANNED` (Roofline verified: `2.56–3.77×` FP8 layer, `1.44–1.82×` E2E ceiling on H200; `1.93–2.42×` NVFP4 layer on B200) | `G1_LAYER_SPEEDUP`, `G2_E2E` |
| **I2** | **Drop-In Library Ecosystem Adoption** | Engineering | Native vLLM `FusedMoEExpertsModular` plugin (+ SGLang path) loading official HF `[128,128]` FP8 & `1×16` NVFP4 checkpoints with **0% extra VRAM** and **zero calibration data** | `DESIGNED` (Extension point verified: `FusedMoEExpertsModular`; `[128,128]` & `1×16` layouts mapped) | `G0_REAL_WEIGHTS`, `G2_PLUGIN` |
| **I3** | **Accuracy & Reliability Integrity** | Engineering + Research | `>= 98.5%` relative benchmark retention (WikiText-2 PPL, GSM8K, MMLU-Pro, LiveCodeBench) vs dense FP8/NVFP4; **zero repetition-loop collapses** via calibration-free top-`k` | `UNVERIFIED` (Synthetic proxy shootout passed; real `Qwen3.5-35B-A3B-FP8` test is next) | `G0_ACCURACY` |
| **I4** | **Zero-VRAM Progressive Bit-Sliced Gating (`Prog8` `4+4` gate & `Prog8-Dual` `2+6` gate+up)** | Research + Engineering | Strictly dominate Park (`2605.08575`), Prox (`2607.27591`), and 2-bit ternary (`2610.37386`) on HBM bytes (`0.50 + 2.50·q_eff`) AND VRAM (**0% overhead**) across `B = 1..256` in 8-bit serving; `Prog8-Dual` (`2+6`) resolves training-free `W_gate`-only accuracy loss (`SGA-Sparse`, Nov 2026) | `VALIDATED_ANALYTICALLY` (Roofline strictly dominates all prior work at 0% VRAM; exact `4+4` and `2+6` bit-splits defined) | `G0_PROG8_VS_TERN8` |
| **I5** | **Blackwell-Native `g=16` NVFP4 Co-Design (`P_e` + TMA `gather4` + PDL)** | Research + Engineering | Offline lossless co-activation neuron permutation (`P_e`) keeps `W_down`'s `1×16` NVFP4 micro-blocks intact for native `tl.dot_scaled` / `tcgen05.mma`, fed via SM100 hardware `TMA gather4` + PDL overlap | `PROTOTYPED_SYNTHETIC` (Exact `5.16e-7` lossless identity; 70% error reduction over naive `g=16` on synthetic clusters) | `G0_PERM_G16`, `G1_B200_NVFP4` |
| **I6** | **Production Robustness & Honest Benchmarking** | Engineering | Per-expert adaptive sparse/dense dispatch (`M_e <= M_thresh` vs `M_e > M_thresh`) eliminating Zipfian stragglers (`max M_e = 37..120`); public L2-flushed, CUDA-Graph benchmark suite vs FlashInfer/DeepGEMM | `IN_PROGRESS` (5 bugs in original `bench_layer.py` audited; Zipfian & L2 contamination proved) | `G1_ZIPF_ROBUST` |

---

## Why These Impacts Matter Right Now (Late-2026 Market & Literature Gap)

1. **BF16 sparse MoE is already solved (`arXiv:2610.37386`, Oct 2026)** — Kim & Hwang shipped calibration-free 2-bit ternary + low-rank BF16 MoE decoding in vLLM (`1.60–2.69×`). **Continuing BF16 or INT4-Prox work has near-zero marginal impact.**
2. **Production serves `[128,128]` FP8 and `1×16` NVFP4 — completely unserved by intra-expert sparsity.**
   - `Qwen3.5/3.6` reasoning models (`35B-A3B`, `122B-A10B`, `397B-A17B`) combine 3:1 Gated DeltaNet (slashing KV-cache traffic ~4×) with long Chain-of-Thought token generation — making **routed MoE 69–83% of total decode time** at serving batch sizes `B = 16..128`!
   - On single-node / workstation GPUs (`1× H200`, `1× B200`, `1× RTX PRO 6000`), VRAM is tight: `+33%` VRAM (INT4 Prox) OOMs `122B` models, and `+17%` VRAM (ternary sketch) eats KV cache. **0% VRAM overhead (`Prog8` / `Gate4`) is a killer production requirement.**
3. **Blackwell SM100 introduced two hardware primitives built for this exact workload** — neither exploited by prior sparse MoE papers:
   - **Hardware TMA `gather4`** (`cp.async.bulk.tensor.2d.tile::gather4` / Triton `async_gather`): asynchronously gathers non-contiguous active neuron rows straight from HBM into SMEM/TMEM without register-gather stalls.
   - **Programmatic Dependent Launch (PDL):** overlaps Stage-1 epilogue with Stage-2 sparse weight prefetching, erasing the 10–15 µs inter-stage launch bubble at low `B`.

---

## Non-Goals (Explicitly Out of Scope — Do Not Drift Here)

- **BF16-only sparse MoE kernels** (already solved by `arXiv:2610.37386`).
- **INT4 duplicate-copy `W_gate+W_up` Prox with Stage-1 input sparsity** (proved to lose to dense-gate Park FP8 at `B >= 128` due to input-sparsity union collapse, while burning `+33%` VRAM).
- **Hyperscale high-batch (`B >= 512`) EP serving** (where `M_e >= 16` everywhere and dense Tensor Cores already dominate).
- **Offline-calibrated static activation thresholds** (proved brittle and prone to repetition loops; use calibration-free top-`k` only).
- **Tuning SM100/SM90 kernels on SM120 (RTX PRO 6000)** (SM120 lacks TMEM and `tcgen05.mma`; use cheap GPUs for Phase-0 accuracy only, and H200/B200 for Phase-1 kernel tuning).

---

## Impact Decision Log

| Date | Decision | Impact Pillars Affected | Evidence / Rationale |
|---|---|---|---|
| 2026-11-25 | **Killed original INT4-Prox + BF16 framing; restarted around `Prog8` (0% VRAM FP8), `Gate4 + P_e` (`g=16` NVFP4 + TMA `gather4` + PDL), and vLLM `FusedMoEExpertsModular` plugin** | I1, I2, I3, I4, I5, I6 | Discovered `arXiv:2610.37386` (Oct 2026, BF16 ternary MoE in vLLM); proved Stage-1 `q_in` union collapse kills INT4 Prox at `B >= 128`; proved `Prog8` beats Park8/Prox8/Tern8 at 0% VRAM; identified SM100 TMA `gather4` + PDL + vLLM `FusedMoEExpertsModular`. |
| 2026-11-25 | **Mandated $5–$10 Phase-0 real-weight kill-switch before writing Phase-1 CUDA/Triton kernels** | I3, I4, I5 | Must verify `Prog8` accuracy, real `g=16` co-activation clustering, real `corr_out`, and real Zipf skew on `Qwen/Qwen3.5-35B-A3B-FP8` before investing GPU hours in kernel optimization. |
