# PLAN.md — Optimized Execution Plan (Post-Phase-0b Reflection)

> Aligned 1-to-1 with [IMPACT.md](./IMPACT.md) (`I1`–`I6`) and grounded in real `Qwen/Qwen3.5-35B-A3B-FP8` measurements ([results/phase0_qwen35_results.json](./results/phase0_qwen35_results.json), [results/phase0b_asymmetric_results.json](./results/phase0b_asymmetric_results.json)).

---

## 1. Key Post-Phase-0b Strategic Insights

1. **The Killer Practical Proof Point (`1× H200 141 GB` serving `Qwen3.5-122B-A10B-FP8`):**
   - Dense FP8 baseline: `122.0 GB` (`19.0 GB` KV cache)
   - **`sparse-moe-mm` (`Prog8-Dual 2+2+4` + `SVD-r16`):** **`123.6 GB` (`17.4 GB` KV cache — FITS!)**
   - `Tern8` (`arXiv:2610.37386` adapted to FP8): `142.4 GB` (**OOM!**)
   - `Prox` (`arXiv:2607.27591` INT4 copy): `162.6 GB` (**OOM!**)
2. **Real Multi-User HBM Ceiling (`Qwen3.5-35B-A3B-FP8`, measured `E_act`, `Cold%`, `corr_out = 0.24`):**
   - `B = 1`: `2.88×` (`s=80%`) / **`2.96×` (3-Tier `2+2+4`)** / `3.28×` (`s=85%`)
   - `B = 16`: `2.27×` (`s=80%`) / **`2.34×` (3-Tier `2+2+4`)** / `2.62×` (`s=85%`)
   - `B = 64`: `1.79×` (`s=80%`) / **`1.85×` (3-Tier `2+2+4`)** / `2.03×` (`s=85%`)
   - `B = 128`: `1.58×` (`s=80%`) / **`1.62×` (3-Tier `2+2+4`)** / `1.74×` (`s=85%`)
   - **Implication:** At `B = 64..128`, the margin above `1.50×` is real *only if* our kernels hit `>= 75%` HBM efficiency with `< 5 µs` metadata overhead.

---

## 2. Four Kernel & Algorithmic Optimizations Before Next GPU Run

| # | Optimization | Problem Solved | Impact |
|---|---|---|---|
| **O1** | **Block-Interleaved 2-bit Packing** (`[j, j+32, j+64, j+96]` per byte) + **BF16 Tensor Core `tl.dot`** in Stage 1 | Eliminates 4× strided `x` global loads and 2× slower TF32 MMA in v2 `kernels.py` | Doubles Stage-1 Tensor Core rate & restores 100% coalesced 64B `x` loads |
| **O2** | **Expert-Level `max`-Reduced Top-`K_u`** (`exp_score[e, f] = max_t |h_s1[t, f]|`) | Eliminates Python/GPU `torch.unique`, `token_mask`, and variable-length union compaction | `< 3 µs` metadata, **100% static CUDA Graph**, AND strictly higher accuracy at identical HBM bytes (every token uses all `K_u` loaded neurons) |
| **O3** | **Power-of-2 `2 + 2 + 4` Progressive Bitplanes** (`hi2` 0.25B, `mid2` 0.25B, `lo4` 0.50B = exact 8-bit) + **3-Tier Precision** | Eliminates unaligned 6-bit packing AND eliminates `+16.7%` duplicate FP8 VRAM; enables 8-bit top-10% + 4-bit next-15% + SVD-r16 rest | **0% extra proxy VRAM** (`122B` fits on `1× H200`), `s=75%` accuracy (`+2.26%` ΔPPL) at `s=82%` bandwidth (`2.34×` at `B=16`, `1.85×` at `B=64`) |
| **O4** | **Native FP8 `1×128` Tensor Core `tl.dot`** in Stage 2 + **Fused `SVD-r16` Epilogue** | Avoids FP8->FP32 register bloat (4× register savings) and eliminates extra launch latency for SVD recovery | Unlocks `>= 75%` HBM efficiency on SM89 (L4) and SM90 (H100/H200) |

---

## 3. Four-Stage Execution Roadmap

### Stage A — Local CPU/Mac Implementation & Unit Tests (Zero GPU Cost)
- [ ] Add **block-interleaved 2-bit packing**, **power-of-2 `2+2+4` bitplane split + 3-tier reconstruction**, and **expert-level `max`-reduced Top-`K_u`** to `sparse_moe_mm/reference.py` + `tests/test_reference.py`.
- [ ] Rewrite `sparse_moe_mm/kernels.py` with **O1–O4** (100% CUDA-Graph native Stage 1 + Expert Top-`K_u` + Stage 2 + SVD-r16 + Cold/Hot Adaptive Dispatch).
- [ ] Upgrade `benchmarks/bench_layer.py` to benchmark against **real dense FP8** (Triton/CUTLASS `[128,128]` FP8) with L2 flushing and CUDA Graphs.

### Stage B — Cheap Native-FP8 Kernel Proving Ground on Colab `L4` (SM89, ~$0.15–$0.30)
- [ ] Spin up Colab `L4` (instant availability, SM89 native FP8 Tensor Cores, 0.30 TB/s memory-bound across `B = 1..64`).
- [ ] Pass **`G1_PARITY`**: exact fp32 parity (`rel_l2 < 3e-2` at `q=1.0`, failing-input check passes) for both `Prog8-Dual` Stage 1, `2+2+4` Stage 2, and fused `SVD-r16`.
- [ ] Pass **100% End-to-End CUDA Graph Capture** (zero host sync, zero allocations).
- [ ] Measure L2-flushed single-layer wall-clock speedup vs dense `[128,128]` FP8 on `Qwen3.5-35B-A3B` & `Qwen3.5-122B-A10B` shapes; iterate tile sizes & warp counts until HBM efficiency `>= 75%`.

### Stage C — Flagship Hardware Gate on `1× H200` (SM90, ~4–6 GPU-hours)
- [ ] Benchmark full L2-flushed CUDA-Graph MoE layer vs **FlashInfer CUTLASS** and **DeepGEMM** dense FP8 `grouped_mm` across `B = 1, 16, 32, 64, 128` under real Zipf routing (`Qwen3.5-35B-A3B` & `Qwen3.5-122B-A10B`).
- [ ] Run `ncu` roofline profiling (`dram__throughput`, occupancy, register pressure) to confirm **`G1_LAYER_SPEEDUP >= 1.50×`**.
- [ ] Run full accuracy suite (`lm-eval`: WikiText-2, GSM8K, MMLU-Pro, LiveCodeBench) verifying **`>= 98.5%`** retention for `Cold-Expert-Only Prog8-Dual (2+2+4) + SVD-r16 + DeltaNet/FullAttn layer policy`.

### Stage D — Drop-In Library Delivery & Release (`I2`)
- [ ] Implement `sparse_moe_mm/vllm_plugin.py` subclassing vLLM's **`FusedMoEExpertsModular`** (`--moe-backend prog8_sparse_fp8`), with load-time in-place `[128,128]` FP8 -> `(hi2, mid2, lo4)` repacking (`0%` extra VRAM) + cached weight-only `SVD-r16`.
- [ ] Verify **`G2_E2E >= 1.25×`** E2E tok/s on `Qwen/Qwen3.5-35B-A3B-FP8` & `Qwen/Qwen3.5-122B-A10B-FP8` (1024/1024 + CoT reasoning) on `1× H200`.
- [ ] Publish pip-installable plugin + reproducible benchmark artifacts + open vLLM RFC/PR + technical paper draft.
