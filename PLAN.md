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

### Stage A — Local CPU/Mac Implementation & Unit Tests (Zero GPU Cost) — **COMPLETE**
- [x] Add **block-interleaved 2-bit/4-bit packing (`O1/O9`)**, **power-of-2 `2+2+4` bitplane split + 3-tier reconstruction (`O3/O10`)**, and **expert-level `max`-reduced Top-`K_u` (`O2`)** to `sparse_moe_mm/reference.py` + `tests/test_reference.py`.
- [x] Rewrite `sparse_moe_mm/kernels.py` with **O1–O4 + O9–O11** (128B cache-line-exact `block_d=512/256`, Marlin `0x6000`/`0x6400` FP16 bitcast, `K=128` TC, 3-Tier `2+2+4` skipping `lo4`, fused active-neuron `h_s1` zeroing, `BLOCK_D=256` SVD-r16, 100% CUDA-Graph native).
- [x] Upgrade `benchmarks/bench_layer.py` to benchmark against **honest dense `[128,128]` FP8** (`205–223 GB/s` = `100%` of L4 hardware copy ceiling `218.5 GB/s`) with 192 MB L2 flushing and 100% CUDA Graphs.

### Stage B — Cheap Native-FP8 Kernel Proving Ground on Colab `L4` (SM89, ~1.1 CU = ~$0.11) — **COMPLETE (CRUSHED ALL GATES!)**
- [x] Passed **`G1_PARITY`** (`Prog8_224 = 0.0292`, 2× tighter than dense FP8 `0.0618`; `SparseFP8_vs_Dense = 0.000000`; `zero_h_s1 = PASS`; 50% drop failing-input = `0.7451`).
- [x] Passed **100% End-to-End CUDA Graph Capture** across all pipelines (`P224`, `P224+SVD`, `3Tier+SVD`, `SpFP8`, `Dense FP8`).
- [x] Achieved **`185.4 GB/s` (85% of L4 copy ceiling)** in Stage-2 `2+2+4` and **`197 GB/s` (90% of L4 copy ceiling)** in Stage-1 `2-bit`.
- [x] Verified **L2-flushed SM89 end-to-end speedups (`zipf=0.8, cold_Me<=8`, `results/l4_iter5_*.json`):**
  - `Qwen3.5-35B-A3B` (`s=0.75`, proven `+2.26%` ΔPPL): **`1.79×–1.94×`** (`3Tier+SVD`), **`1.59×–1.84×`** (`P224+SVD`) at `B = 16..128`
  - `Qwen3.5-35B-A3B` (`s=0.80`): **`1.85×–2.09×`** (`3Tier+SVD`), **`1.67×–2.02×`** (`P224+SVD`) at `B = 16..128`
  - `Qwen3.5-122B-A10B` (`s=0.85`): **`1.97×–2.52×`** (`3Tier+SVD`), **`1.81×–2.35×`** (`P224+SVD`) at `B = 16..128`, **`2.51×`** (`P224`) at `B = 1`
  - **Key Win:** Zero-VRAM `P224+SVD` and `3Tier+SVD` **beat `+17%`-VRAM `SpFP8` by +12–19% at `B = 128`**!

### Stage C — Real-Weight Validation & Flagship `1× H200` Gate — **COMPLETE (`v0.2.0`)**
- [x] **Phase-0c Exact Per-Decode-Step `B=64` Real-Weight Ablation (`Qwen/Qwen3.5-35B-A3B-FP8`, all 40 layers, 128 users, `results/phase0c_ship_ablation.json`):**
  - `q1_rowwise_224` (`O9`, `q=1.0`): **`-0.89%` ΔPPL** (rowwise INT8 beats `[128,128]` FP8!)
  - `O9 + O2` (`s=0.75`, `SVD-r16`): **`+1.25%` ΔPPL** (**-66%** vs per-token union `+3.63%`)
  - `O10 3Tier+SVD` (`s=0.75`): **`-0.15%` ΔPPL**; **`HYBRID DN80-3Tier / FA70-8b + SVD` (default):** **`+0.68%` ΔPPL**; `s=0.80`: **`+1.57%` ΔPPL**
- [x] **Live 40-Layer Production Triton `GSM8K` Gate (`1× L40S 48 GB`, `results/gsm8k_qwen35_35b_l40s.json`):**
  - All 40 layers packed in-place in `18.6s` (deleting dense `gate_up_proj`/`down_proj` from VRAM; `34.09 -> 35.66 GiB`) and evaluated live through `Prog8SparseMoEExperts.apply()` (`N=64, B=16, enable_thinking=False`): **`79.69%` (`51/64`) vs Dense FP8 `75.00%` (`48/64`) — `106.25%` relative retention, `0` repetition loops**!
- [x] **Live `vLLM 0.34.0` Multi-GPU CUDA-Graph Gate (`L4`, `L40S`, `H200 NVL 141 GB`):**
  - **O11 Hopper SM90 (`0x50505050` `lop3.b32` `pack=4` FP8 bitcast + transposed `W@X^T` WGMMA `M=64, N=8` + fused register bitonic TopK + 2-stream CUDA-Graph overlap):** Accelerated H200 Stage-1 **`6.29×`** (`551.9 -> 87.7 µs`, **`2,906 GB/s`**, `0` spills) and Stage-2 **`2.77×`** (`2,251 GB/s`, `0` spills).
  - **H200 NVL 141 GB (`results/h200_vs_vllm034_sota.json`, vs `vLLM 0.34.0` up to `4,129 GB/s`):** `DeepSeek-V3-671B` **`1.05×–1.73×`** (`B=1..128`), `Qwen3.5-122B-A10B` **`1.01×–1.22×`** (`B=16..128`), live 48L `122B` allocation **`113.83 GiB`** (`25.98 GiB` KV headroom vs `ProxSparse` `149.83 GiB` OOM).
  - **L40S 48 GB (`results/l40s_*.json`, vs `vLLM 0.34.0` `589 GB/s`):** FULL `apply()` CUDA Graph `35B` **`1.01×–1.72×`**, `122B` **`1.71×–2.44×`**; **L4 24 GB:** `35B` **`1.50×–2.01×`**.
  - Added hardware-calibrated `should_enable_prog8_sparse(...)` oracle predicate (`>= 180 µs` dense threshold) and EP `expert_map` (`-1`) support.

### Stage D — Ecosystem Delivery & Post-RFC Milestone 1 (`v0.2.0` Live!)
- [x] Published public repo [abaybektursun/sparse-moe-mm](https://github.com/abaybektursun/sparse-moe-mm) (`v0.2.0`, 14/14 unit tests, 8 raw JSON benchmark artifacts) and opened upstream RFC [vllm-project/vllm#70046](https://github.com/vllm-project/vllm/issues/70046).
- [ ] **Post-RFC Milestone 1 (once `@mgoin` / `@robertgshaw2-redhat` confirm OOT hook vs in-tree PR):**
  - Benchmark SM90 `DeepGEMM` and `FlashInfer` CUTLASS alongside `fused_experts`.
  - Wire `oracle/fp8.py` hook and measure end-to-end `vllm serve` tok/s & TPOT on `Qwen3.5-122B-A10B-FP8` (`1× H200`) and `Qwen3.5-35B-A3B-FP8` (`1× L40S`).
  - Run full `lm-eval` (`GSM8K` 1,319 questions + `MMLU-Pro`) and extend to Blackwell SM100 (`NVFP4` + TMA `gather4`).
