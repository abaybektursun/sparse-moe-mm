# sparse-moe-mm (`v0.1.0`)

**Zero-VRAM, calibration-free quantized (`[128,128]` FP8 on SM89/SM90 & `1×16` NVFP4 on SM100) intra-expert sparse `grouped_mm` for fine-grained MoE serving** (`Qwen3.5/3.6-35B-A3B`, `Qwen3.5-122B-A10B`, `DeepSeek-V3`), shipping as a drop-in **`vLLM v0.34+` `FusedMoEExpertsModular`** plugin.

> **Start here:** Read [IMPACT.md](./IMPACT.md) (live North Star impact ledger), [PLAN.md](./PLAN.md) (execution roadmap), and [CHANGELOG.md](./CHANGELOG.md) (`v0.1.0` release notes).

---

## Verified Results (`v0.1.0`)

### 1. Head-to-Head vs Upstream **`vLLM 0.34.0`** (`fused_experts` `[128,128]` FP8, `NVIDIA L4` SM89)
192 MB L2-flushed, 100% CUDA-Graph captured, Zipf `α = 0.8` routing (`cold_Me <= 8`, `Qwen3.5-35B-A3B`, `results/l4_vs_vllm034_35b_*.json`).

| Batch (`B`) | Upstream `vLLM 0.34.0` FP8 | Ours (`3Tier+SVD`, `s=0.75`, **`-0.15%` ΔPPL**) | Speedup vs `vLLM 0.34.0` | Ours (`3Tier+SVD`, `s=0.80`, **`+1.57%` ΔPPL**) | Speedup vs `vLLM 0.34.0` |
|---:|---:|---:|---:|---:|---:|
| **`1`** | `165.9 µs` | `110.6 µs` | **`1.50×`** | `106.5 µs` | **`1.56×`** |
| **`16`** | `1,231.9 µs` | `666.6 µs` | **`1.85×`** | `612.4 µs` | **`2.01×`** |
| **`32`** | `1,881.1 µs` | `982.0 µs` | **`1.92×`** | `937.0 µs` | **`2.01×`** |
| **`64`** | `2,463.7 µs` | `1,380.4 µs` | **`1.78×`** | `1,281.0 µs` | **`1.92×`** |
| **`128`** | `3,100.7 µs` | `2,021.4 µs` | **`1.53×`** | `1,857.5 µs` | **`1.67×`** |

### 2. Scaling to `Qwen3.5-122B-A10B` (L4 SM89 vs `223 GB/s` = `100%` HW-Copy-Ceiling Dense FP8)
`Qwen3.5-122B-A10B-FP8` is `122.0 GB`: **`sparse-moe-mm`** uses **`123.8 GB`** (`0%` extra proxy VRAM + `1.46%` BF16 `SVD-r16` -> **fits on `1× H200 141 GB`**), whereas `Tern8` (`142.4 GB`) and `Prox` (`162.6 GB`) OOM.

| `Qwen3.5-122B-A10B` (`s=0.85`) | `B = 1` | `B = 16` | `B = 32` | `B = 64` | `B = 128` | Extra VRAM |
|---|---:|---:|---:|---:|---:|---:|
| **`P224+SVD` (Exact 8-bit S2)** | **`2.35×`** | `2.33×` | `2.32×` | `2.14×` | `1.81×` | `0%` proxy (+`1.46%` BF16 SVD) |
| **`3Tier+SVD` (8b/4b S2)** | **`2.51×`**\* | **`2.52×`** | **`2.45×`** | **`2.29×`** | **`1.97×`** | `0%` proxy (+`1.46%` BF16 SVD) |
| *Ref: `SpFP8` (`+17%` extra VRAM)* | `2.37×` | `2.26×` | `2.23×` | `2.09×` | `1.76×` | **`+16.7%` (OOMs 1× H200!)** |

*\*At tiny `B = 1` (`n_cold < 16`), `Prog8SparseMoEExperts` automatically runs pure `P224` (`2.51×`). Stage-1 hits `197 GB/s` (90% of L4's `218.5 GB/s` D2D copy ceiling) and Stage-2 `2+2+4` hits `185.4 GB/s` (85%).*

### 3. Exact 40-Layer Real-Weight Validation (`Qwen/Qwen3.5-35B-A3B-FP8`, 128 Independent Users)
Evaluated per-decode-step at `B = 64` (`cold_exp = 88.6%`, `hot_tok = 45.0%`, `results/phase0c_ship_ablation.json`):

| Configuration (All 40 Layers, `cold_Me <= 8`) | Multi-User `B=64` PPL | ΔPPL vs Dense FP8 | HBM Byte Speedup |
|---|---:|---:|---:|
| Official Dense `[128,128]` FP8 Reference | `53.4130` | `0.00%` | `1.00×` |
| **`q1_rowwise_224`** (`O9` rowwise `2+2+4`, `q=1.0`, no sparsity) | **`52.9350`** | **`-0.89%`** | `1.00×` |
| Legacy per-token union (`g=128`, `s=0.75`, `SVD-r16`) | `55.3526` | `+3.63%` | `1.64×` |
| **`O9` Rowwise + `O2` Expert-Level Top-`K_u`** (`s=0.75`, `SVD-r16`) | `54.0782` | **`+1.25%`** (-66%!) | `1.67×` |
| **`O10` `3Tier+SVD`** (`s=0.75`) | **`53.3307`** | **`-0.15%`** | **`1.86×`** |
| **`HYBRID DN80-3Tier / FA70-8b + SVD` (Default Plugin Policy!)** | **`53.7737`** | **`+0.68%`** | **`1.89×`** |
| **`O10` `3Tier+SVD`** (`s=0.80`) | `54.2542` | **`+1.57%`** | **`2.03×`** |

---

## Core Architecture (`O1`–`O11`)

1. **Zero-VRAM `2 + 2 + 4` Progressive Bitplanes (`hi2` + `mid2` + `lo4` == bit-exact rowwise INT8):**
   - **Stage 1:** Reads ONLY `hi2` (`0.25 B/elem` × 2 = `0.50 B`) using **128B cache-line-exact packing (`block_d = 512`) + Marlin `0x6000` (`512.0`, ULP `0.5`) FP16 bitcast + `K = 128` Tensor Cores**.
   - **Stage 2 (3-Tier `2+2+4`):** Tier-1 neurons read `mid2 + lo4` (`0.75 B` -> bit-exact 8-bit via Marlin `0x6400` bitcast); Tier-2 neurons skip `lo4` completely and read ONLY `mid2` (`0.25 B` -> exact 4-bit!). Zeroing active indices in `h_s1` is fused into Stage-2's epilogue.
2. **Expert-Level `max`-Reduced Top-`K_u` (`O2`):** Eliminates `torch.unique` for **100% static CUDA Graphs** AND cuts multi-user ΔPPL by **66%** because every token in an expert uses all `K_u` HBM-loaded neurons for free.
3. **Zero-Cost Stage-1 `h_s1` + Calibration-Free Weight-Only `SVD-r16`:** Computed offline in `18.6s` for all 10,240 experts with zero calibration data; runs in `< 4%` of layer latency.
4. **Cold-Expert-Only Zipf-Adaptive Dispatch (`M_e <= 8` sparse, `M_e > 8` exact dense) + Hybrid DeltaNet/Full-Attn Policy:** Automatically protects hot Zipfian experts and sensitive Full-Attention MoE layers (`Qwen3.5` 3:1 hybrid).

---

## Install & Quickstart

```bash
# Editable install (auto-registers `prog8_sparse_fp8` under `vllm.general_plugins`)
pip install -e .

# Run all 12 CPU unit tests (~1s, no GPU required)
python3 -m pytest -v

# Run standalone drop-in vLLM plugin quickstart (CPU or CUDA SM89/SM90)
python3 examples/quickstart_vllm_plugin.py

# Head-to-head L2-flushed CUDA-Graph benchmark vs upstream vLLM v0.34+ FP8
python3 benchmarks/bench_vs_vllm.py --sparsity 0.80
```

## Repository Layout

- [sparse_moe_mm/kernels.py](./sparse_moe_mm/kernels.py) — `O9`/`O10` CUDA-Graph-native Triton kernels (`2+2+4`, 3-Tier, fused `SVD-r16`, dense `[128,128]` FP8 baseline).
- [sparse_moe_mm/vllm_plugin.py](./sparse_moe_mm/vllm_plugin.py) — Drop-in `vLLM v0.34+` `FusedMoEExpertsModular` backend (`Prog8SparseMoEExperts`) + chunked HF checkpoint packer (`pack_hf_fp8_experts`).
- [sparse_moe_mm/reference.py](./sparse_moe_mm/reference.py) — Exact fp32 ground-truth MoE references and bitplane packing/unpacking.
- [sparse_moe_mm/roofline.py](./sparse_moe_mm/roofline.py) — Analytical HBM + Tensor Core roofline across Hopper/Blackwell.
- [benchmarks/bench_vs_vllm.py](./benchmarks/bench_vs_vllm.py) — Live `vLLM v0.34+` integration & head-to-head L2-flushed benchmark.
- [benchmarks/bench_layer.py](./benchmarks/bench_layer.py) — Standalone L2-flushed CUDA-Graph single-layer benchmark (`35B` & `122B`).
- [benchmarks/phase0c_ship_ablation.py](./benchmarks/phase0c_ship_ablation.py) — Exact 40-layer per-decode-step `B=64` real-weight ablation on `Qwen/Qwen3.5-35B-A3B-FP8`.
- [examples/quickstart_vllm_plugin.py](./examples/quickstart_vllm_plugin.py) — Minimal drop-in plugin usage example.
- [tests/test_reference.py](./tests/test_reference.py) & [tests/test_vllm_plugin.py](./tests/test_vllm_plugin.py) — 12 CPU unit tests.
