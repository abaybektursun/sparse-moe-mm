# sparse-moe-mm (`v0.2.0`)

**Zero-VRAM, calibration-free quantized (`[128,128]` FP8 on SM89/SM90 & `1×16` NVFP4 on SM100) intra-expert sparse `grouped_mm` for fine-grained MoE serving** (`DeepSeek-V3-671B`, `Qwen3.5-122B-A10B`, `Qwen3.5/3.6-35B-A3B`), shipping as a drop-in **`vLLM v0.34+` `FusedMoEExpertsModular`** plugin.

> **Start here:** Read the live upstream RFC ([vllm-project/vllm#70046](https://github.com/vllm-project/vllm/issues/70046) / [docs/VLLM_RFC.md](./docs/VLLM_RFC.md)), [IMPACT.md](./IMPACT.md) (live North Star impact ledger), [PLAN.md](./PLAN.md) (execution roadmap), and [CHANGELOG.md](./CHANGELOG.md) (`v0.2.0` release notes).

---

## Verified Multi-GPU Results vs Upstream `vLLM 0.34.0` (`v0.2.0`)

All measurements below run **FULL end-to-end `Prog8SparseMoEExperts.apply()` inside `torch.cuda.CUDAGraph()` (`require_graph=True`)**, with **L2 cache flushing** (`192–256 MB`), **Zipf `α = 0.8` routing** (`cold_Me <= 8`), and **strict `0.00%` duplicate `w13` VRAM** against live upstream **`vLLM 0.34.0` `fused_experts` (`[128,128]` FP8)**.

### 1. Hopper `1× NVIDIA H200 NVL 141 GB` (SM90, `3,894.3 GB/s` D2D Ceiling)
Powered by **O11** (`0x50505050` `lop3.b32` `pack=4` FP8 bitcast + transposed `W @ X^T` WGMMA `M=64, N=8` hitting **`2,906 GB/s`** Stage-1 & **`2,251 GB/s`** Stage-2 + fused register bitonic TopK + 2-stream CUDA-Graph Cold/Hot overlap; `results/h200_vs_vllm034_sota.json`):

| Model (Strict `0%` Dup `w13`) | Sparsity | Extra VRAM | `B = 1` | `B = 16` | `B = 32` | `B = 64` | `B = 128` | Upstream `vLLM 0.34.0` BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`DeepSeek-V3-671B`** (`d=7168, d_ff=2048`) | `s = 0.80` | `+0.69%` | **`1.05×`** | **`1.52×`** | **`1.68×`** | **`1.44×`** | **`1.10×`** | **`4,128.7 GB/s`** |
| **`DeepSeek-V3-671B`** (`d=7168, d_ff=2048`) | `s = 0.85` | `+0.69%` | **`1.11×`** | **`1.71×`** | **`1.73×`** | **`1.53×`** | **`1.23×`** | **`4,128.7 GB/s`** |
| **`Qwen3.5-122B-A10B`** (`d=3072, d_ff=1024`) | `s = 0.80` | `+1.46%` | `0.78×` | **`1.11×`** | **`1.12×`** | **`1.04×`** | `0.96×` | **`3,788.1 GB/s`** |
| **`Qwen3.5-122B-A10B`** (`d=3072, d_ff=1024`) | `s = 0.85` | `+1.46%` | `0.81×` | **`1.22×`** | **`1.22×`** | **`1.13×`** | **`1.01×`** | **`3,788.1 GB/s`** |

- **Live 48-Layer `Qwen3.5-122B-A10B` VRAM Fit Proof on `1× H200 139.81 GiB`:** **`113.83 GiB`** allocated (**`25.98 GiB`** KV-cache headroom, `3.25×` more KV cache than `Tern8` `131.83 GiB`, while `ProxSparse` `149.83 GiB` physically OOMs `1× H200`!).
- **Hardware-Calibrated Oracle Predicate (`should_enable_prog8_sparse`):** Automatically enables `Prog8` when dense layer latency `>= 180 µs` and `B <= 128`, avoiding overhead on tiny `35B` layers on `4.8 TB/s` H200 (`31–175 µs` dense) while accelerating `122B+`/`DeepSeek-V3` on H200 and all `35B+` models on L4/L40S/A100.

### 2. `NVIDIA L40S 48 GB` (`614.1 GB/s`) & `NVIDIA L4 24 GB` (`218.5 GB/s`, SM89)
*(Sources: `results/l40s_full_apply_vs_vllm034_*.json`, `results/l40s_122b_s85.json`, `results/l4_vs_vllm034_35b_*.json`)*

| GPU | Model & Config | `B = 1` | `B = 16` | `B = 32` | `B = 64` | `B = 128` | Upstream `vLLM 0.34.0` BW |
|---|---|---:|---:|---:|---:|---:|---:|
| **`L40S 48 GB`** | `Qwen3.5-35B-A3B` (`s=0.75`, FULL `apply()`) | `0.99×` | **`1.59×`** | **`1.58×`** | **`1.52×`** | **`1.36×`** | `589.0 GB/s` (`96%` ceiling) |
| **`L40S 48 GB`** | `Qwen3.5-35B-A3B` (`s=0.80`, FULL `apply()`) | **`1.01×`** | **`1.67×`** | **`1.72×`** | **`1.69×`** | **`1.42×`** | `589.0 GB/s` (`96%` ceiling) |
| **`L40S 48 GB`** | `Qwen3.5-122B-A10B` (`s=0.85`) | **`1.71×`** | **`2.28×`** | **`2.44×`** | **`2.25×`** | **`1.98×`** | `624.4 GB/s` (dense FP8) |
| **`L4 24 GB`** | `Qwen3.5-35B-A3B` (`s=0.75`, vs `vLLM 0.34.0`) | **`1.50×`** | **`1.85×`** | **`1.92×`** | **`1.78×`** | **`1.53×`** | `201.3 GB/s` (`92%` ceiling) |
| **`L4 24 GB`** | `Qwen3.5-35B-A3B` (`s=0.80`, vs `vLLM 0.34.0`) | **`1.56×`** | **`2.01×`** | **`2.01×`** | **`1.92×`** | **`1.67×`** | `201.3 GB/s` (`92%` ceiling) |

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

**Live Production Triton 40-Layer `GSM8K` Exact-Match Gate (`results/gsm8k_qwen35_35b_l40s.json`, `1× L40S 48 GB`):**
All 40 layers of `Qwen/Qwen3.5-35B-A3B-FP8` packed **in-place in `18.6s`** (deleting dense `gate_up_proj`/`down_proj` from VRAM) and evaluated live through `Prog8SparseMoEExperts.apply()` (`N = 64, B = 16`):
- Official Dense `[128,128]` FP8: **`75.00%` (`48/64`)**, `0` repetition loops (`34.09 GiB` VRAM)
- **Live Triton `HYBRID DN80-3Tier / FA70-8b + SVD-r16`**: **`79.69%` (`51/64`, `106.25%` relative retention!)**, **`0` repetition loops** (`35.66 GiB` live VRAM, `0%` duplicate `w13`)!

---

## Core Architecture (`O1`–`O11`)

1. **Zero-VRAM `2 + 2 + 4` Progressive Bitplanes (`hi2` + `mid2` + `lo4` == bit-exact rowwise INT8):**
   - **SM89 (L4 / L40S):** 128B cache-line-exact packing (`block_d = 512/256`) + Marlin `0x6000`/`0x6400` FP16 bitcast + `K = 128` Tensor Cores (`85–90%` copy ceiling).
   - **SM90 (H100 / H200 — `O11`):** **`0x50505050` `lop3.b32` (`pack=4`) 1-instruction 4-weight IEEE `float8_e4m3fn` bitcast** (`max_abs_err = 0.0` across all 8 bits via `64*hi2 + 8*mid3 + lo3 - 712`) + **transposed `W @ X^T` Hopper `WGMMA (M=64, N=8, P=128)`** (`6.29×` Stage-1 speedup on H200 -> **`2,906 GB/s`**, `0` spills!).
2. **Fused Register Bitonic TopK + 2-Stream CUDA-Graph Overlap:** `_fast_cold_topk_kernel` (`3.1×–4.4×` faster than `torch.topk`) + child-stream Hot Dense execution captured inside the same `torch.cuda.CUDAGraph()`, with native vLLM Expert Parallelism `expert_map` (`-1` non-local experts) and 64-bit addressing for `>2 GiB` expert tensors.
3. **Zero-Cost Stage-1 `h_s1` + Calibration-Free Weight-Only `SVD-r16`:** Computed offline in `<20s` with zero calibration data (`+0.69%` VRAM on DeepSeek-V3, `+1.46%` on 122B, `+2.72%` on 35B; `0.00%` if `svd_rank=0`).
4. **Cold-Expert-Only Zipf-Adaptive Dispatch (`M_e <= 8` sparse, `M_e > 8` exact zero-VRAM `2+2+4` dense) + Hybrid DeltaNet/Full-Attn Policy.**

---

## Install & Quickstart

```bash
# Editable install (auto-registers `prog8_sparse_fp8` under `vllm.general_plugins`)
pip install -e .

# Run all 14 CPU unit tests (~1.2s, no GPU required)
python3 -m pytest -v

# Run standalone drop-in vLLM plugin quickstart (CPU or CUDA SM89/SM90)
PYTHONPATH=. python3 examples/quickstart_vllm_plugin.py

# Head-to-head L2-flushed FULL apply() CUDA-Graph benchmark vs upstream vLLM v0.34+ FP8
python3 benchmarks/bench_vs_vllm.py --sparsity 0.80

# Flagship Hopper H200 SOTA benchmark (35B, 122B, DeepSeek-V3 + live 48L 122B VRAM fit)
python3 benchmarks/bench_h200_sota.py
```

## Repository Layout

- [docs/VLLM_RFC.md](./docs/VLLM_RFC.md) — Upstream `vllm-project/vllm` RFC with complete L4, L40S, H200, VRAM, and real-weight accuracy evidence.
- [sparse_moe_mm/kernels.py](./sparse_moe_mm/kernels.py) — `O9`/`O10` SM89 & `O11` SM90 Hopper `0x50505050` WGMMA FP8 CUDA-Graph Triton kernels + register bitonic TopK + fused gather-quantize.
- [sparse_moe_mm/vllm_plugin.py](./sparse_moe_mm/vllm_plugin.py) — Drop-in `vLLM v0.34+` `FusedMoEExpertsModular` backend (`Prog8SparseMoEExperts`), `prepare_fp8_moe_layer_for_prog8`, and `should_enable_prog8_sparse`.
- [sparse_moe_mm/reference.py](./sparse_moe_mm/reference.py) — Exact fp32 ground-truth MoE references and bitplane packing/unpacking.
- [sparse_moe_mm/roofline.py](./sparse_moe_mm/roofline.py) — Analytical HBM + Tensor Core roofline across Hopper/Blackwell.
- [benchmarks/bench_h200_sota.py](./benchmarks/bench_h200_sota.py) — Hopper H200 SOTA benchmark (`35B`, `122B`, `DeepSeek-V3`, live 48L `122B` VRAM proof).
- [benchmarks/bench_vs_vllm.py](./benchmarks/bench_vs_vllm.py) — Live `vLLM v0.34+` FULL `apply()` CUDA-Graph + EP `expert_map` benchmark.
- [benchmarks/eval_gsm8k_real.py](./benchmarks/eval_gsm8k_real.py) — Batched 40-layer real-weight `openai/gsm8k` exact-match & repetition-loop gate.
- [benchmarks/phase0c_ship_ablation.py](./benchmarks/phase0c_ship_ablation.py) — Exact 40-layer per-decode-step `B=64` real-weight ablation on `Qwen/Qwen3.5-35B-A3B-FP8`.
- [examples/quickstart_vllm_plugin.py](./examples/quickstart_vllm_plugin.py) — Minimal drop-in plugin usage example.
- [tests/test_reference.py](./tests/test_reference.py) & [tests/test_vllm_plugin.py](./tests/test_vllm_plugin.py) — 14 unit tests.
