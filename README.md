# sparse-moe-mm

Zero-VRAM, calibration-free quantized (**`[128,128]` FP8** on SM89/SM90 & **`1×16` NVFP4** on SM100) intra-expert sparse `grouped_mm` for fine-grained MoE serving (`Qwen3.5/3.6-35B-A3B`, `Qwen3.5-122B-A10B`, `DeepSeek-V3`), shipping as a drop-in **vLLM `FusedMoEExpertsModular`** plugin.

> **Start here:** Read [IMPACT.md](./IMPACT.md) (our live North Star impact ledger), [PLAN.md](./PLAN.md) (4-stage execution roadmap), and [SPEC.md](./SPEC.md) (falsifiable hypotheses & kill-switch gates).

## Verified Results Snapshot (SM89 L4 Native FP8 + Real `Qwen3.5-35B-A3B-FP8`)

### 1. End-to-End MoE Layer Speedup (L4 SM89, 192 MB L2-Flushed, 100% CUDA-Graph, Zipf `α=0.8`, `cold_Me <= 8`)
Measured against an honest dense `[128,128]` FP8 grouped baseline running at **`205–223 GB/s`** (`~100%` of L4's measured `218.5 GB/s` D2D copy ceiling). Stage-2 `2+2+4` hits **`185.4 GB/s` (85% of copy ceiling)** and Stage-1 `2-bit` hits **`197 GB/s` (90% of copy ceiling)**.

| Model & Policy | `B = 1` | `B = 16` | `B = 32` | `B = 64` | `B = 128` | Extra VRAM |
|---|---:|---:|---:|---:|---:|---:|
| **`Qwen3.5-35B-A3B` (`s=0.75`, `+2.26%` ΔPPL)** — `P224+SVD` | `1.40×` | `1.84×` | `1.80×` | `1.66×` | `1.59×` | `0%` proxy (+`2.7%` BF16 SVD) |
| **`Qwen3.5-35B-A3B` (`s=0.75`)** — **`3Tier+SVD`** | `1.52×`* | **`1.92×`** | **`1.94×`** | **`1.84×`** | **`1.79×`** | `0%` proxy (+`2.7%` BF16 SVD) |
| **`Qwen3.5-35B-A3B` (`s=0.80`)** — **`3Tier+SVD`** | `1.64×`* | **`2.09×`** | **`2.02×`** | **`1.96×`** | **`1.85×`** | `0%` proxy (+`2.7%` BF16 SVD) |
| **`Qwen3.5-122B-A10B` (`s=0.85`)** — `P224+SVD` | **`2.35×`** | `2.33×` | `2.32×` | `2.14×` | `1.81×` | `0%` proxy (+`1.46%` BF16 SVD) |
| **`Qwen3.5-122B-A10B` (`s=0.85`)** — **`3Tier+SVD`** | **`2.51×`**\* | **`2.52×`** | **`2.45×`** | **`2.29×`** | **`1.97×`** | `0%` proxy (+`1.46%` BF16 SVD) |
| *Baseline ref: `SpFP8` (`122B`, `+17%` extra VRAM)* | `2.37×` | `2.26×` | `2.23×` | `2.09×` | `1.76×` | **`+16.7%` (OOMs 1× H200!)** |

*\*At tiny `B=1` (`n_cold < 16`), `Prog8SparseMoEExperts` automatically skips `sorted=True` 3-Tier TopK and runs pure `P224` (`1.52×–2.51×`).*

### 2. Real-Weight Validation (`Qwen/Qwen3.5-35B-A3B-FP8`, 40 Layers, 128 Independent Users)
- **Zero-VRAM `Prog8-Dual (2+2+4)`** strictly beats 2-bit ternary (`arXiv:2610.37386`, `+17%` VRAM) on 5/5 layers & 40-layer WikiText-2 PPL at **0% extra proxy VRAM** — enabling `Qwen3.5-122B-A10B-FP8` (`123.8 GB` total with BF16 SVD-r16) to fit on **`1× H200 141 GB`** where `Tern8` (`142.4 GB`) and `Prox` (`162.6 GB`) OOM.
- **Cold-Expert-Only Zipf-Adaptive Dispatch (`M_e <= 8` sparse, `M_e > 8` exact dense):** Real 128-user routing puts `78–89%` of HBM bytes in cold experts while hot experts serve `41–61%` of tokens — cutting multi-user `B=64` ΔPPL by **54%** (`+4.87% -> +2.26%` at `s=75%`) with zero repetition loops and verbatim dense-matched code/math generation.
- **Hybrid DeltaNet vs Full-Attn Asymmetry:** Qwen3.5's 30 Gated DeltaNet MoE layers are ~5× more resilient to intra-expert sparsity than its 10 Full-Attention MoE layers.

## Core Architecture (`O1`–`O11`)

1. **Zero-VRAM `2 + 2 + 4` Progressive Bitplanes (`hi2` + `mid2` + `lo4` == bit-exact INT8):**
   - Stage 1 reads ONLY `hi2` (`0.25 B/elem`) for `W_gate + W_up` (`0.50 B` total) using **128B cache-line-exact packing (`block_d=512`) + Marlin `0x6000` FP16 bitcast + `K=128` Tensor Cores** (`197 GB/s` on L4).
   - Stage 2 reads ONLY `mid2 + lo4` (`0.75 B/elem` for Tier-1 8-bit) or **`mid2` only (`0.25 B/elem` for Tier-2 4-bit, skipping `lo4` completely!)** using **Marlin `0x6400` FP16 bitcast + fused active-neuron `h_s1` zeroing** (`185.4 GB/s` on L4).
2. **Expert-Level `max`-Reduced Top-`K_u`:** Eliminates `torch.unique` and variable-length union compaction -> **100% static CUDA-Graph capture**.
3. **Zero-Cost Stage-1 `h_s1` + Calibration-Free Weight-Only `SVD-r16`:** Recovers dropped-neuron energy in `< 4%` of layer time with zero extra `W_gate`/`W_up` reads.
4. **Drop-In vLLM `FusedMoEExpertsModular` Plugin (`sparse_moe_mm/vllm_plugin.py`):** Registered via `[project.entry-points."vllm.general_plugins"]` with load-time chunked `[128,128]` FP8 -> `2+2+4` conversion and Hybrid DeltaNet/Full-Attn sparsity policy.

## Repository Layout

- [IMPACT.md](./IMPACT.md) — Live North Star ledger tracking all 6 engineering & research impact pillars (`I1`–`I6`).
- [PLAN.md](./PLAN.md) — 4-stage execution roadmap and verified Phase-1 results.
- [SPEC.md](./SPEC.md) — Falsifiable hypotheses, exact HBM ceilings, and go/no-go gates.
- [sparse_moe_mm/kernels.py](./sparse_moe_mm/kernels.py) — O9/O10 CUDA-Graph-native Triton kernels (`2+2+4`, 3-Tier, fused `SVD-r16`, dense `[128,128]` FP8 baseline).
- [sparse_moe_mm/vllm_plugin.py](./sparse_moe_mm/vllm_plugin.py) — Drop-in vLLM `FusedMoEExpertsModular` backend (`Prog8SparseMoEExperts`) + chunked HF checkpoint packer.
- [sparse_moe_mm/reference.py](./sparse_moe_mm/reference.py) — Exact fp32 ground-truth MoE references and bitplane packing/unpacking.
- [sparse_moe_mm/roofline.py](./sparse_moe_mm/roofline.py) — CPU-runnable HBM + Tensor Core roofline across Hopper/Blackwell.
- [benchmarks/bench_layer.py](./benchmarks/bench_layer.py) — L2-flushed, 100% CUDA-Graph single-layer GPU benchmark harness.
- [tests/test_reference.py](./tests/test_reference.py) & [tests/test_vllm_plugin.py](./tests/test_vllm_plugin.py) — 12 CPU unit tests covering parity, VRAM bounds, cold/hot split, 3-Tier, SVD-r16, and plugin registration.

## Quickstart

```bash
# Run all 12 CPU unit tests (~0.7s, no GPU required)
python3 -m pytest -v

# Run analytical roofline model
python3 -m sparse_moe_mm.roofline

# Run L2-flushed CUDA-Graph single-layer GPU benchmark (SM89/SM90)
python3 benchmarks/bench_layer.py --model qwen35_122b_a10b --batches 1,16,32,64,128 --sparsity 0.85 --zipf-alpha 0.8 --cold-me-thresh 8
```
