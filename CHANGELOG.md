# Changelog

All notable milestones of `sparse-moe-mm` are documented here.

## [0.2.0] — 2026-11-26

### Added
- **O11 Hopper SM90 (`H100`/`H200`) Breakthrough Kernels (`sparse_moe_mm/kernels.py`):**
  - **`0x50505050` `lop3.b32` (`pack=4`) IEEE `float8_e4m3fn` Bitcast:** Unpacks 4 weights directly into exact `float8_e4m3fn` (`{8,9,10,11} = (hi2+0.5)+9.5`, exact `0.0` reconstruction error across all 8 bits via `64*hi2 + 8*mid3 + lo3 - 712`) in 1–2 instructions with zero K-loop subtraction.
  - **Transposed `W @ X^T` Hopper `WGMMA (M=64, N=8, P=128)`:** Accelerates Stage-1 on H200 by **`6.29×`** (`551.9 µs -> 87.7 µs`, **`2,906.1 GB/s`** = 95% raw HBM ceiling, `0` register spills) and Stage-2 by **`2.77×`** (`2,250.6 GB/s`, `0` spills).
  - **Fused 1-Kernel Register Bitonic TopK (`_fast_cold_topk_kernel`):** Early-exits empty/hot experts (`3.1×–4.4×` faster than `torch.topk`, 20/20 exact match).
  - **2-Stream CUDA-Graph Fork/Join Overlap & Fused Gather-Quantize:** Runs disjoint Hot Dense experts concurrently on a child stream captured inside the exact same `torch.cuda.CUDAGraph()`, eliminates `x_pairs` copy on SM90, and uses `int16` routing sort (`1.9×` faster).
  - **64-Bit Expert/Neuron Pointer Offsets (`tl.int64`):** Verified on `3.76 GiB` `DeepSeek-V3-671B` expert tensors.
- **Strict `0.00%` Duplicate-`w13` CUDA-Graph `apply()` + Expert Parallelism (`sparse_moe_mm/vllm_plugin.py`):**
  - Eliminated `torch.bincount` (replaced with static `scatter_add_`) and fused 1-pass zero-VRAM `_dense_eg_gate_up_224_kernel` / `_dense_sm90_224_gu_kernel` so `Prog8SparseMoEExperts.apply()` captures 100% inside `torch.cuda.CUDAGraph()` (`require_graph=True`) with strict `0.00%` duplicate `w13` VRAM.
  - Native vLLM Expert Parallelism `expert_map` (`-1` non-local experts) support (`rel_L2 = 0.0123` on SM89 / `0.0390` on SM90).
  - Added vLLM `oracle/fp8.py`-style `prepare_fp8_moe_layer_for_prog8(...)` and hardware-calibrated `should_enable_prog8_sparse(...)` dispatch predicate.
- **Multi-GPU SOTA Proof & Upstream vLLM RFC (`docs/VLLM_RFC.md`, `benchmarks/bench_h200_sota.py`):**
  - **Hopper `1× H200 NVL 141 GB` (`3,894 GB/s` D2D ceiling, 256 MB L2 flush, vs `vLLM 0.34.0` up to `4,129 GB/s`):** `DeepSeek-V3-671B` **`1.05×–1.68×`** (`s=0.80`) / **`1.11×–1.73×`** (`s=0.85`) across `B=1..128`; `Qwen3.5-122B-A10B` **`1.04×–1.22×`** across `B=16..128`; live 48-layer `122B` allocation proven at **`113.83 GiB`** (`25.98 GiB` KV-cache headroom vs `ProxSparse` `149.83 GiB` OOM).
  - **`NVIDIA L40S 48 GB` (`614 GB/s` D2D ceiling, vs `vLLM 0.34.0` `589 GB/s`):** FULL `apply()` CUDA Graph `Qwen3.5-35B-A3B` **`1.01×–1.72×`** (`s=0.80`) across `B=1..128`; `Qwen3.5-122B-A10B` **`1.71×–2.44×`** (`s=0.85`).
  - **Live 40-Layer Production Triton `GSM8K` Exact-Match Gate (`results/gsm8k_qwen35_35b_l40s.json`):** Real `Qwen/Qwen3.5-35B-A3B-FP8` packed in-place in `18.6s` (`0%` duplicate `w13` VRAM) and evaluated live through `Prog8SparseMoEExperts.apply()` (`N=64, B=16`): **`79.69%` (`51/64`) vs Dense FP8 `75.00%` (`48/64`) — `106.25%` relative retention, `0` repetition loops**!
  - Expanded unit test suite to 14 tests (`tests/test_reference.py`, `tests/test_vllm_plugin.py`).

## [0.1.0] — 2026-11-25

### Added
- **Zero-VRAM `2 + 2 + 4` Progressive Bitplane Triton Kernels (`sparse_moe_mm/kernels.py`):**
  - `O9`: 128B cache-line-exact packing (`block_d=512` for 2-bit `hi2`/`mid2`, `block_d=256` for 4-bit `lo4`) + Marlin `0x6000`/`0x6400` FP16 bitcast + `K=128` Tensor Core `tl.dot` (`197 GB/s` Stage-1, `185.4 GB/s` Stage-2 on SM89 L4 — 85–90% of hardware D2D copy ceiling).
  - `O10`: 3-Tier `2+2+4` Stage-2 precision (Tier-1 8-bit `mid2+lo4`, Tier-2 4-bit `mid2` skipping `lo4` completely) with fused active-neuron `h_s1` zeroing for mathematically exact dropped-neuron `SVD-r16` recovery.
  - `O2`: Expert-level `max`-reduced Top-`K_u` selection enabling 100% static CUDA Graph capture.
- **Drop-In `vLLM v0.34.0` `FusedMoEExpertsModular` Plugin (`sparse_moe_mm/vllm_plugin.py`):**
  - Concrete implementation of all `vLLM v0.34.0` `FusedMoEExpertsModular` abstract methods (`activation_format`, `workspace_shapes`, `finalize_weight_and_reduce_impl`, `_supports_*`, `apply`), auto-registered via `[project.entry-points."vllm.general_plugins"]`.
  - Chunked load-time HF `[128,128]` FP8 -> `2+2+4` + calibration-free weight-only `SVD-r16` packer (`0%` extra proxy VRAM; `+1.46%` BF16 SVD on `Qwen3.5-122B-A10B`).
  - Default **Hybrid DeltaNet/Full-Attn** policy (`DeltaNet s=0.80` 3-Tier + `Full-Attn s=0.70` 8-bit + `cold_Me <= 8` Zipf-adaptive dispatch + `SVD-r16`).
- **Benchmarks, Examples & Tests:**
  - `benchmarks/bench_vs_vllm.py`: Verified live against upstream `vLLM 0.34.0` (`1.50×–2.01×` faster than `fused_experts` FP8 across `B=1..128` on SM89 L4; live GPU `apply(q=1.0)` parity `rel_L2 = 0.0126`).
  - `benchmarks/phase0c_ship_ablation.py`: Verified across all 40 layers of real `Qwen/Qwen3.5-35B-A3B-FP8` at exact per-decode-step `B=64` (`-0.15%` ΔPPL at `s=0.75`, `+0.68%` ΔPPL for Hybrid `DN80/FA70`, `+1.57%` ΔPPL at `s=0.80`).
  - `examples/quickstart_vllm_plugin.py` and 12 CPU unit tests (`tests/test_reference.py`, `tests/test_vllm_plugin.py`).
