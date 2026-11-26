# Changelog

All notable milestones of `sparse-moe-mm` are documented here.

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
