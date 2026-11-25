# sparse-moe-mm

Quantized (**FP8 / NVFP4**) proxy-gated all-3-matrix sparse `grouped_mm` for fine-grained Mixture-of-Experts (MoE) inference (`Qwen3.5/3.6-35B-A3B`, `Qwen3.5-122B-A10B`, `DeepSeek-V3`).

## Why This Exists

Fine-grained MoEs (`E=256, K=8`) route $M_e = B \cdot K / E = B / 32$ tokens per expert on average, keeping routed experts memory-bandwidth bound (`M_e = 1..4`) at serving batch sizes `B = 32..128` where dense activation sparsity saturates.

Prior work leaves three concrete gaps:
1. **Park et al. (2026, `arXiv:2605.08575`)** shipped BF16 intra-expert sparse `up+down` Triton kernels in vLLM (`1.55–2.5×` MoE layer, `1.20×` end-to-end on H200), but left `W_gate` dense (`3×` hard ceiling), ran BF16 only, and fell back to dense GEMM once per-neuron gather lost coalescing.
2. **Prox (2026, `arXiv:2607.27591`)** sparsified all three SwiGLU matrices via an INT4 proxy (`1.99×` end-to-end on A6000), but is dense-only, `B = 1` only, and uses FP16 weights in Stage 2.
3. Neither provides **block-grouped intra-expert sparsity (`g = 8, 16, 32` neurons)** aligned to Tensor Core tiles or **joint FP8/NVFP4 dequantization inside the sparse gather load**.

## Layout

- `SPEC.md` — Falsifiable hypotheses, analytical ceilings, and Phase 0/1 go/no-go gates.
- `sparse_moe_mm/roofline.py` — CPU-runnable HBM byte + FLOP roofline model across models, precisions, proxies, and GPUs.
- `sparse_moe_mm/reference.py` — Exact fp32 ground-truth MoE reference, INT4 proxy packing/selection, and block-grouped (`g = 1..32`) masking.
- `sparse_moe_mm/kernels.py` — CUDA-Graph-safe Triton Phase-1 prototype kernels (`threshold_pack`, `int4_proxy_gate_up`, `sparse_up_down_fp8`).
- `benchmarks/bench_layer.py` — Single-layer GPU benchmark harness against dense BF16/FP8 `grouped_mm` (`B = 1..256`).
- `tests/test_reference.py` — CPU unit tests (exact identity parity, failing-input gate, INT4 pack/unpack, block-group shapes).

## Local CPU Dry-Run (No GPU Required)

```bash
python3 -m pytest -q
python3 -m sparse_moe_mm.roofline
```

## Phase-1 GPU Feasibility Probe (5–8 GPU-hours on `1× H200` or `1× RTX PRO 6000`)

```bash
python3 benchmarks/bench_layer.py --model qwen35_35b_a3b --dtype fp8 --batches 1,4,16,32,64,128,256 --sparsity 0.90 --group-size 1,16
```
