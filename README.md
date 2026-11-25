# sparse-moe-mm

Zero-VRAM, calibration-free quantized (**`[128,128]` FP8** on Hopper SM90 & **`1×16` NVFP4** on Blackwell SM100) intra-expert sparse `grouped_mm` for fine-grained MoE serving (`Qwen3.5/3.6-35B-A3B`, `Qwen3.5-122B-A10B`, `DeepSeek-V3`), targeting drop-in integration via **vLLM `FusedMoEExpertsModular`**.

> **Start here:** Read [IMPACT.md](./IMPACT.md) (our live North Star impact ledger) and [SPEC.md](./SPEC.md) (v2 impact-aligned technical spec and kill-switch gates).

## Why This Exists (Late-2026 Landscape)

Fine-grained MoEs (`E=256, K=8`) route $M_e = B \cdot K / E = B / 32$ tokens per expert on average — keeping routed MoE **memory-bandwidth bound (`69–83%` of decode time on `Qwen3.5-35B-A3B`)** across serving batch sizes `B = 1..128`.

Prior work leaves the **production quantized serving stack** completely unsolved:
1. **Park et al. (May 2026, `arXiv:2605.08575`)** — BF16 only, dense `W_gate` (`3×` ceiling), threshold-based masking, global batch fallback.
2. **Prox (Jul 2026, `arXiv:2607.27591`)** — Dense-only, `B = 1`, FP16 Stage 2, `+33%` VRAM vs FP8; Stage-1 input sparsity collapses at `B >= 64` due to token-union expansion.
3. **Kim & Hwang (Oct 2026, `arXiv:2610.37386`, `calibration-free-moe`)** — Proved calibration-free top-`k` + 2-bit ternary sketch + low-rank recovery in vLLM (`1.60–2.69×` MoE speedup on Qwen3.5), **but BF16-only** with extra sketch VRAM and no `[128,128]` FP8 or `1×16` NVFP4 kernel support.

## Core Impact-Driven Architecture

1. **`Prog8` (Zero-VRAM Progressive `4+4` Bit-Sliced Gate for 8-bit MoEs):** Splits `W_gate` into `4-bit coarse + 4-bit fine` (`0%` extra VRAM), beating Park8, INT4 Prox8, and 2-bit Ternary8 across `B = 1..256` (`2.56×–3.77×` H200 FP8 ceiling).
2. **`Gate4` + Lossless Co-Activation Permutation (`g = 16` NVFP4 on Blackwell SM100):** Uses exact permutation invariance of SwiGLU to cluster co-activating neurons into contiguous 16-neuron blocks aligned to `W_down`'s `1×16` NVFP4 micro-blocks, enabling native `tl.dot_scaled` / `tcgen05.mma` + **hardware TMA `gather4`** + **PDL** stage overlap at `0%` extra VRAM.
3. **Per-Expert Adaptive Sparse/Dense Dispatch:** Routes cold/medium experts (`M_e <= 8`) to an expert-grouped sparse Tensor Core kernel and hot Zipfian experts (`M_e > 8`) to dense FP8/NVFP4 `grouped_mm`.
4. **Drop-In vLLM `FusedMoEExpertsModular` Plugin:** Loads official HuggingFace `[128,128]` FP8 and `1×16` NVFP4 checkpoints with zero calibration data.

## Repository Layout

- `IMPACT.md` — Live North Star ledger tracking all 6 engineering & research impact pillars (`I1`–`I6`).
- `SPEC.md` — Falsifiable hypotheses, exact HBM ceilings (with scale bytes & union expansion), and Phase 0/1/2 go/no-go gates.
- `sparse_moe_mm/roofline.py` — CPU-runnable HBM + Tensor Core roofline across Hopper/Blackwell, routing skew, and E2E Amdahl budgets.
- `sparse_moe_mm/reference.py` — Exact fp32 ground-truth MoE references, bit-sliced/quantized proxies, and lossless co-activation permutation.
- `sparse_moe_mm/kernels.py` — CUDA-Graph-safe Triton sparse `grouped_mm` kernels.
- `benchmarks/bench_layer.py` — L2-flushed, CUDA-Graph single-layer benchmark harness vs dense FP8/NVFP4 `grouped_mm`.
- `tests/test_reference.py` — CPU unit tests (exact identity parity, failing-input gates, quantization & permutation invariance).

## Local CPU Dry-Run (No GPU Required)

```bash
python3 -m pytest -q
python3 -m sparse_moe_mm.roofline
```
