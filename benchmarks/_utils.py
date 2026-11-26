"""Shared GPU benchmarking, quantization, routing, and HF model-inspection utilities."""

from __future__ import annotations

import torch

from sparse_moe_mm.reference import (
    pack_2bit_interleaved,
    pack_4bit_interleaved,
    quantize_prog8_224,
)


def measure_d2d_ceiling_gbs(size_mb: int = 512, iters: int = 40) -> float:
    """Measure practical device-to-device HBM copy bandwidth ceiling in GB/s."""
    n = (size_mb * 1024 * 1024) // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        b.copy_(a)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    gbs = (2.0 * size_mb * 1024 * 1024 / (ms * 1e-3)) / 1e9
    del a, b
    torch.cuda.empty_cache()
    return round(gbs, 1)


def quantize_fp8_block128_hw_chunked(
    w: torch.Tensor, chunk_e: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked `[128,128]` FP8 E4M3 quantization keeping peak temporary VRAM low."""
    e, m, n = w.shape
    bm, bn = m // 128, n // 128
    q_out = torch.empty((e, m, n), dtype=torch.float8_e4m3fn, device=w.device)
    s_out = torch.empty((e, bm, bn), dtype=torch.float32, device=w.device)
    for start in range(0, e, chunk_e):
        end = min(e, start + chunk_e)
        blocks = w[start:end].float().view(end - start, bm, 128, bn, 128)
        scale = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
        q_out[start:end] = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(end - start, m, n)
        s_out[start:end] = scale.squeeze(4).squeeze(2)
    return q_out, s_out


def pack_prog8_224_chunked(w: torch.Tensor, chunk_e: int = 32):
    """Chunked 2+2+4 bitplane packing with 128B cache-line blocks (block_d=512 for 2-bit, 256 for 4-bit)."""
    e, m, n = w.shape
    hi2_p = torch.empty((e, m, n // 4), dtype=torch.uint8, device=w.device)
    mid2_p = torch.empty((e, m, n // 4), dtype=torch.uint8, device=w.device)
    lo4_p = torch.empty((e, m, n // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((e, m), dtype=torch.float32, device=w.device)
    for start in range(0, e, chunk_e):
        end = min(e, start + chunk_e)
        hi2, mid2, lo4, sc = quantize_prog8_224(w[start:end], group_size=None)
        hi2_p[start:end] = pack_2bit_interleaved(hi2, signed=True, block_d=512)
        mid2_p[start:end] = pack_2bit_interleaved(mid2, signed=False, block_d=512)
        lo4_p[start:end] = pack_4bit_interleaved(lo4, block_d=256)
        scales[start:end] = sc.squeeze(-1)
    return hi2_p, mid2_p, lo4_p, scales


def bench_cuda_graph_l2_flushed(
    fn,
    warmup: int = 10,
    iters: int = 40,
    l2_flush_mb: int = 192,
    require_graph: bool = True,
) -> float:
    """Measure median latency in microseconds with L2 cache flushing and CUDA Graph capture."""
    flush_buf = torch.empty((l2_flush_mb * 1024 * 1024 // 4,), dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    runner = fn
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        runner = g.replay
    except RuntimeError:
        if require_graph:
            raise

    times_ms = []
    for _ in range(iters):
        flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    del flush_buf
    return times_ms[len(times_ms) // 2] * 1000.0


def sample_zipf_routing(
    b: int, num_experts: int, top_k: int, alpha: float, device: torch.device
) -> torch.Tensor:
    """Sample Gumbel-TopK Zipfian expert routing (alpha=0.8 matches real multi-user skew)."""
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-alpha)
    probs = probs / probs.sum()
    u = torch.rand((b, num_experts), device=device).clamp(min=1e-7)
    keys = torch.log(probs)[None, :] - torch.log(-torch.log(u))
    return torch.topk(keys, k=top_k, dim=-1).indices


def find_moe_layers(model) -> list:
    """Locate routed MoE decoder layers across Qwen3.5/3.6, DeepSeek-V3, and standard HF models."""
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
        model,
    ]
    for root in candidates:
        layers = getattr(root, "layers", None)
        if layers is not None and len(layers) > 0:
            moe_layers = [ly for ly in layers if hasattr(getattr(ly, "mlp", None), "experts")]
            if moe_layers:
                return moe_layers
    raise RuntimeError("Could not locate MoE layers with `layer.mlp.experts` on model.")


def is_full_attn_layer(layer, layer_idx: int) -> bool:
    """Detect whether a hybrid DeltaNet/Full-Attention MoE layer uses full self-attention."""
    lt = getattr(layer, "layer_type", None)
    if isinstance(lt, str):
        return "full" in lt.lower() or "self_attn" in lt.lower()
    if hasattr(layer, "linear_attn"):
        return False
    if hasattr(layer, "self_attn"):
        return True
    # Qwen3.5 3:1 hybrid pattern: every 4th layer (3, 7, 11, ..., 39) is Full Attention
    return ((layer_idx + 1) % 4) == 0
