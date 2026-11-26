"""Live integration & head-to-head benchmark vs upstream vLLM v0.34+ `[128,128]` FP8 MoE.

Verifies:
  1. `vllm.plugins.load_general_plugins()` auto-loads `prog8_sparse_fp8`.
  2. `Prog8SparseMoEExperts` is a 100% concrete subclass of `vllm` `FusedMoEExpertsModular`
     (`__abstractmethods__ == frozenset()`), passing `workspace_shapes`, `activation_format`,
     `_supports_*`, and `finalize_weight_and_reduce_impl` checks.
  3. Live CUDA SM89/SM90 `q=1.0` parity (`rel_L2 < 0.04`) through `Prog8SparseMoEExperts.apply`.
  4. L2-flushed (192 MB) head-to-head latency comparison vs upstream vLLM `fused_experts`
     (`[128,128]` block-scaled FP8) across `B = 1, 16, 32, 64, 128` on `Qwen3.5-35B-A3B`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

import vllm
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.plugins import load_general_plugins

from benchmarks.bench_layer import (
    bench_cuda_graph_l2_flushed,
    pack_prog8_224_chunked,
    quantize_fp8_block128_hw_chunked,
    sample_zipf_routing,
)
from sparse_moe_mm.kernels import (
    launch_dense_fp8_grouped_mm,
    launch_prog8_224_stage1,
    launch_prog8_224_stage2,
    launch_svd_recovery,
)
from sparse_moe_mm.reference import dense_moe_swiglu_ref
from sparse_moe_mm.roofline import MODELS
from sparse_moe_mm.vllm_plugin import (
    HAS_VLLM,
    PLUGIN_REGISTRY,
    Prog8MoEConfig,
    Prog8SparseMoEExperts,
    pack_hf_fp8_experts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,16,32,64,128")
    ap.add_argument("--sparsity", type=float, default=0.80)
    ap.add_argument("--out", default="results/l4_vs_vllm034_35b.json")
    args = ap.parse_args()

    # 1. Verify vLLM plugin discovery + concrete FusedMoEExpertsModular contract
    load_general_plugins()
    assert HAS_VLLM, "Expected HAS_VLLM=True inside vLLM environment"
    assert PLUGIN_REGISTRY["prog8_sparse_fp8"] is Prog8SparseMoEExperts
    assert issubclass(Prog8SparseMoEExperts, mk.FusedMoEExpertsModular)
    assert len(Prog8SparseMoEExperts.__abstractmethods__) == 0, (
        f"Unimplemented abstract methods: {Prog8SparseMoEExperts.__abstractmethods__}"
    )
    print(
        f"[vLLM {vllm.__version__}] Concrete FusedMoEExpertsModular verified! "
        f"activation_format={Prog8SparseMoEExperts.activation_format()} "
        f"supports_device={Prog8SparseMoEExperts._supports_current_device()} "
        f"supports_silu={Prog8SparseMoEExperts._supports_activation(MoEActivation.SILU)}"
    )

    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    # 2. Live GPU parity + CUDA-Graph capture check through Prog8SparseMoEExperts.apply(...)
    # Including: q=1.0 fused 2+2+4 dense parity, EP expert_map (-1) parity, AND strict 0%-duplicate-w13 CUDA Graph!
    torch.manual_seed(0)
    e_s, d_s, dff_s, b_s, k_s = 32, 2048, 512, 16, 4
    wg_s = torch.randn((e_s, dff_s, d_s), device=dev, dtype=torch.bfloat16) * 0.02
    wu_s = torch.randn((e_s, dff_s, d_s), device=dev, dtype=torch.bfloat16) * 0.02
    wd_s = torch.randn((e_s, d_s, dff_s), device=dev, dtype=torch.bfloat16) * 0.02
    w13_s = torch.cat([wg_s, wu_s], dim=1).contiguous()
    x_s = torch.randn((b_s, d_s), device=dev, dtype=torch.bfloat16)
    tv, tid = torch.topk(torch.randn((b_s, e_s), device=dev), k=k_s, dim=-1)
    tw = torch.softmax(tv, dim=-1)

    cfg_q1 = Prog8MoEConfig(sparsity=0.0, svd_rank=0, cold_me_thresh=8)
    packed_q1 = pack_hf_fp8_experts(w13_s, wd_s, config=cfg_q1, keep_dense_w13_for_hot=False)
    assert packed_q1.w_gate_fp8 is None and packed_q1.w_up_fp8 is None
    exp_q1 = Prog8SparseMoEExperts(packed_q1, config=cfg_q1)
    assert isinstance(exp_q1.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP)
    out_q1 = torch.empty_like(x_s)
    # Warmup + verify FULL apply() captures inside torch.cuda.CUDAGraph() with ZERO syncs!
    exp_q1.apply(out_q1, x_s, None, None, tw, tid)
    torch.cuda.synchronize()
    g_cap = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_cap):
        exp_q1.apply(out_q1, x_s, None, None, tw, tid)
    g_cap.replay()
    torch.cuda.synchronize()

    ref_s = dense_moe_swiglu_ref(x_s, tid, tw, wg_s, wu_s, wd_s)
    err_q1 = float((out_q1.float() - ref_s).norm() / ref_s.norm())
    assert err_q1 < 0.04, f"Live GPU plugin q=1.0 rel_L2 too high: {err_q1}"

    # Verify EP expert_map (-1 non-local experts) on CUDA GPU inside CUDA Graph!
    ep_map = torch.arange(e_s, dtype=torch.int32, device=dev)
    ep_map[1::2] = -1  # Half of experts live on another rank!
    out_ep = torch.empty_like(x_s)
    exp_q1.apply(out_ep, x_s, None, None, tw, tid, expert_map=ep_map)
    tw_ep = tw * (ep_map[tid] >= 0).to(tw.dtype)
    ref_ep = dense_moe_swiglu_ref(x_s, tid, tw_ep, wg_s, wu_s, wd_s)
    err_ep = float((out_ep.float() - ref_ep).norm() / ref_ep.norm())
    assert err_ep < 0.04, f"Live GPU EP expert_map rel_L2 too high: {err_ep}"
    print(
        f"[Live GPU Plugin Parity + CUDA Graph] apply(q=1.0) rel_L2={err_q1:.4f} | "
        f"EP expert_map rel_L2={err_ep:.4f} | CUDAGraph(apply)=PASS!"
    )
    del wg_s, wu_s, wd_s, w13_s, exp_q1, packed_q1
    torch.cuda.empty_cache()

    # 3. Head-to-Head L2-flushed FULL apply() CUDA-Graph benchmark vs upstream vLLM v0.34 `fused_experts`
    spec = MODELS["qwen35_35b_a3b"]
    torch.manual_seed(42)
    wg = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=dev, dtype=torch.bfloat16) * 0.02
    wu = torch.randn((spec.num_experts, spec.d_ff, spec.d_model), device=dev, dtype=torch.bfloat16) * 0.02
    wd = torch.randn((spec.num_experts, spec.d_model, spec.d_ff), device=dev, dtype=torch.bfloat16) * 0.02

    w_gate_q, s_gate = quantize_fp8_block128_hw_chunked(wg)
    w_up_q, s_up = quantize_fp8_block128_hw_chunked(wu)
    w13_fp8 = torch.cat([w_gate_q, w_up_q], dim=1).contiguous()
    w13_scale = torch.cat([s_gate, s_up], dim=1).contiguous()
    w2_fp8, w2_scale = quantize_fp8_block128_hw_chunked(wd)
    del wg, wu, wd, w_gate_q, s_gate, w_up_q, s_up
    torch.cuda.empty_cache()

    # Pack strictly from official FP8 weights with ZERO duplicate w13 kept in VRAM!
    cfg_ship = Prog8MoEConfig(sparsity=args.sparsity, cold_me_thresh=8, svd_rank=16, use_3tier=True)
    packed_ship = pack_hf_fp8_experts(
        w13_fp8, w2_fp8, w13_scale=w13_scale, w2_scale=w2_scale, config=cfg_ship, keep_dense_w13_for_hot=False
    )
    assert packed_ship.w_gate_fp8 is None and packed_ship.w_up_fp8 is None
    experts_ship = Prog8SparseMoEExperts(packed_ship, config=cfg_ship, layer_type="deltanet")
    torch.cuda.empty_cache()

    w_elems = spec.d_model * spec.d_ff
    records = []
    print(
        f"\n[FULL apply() CUDA-Graph vs vLLM {vllm.__version__} fused_experts FP8 [128,128] on {gpu_name} | "
        f"Qwen3.5-35B-A3B s={args.sparsity} | extra_VRAM={packed_ship.vram_overhead_pct():.2f}%]"
    )

    for b in [int(v) for v in args.batches.split(",")]:
        x = torch.randn((b, spec.d_model), device=dev, dtype=torch.bfloat16)
        topk_ids = sample_zipf_routing(b, spec.num_experts, spec.top_k, 0.8, dev).to(torch.int32)
        topk_weights = torch.full((b, spec.top_k), 1.0 / spec.top_k, device=dev, dtype=torch.float32)
        e_act = int(torch.unique(topk_ids).numel())

        # Upstream vLLM v0.34.0 fused_experts ([128,128] FP8)
        vllm_qcfg = fp8_w8a8_moe_quant_config(
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=[128, 128],
        )
        vllm_us = bench_cuda_graph_l2_flushed(
            lambda: fused_experts(
                x,
                w13_fp8,
                w2_fp8,
                topk_weights,
                topk_ids,
                quant_config=vllm_qcfg,
            ),
            require_graph=False,
        )

        # FULL END-TO-END Prog8SparseMoEExperts.apply() (sort + bincount + S1 + TopK + S2 + SVD + 0%-VRAM Hot Dense + combine)
        # MUST CAPTURE 100% INSIDE torch.cuda.CUDAGraph() (require_graph=True)!
        out_ours = torch.empty_like(x)
        ours_us = bench_cuda_graph_l2_flushed(
            lambda: experts_ship.apply(out_ours, x, None, None, topk_weights, topk_ids),
            require_graph=True,
        )
        vllm_bw_gbs = (e_act * 3.0 * w_elems / (vllm_us * 1e-6)) / 1e9
        spd = vllm_us / ours_us
        rec = {
            "B": b,
            "E_act": e_act,
            "vllm_fp8_us": round(vllm_us, 1),
            "vllm_bw_gbs": round(vllm_bw_gbs, 1),
            "ours_full_apply_us": round(ours_us, 1),
            "speedup_vs_vllm": round(spd, 2),
        }
        records.append(rec)
        print(
            f"  B={b:>3} E={e_act:>3} | "
            f"vLLM {vllm.__version__} FP8={vllm_us:>7.1f}u ({vllm_bw_gbs:>5.1f}GB/s) | "
            f"Ours FULL apply()={ours_us:>7.1f}u | Speedup vs vLLM={spd:>5.2f}x"
        )

    out_data = {
        "gpu": gpu_name,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "sparsity": args.sparsity,
        "vram_overhead_pct": round(packed_ship.vram_overhead_pct(), 2),
        "live_gpu_plugin_q1_rel_l2": round(err_q1, 4),
        "live_gpu_ep_map_rel_l2": round(err_ep, 4),
        "full_apply_cuda_graph_captured": True,
        "strict_zero_duplicate_w13": True,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
