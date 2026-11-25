"""Phase-0 Kill-Switch: Real-Weight Validation on Qwen/Qwen3.5-35B-A3B-FP8 (Colab A100-80GB).

Validates ALL 3 make-or-break gates in SPEC.md BEFORE investing GPU hours in kernels:
  1. G0_ACCURACY (I3, I4):
       - Part A: Per-expert rel_L2 & energy across 40 real experts (5 layers x 8 experts)
       - Part B: True 40-layer END-TO-END WikiText-2 PPL (Delta-PPL) at 85% & 90% sparsity for:
           Dense FP8, Oracle, Park8, Prog8 (4+4 gate, 0% VRAM), Prog8-Dual (2+6 gate+up, 0% VRAM),
           Tern8 (2-bit ternary, +17% VRAM), and Gate4 (NVFP4 1x16, 0% VRAM).
  2. G0_PERM_G16 (I5):
       - Offline lossless co-activation neuron permutation P_e on real Qwen3.5 experts
         comparing g=1 vs naive g=16 vs permuted g=16 (FP8 and NVFP4).
  3. G0_REAL_ROUTING (I1, I6):
       - Real routing Zipf skew (E_act, max M_e, p95 M_e) at B = 16, 64, 128
       - Real intra-expert active-neuron union correlation `corr_out` at M_e = 4 and M_e = 8.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import types
import torch
import torch.nn.functional as F

from sparse_moe_mm.reference import (
    apply_expert_permutation,
    block_group_mask,
    compute_coactivation_permutation,
    dequantize_prog8_4_4,
    dequantize_prog8_dual_2_6,
    quantize_nvfp4_block16,
    quantize_prog8_4_4,
    quantize_prog8_dual_2_6,
    quantize_ternary_2bit,
)


def _dequant_fp8_weight_if_needed(weight: torch.Tensor, scale_inv: torch.Tensor | None, block_size: int = 128) -> torch.Tensor:
    w = weight.float()
    if scale_inv is None:
        return w
    s = scale_inv.float()
    if s.ndim == w.ndim and s.shape[-2] != w.shape[-2]:
        m, n = w.shape[-2], w.shape[-1]
        s_exp = s.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)[..., :m, :n]
        return w * s_exp
    return w * s


def evaluate_expert_methods(
    x_cal: torch.Tensor,
    x_eval: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    sparsities: tuple[float, ...] = (0.80, 0.85, 0.90),
    quant_group: int = 128,
) -> dict:
    """Evaluate all candidate selectors + lossless g=16 permutation on a single expert's real weights."""
    device = x_eval.device
    wg = w_gate.to(device=device, dtype=torch.float32)
    wu = w_up.to(device=device, dtype=torch.float32)
    wd = w_down.to(device=device, dtype=torch.float32)
    d_ff, d_model = wg.shape
    qg = quant_group if (d_model % quant_group == 0) else 32

    g_exact = F.silu(x_eval @ wg.T)
    u_exact = x_eval @ wu.T
    h_exact = g_exact * u_exact
    y_dense = h_exact @ wd.T
    y_norm = y_dense.norm().clamp(min=1e-8)
    h_energy_tot = h_exact.pow(2).sum(dim=-1).clamp(min=1e-12)

    # 1. Prog8 (4+4 exact INT8 gate, 0% VRAM)
    wg_hi4, wg_lo4, wg_s4 = quantize_prog8_4_4(wg, group_size=qg)
    wg_p4_s1 = dequantize_prog8_4_4(wg_hi4, None, wg_s4, group_size=qg)
    wg_p4_s2 = dequantize_prog8_4_4(wg_hi4, wg_lo4, wg_s4, group_size=qg)
    s_prog8 = F.silu(x_eval @ wg_p4_s1.T)
    h_prog8 = F.silu(x_eval @ wg_p4_s2.T) * u_exact

    # 2. Prog8-Dual (2+6 exact INT8 gate+up, 0% VRAM)
    wg_hi2, wg_lo6, wg_s2 = quantize_prog8_dual_2_6(wg, group_size=qg)
    wu_hi2, wu_lo6, wu_s2 = quantize_prog8_dual_2_6(wu, group_size=qg)
    wg_p2_s1 = dequantize_prog8_dual_2_6(wg_hi2, None, wg_s2, group_size=qg)
    wu_p2_s1 = dequantize_prog8_dual_2_6(wu_hi2, None, wu_s2, group_size=qg)
    wg_p2_s2 = dequantize_prog8_dual_2_6(wg_hi2, wg_lo6, wg_s2, group_size=qg)
    wu_p2_s2 = dequantize_prog8_dual_2_6(wu_hi2, wu_lo6, wu_s2, group_size=qg)
    s_prog8_dual = F.silu(x_eval @ wg_p2_s1.T) * (x_eval @ wu_p2_s1.T)
    h_prog8_dual = F.silu(x_eval @ wg_p2_s2.T) * (x_eval @ wu_p2_s2.T)

    # 3. Tern8 (2-bit TWN ternary gate+up, +17% VRAM — arXiv:2610.37386)
    wg_t2 = quantize_ternary_2bit(wg)
    wu_t2 = quantize_ternary_2bit(wu)
    s_tern8 = F.silu(x_eval @ wg_t2.T) * (x_eval @ wu_t2.T)

    # 4. Gate4 (Native NVFP4 1x16 W_gate, W_up, W_down, 0% VRAM)
    wg_fp4, _ = quantize_nvfp4_block16(wg, block_size=16)
    wu_fp4, _ = quantize_nvfp4_block16(wu, block_size=16)
    wd_fp4, _ = quantize_nvfp4_block16(wd, block_size=16)
    g_fp4 = F.silu(x_eval @ wg_fp4.T)
    h_fp4 = g_fp4 * (x_eval @ wu_fp4.T)

    # 5. Offline lossless co-activation permutation for g=16 (on x_cal!)
    h_cal = (F.silu(x_cal @ wg.T) * (x_cal @ wu.T)).abs()
    perm = compute_coactivation_permutation(h_cal, group_size=16)
    wg_p, wu_p, wd_p = apply_expert_permutation(wg, wu, wd, perm)
    h_exact_p = F.silu(x_eval @ wg_p.T) * (x_eval @ wu_p.T)
    y_perm_dense = h_exact_p @ wd_p.T
    perm_inv_err = float((y_perm_dense - y_dense).norm() / y_norm)

    wg_p_fp4, _ = quantize_nvfp4_block16(wg_p, block_size=16)
    wu_p_fp4, _ = quantize_nvfp4_block16(wu_p, block_size=16)
    wd_p_fp4, _ = quantize_nvfp4_block16(wd_p, block_size=16)
    g_p_fp4 = F.silu(x_eval @ wg_p_fp4.T)
    h_p_fp4 = g_p_fp4 * (x_eval @ wu_p_fp4.T)

    res: dict = {"perm_invariance_err": perm_inv_err, "by_sparsity": {}}
    for sp in sparsities:
        q = 1.0 - sp
        entry = {}

        def _score(name: str, score_mat: torch.Tensor, h_mat: torch.Tensor, w_dn: torch.Tensor, g_sz: int):
            m = block_group_mask(score_mat, keep_frac=q, group_size=g_sz)
            y_hat = (h_mat * m.float()) @ w_dn.T
            rel_l2 = float((y_hat - y_dense).norm() / y_norm)
            energy = float(((h_exact * m.float()).pow(2).sum(dim=-1) / h_energy_tot).mean().item()) if (w_dn.data_ptr() == wd.data_ptr()) else float("nan")
            entry[name] = {"rel_l2": rel_l2, "energy": energy}

        _score("oracle_g1", h_exact, h_exact, wd, 1)
        _score("park8_g1", g_exact, h_exact, wd, 1)
        _score("prog8_4_4_g1", s_prog8, h_prog8, wd, 1)
        _score("prog8_dual_2_6_g1", s_prog8_dual, h_prog8_dual, wd, 1)
        _score("tern8_g1", s_tern8, h_exact, wd, 1)
        _score("gate4_nvfp4_g1", g_fp4, h_fp4, wd_fp4, 1)
        _score("oracle_g16_naive", h_exact, h_exact, wd, 16)
        _score("oracle_g16_permuted", h_exact_p, h_exact_p, wd_p, 16)
        _score("gate4_nvfp4_g16_naive", g_fp4, h_fp4, wd_fp4, 16)
        _score("gate4_nvfp4_g16_permuted", g_p_fp4, h_p_fp4, wd_p_fp4, 16)

        # Measure real intra-expert union correlation `corr_out` at M_e = 4 and M_e = 8
        m_all = block_group_mask(h_exact, keep_frac=q, group_size=1)
        for m_grp in (4, 8):
            n_grp = m_all.shape[0] // m_grp
            if n_grp >= 2:
                mg = m_all[: n_grp * m_grp].view(n_grp, m_grp, d_ff)
                q_union_emp = float(mg.any(dim=1).float().mean().item())
                q_indep = 1.0 - (1.0 - q) ** m_grp
                corr_emp = (q_indep - q_union_emp) / max(q_indep - q, 1e-8)
                entry[f"empirical_union_M{m_grp}"] = q_union_emp
                entry[f"empirical_corr_out_M{m_grp}"] = max(0.0, min(1.0, corr_emp))
        res["by_sparsity"][f"{sp:.2f}"] = entry
    return res


def run_synthetic_smoke(out_path: str | None) -> dict:
    """Fast CPU smoke test validating all Phase-0 logic without downloading weights."""
    torch.manual_seed(0)
    d, d_ff, g = 512, 256, 16
    nc = d_ff // g
    latents = F.normalize(torch.randn(nc, d), dim=-1)
    wg = torch.cat([latents[c] + 0.03 * torch.randn(g, d) for c in range(nc)], dim=0) * 0.05
    wu = torch.randn(d_ff, d) * 0.04
    wd = torch.randn(d, d_ff) * 0.04
    scr = torch.randperm(d_ff)
    wg, wu, wd = apply_expert_permutation(wg, wu, wd, scr)

    def _gen(n: int):
        x = torch.randn(n, d) * 0.2
        for i in range(n):
            x[i] += latents[torch.randperm(nc)[:2]].sum(dim=0) * 3.0
        return x

    res = evaluate_expert_methods(_gen(256), _gen(128), wg, wu, wd)
    out = {"mode": "synthetic_smoke", "timestamp": time.time(), "expert_metrics": res}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _find_moe_layers(model):
    candidates = [
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers
    raise RuntimeError(f"Could not locate decoder layers in model type {type(model).__name__}")


def _extract_expert_weights(mlp_mod, expert_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts = getattr(mlp_mod, "experts", None)
    if experts is None:
        raise RuntimeError(f"No .experts attribute found on {type(mlp_mod).__name__}")

    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        gu_p = experts.gate_up_proj
        dn_p = experts.down_proj
        gu_w = gu_p.weight if hasattr(gu_p, "weight") else gu_p
        dn_w = dn_p.weight if hasattr(dn_p, "weight") else dn_p
        gu_s = getattr(experts, "gate_up_proj_scale_inv", getattr(gu_p, "weight_scale_inv", None))
        dn_s = getattr(experts, "down_proj_scale_inv", getattr(dn_p, "weight_scale_inv", None))
        gu_e = _dequant_fp8_weight_if_needed(gu_w[expert_id], gu_s[expert_id] if gu_s is not None else None)
        dn_e = _dequant_fp8_weight_if_needed(dn_w[expert_id], dn_s[expert_id] if dn_s is not None else None)
        if gu_e.shape[0] == dn_e.shape[0] * 2 or gu_e.shape[0] == 2 * dn_e.shape[1]:
            wg, wu = gu_e.chunk(2, dim=0)
        else:
            wg, wu = gu_e.T.chunk(2, dim=0)
        if dn_e.shape[1] != wg.shape[0]:
            dn_e = dn_e.T
        return wg.contiguous(), wu.contiguous(), dn_e.contiguous()

    exp = experts[expert_id]
    def _lin_fp32(mod):
        w = mod.weight
        s = getattr(mod, "weight_scale_inv", getattr(mod, "scale_inv", None))
        return _dequant_fp8_weight_if_needed(w, s)

    wg = _lin_fp32(getattr(exp, "gate_proj", getattr(exp, "w1", None)))
    wu = _lin_fp32(getattr(exp, "up_proj", getattr(exp, "w3", None)))
    wd = _lin_fp32(getattr(exp, "down_proj", getattr(exp, "w2", None)))
    return wg.contiguous(), wu.contiguous(), wd.contiguous()


def _make_patched_experts_forward(mode: str, sparsity: float, group_size: int = 1):
    """Return a fast monkey-patched Qwen3_5MoeExperts.forward evaluating `mode` across ALL 40 layers."""
    keep_frac = 1.0 - sparsity

    def _patched_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            eid = int(expert_idx[0].item())
            if eid == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[eid])
            x_e = hidden_states[token_idx]
            gu_w = self.gate_up_proj[eid]  # [2*d_ff, d]
            dn_w = self.down_proj[eid]  # [d, d_ff]
            wg, wu = gu_w.chunk(2, dim=0)
            x_f = x_e.float()

            if mode == "oracle":
                g = F.silu(F.linear(x_e, wg))
                u = F.linear(x_e, wu)
                h = g * u
                m = block_group_mask(h, keep_frac=keep_frac, group_size=group_size)
                out_e = F.linear(h * m.to(h.dtype), dn_w)
            elif mode == "park8":
                g = F.silu(F.linear(x_e, wg))
                u = F.linear(x_e, wu)
                m = block_group_mask(g, keep_frac=keep_frac, group_size=group_size)
                out_e = F.linear((g * u) * m.to(g.dtype), dn_w)
            elif mode == "prog8_4_4":
                hi4, lo4, s4 = quantize_prog8_4_4(wg, group_size=128)
                wg_s1 = dequantize_prog8_4_4(hi4, None, s4, group_size=128)
                wg_s2 = dequantize_prog8_4_4(hi4, lo4, s4, group_size=128)
                score = F.silu(F.linear(x_f, wg_s1))
                m = block_group_mask(score, keep_frac=keep_frac, group_size=group_size)
                h = (F.silu(F.linear(x_f, wg_s2)) * F.linear(x_f, wu.float())) * m.float()
                out_e = F.linear(h.to(dn_w.dtype), dn_w)
            elif mode == "prog8_dual_2_6":
                wg_hi, wg_lo, wg_s = quantize_prog8_dual_2_6(wg, group_size=128)
                wu_hi, wu_lo, wu_s = quantize_prog8_dual_2_6(wu, group_size=128)
                score = F.silu(F.linear(x_f, dequantize_prog8_dual_2_6(wg_hi, None, wg_s, group_size=128))) * F.linear(
                    x_f, dequantize_prog8_dual_2_6(wu_hi, None, wu_s, group_size=128)
                )
                m = block_group_mask(score, keep_frac=keep_frac, group_size=group_size)
                h = (
                    F.silu(F.linear(x_f, dequantize_prog8_dual_2_6(wg_hi, wg_lo, wg_s, group_size=128)))
                    * F.linear(x_f, dequantize_prog8_dual_2_6(wu_hi, wu_lo, wu_s, group_size=128))
                ) * m.float()
                out_e = F.linear(h.to(dn_w.dtype), dn_w)
            elif mode == "tern8":
                score = F.silu(F.linear(x_f, quantize_ternary_2bit(wg))) * F.linear(x_f, quantize_ternary_2bit(wu))
                m = block_group_mask(score, keep_frac=keep_frac, group_size=group_size)
                h = (F.silu(F.linear(x_e, wg)) * F.linear(x_e, wu)) * m.to(x_e.dtype)
                out_e = F.linear(h, dn_w)
            elif mode == "gate4_nvfp4":
                wg4, _ = quantize_nvfp4_block16(wg, block_size=16)
                wu4, _ = quantize_nvfp4_block16(wu, block_size=16)
                wd4, _ = quantize_nvfp4_block16(dn_w, block_size=16)
                g4 = F.silu(F.linear(x_f, wg4))
                m = block_group_mask(g4, keep_frac=keep_frac, group_size=group_size)
                h4 = (g4 * F.linear(x_f, wu4)) * m.float()
                out_e = F.linear(h4, wd4).to(hidden_states.dtype)
            else:
                raise ValueError(f"Unknown E2E mode: {mode}")

            out_e = out_e * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, out_e.to(final_hidden_states.dtype))

        return final_hidden_states

    return _patched_forward


def _eval_wikitext2_ppl(model, input_ids: torch.Tensor, chunk_size: int = 512) -> float:
    nll_sum, n_tok = 0.0, 0
    emb = model.get_input_embeddings()
    first_dev = emb.weight.device if emb is not None else next(model.parameters()).device
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        for start in range(0, seq_len - 1, chunk_size):
            end = min(seq_len, start + chunk_size)
            ids = input_ids[:, start:end].to(first_dev)
            out = model(input_ids=ids, labels=ids)
            if out.loss is not None:
                n_here = ids.shape[1] - 1
                nll_sum += float(out.loss.item()) * n_here
                n_tok += n_here
    return math.exp(nll_sum / max(n_tok, 1))


def run_real_model(
    model_id: str,
    sample_layers: list[int],
    experts_per_layer: int,
    num_tokens: int,
    e2e_tokens: int,
    out_path: str | None,
) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

    t0 = time.time()
    print(f"[Phase 0] Loading tokenizer & model: {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # On SM80 (A100), FP8 checkpoints require dequantize=True -> BF16 (~70 GB).
    # Cap GPU memory to 66GiB with CPU overflow (140GiB RAM) to prevent allocator fragmentation OOM.
    cc_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    load_kwargs: dict = {
        "device_map": "auto",
        "max_memory": {0: "66GiB", "cpu": "140GiB"} if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if "FP8" in model_id.upper() and cc_major < 9:
        print(f"[Phase 0] GPU SM{cc_major}x detected: loading FP8 weights with FineGrainedFP8Config(dequantize=True) -> BF16", flush=True)
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = "auto"

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = getattr(cfg, "architectures", []) or []
    if any("ConditionalGeneration" in a for a in archs):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    torch.cuda.empty_cache()
    print(
        f"[Phase 0] Loaded in {time.time() - t0:.1f}s | VRAM allocated: "
        f"{torch.cuda.memory_allocated() / 1e9:.1f} GB",
        flush=True,
    )

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x for x in ds["text"] if len(x.strip()) > 40)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=num_tokens)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"[Phase 0] Tokenized {seq_len} real WikiText-2 tokens", flush=True)

    layers = _find_moe_layers(model)
    valid_layers = [idx for idx in sample_layers if idx < len(layers)]
    print(f"[Phase 0] Found {len(layers)} layers; sampling layers {valid_layers}", flush=True)

    # --- PART A: Capture real activations & routing on dense pass ---
    captured: dict[int, dict[str, list[torch.Tensor]]] = {i: {"x": [], "selected_experts": []} for i in valid_layers}
    hooks = []

    def _make_mlp_hook(layer_idx: int):
        def _hook(_mod, inp, _out):
            captured[layer_idx]["x"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
        return _hook

    def _make_gate_hook(layer_idx: int):
        def _hook(_mod, _inp, out):
            # Qwen3.5 gate returns (logits, routing_weights, selected_experts)
            if isinstance(out, tuple) and len(out) >= 3:
                sel = out[2].detach().cpu()
            else:
                logits = out[0] if isinstance(out, tuple) else out
                top_k = getattr(getattr(model.config, "text_config", model.config), "num_experts_per_tok", 8)
                sel = torch.topk(logits.detach().float(), k=top_k, dim=-1).indices.cpu()
            captured[layer_idx]["selected_experts"].append(sel)
        return _hook

    for li in valid_layers:
        mlp = layers[li].mlp
        hooks.append(mlp.register_forward_hook(_make_mlp_hook(li)))
        gate_mod = getattr(mlp, "gate", getattr(mlp, "router", None))
        if gate_mod is not None:
            hooks.append(gate_mod.register_forward_hook(_make_gate_hook(li)))

    dense_ppl_full = _eval_wikitext2_ppl(model, input_ids, chunk_size=512)
    for h in hooks:
        h.remove()
    print(f"[Phase 0] Dense WikiText-2 PPL ({seq_len} tok): {dense_ppl_full:.3f}", flush=True)

    eval_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = getattr(model.config, "text_config", model.config)
    num_experts = getattr(tc, "num_experts", 256)
    top_k = getattr(tc, "num_experts_per_tok", 8)

    layer_reports = {}
    for li in valid_layers:
        x_all = torch.cat(captured[li]["x"], dim=0)
        topk_ids = torch.cat(captured[li]["selected_experts"], dim=0)  # [T, K]

        routing_stats = {}
        for B in (16, 64, 128):
            nb = topk_ids.shape[0] // B
            if nb < 1:
                continue
            e_acts, max_mes, p95_mes = [], [], []
            for bi in range(nb):
                cnt = torch.bincount(topk_ids[bi * B : (bi + 1) * B].reshape(-1), minlength=num_experts)
                nz = cnt[cnt > 0].float()
                e_acts.append(float(nz.numel()))
                max_mes.append(float(nz.max().item()))
                p95_mes.append(float(torch.quantile(nz, 0.95).item()))
            routing_stats[f"B={B}"] = {
                "E_act_mean": round(sum(e_acts) / len(e_acts), 1),
                "max_Me_mean": round(sum(max_mes) / len(max_mes), 1),
                "p95_Me_mean": round(sum(p95_mes) / len(p95_mes), 1),
            }

        total_counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
        hot_experts = torch.topk(total_counts, k=min(experts_per_layer, num_experts)).indices.tolist()
        expert_results = []
        mlp = layers[li].mlp
        for eid in hot_experts:
            tok_mask = (topk_ids == eid).any(dim=-1)
            x_e = x_all[tok_mask]
            if x_e.shape[0] < 32:
                continue
            mid = x_e.shape[0] // 2
            x_cal_e = x_e[:mid].to(eval_dev, dtype=torch.float32)
            x_eval_e = x_e[mid:].to(eval_dev, dtype=torch.float32)
            wg, wu, wd = _extract_expert_weights(mlp, eid)
            em = evaluate_expert_methods(x_cal_e, x_eval_e, wg, wu, wd)
            em["expert_id"] = eid
            em["num_tokens"] = int(x_e.shape[0])
            expert_results.append(em)

        layer_reports[f"layer_{li}"] = {
            "routing_stats": routing_stats,
            "num_experts_evaluated": len(expert_results),
            "experts": expert_results,
        }
        print(f"[Phase 0] Layer {li}: {len(expert_results)} experts | routing={routing_stats}", flush=True)

    summary = {}
    for sp_key in ("0.80", "0.85", "0.90"):
        agg_l2: dict[str, list[float]] = {}
        agg_en: dict[str, list[float]] = {}
        corrs4, corrs8 = [], []
        for lr in layer_reports.values():
            for em in lr["experts"]:
                sp_d = em["by_sparsity"][sp_key]
                if "empirical_corr_out_M4" in sp_d:
                    corrs4.append(sp_d["empirical_corr_out_M4"])
                if "empirical_corr_out_M8" in sp_d:
                    corrs8.append(sp_d["empirical_corr_out_M8"])
                for mname, mvals in sp_d.items():
                    if isinstance(mvals, dict):
                        agg_l2.setdefault(mname, []).append(mvals["rel_l2"])
                        if not math.isnan(mvals["energy"]):
                            agg_en.setdefault(mname, []).append(mvals["energy"])
        summary[f"sparsity_{sp_key}"] = {
            "mean_rel_l2": {k: round(sum(v) / len(v), 4) for k, v in agg_l2.items()},
            "mean_energy": {k: round(sum(v) / len(v), 4) for k, v in agg_en.items()},
            "mean_empirical_corr_out_M4": round(sum(corrs4) / len(corrs4), 4) if corrs4 else None,
            "mean_empirical_corr_out_M8": round(sum(corrs8) / len(corrs8), 4) if corrs8 else None,
        }

    # --- PART B: TRUE 40-LAYER END-TO-END WIKITEXT-2 PERPLEXITY ---
    e2e_ids = input_ids[:, : min(e2e_tokens, seq_len)]
    dense_e2e_ppl = _eval_wikitext2_ppl(model, e2e_ids, chunk_size=512)
    e2e_ppl_results: dict[str, dict[str, float]] = {
        "dense_fp8": {"ppl": round(dense_e2e_ppl, 4), "delta_ppl_pct": 0.0}
    }
    print(f"\n[Phase 0 Part B] True 40-layer E2E WikiText-2 PPL ({e2e_ids.shape[1]} tok) | dense_fp8={dense_e2e_ppl:.4f}", flush=True)

    orig_forwards = [layer.mlp.experts.forward for layer in layers]
    e2e_configs = [
        ("oracle", 0.85, 1),
        ("park8", 0.85, 1),
        ("prog8_4_4", 0.85, 1),
        ("prog8_dual_2_6", 0.85, 1),
        ("tern8", 0.85, 1),
        ("oracle", 0.90, 1),
        ("park8", 0.90, 1),
        ("prog8_4_4", 0.90, 1),
        ("prog8_dual_2_6", 0.90, 1),
        ("tern8", 0.90, 1),
        ("gate4_nvfp4", 0.85, 1),
        ("gate4_nvfp4", 0.90, 1),
        ("oracle", 0.85, 16),
    ]
    try:
        for mode, sp, gs in e2e_configs:
            ts = time.time()
            patched_fn = _make_patched_experts_forward(mode, sp, group_size=gs)
            for layer in layers:
                layer.mlp.experts.forward = types.MethodType(patched_fn, layer.mlp.experts)
            ppl = _eval_wikitext2_ppl(model, e2e_ids, chunk_size=512)
            dpct = 100.0 * (ppl - dense_e2e_ppl) / dense_e2e_ppl
            key = f"{mode}_s{int(sp*100)}_g{gs}"
            e2e_ppl_results[key] = {"ppl": round(ppl, 4), "delta_ppl_pct": round(dpct, 3)}
            print(f"  {key:<26} | PPL={ppl:>7.4f} | delta={dpct:>+6.2f}% ({time.time()-ts:.1f}s)", flush=True)
    finally:
        for layer, orig in zip(layers, orig_forwards):
            layer.mlp.experts.forward = orig

    report = {
        "model_id": model_id,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dense_wikitext2_ppl_full": round(dense_ppl_full, 4),
        "num_tokens": seq_len,
        "e2e_tokens": int(e2e_ids.shape[1]),
        "elapsed_sec": round(time.time() - t0, 1),
        "e2e_40layer_ppl": e2e_ppl_results,
        "summary_40experts": summary,
        "layers": layer_reports,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"[Phase 0] Wrote full report to {out_path}", flush=True)
    print("\n=== PHASE-0 40-EXPERT SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\n=== PHASE-0 40-LAYER E2E PPL ===", flush=True)
    print(json.dumps(e2e_ppl_results, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--layers", default="2,10,20,30,38")
    ap.add_argument("--experts-per-layer", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--e2e-tokens", type=int, default=1536)
    ap.add_argument("--out", default="/content/phase0_qwen35_results.json")
    args = ap.parse_args()

    if args.synthetic_smoke:
        run_synthetic_smoke(args.out)
    else:
        sample_layers = [int(x) for x in args.layers.split(",")]
        run_real_model(args.model, sample_layers, args.experts_per_layer, args.num_tokens, args.e2e_tokens, args.out)


if __name__ == "__main__":
    main()
