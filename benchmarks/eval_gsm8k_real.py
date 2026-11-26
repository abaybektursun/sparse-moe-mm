"""Live End-to-End 40-Layer GSM8K Exact-Match Gate Through Production Triton `Prog8SparseMoEExperts`.

Evaluates batched multi-user decoding on `openai/gsm8k` (test split) on real
`Qwen/Qwen3.5-35B-A3B-FP8` (`1x NVIDIA L40S 48 GB`):
  1. Official Dense `[128,128]` FP8 (`baseline`)
  2. Live Production Triton `Prog8SparseMoEExperts.apply()` (`HYBRID DN80-3Tier / FA70-8b + SVD-r16`,
     `cold_Me <= 8`, strict `0.00%` duplicate `w13` VRAM — deletes `gate_up_proj` after packing!).

Uses `enable_thinking=False` chat template so 64 GSM8K problems evaluate cleanly within
`max_new_tokens=384` (avoiding truncated thinking traces) and reports exact-match accuracy,
relative retention, repetition-loop count, and live VRAM before/after in-place packing.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
import time
import types
import torch

from benchmarks.phase0b_asymmetric_bets import _find_moe_layers
from benchmarks.phase0c_ship_ablation import _is_full_attn_layer
from sparse_moe_mm.vllm_plugin import (
    Prog8MoEConfig,
    Prog8SparseMoEExperts,
    pack_hf_fp8_experts,
)


def _extract_gold(answer_str: str) -> str | None:
    m = re.search(r"####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)", answer_str)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def _extract_pred(text: str) -> str | None:
    for pat in (
        r"####\s*\$?([-+]?[0-9,]+(?:\.[0-9]+)?)",
        r"\\boxed\{([^}]+)\}",
        r"(?:Answer|answer|Final Answer)\s*[:=]\s*\$?([-+]?[0-9,]+(?:\.[0-9]+)?)",
    ):
        matches = re.findall(pat, text)
        if matches:
            num = re.findall(r"[-+]?[0-9,]+(?:\.[0-9]+)?", matches[-1])
            if num:
                return num[-1].replace(",", "").strip()
    nums = re.findall(r"[-+]?[0-9,]+(?:\.[0-9]+)?", text)
    return nums[-1].replace(",", "").strip() if nums else None


def _has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 24:
        return False
    for n in (4, 6, 8):
        for i in range(0, len(toks) - 4 * n):
            chunk = toks[i : i + n]
            if all(toks[i + k * n : i + (k + 1) * n] == chunk for k in range(1, 5)):
                return True
    return False


def _format_prompts(tok, questions: list[str]) -> list[str]:
    prompts = []
    for q in questions:
        msgs = [
            {
                "role": "system",
                "content": "You are a concise math solver. Solve step by step and end with `#### <number>`.",
            },
            {"role": "user", "content": q},
        ]
        try:
            p = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    return prompts


def _eval_gsm8k_batched(
    model, tok, questions: list[str], golds: list[str], batch_size: int, max_new_tokens: int
) -> dict:
    correct = 0
    loops = 0
    samples = []
    first_dev = model.get_input_embeddings().weight.device
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    t0 = time.time()
    for start in range(0, len(questions), batch_size):
        end = min(len(questions), start + batch_size)
        prompts = _format_prompts(tok, questions[start:end])
        enc = tok(prompts, return_tensors="pt", padding=True).to(first_dev)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        for j, seq in enumerate(gen):
            txt = tok.decode(seq[in_len:], skip_special_tokens=True)
            pred = _extract_pred(txt)
            gold = golds[start + j]
            ok = (pred is not None) and (gold is not None) and (
                pred == gold or pred.replace(".0", "") == gold.replace(".0", "")
            )
            if ok:
                correct += 1
            if _has_repetition_loop(txt):
                loops += 1
            if len(samples) < 5:
                samples.append({"q": questions[start + j][:90], "gold": gold, "pred": pred, "ok": ok})
        print(
            f"    progress {end}/{len(questions)} | acc={100.0 * correct / end:.1f}% | "
            f"loops={loops} | {time.time() - t0:.1f}s",
            flush=True,
        )

    return {
        "n": len(questions),
        "correct": correct,
        "accuracy_pct": round(100.0 * correct / max(1, len(questions)), 2),
        "repetition_loops": loops,
        "elapsed_s": round(time.time() - t0, 1),
        "samples": samples,
    }


def _make_live_triton_forward(
    sparse_experts: Prog8SparseMoEExperts,
    dense_experts: Prog8SparseMoEExperts,
):
    """Routes prefill (N > 128) through zero-VRAM dense 2+2+4 and decode (N <= 128) through live Triton Prog8!"""

    def _patched_forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(hidden_states)
        n_tok = hidden_states.shape[0]
        if n_tok > 128:
            # Chunked dense 2+2+4 for prefill (strict 0% duplicate w13 VRAM!)
            for s in range(0, n_tok, 128):
                e = min(n_tok, s + 128)
                dense_experts.apply(
                    out[s:e],
                    hidden_states[s:e],
                    None,
                    None,
                    top_k_weights[s:e],
                    top_k_index[s:e],
                )
        else:
            sparse_experts.apply(out, hidden_states, None, None, top_k_weights, top_k_index)
        return out

    return _patched_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/gsm8k_qwen35_35b_l40s.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[Live Triton GSM8K Gate] Loading {args.model} on {gpu_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cc = torch.cuda.get_device_capability(0)
    load_kwargs: dict = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if "FP8" in args.model.upper() and cc < (8, 9):
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = "auto"

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    archs = getattr(cfg, "architectures", []) or []
    if any("ConditionalGeneration" in a for a in archs):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    layers = _find_moe_layers(model)
    n_layers = len(layers)
    fa_flags = [_is_full_attn_layer(layers[i], i) for i in range(n_layers)]
    vram_dense_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
    print(
        f"[Live Triton GSM8K Gate] Loaded {n_layers} layers in {time.time() - t0:.1f}s | "
        f"Dense VRAM={vram_dense_gb:.2f} GiB",
        flush=True,
    )

    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [ds[i]["question"] for i in range(args.num_samples)]
    golds = [_extract_gold(ds[i]["answer"]) for i in range(args.num_samples)]

    print("\n[1/2] Evaluating Official Dense [128,128] FP8...", flush=True)
    res_dense = _eval_gsm8k_batched(model, tok, questions, golds, args.batch_size, args.max_new_tokens)

    # Convert ALL 40 layers IN-PLACE into Live Production Triton Prog8SparseMoEExperts (strict 0% dup w13!)
    print("\n[2/2] Packing ALL 40 layers in-place into Live Production Triton Prog8SparseMoEExperts...", flush=True)
    tp = time.time()
    cfg_ship = Prog8MoEConfig(
        sparsity=0.80,
        full_attn_sparsity=0.70,
        cold_me_thresh=8,
        svd_rank=16,
        use_3tier=True,
        full_attn_use_3tier=False,
    )
    cfg_dense_prefill = Prog8MoEConfig(sparsity=0.0, cold_me_thresh=8, svd_rank=0)

    def _set_fwd(exps, fn):
        if hasattr(exps, "_old_forward"):
            exps._old_forward = fn
        else:
            exps.forward = fn

    with torch.no_grad():
        for i in range(n_layers):
            exps = layers[i].mlp.experts
            w13 = exps.gate_up_proj.data
            w2 = exps.down_proj.data
            w13_s = getattr(exps, "gate_up_proj_scale_inv", None)
            w2_s = getattr(exps, "down_proj_scale_inv", None)
            packed = pack_hf_fp8_experts(
                w13,
                w2,
                w13_scale=w13_s.data if w13_s is not None else None,
                w2_scale=w2_s.data if w2_s is not None else None,
                config=cfg_ship,
                keep_dense_w13_for_hot=False,
            )
            # Physically free original dense FP8 gate_up_proj & down_proj from VRAM!
            del exps.gate_up_proj
            del exps.down_proj
            if hasattr(exps, "gate_up_proj_scale_inv"):
                del exps.gate_up_proj_scale_inv
            if hasattr(exps, "down_proj_scale_inv"):
                del exps.down_proj_scale_inv
            lt = "full_attn" if fa_flags[i] else "deltanet"
            sp_mod = Prog8SparseMoEExperts(packed, config=cfg_ship, layer_type=lt)
            dn_mod = Prog8SparseMoEExperts(packed, config=cfg_dense_prefill, layer_type=lt)
            exps._prog8_sparse = sp_mod
            exps._prog8_dense = dn_mod
            _set_fwd(exps, types.MethodType(_make_live_triton_forward(sp_mod, dn_mod), exps))
    gc.collect()
    torch.cuda.empty_cache()
    vram_prog8_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
    print(
        f"[Live Triton GSM8K Gate] Packed all {n_layers} layers in {time.time() - tp:.1f}s | "
        f"Live VRAM: Dense={vram_dense_gb:.2f} GiB -> Prog8+SVD16={vram_prog8_gb:.2f} GiB "
        f"(+{100.0 * (vram_prog8_gb - vram_dense_gb) / vram_dense_gb:.2f}%)",
        flush=True,
    )

    print("\n[2/2] Evaluating Live Production Triton Prog8SparseMoEExperts (HYBRID DN80-3Tier / FA70-8b + SVD-r16)...", flush=True)
    res_hybrid = _eval_gsm8k_batched(model, tok, questions, golds, args.batch_size, args.max_new_tokens)
    retention = 100.0 * res_hybrid["accuracy_pct"] / max(res_dense["accuracy_pct"], 1e-6)
    report = {
        "model": args.model,
        "gpu": gpu_name,
        "live_triton_kernel": True,
        "strict_zero_duplicate_w13": True,
        "vram_dense_gib": vram_dense_gb,
        "vram_prog8_svd16_gib": vram_prog8_gb,
        "dense_fp8": res_dense,
        "hybrid_dn80_fa70_svd16_live_triton": res_hybrid,
        "relative_retention_pct": round(retention, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(
        f"\n[LIVE TRITON GSM8K SUMMARY] Dense={res_dense['accuracy_pct']:.2f}% | "
        f"Live Triton HYBRID={res_hybrid['accuracy_pct']:.2f}% | Retention={retention:.2f}% | "
        f"Loops={res_hybrid['repetition_loops']} | VRAM={vram_dense_gb:.2f}->{vram_prog8_gb:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
