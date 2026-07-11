"""Application: circuit-aware FP4 quantization on Llama-3.2-1B.

Compares per-algorithm circuit-aware quantization vs uniform-FP4
baseline. The hypothesis from scripts/q3 (canonical paper): keeping
the top-N circuit-relevant layers at fp16 and FP4-quantizing the rest
preserves more task accuracy than uniform FP4.

Cell stages:
  1. Discover the algorithm's circuit on a real custom task (MMLU).
  2. Identify the top-30% layers that contain the highest-scoring
     circuit components.
  3. Build a BitsAndBytesConfig that skips those layers (they stay
     fp16) and FP4-quantizes the rest.
  4. Load Llama-3.2-1B with that config and measure MMLU answer
     logit gap on a held-out batch.
  5. Compare against uniform-FP4 (no skips) baseline.

Per-algo metric: (circuit_skip_score - uniform_fp4_score). Higher
means the algorithm-located layers carry more of the MMLU-relevant
computation.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "19_quantization_llama1b"
MODEL = "meta-llama/Llama-3.2-1B"
TASK = "mmlu"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _circuit_top_layers(node_scores, n_top: int) -> list:
    """Return the top-N layer indices ranked by sum-of-abs scores at that layer."""
    by_layer = {}
    for name, score in node_scores.items():
        m = re.match(r"A(\d+)\.\d+", name) or re.match(r"MLP (\d+)", name)
        if not m:
            continue
        l = int(m.group(1))
        by_layer[l] = by_layer.get(l, 0.0) + abs(score)
    return [l for l, _ in sorted(by_layer.items(), key=lambda kv: kv[1],
                                 reverse=True)[:n_top]]


def _build_skip_modules(layer_idxs: list) -> list:
    base = ["lm_head"]
    for i in layer_idxs:
        base.append(f"model.layers.{i}")
    return base


def _load_quantized(model_name: str, skip_modules: list, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=cfg, device_map=device,
    )
    return model, tok


def _mmlu_score(model, tok, n=8) -> float:
    """Mean answer-letter logit gap on a tiny MMLU subset."""
    from datasets import load_dataset
    ds = list(load_dataset("cais/mmlu", "high_school_world_history",
                           split="test", streaming=True).take(n))
    gaps = []
    LETTERS = ["A", "B", "C", "D"]
    for ex in ds:
        question = ex["question"]
        choices = ex["choices"]
        ans_idx = ex["answer"]
        prompt = f"{question}\n"
        for i, c in enumerate(choices):
            prompt += f"{LETTERS[i]}. {c}\n"
        prompt += "Answer:"
        in_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            logits = model(in_ids).logits
        last_logits = logits[0, -1]
        choice_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][0]
                      for L in LETTERS]
        correct_logit = last_logits[choice_ids[ans_idx]].item()
        wrong_logits = [last_logits[c].item() for i, c in enumerate(choice_ids)
                        if i != ans_idx]
        gaps.append(correct_logit - max(wrong_logits))
    return sum(gaps) / max(1, len(gaps))


def _run_one_algo(algo: str):
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=12,
                                    precision="bfloat16")
    except Exception as exc:  # noqa: BLE001
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"algorithm": algo,
                "error": f"discovery: {type(exc).__name__}: {str(exc)[:200]}"}
    cs = cell["scores"]
    t0 = time.time()
    n_layers = 16  # Llama-3.2-1B has 16 layers
    n_top = max(1, n_layers * 3 // 10)  # top-30%
    keep_layers = _circuit_top_layers(cs.node_scores, n_top)
    if not keep_layers:
        return {"algorithm": algo, "error": "no layers in circuit"}

    skip = _build_skip_modules(keep_layers)
    try:
        m, t = _load_quantized(MODEL, skip)
        score = _mmlu_score(m, t, n=8)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "kept_layers": keep_layers,
        "circuit_aware_score": round(score, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)

    # Uniform FP4 baseline (no skips beyond lm_head). Run once.
    print("Building uniform-FP4 baseline...")
    try:
        m, t = _load_quantized(MODEL, ["lm_head"])
        baseline_score = _mmlu_score(m, t, n=8)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  uniform FP4 score: {baseline_score:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  uniform FP4 baseline FAILED: {exc}")
        baseline_score = float("nan")

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo)
        if "error" not in row and baseline_score == baseline_score:
            row["uniform_fp4_score"] = round(baseline_score, 4)
            row["circuit_advantage"] = round(
                row["circuit_aware_score"] - baseline_score, 4
            )
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  ca={row['circuit_aware_score']}  "
                  f"adv={row.get('circuit_advantage', '—')}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = ["# Application 19 — Circuit-aware FP4 quantization on Llama-3.2-1B",
             "",
             f"Uniform FP4 baseline MMLU answer-logit gap: **{baseline_score:.4f}**",
             "",
             "| Algorithm | Kept layers (fp16) | Circuit-aware | Δ vs uniform |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['algorithm']}` | "
                     f"{r.get('kept_layers', '—')} | "
                     f"{r.get('circuit_aware_score', '—')} | "
                     f"{r.get('circuit_advantage', '—')} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))
    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.quantization_llama1b",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS)},
        "metrics": {"baseline_score": (round(baseline_score, 4)
                                       if baseline_score == baseline_score else None),
                    "n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
