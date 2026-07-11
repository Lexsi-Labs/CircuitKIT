"""Circuit-aware hallucination detection on TruthfulQA, Llama-3.2-3B-Instruct.

Pipeline per (method, seed):
  1. Register TruthfulQA via the safety_datasets entry point and run
     discover_circuit() to get a circuit for the truthfulness task.
  2. Instantiate the HallucinationDetector with the top-K circuit
     components as the "on-circuit" mask.
  3. Score each TruthfulQA MC1 question: detector emits a probability
     that the model's answer is hallucinated based on off-circuit
     activation dominance.
  4. Use the MC1 ground truth (whether the model's top-1 answer is
     in the truthful_answers set) as the binary label.
  5. Report AUROC and accuracy at the optimal threshold.

Baselines:
  - Random-circuit detector (top-K random nodes, same budget)
  - Whole-model entropy baseline (token-entropy as hallucination signal)

Headline question: do circuit-aware detectors beat the entropy
baseline, and does method choice change AUROC by more than 5 points?
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "27_hallucination_truthfulqa_llama"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TASK = "truthfulqa"
SEEDS = [42, 143, 256]
METHODS = ["eap", "eap-ig", "eap-clean-corrupted", "atp-gd",
           "eap-gp", "relp"]
TOP_K = 32
N_EVAL = 200  # held-out TruthfulQA MC1 items


def _register_truthfulqa():
    from circuitkit.tasks.safety_datasets import register_truthfulqa
    from circuitkit.tasks.registry import is_task_registered
    if not is_task_registered(TASK):
        register_truthfulqa(max_records=400)


def _load_truthfulqa_eval(n: int) -> List[Dict]:
    """Load TruthfulQA MC1 split for evaluation (separate from discovery)."""
    from datasets import load_dataset
    raw = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for r in list(raw)[:n]:
        items.append({
            "question": r["question"],
            "choices": r["mc1_targets"]["choices"],
            "labels": r["mc1_targets"]["labels"],  # 1 == truthful
        })
    return items


def _model_top1(model, prompt: str) -> int:
    """Return top-1 next-token id."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return int(logits[0, -1].argmax().item())


def _is_truthful(model, item: Dict) -> bool:
    """Heuristic: model is 'truthful' if its top-token decoding starts
    with a token that prefixes a truthful choice."""
    pred_id = _model_top1(model, item["question"])
    pred = model.tokenizer.decode([pred_id]).strip().lower()
    for choice, label in zip(item["choices"], item["labels"]):
        if label == 1 and choice.lower().startswith(pred[:5]):
            return True
    return False


def _entropy_baseline_score(model, prompt: str) -> float:
    """Token-entropy at the answer position; higher = more uncertain."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, -1], dim=-1)
    ent = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return ent


def _auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC without sklearn dependency."""
    paired = sorted(zip(scores, labels), key=lambda sl: sl[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fpr = 0.0
    for _, lbl in reversed(paired):
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * tpr
        prev_fpr = fpr
    return float(auc)


def _circuit_off_circuit_ratio(
    model, prompt: str, circuit_mask: Dict[str, float],
) -> float:
    """Circuit-awareness score: off-circuit / (on-circuit + off-circuit) norms.

    High value (off-circuit dominates) = model is likely hallucinating
    because the task-specific circuit is not driving the prediction.
    Low value (circuit dominates) = model routes through the relevant
    circuit = more likely to be factual.

    Uses TransformerLens `run_with_cache` to extract per-head and per-MLP
    output norms at the last token position without requiring separate probes
    or a second model pass.
    """
    tok = model.tokenizer
    in_ids = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=512)["input_ids"].to(model.cfg.device)
    target_pos = in_ids.shape[1] - 1

    with torch.inference_mode():
        _, cache = model.run_with_cache(in_ids)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    circuit_norm = off_circuit_norm = 0.0
    for l in range(n_layers):
        # Attention heads
        result_key = f"blocks.{l}.attn.hook_result"
        if result_key in cache:
            head_out = cache[result_key][0, target_pos]  # [n_heads, d_head]
            for h in range(n_heads):
                norm = float(head_out[h].norm().item())
                if f"A{l}.{h}" in circuit_mask:
                    circuit_norm += norm
                else:
                    off_circuit_norm += norm
        # MLP
        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            norm = float(cache[mlp_key][0, target_pos].norm().item())
            if f"MLP {l}" in circuit_mask:
                circuit_norm += norm
            else:
                off_circuit_norm += norm

    total = circuit_norm + off_circuit_norm + 1e-8
    return off_circuit_norm / total  # high = off-circuit dominance = hallu-prone


def _cell(method: str, model_name: str, seed: int,
          out_dir: Path) -> Dict[str, object]:
    from transformer_lens import HookedTransformer

    t0 = time.time()
    cell = {"method": method, "seed": seed, "top_k": TOP_K}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        _register_truthfulqa()
        discovery = get_or_run_discovery(
            algorithm=method, model=model_name, task=TASK,
            num_examples=64, batch_size=1,
        )
        node_scores = discovery["node_scores"]
        top_nodes = sorted(node_scores.items(),
                            key=lambda kv: kv[1], reverse=True)[:TOP_K]
        circuit_mask = {n: s for n, s in top_nodes}
        cell["circuit_nodes"] = list(circuit_mask.keys())

        model = HookedTransformer.from_pretrained(model_name, device=device,
                                                   dtype=torch.float32)
        model.cfg.use_attn_result = True
        eval_items = _load_truthfulqa_eval(N_EVAL)

        # Circuit-aware detector: off-circuit dominance score.
        circ_scores = [
            _circuit_off_circuit_ratio(model, item["question"], circuit_mask)
            for item in eval_items
        ]
        labels = [int(_is_truthful(model, item)) for item in eval_items]
        # Higher off-circuit score = more likely hallucinated.
        # AUROC: treat hallu score as classifier, label=1 means truthful.
        # So we negate: lower circuit score (=truthful) should get higher rank.
        cell["circuit_auroc"] = round(_auroc([-s for s in circ_scores], labels), 4)

        # Entropy baseline
        ent_scores = [_entropy_baseline_score(model, item["question"])
                       for item in eval_items]
        cell["entropy_auroc"] = round(_auroc([-s for s in ent_scores], labels), 4)

        # Random-circuit detector
        random.seed(seed)
        all_nodes = list(node_scores.keys())
        rand_mask = {n: 1.0 for n in random.sample(all_nodes, min(TOP_K, len(all_nodes)))}
        rand_scores = [
            _circuit_off_circuit_ratio(model, item["question"], rand_mask)
            for item in eval_items
        ]
        cell["random_auroc"] = round(_auroc([-s for s in rand_scores], labels), 4)

        cell["circuit_advantage_over_entropy"] = round(
            cell["circuit_auroc"] - cell["entropy_auroc"], 4)
        cell["circuit_advantage_over_random"] = round(
            cell["circuit_auroc"] - cell["random_auroc"], 4)
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"

    cell["wall_s"] = round(time.time() - t0, 2)
    return cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    out_dir = make_results_dir(SCRIPT_NAME)
    rows: List[Dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            cell = _cell(method, args.model, seed, out_dir)
            rows.append(cell)
            print(f"[{method} seed={seed}] "
                  f"AUROC circuit={cell.get('circuit_auroc')} "
                  f"vs entropy={cell.get('entropy_auroc')} "
                  f"{cell.get('error') or 'ok'}")

    payload = {"script": SCRIPT_NAME, "model": args.model, "task": TASK,
               "top_k": TOP_K, "n_eval": N_EVAL,
               "method_seed_rows": rows}
    write_status(out_dir, payload)
    print(f"Wrote {out_dir / 'status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
