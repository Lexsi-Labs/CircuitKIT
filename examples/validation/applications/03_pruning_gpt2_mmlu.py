"""Application: structural pruning on GPT-2 / MMLU (real custom HF data).

Mirrors `01_pruning_gpt2_ioi.py` but on a real HF dataset. The metric is
the per-example "answer logit gap": for each MCQ, the logit of the correct
choice token minus the maximum logit of the incorrect choice tokens.
Higher means the pruned model still favours the correct answer.

For each algorithm:
  1. Pull CircuitScores from cache (or run discovery).
  2. Weight-zero-prune the lowest-scored 30% of nodes.
  3. Measure the answer-logit gap on a held-out MMLU batch.
  4. Compare against random pruning at the same sparsity.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "03_pruning_gpt2_mmlu"
MODEL = "gpt2"
TASK = "mmlu"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _mmlu_logit_gap(model: HookedTransformer, dataloader) -> float:
    """Mean per-example (correct_logit - max_incorrect_logit) on the dataloader.

    Labels are [batch, n_choices]: column 0 is the correct token id, columns
    1.. are incorrect. Higher gap means the model favours the correct choice.
    """
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    gaps = []
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            logits = model(clean_tokens, attention_mask=attn_mask)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            row = label[i] if label.dim() == 2 else label[i:i + 1]
            correct = int(row[0].item())
            incorrect = [int(c.item()) for c in row[1:]] if row.numel() > 1 else []
            if not incorrect:
                continue
            correct_logit = logits[i, last, correct].item()
            max_incorrect = max(logits[i, last, c].item() for c in incorrect)
            gaps.append(correct_logit - max_incorrect)
    return sum(gaps) / max(1, len(gaps))


def _bottom_k(node_scores, sparsity):
    n = max(1, int(len(node_scores) * sparsity))
    return [k for k, _ in sorted(node_scores.items(), key=lambda kv: abs(kv[1]))[:n]]


def _run_one_algo(algo: str, dataloader, model_clean: HookedTransformer):
    from circuitkit.applications.pruning.weight_pruner import apply_node_pruning_to_weights

    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    except Exception as exc:
        return {"algorithm": algo, "error": f"discovery: {type(exc).__name__}: {exc}"}
    cs = cell["scores"]
    t0 = time.time()

    nodes_to_prune = _bottom_k(cs.node_scores, sparsity=0.3)
    pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(pruned, nodes_to_prune)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"prune: {exc}"}
    circuit_gap = _mmlu_logit_gap(pruned, dataloader)
    del pruned

    rng = random.Random(42)
    random_nodes = rng.sample(list(cs.node_scores.keys()),
                              max(1, int(len(cs.node_scores) * 0.3)))
    rand_pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(rand_pruned, random_nodes)
        random_gap = _mmlu_logit_gap(rand_pruned, dataloader)
    except Exception as exc:  # noqa: BLE001
        random_gap = None
    del rand_pruned

    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_nodes_pruned": len(nodes_to_prune),
        "circuit_logit_gap": round(circuit_gap, 4),
        "random_logit_gap": round(random_gap, 4) if random_gap is not None else None,
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    task_spec = get_task(TASK)
    dl_cfg = {
        "algorithm": "eap", "task": TASK, "level": "node", "batch_size": 4,
        "data_params": {"num_examples": 32},
    }
    dataloader = task_spec.build_dataloader(model, dl_cfg, device)

    baseline_gap = _mmlu_logit_gap(model, dataloader)
    print(f"GPT-2 baseline MMLU answer-logit gap: {baseline_gap:.4f}\n", flush=True)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, dataloader, model)
        row["baseline_logit_gap"] = round(baseline_gap, 4)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(
                f"  {algo:25s}  circuit={row['circuit_logit_gap']:>7}  "
                f"random={row['random_logit_gap']!s:>7}  "
                f"({row['wall_seconds']}s)",
                flush=True,
            )

    lines = [
        f"# Application 3 — Structural Pruning on GPT-2 / MMLU (sparsity=0.3)",
        "",
        f"Baseline (no pruning) MMLU answer-logit gap: **{baseline_gap:.4f}**",
        "",
        "| Algorithm | Circuit-pruned | Random-pruned | Note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        note = r.get("error", "")
        lines.append(
            f"| `{r['algorithm']}` | "
            f"{r.get('circuit_logit_gap', '—')} | "
            f"{r.get('random_logit_gap', '—')} | "
            f"{note} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.structural_pruner",
        "input": {"model": MODEL, "task": TASK, "sparsity": 0.3, "n_algos": len(ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {
            "baseline_logit_gap": round(baseline_gap, 4),
            "n_algos_ok": n_ok, "n_algos_total": len(ALGOS),
        },
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
