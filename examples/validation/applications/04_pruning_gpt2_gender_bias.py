"""Application: structural pruning on GPT-2 / gender_bias for every algorithm.

Pruning preserves gender-bias circuit (built-in TaskSpec).

Metric: per-example logit gap between the correct-answer token and the
strongest competing-answer token. Higher means the pruned model still
favours the correct choice. Random-pruning serves as the contextual
baseline.
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

# Reuse the custom-task registration helper from the benchmark layer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

SCRIPT_NAME = "04_pruning_gpt2_gender_bias"
MODEL = "gpt2"
TASK = "gender_bias"
NEEDS_CUSTOM_REGISTRATION = False

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _logit_gap(model, dataloader) -> float:
    """Per-example (correct_logit - max_incorrect_logit), averaged."""
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


def _run_one_algo(algo, dataloader, model_clean):
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
    except Exception as exc:
        return {"algorithm": algo, "error": f"prune: {exc}"}
    circuit_gap = _logit_gap(pruned, dataloader)
    del pruned

    rng = random.Random(42)
    random_nodes = rng.sample(list(cs.node_scores.keys()),
                              max(1, int(len(cs.node_scores) * 0.3)))
    rand_pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(rand_pruned, random_nodes)
        random_gap = _logit_gap(rand_pruned, dataloader)
    except Exception:
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
    if NEEDS_CUSTOM_REGISTRATION:
        from _custom_tasks import register_all
        register_all()

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
        "model_name": MODEL,
    }
    dataloader = task_spec.build_dataloader(model, dl_cfg, device)

    baseline = _logit_gap(model, dataloader)
    print(f"GPT-2 baseline {TASK} logit gap: {baseline:.4f}\n", flush=True)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, dataloader, model)
        row["baseline_logit_gap"] = round(baseline, 4)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  circuit={row['circuit_logit_gap']:>7}  "
                  f"random={row['random_logit_gap']!s:>7}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = [
        f"# Application: pruning on GPT-2 / {TASK} (sparsity=0.3)",
        "",
        f"Baseline (no pruning) logit gap: **{baseline:.4f}**",
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
        "module": "applications.weight_pruner",
        "input": {"model": MODEL, "task": TASK, "sparsity": 0.3, "n_algos": len(ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {
            "baseline_logit_gap": round(baseline, 4),
            "n_algos_ok": n_ok, "n_algos_total": len(ALGOS),
        },
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
