"""Application: structural pruning on GPT-2 / IOI for every algorithm.

For each discovery algorithm:
  1. Pull CircuitScores from cache (or run discovery if missing).
  2. Run StructuralPruner at target sparsity 30%.
  3. Compare pruned-model IOI logit-difference vs the original model
     and vs a randomly-pruned baseline at the same sparsity.

The metric: per-example logit_diff(IO - S). Higher is better.
- Original baseline: full GPT-2 — should be high positive (model knows IOI).
- Circuit-pruned (algo-guided): should be higher than random-pruned.
- Random-pruned: usually destroys the IOI signal.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "01_pruning_gpt2_ioi"
MODEL = "gpt2"
TASK = "ioi"

# Smaller algo set for the application sweep — eap-exact / atp-gd / eap-gp
# are slow and we already have them cached from the benchmark layer.
ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _ioi_logit_diff(model: HookedTransformer, dataloader) -> float:
    """Mean logit difference (correct - incorrect) over the dataloader.

    EAP collate_EAP returns labels as a tensor [batch, 2] for IOI:
    column 0 = correct (IO) token id, column 1 = incorrect (S) token id.
    """
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    diffs = []
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            logits = model(clean_tokens, attention_mask=attn_mask)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            io_id = int(label[i, 0].item())
            s_id = int(label[i, 1].item())
            d = logits[i, last, io_id].item() - logits[i, last, s_id].item()
            diffs.append(d)
    return sum(diffs) / max(1, len(diffs))


def _bottom_k_node_names(node_scores, sparsity: float):
    """Return the lowest-scoring fraction of nodes (these are pruned)."""
    n_drop = max(1, int(len(node_scores) * sparsity))
    return [n for n, _ in sorted(
        node_scores.items(), key=lambda kv: abs(kv[1])
    )[:n_drop]]


def _run_one_algo(algo: str, dataloader, model_clean: HookedTransformer):
    """Weight-zero pruning: keeps topology intact so use_split_qkv_input
    still works on the pruned model. We deepcopy → prune → measure → discard.
    """
    import copy
    import random
    from circuitkit.applications.pruning.weight_pruner import apply_node_pruning_to_weights

    cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=32)
    cs = cell["scores"]
    t0 = time.time()

    # Circuit-guided: prune the lowest-scored 30% of nodes.
    nodes_to_prune = _bottom_k_node_names(cs.node_scores, sparsity=0.3)
    pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(pruned, nodes_to_prune)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"prune: {exc}"}
    circuit_diff = _ioi_logit_diff(pruned, dataloader)
    del pruned

    # Random baseline at the same sparsity, fixed seed for fairness.
    rng = random.Random(42)
    random_nodes = rng.sample(list(cs.node_scores.keys()),
                              max(1, int(len(cs.node_scores) * 0.3)))
    rand_pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(rand_pruned, random_nodes)
        random_diff = _ioi_logit_diff(rand_pruned, dataloader)
    except Exception as exc:  # noqa: BLE001
        random_diff = None
    del rand_pruned

    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_nodes_pruned": len(nodes_to_prune),
        "circuit_logit_diff": round(circuit_diff, 4),
        "random_logit_diff": round(random_diff, 4) if random_diff is not None else None,
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model ONCE — pruner deepcopies it per cell.
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    # Build one IOI dataloader to evaluate on.
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    task_spec = get_task(TASK)
    dl_cfg = {
        "algorithm": "eap", "task": TASK, "level": "node", "batch_size": 4,
        "data_params": {"num_examples": 64},
    }
    dataloader = task_spec.build_dataloader(model, dl_cfg, device)

    baseline_diff = _ioi_logit_diff(model, dataloader)
    print(f"GPT-2 baseline IOI logit-diff: {baseline_diff:.4f}\n", flush=True)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, dataloader, model)
        row["baseline_logit_diff"] = round(baseline_diff, 4)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(
                f"  {algo:25s}  circuit={row['circuit_logit_diff']:>7}  "
                f"random={row['random_logit_diff']!s:>7}  "
                f"({row['wall_seconds']}s)",
                flush=True,
            )

    # Markdown table.
    lines = [
        f"# Application 1 — Structural Pruning on GPT-2 / IOI (sparsity=0.3)",
        "",
        f"Baseline (no pruning) IOI logit-diff: **{baseline_diff:.4f}**",
        "",
        "| Algorithm | Circuit-pruned | Random-pruned | Note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        note = r.get("error", "")
        lines.append(
            f"| `{r['algorithm']}` | "
            f"{r.get('circuit_logit_diff', '—')} | "
            f"{r.get('random_logit_diff', '—')} | "
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
            "baseline_logit_diff": round(baseline_diff, 4),
            "n_algos_ok": n_ok, "n_algos_total": len(ALGOS),
        },
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
