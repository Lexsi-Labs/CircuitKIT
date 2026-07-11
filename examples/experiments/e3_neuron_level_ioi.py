"""E3 — Neuron-level IOI on Llama-3.2-1B.

Compares neuron-level discovery (with post_act MLP hook) against a
node-level reference using both EAP and EAP-IG on Llama-3.2-1B / IOI.

For each of the four runs (2 algos × 2 levels) it reports:
  - Pillar-1 (patching) and Pillar-2 (ablation) faithfulness
  - Number of pruned components
  - Wall-clock time

Output
------
results/e3_neuron_level_ioi/results.json

Run
---
    python examples/experiments/e3_neuron_level_ioi.py
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from circuitkit.api import discover_circuit, evaluate_circuit

MODEL = "meta-llama/Llama-3.2-1B"
TASK = "ioi"
ALPHA = 0.3
N_EXAMPLES = 256
SEED = 42
BATCH_SIZE = 1
IG_STEPS = 3

OUTPUT_DIR = Path("results/e3_neuron_level_ioi")

RUNS = [
    {"algorithm": "eap",    "level": "node"},
    {"algorithm": "eap",    "level": "neuron"},
    {"algorithm": "eap-ig", "level": "node"},
    {"algorithm": "eap-ig", "level": "neuron"},
]


def run_one(algo: str, level: str) -> dict:
    tag = f"{algo}_{level}"
    artifact = OUTPUT_DIR / f"{tag}.pt"

    discovery_cfg: dict = {
        "algorithm": algo,
        "task": TASK,
        "level": level,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "data_params": {"num_examples": N_EXAMPLES},
    }

    if algo == "eap-ig":
        discovery_cfg["ig_steps"] = IG_STEPS

    if level == "neuron":
        discovery_cfg["mlp_hook"] = "post_act"

    config = {
        "model": {"name": MODEL, "precision": "bfloat16"},
        "discovery": discovery_cfg,
        "pruning": {"target_sparsity": ALPHA, "scope": "both"},
        "output_path": str(artifact),
        "eval": {
            "full_faithfulness_eval": True,
            "pillars": ["patching", "ablation"],
        },
    }

    t0 = time.time()
    pruned = discover_circuit(config)
    wall_discover = time.time() - t0

    t1 = time.time()
    report = evaluate_circuit(config)
    wall_eval = time.time() - t1

    if isinstance(pruned, dict):
        n_heads_pruned = sum(len(v) for v in pruned.get("heads", {}).values())
        n_mlp_pruned = sum(len(v) for v in pruned.get("mlp", {}).values())
        n_pruned = n_heads_pruned + n_mlp_pruned
    else:
        n_pruned = len(pruned) if pruned else 0

    result = {
        "algorithm": algo,
        "level": level,
        "wall_discover_s": round(wall_discover, 2),
        "wall_eval_s": round(wall_eval, 2),
        "n_pruned_components": n_pruned,
        "P1_patching": float(report.patching_score),
        "P2_ablation": float(report.ablation_score),
    }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    header = (
        f"{'algo':10s}  {'level':8s}  {'P1':>8s}  {'P2':>8s}  "
        f"{'#pruned':>8s}  {'disc(s)':>8s}  {'eval(s)':>8s}"
    )
    print(header)
    print("-" * len(header))

    for run_cfg in RUNS:
        algo, level = run_cfg["algorithm"], run_cfg["level"]
        tag = f"{algo}/{level}"
        try:
            r = run_one(algo, level)
            results.append(r)
            print(
                f"{r['algorithm']:10s}  {r['level']:8s}  "
                f"{r['P1_patching']:>8.4f}  {r['P2_ablation']:>8.4f}  "
                f"{r['n_pruned_components']:>8d}  "
                f"{r['wall_discover_s']:>8.1f}  {r['wall_eval_s']:>8.1f}"
            )
        except Exception as exc:
            print(f"{tag:20s}  ERROR: {type(exc).__name__}: {str(exc)[:60]}")
            results.append({"algorithm": algo, "level": level, "error": str(exc)})

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "E3",
                "model": MODEL,
                "task": TASK,
                "alpha": ALPHA,
                "seed": SEED,
                "results": results,
            },
            f, indent=2, default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
