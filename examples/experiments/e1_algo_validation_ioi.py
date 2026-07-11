"""E1 — Algorithm validation on IOI (node level).

Runs all six stable algorithms on GPT-2 Small with α=0.3, seed 42,
256 clean / 256 corrupt examples.  Reports Pillar-1 (patching),
Pillar-2 (ablation), wall-clock time, and Jaccard overlap of the
top-k circuit nodes with the canonical IOI head classes.

Output
------
results/e1_algo_validation_ioi/results.json   — structured metrics
stdout                                        — human-readable table

Run
---
    python examples/experiments/e1_algo_validation_ioi.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from circuitkit import Pipeline

ALGOS = ["eap", "eap-ig", "eap-gp", "cdt","ibcircuit", "acdc"]

SEED = 42
ALPHA = 0.3
N_EXAMPLES = 256
MODEL = "gpt2"
TASK = "ioi"

OUTPUT_DIR = Path("results/e1_algo_validation_ioi")

# Canonical IOI head classes (from Wang et al., 2022).
CANONICAL_IOI = {
    "name_movers":      ["A9.9", "A10.0", "A9.6"],
    "negative_movers":  ["A10.7", "A11.10"],
    "s_inhibition":     ["A7.3", "A7.9", "A8.6", "A8.10"],
    "induction":        ["A5.5", "A6.9"],
    "duplicate_token":  ["A0.1", "A3.0"],
    "previous_token":   ["A2.2", "A4.11"],
}
ALL_CANONICAL = {h for heads in CANONICAL_IOI.values() for h in heads}


def overlap_with_canonical(top_nodes: list[str]) -> dict:
    top_set = set(top_nodes)
    per_class = {}
    for cls, heads in CANONICAL_IOI.items():
        cls_set = set(heads)
        inter = top_set & cls_set
        per_class[cls] = {
            "overlap": sorted(inter),
            "recall": len(inter) / len(cls_set) if cls_set else 0.0,
        }
    overall = top_set & ALL_CANONICAL
    return {
        "per_class": per_class,
        "overall_recall": len(overall) / len(ALL_CANONICAL),
        "overall_precision": len(overall) / len(top_set) if top_set else 0.0,
        "overall_jaccard": (
            len(overall) / len(top_set | ALL_CANONICAL)
            if (top_set | ALL_CANONICAL) else 0.0
        ),
    }


def run_one(algo: str) -> dict:
    out = OUTPUT_DIR / algo
    out.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(MODEL, task=TASK, output_dir=str(out))

    discover_kw: dict = dict(
        algorithm=algo,
        level="node",
        sparsity=ALPHA,
        n_examples=N_EXAMPLES,
        batch_size=1,
        seed=SEED,
        scope="both",
    )
    if algo in ("eap-ig", "eap-gp"):
        discover_kw["ig_steps"] = 3
    if algo == "ibcircuit":
        discover_kw["num_epochs"] = 1000
        discover_kw["scope"] = "heads"
        # IBCircuit uses batch_size as a fixed-batch truncation cap, not as an
        # EAP accumulation step. batch_size=1 makes torch.std return NaN (N=1
        # sample std undefined). Use ≥2; 8 is the selector default and fits
        # comfortably on a single GPU for GPT-2.
        discover_kw["batch_size"] = 8

    t0 = time.time()
    pipe.discover(**discover_kw)
    wall = time.time() - t0

    pipe.evaluate(
        pillars=["patching", "ablation"],
        n_examples=N_EXAMPLES,
    )

    report = pipe.report
    top_nodes = list(pipe.circuit.top_nodes(len(ALL_CANONICAL)).keys())
    canonical = overlap_with_canonical(top_nodes)

    return {
        "algorithm": algo,
        "wall_seconds": round(wall, 2),
        "n_circuit_nodes": len(pipe.circuit),
        "patching_score": float(report.patching_score),
        "ablation_score": float(report.ablation_score),
        "top_nodes": top_nodes,
        "canonical_overlap": canonical,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    header = (
        f"{'algo':15s}  {'P1(patch)':>10s}  {'P2(ablat)':>10s}  "
        f"{'time(s)':>8s}  {'IOI Jacc':>9s}  {'IOI Rec':>8s}"
    )
    print(header)
    print("-" * len(header))

    for algo in ALGOS:
        try:
            r = run_one(algo)
            results.append(r)
            co = r["canonical_overlap"]
            print(
                f"{r['algorithm']:15s}  {r['patching_score']:>10.4f}  "
                f"{r['ablation_score']:>10.4f}  {r['wall_seconds']:>8.1f}  "
                f"{co['overall_jaccard']:>9.4f}  {co['overall_recall']:>8.4f}"
            )
        except Exception as exc:
            print(f"{algo:15s}  ERROR: {type(exc).__name__}: {str(exc)[:60]}")
            results.append({"algorithm": algo, "error": str(exc)})

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"experiment": "E1", "seed": SEED, "alpha": ALPHA, "results": results},
            f, indent=2, default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
