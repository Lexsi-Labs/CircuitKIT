"""E2 — Cross-family discovery.

Runs EAP-IG at node level (α=0.3, n=3 IG steps) across five model
families on both IOI and Greater-Than.  Reports Pillar-1 (patching),
Pillar-2 (ablation), and Pillar-3 (stability) for each combination.

IOI is an attention-routing task (duplicate-token → induction →
S-inhibition → name-mover heads); Greater-Than is an MLP-driven numeric
magnitude reasoning task.  Together they probe orthogonal circuit types
across model families.

Models
------
- GPT-2              (117 M, OpenAI)
- pythia-1.4b        (1.4 B, EleutherAI)
- Llama-3.2-1B       (1.2 B, Meta)
- gemma-2-2b         (2.6 B, Google)
- Qwen2.5-1.5B       (1.5 B, Alibaba)
- microsoft/phi-2    (2.7 B, Microsoft)

Output
------
results/e2_cross_family/results.json

Run
---
    python examples/experiments/e2_cross_family_discovery.py
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import torch

from circuitkit import Pipeline

MODELS = [
    "gpt2",
    "EleutherAI/pythia-1.4b",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-1.5B",
    "microsoft/phi-2",
]

TASKS = ["ioi", "greater_than"]

ALGO = "eap-ig"
ALPHA = 0.3
IG_STEPS = 3
N_EXAMPLES = 256
SEED = 42

OUTPUT_DIR = Path("results/e2_cross_family")

PILLARS = ["patching", "ablation", "baselines"]
N_STABILITY_RUNS = 3


def short_name(model: str) -> str:
    return model.split("/")[-1]


def run_one(model: str, task: str) -> dict:
    tag = f"{short_name(model)}_{task}"
    out = OUTPUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    
    n_examples = N_EXAMPLES

    pipe = Pipeline(model, task=task, output_dir=str(out))

    t0 = time.time()
    pipe.discover(
        algorithm=ALGO,
        level="node",
        sparsity=ALPHA,
        n_examples=n_examples,
        batch_size=1,
        ig_steps=IG_STEPS,
        seed=SEED,
        scope="both",
    )
    wall = time.time() - t0

    pipe.evaluate(
        pillars=PILLARS,
        n_examples=n_examples,
        # n_stability_runs=N_STABILITY_RUNS,
    )

    report = pipe.report
    stability_jaccard = None
    if report.stability:
        stability_jaccard = report.stability.get("mean_jaccard")

    result = {
        "model": model,
        "model_short": short_name(model),
        "task": task,
        "algorithm": ALGO,
        "wall_seconds": round(wall, 2),
        "n_circuit_nodes": len(pipe.circuit),
        "P1_patching": float(report.patching_score) if report.patching_score is not None else None,
        "P2_ablation": float(report.ablation_score) if report.ablation_score is not None else None,
        "P3_stability_jaccard": float(stability_jaccard) if stability_jaccard is not None else None,
        "top_10_nodes": list(pipe.circuit.top_nodes(10).keys()),
    }

    # Free GPU memory before moving to next model.
    pipe._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    header = (
        f"{'model':20s}  {'task':8s}  {'P1':>8s}  {'P2':>8s}  "
        f"{'P3(Jacc)':>8s}  {'time':>7s}"
    )
    print(header)
    print("-" * len(header))

    for model in MODELS:
        for task in TASKS:
            tag = f"{short_name(model)}/{task}"
            try:
                r = run_one(model, task)
                results.append(r)
                p1 = f"{r['P1_patching']:.4f}" if r["P1_patching"] is not None else "invalid"
                p2 = f"{r['P2_ablation']:.4f}" if r["P2_ablation"] is not None else "invalid"
                p3 = f"{r['P3_stability_jaccard']:.4f}" if r["P3_stability_jaccard"] is not None else "n/a"
                print(
                    f"{r['model_short']:20s}  {task:8s}  "
                    f"{p1:>8s}  {p2:>8s}  "
                    f"{p3:>8s}  {r['wall_seconds']:>6.1f}s"
                )
            except Exception as exc:
                print(f"{tag:30s}  ERROR: {type(exc).__name__}: {str(exc)[:60]}")
                results.append({"model": model, "task": task, "error": str(exc)})

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "E2",
                "algorithm": ALGO,
                "alpha": ALPHA,
                "ig_steps": IG_STEPS,
                "seed": SEED,
                "results": results,
            },
            f, indent=2, default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
