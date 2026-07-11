"""E4 — Custom data end-to-end: jailbreak CSV on Qwen2.5-1.5B-Instruct.

Demonstrates the two custom-data paths side by side:
  A. Paired (EAP-IG)  — clean + corrupt prompts
  B. Clean-only (IBCircuit) — no corruption pairs needed

Both use the bundled jailbreak_binary.csv on Qwen2.5-1.5B-Instruct.
For each path, 5-pillar faithfulness evaluation is run.

Output
------
results/e4_custom_data_jailbreak/results.json

Run
---
    python examples/experiments/e4_custom_data_jailbreak.py
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import torch

from circuitkit import Pipeline

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
N_EXAMPLES = 200
ALPHA = 0.3
OUTPUT_DIR = Path("results/e4_custom_data_jailbreak")

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "src", "circuitkit", "data", "task_data",
    "tasks", "binary_align", "jailbreak_binary.csv",
)

CLEAN_PROMPT = (
    "User: {harmful_req}\n"
    "Please answer with only 'Yes' or 'No'.\n"
    "Assistant:"
)
CORRUPT_PROMPT = (
    "User: {benign_req}\n"
    "Please answer with only 'Yes' or 'No'.\n"
    "Assistant:"
)
CLEAN_ANSWER = " No"
CORRUPT_ANSWER = " Yes"

print("Clean prompt template (harmful request -> refusal):")
print(f"  {CLEAN_PROMPT!r}")
print(f"  Expected answer: {CLEAN_ANSWER!r}")
print(f"\nCorrupt prompt template (benign request -> compliance):")
print(f"  {CORRUPT_PROMPT!r}")
print(f"  Expected answer: {CORRUPT_ANSWER!r}")

PILLARS = ["patching", "ablation", "baselines", "robustness"]
N_STABILITY_RUNS = 3


def run_paired() -> dict:
    """Section A: Paired data with EAP-IG."""
    out = OUTPUT_DIR / "paired"
    out.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline.from_custom_data(
        MODEL,
        data_path=CSV_PATH,
        clean_prompt=CLEAN_PROMPT,
        corrupt_prompt=CORRUPT_PROMPT,
        clean_answer=CLEAN_ANSWER,
        corrupt_answer=CORRUPT_ANSWER,
        task_name="jailbreak_paired",
        output_dir=str(out),
    )

    t0 = time.time()
    pipe.discover(
        algorithm="eap-ig",
        level="node",
        sparsity=ALPHA,
        n_examples=N_EXAMPLES,
        batch_size=1,
        ig_steps=3,
        seed=SEED,
    )
    wall = time.time() - t0

    pipe.evaluate(
        pillars=PILLARS,
        n_examples=N_EXAMPLES,
    )

    report = pipe.report
    result = {
        "method": "paired_eap-ig",
        "algorithm": "eap-ig",
        "wall_seconds": round(wall, 2),
        "n_circuit_nodes": len(pipe.circuit),
        "top_10_nodes": list(pipe.circuit.top_nodes(10).keys()),
        "P1_patching": float(report.patching_score) if report.patching_score is not None else None,
        "P2_ablation": float(report.ablation_score) if report.ablation_score is not None else None,
        "P3_stability": report.stability,
        "P4_robustness": report.robustness if hasattr(report, "robustness") else None,
        "P5_baselines": report.baseline_comparison,
    }

    pipe._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_clean_only() -> dict:
    """Section B: Clean-only data with IBCircuit."""
    out = OUTPUT_DIR / "clean_only"
    out.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline.from_custom_data(
        MODEL,
        data_path=CSV_PATH,
        clean_prompt=CLEAN_PROMPT,
        clean_answer=CLEAN_ANSWER,
        task_name="jailbreak_clean_only",
        output_dir=str(out),
    )

    t0 = time.time()
    pipe.discover(
        algorithm="ibcircuit",
        level="neuron",
        scope="heads",
        sparsity=ALPHA,
        n_examples=N_EXAMPLES,
        batch_size=8,
        num_epochs=1000,
        learning_rate=0.05,
        seed=SEED,
    )
    wall = time.time() - t0

    # Clean-only data: only sufficiency evaluation is available.
    # patching_score = full-model P(correct); ablation_score = circuit P(correct).
    pipe.evaluate(
        pillars=["patching", "ablation"],
        n_examples=N_EXAMPLES,
    )

    report = pipe.report
    result = {
        "method": "clean_only_ibcircuit",
        "algorithm": "ibcircuit",
        "wall_seconds": round(wall, 2),
        "n_circuit_nodes": len(pipe.circuit),
        "top_10_nodes": list(pipe.circuit.top_nodes(10).keys()),
        # For clean-only data these are sufficiency scores, not interchange ratios:
        # full_model_P_correct = P(correct answer | full model clean run)
        # circuit_P_correct    = P(correct answer | mean-ablated circuit)
        "full_model_P_correct": float(report.patching_score) if report.patching_score is not None else None,
        "circuit_P_correct": float(report.ablation_score) if report.ablation_score is not None else None,
        # Patching-based pillars are not applicable for clean-only data.
        "P1_patching": None,
        "P2_ablation": None,
    }

    pipe._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E4: Custom data — jailbreak CSV on Qwen2.5-1.5B-Instruct")
    print("=" * 70)

    results = []

    print("\n--- Section A: Paired (EAP-IG, node-level) ---")
    try:
        r = run_paired()
        results.append(r)
        p1 = f"{r['P1_patching']:.4f}" if r["P1_patching"] is not None else "invalid"
        p2 = f"{r['P2_ablation']:.4f}" if r["P2_ablation"] is not None else "invalid"
        print(f"  P1(patching)={p1}  P2(ablation)={p2}  "
              f"nodes={r['n_circuit_nodes']}  time={r['wall_seconds']:.1f}s")
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        results.append({"method": "paired_eap-ig", "error": str(exc)})

    print("\n--- Section B: Clean-only (IBCircuit, neuron-level) ---")
    try:
        r = run_clean_only()
        results.append(r)
        fm = f"{r['full_model_P_correct']:.4f}" if r.get("full_model_P_correct") is not None else "n/a"
        ci = f"{r['circuit_P_correct']:.4f}" if r.get("circuit_P_correct") is not None else "n/a"
        print(f"  full-model P(correct)={fm}  circuit P(correct)={ci}  "
              f"nodes={r['n_circuit_nodes']}  time={r['wall_seconds']:.1f}s")
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        results.append({"method": "clean_only_ibcircuit", "error": str(exc)})

    # Summary comparison
    # Section A uses interchange-patching faithfulness ratios.
    # Section B uses sufficiency (P(correct) under mean ablation), not ratios —
    # the two columns are not directly comparable across sections.
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Method':25s}  {'metric':>26s}  {'#nodes':>7s}  {'time':>7s}")
    print("-" * 70)
    for r in results:
        if "error" not in r:
            if r.get("method") == "paired_eap-ig":
                p1 = f"{r['P1_patching']:.4f}" if r["P1_patching"] is not None else "invalid"
                p2 = f"{r['P2_ablation']:.4f}" if r["P2_ablation"] is not None else "invalid"
                metric_str = f"P1={p1}  P2={p2}"
            else:
                fm = f"{r.get('full_model_P_correct', 'n/a'):.4f}" if r.get("full_model_P_correct") is not None else "n/a"
                ci = f"{r.get('circuit_P_correct', 'n/a'):.4f}" if r.get("circuit_P_correct") is not None else "n/a"
                metric_str = f"P(full)={fm}  P(circ)={ci}"
            print(
                f"{r['method']:25s}  {metric_str:>26s}  {r['n_circuit_nodes']:>7d}  "
                f"{r['wall_seconds']:>6.1f}s"
            )

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "E4",
                "model": MODEL,
                "csv": CSV_PATH,
                "seed": SEED,
                "alpha": ALPHA,
                "results": results,
            },
            f, indent=2, default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
