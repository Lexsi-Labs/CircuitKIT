#!/usr/bin/env python3
"""
04 - Faithfulness evaluation with the 6-pillar framework.

evaluate_circuit() scores a discovered circuit. By default it runs the two
fast pillars (causal patching + ablation). Setting eval.full_faithfulness_eval
turns on the full 6-pillar framework; you can also pick a subset via eval.pillars.

The 6 pillars (circuitkit.evaluation.pillars):
    1. patching        - does the circuit explain model behavior?
    2. ablation        - does ablating the circuit degrade behavior?
    3. stability       - is the circuit stable across re-discovery seeds?
    4. robustness      - does the circuit survive input corruptions?
    5. baselines       - how does it compare to random/magnitude baselines?
    6. generalization  - does the circuit transfer to a related task?

This script discovers a circuit, then runs a faithfulness evaluation.

Run:
    python examples/05-evaluate-faithfulness.py
"""

import os

from circuitkit.api import discover_circuit, evaluate_circuit

CONFIG = {
    "model": {"name": "gpt2", "precision": "bfloat16"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "level": "node",
        "batch_size": 2,
        "ig_steps": 2,
        "data_params": {"num_examples": 16},
    },
    "pruning": {"target_sparsity": 0.2, "scope": "heads"},
    "output_path": "./results/example_eval_circuit.pt",
    # The `eval` block controls faithfulness evaluation.
    "eval": {
        "full_faithfulness_eval": True,                      # 6-pillar report
        "pillars": ["patching", "ablation", "baselines", "stability"],
        "n_stability_runs": 2,                               # keep small for the demo
    },
}


def main():
    os.makedirs("./results", exist_ok=True)

    # 1. Discover a circuit (writes the artifact + scores side-car).
    print("Step 1: discovering circuit ...")
    discover_circuit(CONFIG)

    # 2. Evaluate its faithfulness. Because full_faithfulness_eval is set,
    #    evaluate_circuit returns a FaithfulnessReport.
    print("\nStep 2: running faithfulness evaluation ...")
    report = evaluate_circuit(CONFIG)

    print("\n=== Faithfulness report ===")
    print(f"Pillar 1 (patching) - circuit faithfulness score : {report.patching_score:.4f}")
    print(f"Pillar 2 (ablation) - ablated-circuit score       : {report.ablation_score:.4f}")
    if report.baseline_comparison:
        bc = report.baseline_comparison
        print(f"Pillar 5 (baselines): {bc.get('summary', 'n/a')}")
    if report.stability:
        print(f"Pillar 3 (stability) - mean Jaccard : {report.stability['mean_jaccard']:.4f}")

    # The simple, non-full path returns the same FaithfulnessReport with only
    # the Pillar-1/Pillar-2 fields populated:
    #   res = evaluate_circuit({..., "eval": {"full_faithfulness_eval": False}})
    #   res.patching_score, res.ablation_score  # other pillars are None
    print("\nDone.")


if __name__ == "__main__":
    main()
