#!/usr/bin/env python3
"""Case study: Tabular model faithfullness audit (Orion-MSP).

Lexsi Labs' Orion-MSP is a tabular foundation model for in-context learning
on structured data. In regulated settings (medical, financial), auditors
need to verify that predictions are grounded in genuine feature relationships,
not spurious correlations.

This example shows how to audit a tabular classification model's circuits
to build trust for deployment.
"""

from circuitkit.api import discover_circuit, evaluate_circuit
import json

# Simulated tabular model — in practice use a HuggingFace tabular model
# or a small LLM fine-tuned on tabular data.
MODEL_NAME = "gpt2"
TASK = "ioi"

print("=" * 60)
print("TABULAR MODEL FAITHFULNESS AUDIT")
print("= " * 30)
print(f"Model:   Orion-MSP (via {MODEL_NAME})")
print(f"Task:    {TASK}")
print("Goal:    Verify predictions rely on meaningful feature circuits")
print()

# Step 1: Discover the circuit
circuit = discover_circuit({
    "model": {"name": MODEL_NAME, "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": TASK,
        "data_params": {"num_examples": 32},
    },
    "output_path": "./results/tabular_audit_circuit.pt",
})

n_nodes = len(circuit.node_scores) if hasattr(circuit, "node_scores") else 0
print(f"Discovered circuit:   {n_nodes} nodes identified")

# Step 2: Evaluate faithfulness
results = evaluate_circuit({
    "model": {"name": MODEL_NAME},
    "discovery": {"algorithm": "eap-ig", "task": TASK},
    "output_path": "./results/tabular_audit_circuit.pt",
})

# Step 3: Produce audit report
# evaluate_circuit returns a FaithfulnessReport; the default eval runs exactly
# the two Pillar-1/Pillar-2 scores.
pillar_scores = {"patching": results.patching_score, "ablation": results.ablation_score}
pillar_scores = {k: v for k, v in pillar_scores.items() if v is not None}

audit = {
    "model": "Orion-MSP (via GPT-2 proxy)",
    "task": TASK,
    "algorithm": "eap-ig",
    "circuit_size": n_nodes,
    "pillar_results": {},
    # PASS only if at least one pillar ran AND every pillar clears the bar —
    # an empty pillar set is not a pass.
    "verdict": "PASS" if pillar_scores and all(
        s >= 0.6 for s in pillar_scores.values()
    ) else "REVIEW",
}

for pillar, score in pillar_scores.items():
    audit["pillar_results"][pillar] = {
        "score": score,
        "threshold": 0.6,
        "status": "PASS" if score >= 0.6 else "REVIEW",
    }

with open("./results/tabular_faithfulness_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print(f"\nAudit verdict: {audit['verdict']}")
print(f"Report:        ./results/tabular_faithfulness_audit.json")
print()
print("Interpretation: High faithfulness scores mean the model's")
print("decisions rely on stable, causally relevant circuits —")
print("a strong signal for regulated deployment.")
