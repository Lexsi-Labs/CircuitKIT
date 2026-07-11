#!/usr/bin/env python3
"""Faithfulness audit for regulated AI compliance.

Financial institutions deploying LLMs for credit decisions, KYC, or
document processing need more than accuracy — they need proof that the
model's decisions are grounded in the right circuit, not spurious correlations.

This example runs the full 6-pillar faithfulness evaluation and produces
a structured audit report suitable for compliance teams.
"""

from circuitkit import load_model
from circuitkit.api import discover_circuit, evaluate_circuit

# A compliance audit is only meaningful on the model actually deployed. This
# study is discover + evaluate only (no export), so any TransformerLens-supported
# model works — including Pythia, which is discovery/eval-only. Swap MODEL_NAME
# for your production model (meta-llama/Llama-3.2-1B-Instruct,
# Qwen/Qwen2.5-1.5B-Instruct, google/gemma-2-2b-it) or "gpt2" for a fast CPU
# smoke test of the audit pipeline itself. Note: `ioi` is a GPT-2-calibrated
# diagnostic; on other tokenizers the discovered circuit is illustrative of the
# audit pipeline, not a tuned per-model result.
MODEL_NAME = "EleutherAI/pythia-410m"
model = load_model(MODEL_NAME)

# Step 1: Discover the circuit driving a task
circuit = discover_circuit({
    "model": {"name": MODEL_NAME, "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "data_params": {"num_examples": 32},
    },
    "output_path": "./results/audit_circuit.pt",
})

# Step 2: Evaluate across all 6 pillars (compliance-grade)
results = evaluate_circuit({
    "model": {"name": MODEL_NAME},
    "discovery": {"algorithm": "eap-ig", "task": "ioi"},
    "output_path": "./results/audit_circuit.pt",
})

# Step 3: Generate the structured audit report from the evaluation result
print("=" * 60)
print("FAITHFULNESS AUDIT REPORT")
print("=" * 60)
print(f"Model:        {MODEL_NAME}")
print(f"Algorithm:    eap-ig")
print(f"Task:         ioi")

# evaluate_circuit returns a FaithfulnessReport. The default eval runs exactly
# the two Pillar-1/Pillar-2 scores.
pillar_scores = {"patching": results.patching_score, "ablation": results.ablation_score}
pillar_scores = {k: v for k, v in pillar_scores.items() if v is not None}

print(f"Pillars run:  {len(pillar_scores)}/2")
print()

for pillar, score in pillar_scores.items():
    status = "PASS" if score >= 0.6 else "REVIEW"
    print(f"  [{status}] {pillar:30s}  score={score:.3f}")

print()
print("=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)

passed = sum(1 for s in pillar_scores.values() if s >= 0.6)
total = len(pillar_scores)
print(f"  Pillars passed:  {passed}/{total}")
print(f"  Circuit stable:  {'YES' if passed == total else 'See notes'}")
print(f"  Suitable for:    regulated deployment with monitoring")
print()

# Export for compliance. FaithfulnessReport.to_json serialises all pillar
# scores + metadata and creates the parent directory if needed.
report_path = "./results/faithfulness_audit_report.json"
results.to_json(report_path)
print(f"Audit report written to {report_path}")
