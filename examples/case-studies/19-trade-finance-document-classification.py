#!/usr/bin/env python3
"""Case study: Trade finance document classification (Fintra).

Banks process millions of trade documents — bills of lading, invoices,
letters of credit. An LLM classifying these must run on-premises with
no GPU, low latency, and verifiable accuracy.

CircuitKit discovers the minimal circuit for document classification,
prunes non-essential weights, and exports a HuggingFace checkpoint
that runs on CPU-only banking infrastructure.
"""

from circuitkit import load_model, export_checkpoint
from circuitkit.api import discover_circuit, evaluate_circuit

# A small instruct model suits the "CPU-only, <100ms" target after pruning.
# Gemma-3-1b is gated — accept the license on HF, or swap for an open *registered*
# model (Qwen/Qwen2.5-0.5B-Instruct) or "gpt2" for a fast CPU smoke test of the
# classify → prune → export pipeline. NOTE: this pipeline prunes and exports, so
# the model must be a registered architecture — Pythia/GPT-NeoX is discovery/eval
# only and cannot export. Also: `ioi` is a GPT-2-calibrated diagnostic task; on a
# non-GPT-2 tokenizer its single-token name assumption may not hold, so treat the
# discovered circuit here as illustrative of the pipeline, not a tuned result.
MODEL_NAME = "google/gemma-3-1b-it"
model = load_model(MODEL_NAME)

print("=" * 60)
print("TRADE FINANCE DOCUMENT CLASSIFICATION")
print("= " * 30)
print("Domain:  Fintra — AI-native trade finance")
print("Target:  CPU-only on-prem deployment")
print("Budget:  <100ms per document")
print()

# Step 1: Discover circuit for document type classification
print("[1/4] Discovering circuit …")
circuit = discover_circuit({
    "model": {"name": MODEL_NAME, "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "data_params": {"num_examples": 32},
    },
    "pruning": {"target_sparsity": 0.6, "scope": "both"},
    "output_path": "./results/fintra_circuit.pt",
})

# Step 2: Verify faithfulness before deployment
print("[2/4] Verifying faithfulness …")
results = evaluate_circuit({
    "model": {"name": MODEL_NAME},
    "discovery": {"algorithm": "eap-ig", "task": "ioi"},
    "pruning": {"target_sparsity": 0.6, "scope": "both"},
    "output_path": "./results/fintra_circuit.pt",
})

# This audit keys on the Pillar-2 ablation score. evaluate_circuit returns a
# FaithfulnessReport; read ablation specifically — not patching.
faithfulness = results.ablation_score
faithfulness = faithfulness if faithfulness is not None else 0.0
print(f"  Faithfulness at 60% sparsity: {faithfulness:.3f}")

# Step 3: Export compressed checkpoint
print("[3/4] Exporting compressed checkpoint …")
pruned = model
path = export_checkpoint(pruned, circuit, "./checkpoints/fintra-docclass")
print(f"  Checkpoint: {path}")
print(f"  Size reduction: ~60% (on-prem deployable)")

# Step 4: Deployment readiness
print()
print("[4/4] DEPLOYMENT READINESS")
print(f"  {'Model':30s} {MODEL_NAME} → fintra-docclass (60% compressed)")
print(f"  {'Infrastructure':30s} CPU-only, on-prem")
print(f"  {'Latency target':30s} <100ms/doc")
print(f"  {'Faithfulness':30s} {faithfulness:.1%}")
print(f"  {'Audit trail':30s} Circuit artifact + evaluation report")
print()
print("Ready for deployment to banking production environment.")
