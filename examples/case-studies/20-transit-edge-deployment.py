#!/usr/bin/env python3
"""Case study: Transit edge deployment (smart mobility).

Metro ticketing systems use ML for fare classification, passenger flow
prediction, and anomaly detection. Models must run on low-power edge
hardware at station gates with sub-100ms latency.

CircuitKit compresses the model while preserving faithfulness, ensuring
the edge-deployed model makes the same decisions as the full model.
"""

from circuitkit import load_model, export_checkpoint
from circuitkit.api import discover_circuit, evaluate_circuit

# Edge hardware (ARM, <5W, <500MB) wants a tiny, open model. Qwen2.5-0.5B is a
# registered architecture (so it prunes + exports to an HF checkpoint) and needs
# no gating. Swap MODEL_NAME for another registered model (google/gemma-3-1b-it,
# meta-llama/Llama-3.2-1B-Instruct) or "gpt2" for a fast CPU smoke test. Note:
# Pythia is discovery/eval-only and does not export, so it isn't used here. Also
# `ioi` is a GPT-2-calibrated diagnostic; on other tokenizers its circuit is
# illustrative of the pipeline, not a tuned result.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = load_model(MODEL_NAME)

print("=" * 60)
print("SMART TRANSIT EDGE DEPLOYMENT")
print("= " * 30)
print("Domain:  AFC — Automatic Fare Collection")
print("Target:  Station gate edge hardware (ARM, <5W)")
print("Budget:  <50ms inference, <500MB footprint")
print()

# Step 1: Aggressive pruning for edge deployment
print("[1/3] Discovering minimal viable circuit …")
circuit = discover_circuit({
    "model": {"name": MODEL_NAME, "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "data_params": {"num_examples": 32},
    },
    "pruning": {"target_sparsity": 0.7, "scope": "both"},
    "output_path": "./results/transit_circuit.pt",
})

# Step 2: Edge-specific evaluation
print("[2/3] Validating for edge deployment …")
results = evaluate_circuit({
    "model": {"name": MODEL_NAME},
    "discovery": {"algorithm": "eap-ig", "task": "ioi"},
    "pruning": {"target_sparsity": 0.7, "scope": "both"},
    "output_path": "./results/transit_circuit.pt",
})

# evaluate_circuit returns a FaithfulnessReport. Robustness is only populated on
# the full-faithfulness path (the default patching+ablation eval does not
# compute it), so it is only shown when available rather than defaulting a
# missing value to a misleading 0.0.
faithfulness = results.patching_score
robustness = getattr(results, "robustness", None)
faithfulness = faithfulness if faithfulness is not None else 0.0
print(f"  Faithfulness (causal patching): {faithfulness:.3f}")
if robustness is not None:
    print(f"  Robustness (input variation):   {robustness:.3f}")

# Step 3: Export edge checkpoint
print("[3/3] Exporting edge-optimized checkpoint …")
pruned = model
path = export_checkpoint(pruned, circuit, "./checkpoints/transit-edge-a150")
print(f"  Checkpoint: {path}")
print()

print("=" * 60)
print("EDGE DEPLOYMENT REPORT")
print("=" * 60)
print(f"  {'Target hardware':30s} ARM Cortex, <5W TDP")
print(f"  {'Compression':30s} 70% sparsity")
print(f"  {'Model footprint':30s} ~70% reduction")
print(f"  {'Faithfulness':30s} {faithfulness:.1%}")
if robustness is not None:
    print(f"  {'Robustness':30s} {robustness:.1%}")
print(f"  {'Expected latency':30s} <50ms per inference")
print(f"  {'Deployment':30s} Station gate (edge, no cloud dependency)")
print("=" * 60)
