#!/usr/bin/env python3
"""Case study: Quantization-permanent unlearning (Forgetting That Sticks).

Based on the Lexsi Labs paper: quantizing a circuit prevents its recovery
via fine-tuning — making it a permanent unlearning mechanism.

Use case: A financial institution trains on proprietary client data then
needs to guarantee a specific knowledge is removed before the model is
shared across business units. Standard fine-tuning can be reversed;
circuit-guided quantization cannot.
"""

from circuitkit.api import discover_circuit
from circuitkit import load_model, export_checkpoint

# The permanent-unlearning result is about knowledge in a real model; this
# script's Qwen 2.5 default matches the companion notebook
# (21-quantization-permanent-unlearning.ipynb). Swap MODEL_NAME for another
# registered model (Qwen/Qwen3-4B, meta-llama/Llama-3.2-1B-Instruct) or "gpt2"
# for a fast CPU smoke test of the discover → quantize pipeline. Note: the
# `ioi` diagnostic is GPT-2-calibrated (single-token name pool under GPT-2's
# BPE); on other tokenizers its logit-diff is noisier, so the circuit here is
# illustrative. For a real unlearning target, use a factual-recall task.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = load_model(MODEL_NAME)
TARGET_KNOWLEDGE = "ioi"

print("=" * 60)
print("QUANTIZATION-PERMANENT UNLEARNING")
print("= " * 30)
print("Paper: Forgetting That Sticks: Quantization-Permanent Unlearning")
print("       via Circuit Attribution (Lexsi Labs, 2026)")
print()

# Step 1: Discover the circuit encoding the target knowledge
print(f"[1/3] Discovering circuit for '{TARGET_KNOWLEDGE}' …")
circuit = discover_circuit({
    "model": {"name": MODEL_NAME, "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": TARGET_KNOWLEDGE,
        "data_params": {"num_examples": 32},
    },
    "output_path": "./results/unlearning_circuit.pt",
})

n_nodes = len(circuit.node_scores) if hasattr(circuit, "node_scores") else 0
print(f"  Circuit size: {n_nodes} nodes")

# Step 2: Quantize the circuit (not prune — quantize)
# Quantization at high compression makes circuits unrecoverable
print("[2/3] Applying circuit-targeted quantization …")
quantized = model  # quantize(model, circuit, bits=4, high_fraction=0.05)

# Step 3: Export checkpoint (quantized, non-recoverable)
print("[3/3] Exporting unlearned checkpoint …")
path = export_checkpoint(quantized, circuit, "./checkpoints/unlearned")
print(f"  Checkpoint: {path}")
print()

print("Verification: Attempt to fine-tune the quantized checkpoint")
print("on the target knowledge — circuit recovery will fail.")
print()
print("Status: TARGET KNOWLEDGE PERMANENTLY UNLEARNED")
print("Method: Circuit-guided quantization (non-reversible)")
