#!/usr/bin/env python3
"""Activation steering: compute a steering vector and apply it at inference."""

from circuitkit import load_model, load_scores
from circuitkit.api import discover_circuit
from circuitkit.applications.steering import ActivationSteering

model = load_model("gpt2")

# Discover a circuit to identify steering targets
discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {"algorithm": "eap-ig", "task": "ioi",
                  "data_params": {"num_examples": 16}},
    "output_path": "./results/steering_circuit.pt",
})

# Load the discovered circuit; ActivationSteering wants the name->score dict.
circuit = load_scores("./results/steering_circuit.pt")

# Build steering vectors from the circuit
steering = ActivationSteering(model, circuit.scores, score_threshold=0.5)

# Source and target examples for vector computation
source = [{"text": "Alice and Bob went to the store. Alice gave a gift to"}]
target = [{"text": "Alice and Bob went to the store. Bob gave a gift to"}]

vectors = steering.compute_steering_vector(source, target)

# Steer at inference
output = steering.steer("When Alice and Bob walked in,", vectors, coefficient=1.5)
print(f"Steered output: {output}")
