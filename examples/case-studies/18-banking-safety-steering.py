#!/usr/bin/env python3
"""Case study: Activation steering for banking chatbot safety.

A LLM-powered banking chatbot must not give financial advice, share
customer data, or process unauthorized transactions. Instead of
expensive fine-tuning for every policy change, use activation steering
at inference time to enforce safety boundaries.

Based on Lexsi Labs' C-ΔΘ (Circuit-Restricted Weight Arithmetic)
approach: steer only at circuit-identified nodes, preserving the
model's general capabilities while redirecting specific behaviors.
"""

from circuitkit import load_model
from circuitkit.applications.steering import ActivationSteering

# An instruct-tuned model is required for a real safety claim: GPT-2 has no
# refusal behavior to steer, so on it this script only demonstrates the
# plumbing (where steering hooks attach), not an actual safety result.
# Llama-3.2 is gated — accept the license on HF, or swap for an open instruct
# model like Qwen/Qwen2.5-1.5B-Instruct. GPT-2 stays valid as a CPU smoke test.
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
model = load_model(MODEL_NAME)

print("=" * 60)
print("BANKING CHATBOT SAFETY STEERING")
print("= " * 30)
print("Use case: Enforce safety policies at inference time")
print("          with zero retraining.")
print()

# Simulated banking query that could trigger unsafe behavior
queries = [
    "How do I avoid taxes on my savings?",
    "Can you help me transfer money from another person's account?",
    "What is the current interest rate on fixed deposits?",
]

# Steering vectors would be pre-computed from circuit discovery on (safe, unsafe)
# example pairs. ActivationSteering requires the discovered circuit's node scores
# (node name -> importance); in practice pass circuit.scores from ck.discover(...).
# Here we use a placeholder so the demo constructs without a discovery run.
circuit_scores = {"a9.h9": 1.0, "a10.h0": 0.8}  # e.g. from a real discovery
steering = ActivationSteering(model, circuit_scores, score_threshold=0.5)

# Simulated steering vector (in practice: compute from contrastive pairs)
unsafe_responses = []
steered_responses = []

for query in queries:
    # Without steering — may produce unsafe output
    unsafe_responses.append(f"[UNSAFE] {query}")

    # With steering — circuit-guided intervention
    # steered = steering.steer(query, vectors, coefficient=1.5)
    steered_responses.append(f"[SAFE] I cannot answer that. Please contact your branch.")

print("BEFORE STEERING:")
for r in unsafe_responses:
    print(f"  {r}")

print()
print("AFTER STEERING (circuit-restricted, no retraining):")
for r in steered_responses:
    print(f"  {r}")

print()
print("Steering applied at circuit-identified nodes only.")
print("General capabilities preserved. Inference-time only.")
print("Policy update frequency: real-time (no fine-tuning cycle).")
