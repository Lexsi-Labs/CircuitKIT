#!/usr/bin/env python3
"""Knowledge editing: rewrite a fact using ROME/MEMIT at circuit-identified layers."""

from circuitkit import load_model
from circuitkit.applications.editing import CircuitKnowledgeEditor

model = load_model("gpt2")

editor = CircuitKnowledgeEditor(model)

# Edit a fact using ROME at the middle MLP layer
result = editor.edit_via_circuit(
    prompt="The capital of France is",
    subject="France",
    target="Lyon",
    method="rome",  # "rome", "memit", or "ft"
)

print(f"Success:      {result.success}")
print(f"Confidence:   {result.confidence_before:.3f} → {result.confidence_after:.3f}")
print(f"Target layer: {result.target_layer}")

# Verify the edit
logits = model("The capital of France is")
answer = model.tokenizer.decode(int(logits.argmax(-1).squeeze()[-1]))
print(f"Model says:   {answer}")
