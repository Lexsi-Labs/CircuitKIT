# Hallucination Detection

CircuitKit's `HallucinationDetector` uses circuit activations and trained linear probes to detect probable hallucinations during model generation. Rather than using an external judge, it monitors the circuit components that were identified as task-relevant and flags outputs where those components show patterns associated with incorrect generation.

---

## How It Works

1. **Discover a circuit** for a factual task (e.g., `capital_country`, `truthfulqa`)
2. **Train linear probes** on circuit-layer activations, using labeled (factual, hallucination) examples
3. **At generation time**, hook into the circuit layers and run each new token through the trained probes
4. **Aggregate** per-token probabilities into a generation-level hallucination score

The probes are simple linear classifiers: `activation_tensor → Linear(hidden_dim, 1) → Sigmoid → [0, 1]`. They add minimal overhead (~1-2% slower generation).

---

## Quick Start

`HallucinationDetector` expects a `CircuitArtifact` (whose nodes carry `.layer_idx`) and a HuggingFace model (it reads `model.config.hidden_size`). The flat-API convenience functions `ck.load_scores()` (returns a `Circuit` with plain node-name strings, no `.layer_idx`) and `ck.load_model()` (returns a TransformerLens `HookedTransformer`, no `.config.hidden_size`) are **not** compatible here — both raise `AttributeError` if passed directly.

```python
from circuitkit.applications.common_utils.hallucination_detection import HallucinationDetector
from circuitkit.artifacts import eap_to_artifact
import circuitkit as ck
from transformers import AutoModelForCausalLM

# 1. Load circuit scores, then convert to a CircuitArtifact (has layer_idx per node)
circuit = ck.load_scores("./circuit.pt")
artifact = eap_to_artifact(
    node_scores=circuit.scores,
    model_id="gpt2",
    task=circuit.task or "ioi",
    dataset="ioi_dataset",
)

# 2. Load the model via HuggingFace, not ck.load_model() (HookedTransformer lacks .config.hidden_size)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 3. Prepare labeled training data
train_data = [
    {"text": "The capital of France is Paris", "is_hallucination": False},
    {"text": "The capital of France is Berlin", "is_hallucination": True},
    # ... 50+ examples per class
]
val_data = [
    {"text": "2 + 2 equals 4", "is_hallucination": False},
    {"text": "2 + 2 equals 5", "is_hallucination": True},
]

# 4. Create detector and train probes
from circuitkit.applications import get_arch_config
arch_cfg = get_arch_config("gpt2")

detector = HallucinationDetector(model, artifact, arch_cfg, device="cuda")

results = detector.train_probes(
    train_data=train_data,
    val_data=val_data,
    batch_size=32,
    epochs=10,
    patience=3,
)
print(f"Best validation AUROC: {results['best_val_auroc']:.4f}")

# 5. Save and load probes
detector.save_probes("./hallucination_probes.pt")
detector.load_probes("./hallucination_probes.pt")

# 6. Detect hallucinations
result = detector.detect_hallucinations(
    "The capital of France is London",
    threshold=0.5,
)
print(f"Hallucination probability: {result['hallucination_prob']:.2%}")
print(f"Is hallucination: {result['is_hallucination']}")
print(f"Explanation: {result['explanation']}")
```

---

## Training Data Requirements

| Requirement | Details |
|-------------|---------|
| Minimum examples | 20–50 per class (factual / hallucination) |
| Balance | Equal or near-equal class distribution |
| Label quality | Must be manually verified — mislabeled data dominates performance |
| Diversity | Multiple domains/topics so probes generalize |

**Target metric:** Validation AUROC ≥ 0.85. Below 0.75, the probes are unreliable.

---

## Detection API

```python
result = detector.detect_hallucinations("text", threshold=0.5)

# result fields:
result["is_hallucination"]    # bool: True if hallucination_prob > threshold
result["hallucination_prob"]  # float in [0, 1]
result["per_token_probs"]     # List[float]: per-token probabilities
result["flagged_tokens"]      # List[int]: token indices above threshold
result["explanation"]         # str: human-readable explanation
result["circuit_activations"] # Dict: raw activation data per circuit layer
```

---

## Integration with Generation

Monitor hallucinations in a generation loop:

```python
def generate_with_monitoring(model, prompt, detector, max_tokens=100):
    generated = prompt

    for _ in range(max_tokens):
        token = model.generate_one(generated)
        generated += token

        result = detector.detect_hallucinations(generated)
        if result["hallucination_prob"] > 0.7:
            print(f"WARNING: High hallucination probability ({result['hallucination_prob']:.0%})")
            print(f"Flagged tokens: {result['flagged_tokens']}")
            break

    return generated
```

---

## Best Practices

1. **Discover the circuit on a factual task** — `capital_country`, `truthfulqa`, or `wmdp` work well. Avoid diagnostic tasks like IOI (they don't capture factuality).

2. **Curate labels carefully** — mislabeled examples are the primary failure mode. For factual claims, use a database lookup or expert review.

3. **Monitor AUROC, not accuracy** — AUROC is threshold-independent. Target > 0.85.

4. **Update probes after fine-tuning** — if the model is fine-tuned, the circuit activations shift. Re-train probes on the fine-tuned model.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| AUROC < 0.70 | Insufficient or mislabeled data | Add more examples; verify labels |
| All predictions high | Class imbalance or training instability | Balance data; reduce learning rate to 1e-4 |
| OOM during training | Batch size too large | Reduce to 8 or 16 |
| Probes overfit quickly | Too few training examples | Increase `patience`; add regularization |

---

## Next Steps

- [User Guide: Applications](../user-guide/applications.md) — other circuit-based applications
- [Evaluation: Causal Patching](../evaluation/causal-patching.md) — faithfulness before using for detection
- [Advanced: Circuit Artifacts](circuit-artifacts.md) — circuit artifact format
