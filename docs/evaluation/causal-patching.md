# Pillar 1: Causal Patching

Causal patching is the **fastest** faithfulness test and the most commonly reported metric. It answers the question: *if we patch back only the circuit nodes to their clean activation values (leaving all other nodes at their corrupted values), does the model's behavior recover?*

---

## The Measurement

**Setup:**
1. Run the model on the corrupted input $x^*$ — all activations are now "corrupted"
2. Patch circuit nodes back to their clean activation values (from $x$)
3. Measure the logit-difference metric $f$ at the output

If the circuit is faithful, the metric after patching should be close to the full-model clean metric:

$$\text{patching score} = \frac{f(\text{corrupted} + \text{circuit nodes patched}) - f(x^*)}{f(x) - f(x^*)}$$

A score of 1.0 means the circuit alone can fully recover the clean behavior. A score of 0.0 means patching back the circuit has no effect.

---

## Running Pillar 1

```python
# Via Pipeline
pipe.evaluate(pillars=["patching"], n_examples=256)
print(pipe.report.patching_score)

# Via evaluate_circuit
results = evaluate_circuit({
    ...,
    "eval": {"pillars": ["patching"]},
})
print(results.patching_score)  # Pillar 1 patching score
```

Low-level:

```python
from circuitkit.evaluation.evaluate import evaluate_graph

# Apply circuit to graph
graph.apply_topn(n=30, level="node")  # keep top 30 nodes (default level="edge")

raw_metric = evaluate_graph(
    model,
    graph,
    dataloader,
    metrics=[metric_fn],
    intervention="patching",   # the patching intervention
    quiet=True,
)
```

!!! note "`evaluate_graph` returns the raw metric, not the normalized score"
    `evaluate_graph` gives you the raw per-sample metric under the patching intervention — **not** the 0-1 faithfulness ratio. The normalization against the clean and corrupt baselines happens inside `Pillar1_CausalPatching.run`, which is what populates `report.patching_score`. If you call `evaluate_graph` directly, you still need to normalize yourself: `(raw_metric - corrupt_baseline) / (clean_baseline - corrupt_baseline)`.

---

## Interpreting the Score

| Score | Interpretation |
|-------|---------------|
| ≥ 0.85 | **Strong.** The circuit accounts for ≥ 85% of the behavior. |
| 0.70 – 0.85 | **Moderate.** The circuit captures the main computation but misses some components. |
| 0.50 – 0.70 | **Weak.** A meaningful portion of the behavior is outside the circuit. |
| < 0.50 | **Poor.** The circuit does not explain the behavior well. |

**Typical values in the CircuitKit audit:**

| Model | Task | Algorithm | Pillar 1 Score |
|-------|------|-----------|:-----------:|
| GPT-2 | IOI | `eap-ig` | 0.83 |
| GPT-2 | SVA | `eap-ig` | 0.79 |
| Llama-1B | MMLU | `eap-ig` | 0.71 |
| Gemma-2B | IOI | `eap-ig` | 0.76 |

---

## What Causal Patching Tests (and Does Not Test)

**Tests:**
- Whether the circuit nodes are sufficient to reproduce the behavior (sufficiency)
- The circuit's ability to carry the relevant information

**Does NOT test:**
- Circuit *sufficiency* under ablation of out-of-circuit nodes (that's Pillar 2: Ablation)
- Whether the circuit is stable across seeds (Pillar 3)
- Whether the circuit is better than a random selection of the same size (Pillar 5)

A circuit can pass Pillar 1 and fail Pillar 2 if there are other components outside the circuit that *also* carry the information: the model has redundancy.

---

## Next Steps

- [Ablation (Pillar 2)](ablation.md) — testing necessity
- [Baselines (Pillar 5)](baselines.md) — comparing to random selection
- [Framework Overview](framework.md) — all 6 pillars
