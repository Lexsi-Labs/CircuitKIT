# Pillar 5: Baselines

Pillar 5 answers the question: *is this circuit actually better than chance?* A circuit that passes Pillars 1 and 2 might still not be special — maybe any same-size selection of components would do equally well.

Baselines run the Pillar 1 patching evaluation with two comparison circuits: a **random** selection of the same size, and a **magnitude** selection (the components with the largest weight norms, a standard heuristic).

---

## The Measurement

Three patching scores are computed:

| Comparison | What it tests |
|-----------|--------------|
| **Circuit** (discovered) | The circuit found by the discovery algorithm |
| **Random baseline** | A random selection of the same number of nodes |
| **Magnitude baseline** | The same number of nodes, selected by RMS weight norm |

A faithful circuit should score higher than both baselines. The gap between the circuit and random baseline is the primary indicator.

---

## Running Pillar 5

```python
# Via Pipeline
pipe.evaluate(pillars=["baselines"], n_examples=256)
print(pipe.report.baseline_comparison)
# {
#   "circuit_score": 0.83,
#   "baselines": {
#     "random":    {"score": 0.31, "percentage": 267.7, "improvement": 2.68},
#     "magnitude": {"score": 0.48, "percentage": 172.9, "improvement": 1.73},
#   },
#   "best_baseline_score": 0.48,
#   "circuit_advantage": 0.35,       # circuit_score - best_baseline_score
#   "circuit_sparsity": 0.02,
#   "summary": "Circuit substantially outperforms random baseline (2.68x improvement)",
#   "baseline_types": ["random", "magnitude"],
# }
```

---

## Interpreting the Results

**Key metric:** `circuit_score - baselines["random"]["score"]` (the "baseline gap")

| Gap | Interpretation |
|:---:|----------------|
| ≥ 0.40 | **Strong** — circuit is substantially better than chance |
| 0.20 – 0.40 | **Moderate** — circuit outperforms random but not dramatically |
| 0.10 – 0.20 | **Weak** — marginal improvement over random |
| < 0.10 | **Fails baseline** — circuit may not be meaningful |

**Circuit vs. magnitude:** If the magnitude baseline is close to the circuit score, the discovery algorithm may be largely selecting high-weight components — which magnitude selection also does. This suggests the algorithm is not providing much value beyond a simple weight-norm heuristic.

**Typical values from the CircuitKit audit:**

| Model | Task | Circuit | Random | Magnitude |
|-------|------|:-------:|:------:|:---------:|
| GPT-2 | IOI | 0.83 | 0.31 | 0.48 |
| GPT-2 | SVA | 0.79 | 0.28 | 0.45 |
| Llama-1B | MMLU | 0.71 | 0.33 | 0.50 |

---

## What Failing Pillar 5 Means

If the circuit does not beat the random baseline:

1. **Too high sparsity** — at very high sparsity (e.g., 50%+), any random selection of that many components does well
2. **Task is distributed** — the task uses diffuse computation; no small subset is special
3. **Algorithm issue** — the algorithm may not be correctly identifying important components; try `eap-ig`
4. **Too few examples** — attribution scores are noisy; increase `n_examples`

---

## The Magnitude Baseline

The magnitude selector (`magnitude`) scores components by their RMS weight magnitude — the Frobenius norm divided by the square root of the parameter count, so heads and MLPs stay on a comparable scale:

$$\text{score}(A_{l,h}) = \frac{\|W_{l,h}\|_F}{\sqrt{|W_{l,h}|}}$$

This is the simplest possible heuristic and acts as a "free" baseline. A good discovery algorithm should substantially outperform magnitude on faithfulness, not just on circuit size.

From `circuitkit.selection`:

```python
from circuitkit.selection import get_selector

magnitude_selector = get_selector("magnitude")
baseline_scores = magnitude_selector(model, "ioi", {"level": "node"})
```

---

## Next Steps

- [Generalization (Pillar 6)](generalization.md) — cross-task transfer
- [Causal Patching (Pillar 1)](causal-patching.md) — the core patching measurement
- [Framework Overview](framework.md) — all 6 pillars
