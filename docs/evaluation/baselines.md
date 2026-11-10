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
# GPT-2, ioi, eap-ig, seed 42, 256 examples, 30% pruned (Tesla T4, 2026-10-11):
# {
#   "circuit_score": 4.044,          # mean logit difference, not a 0-to-1 ratio
#   "baselines": {
#     "random":    {"score": 1.172, "score_std": 1.288, "n_draws": 5,
#                   "percentage": 345.0, "improvement": 3.45, "z_score": 2.23, ...},
#     "magnitude": {"score": -0.493, "percentage": None, "improvement": None, ...},
#   },
#   "best_baseline_score": 1.172,
#   "circuit_advantage": 2.872,      # circuit_score - best_baseline_score
#   "circuit_sparsity": 0.696,
#   "summary": "Circuit substantially outperforms random baseline (3.45x improvement)",
#   "baseline_types": ["random", "magnitude"],
# }
```

---

## Interpreting the Results

The scores in `baseline_comparison` are the task metric itself (for `ioi`, a mean logit
difference). To put them on the 0-to-1 scale of the patching score, divide the distance from the
corrupt baseline by `clean - corrupt`; both are printed in the Pillar 1 log line. The table below
is on that scale.

**Key metric:** the circuit's ratio minus the random baseline's ratio (the "baseline gap")

| Gap | Interpretation |
|:---:|----------------|
| ≥ 0.40 | **Strong** — circuit is substantially better than chance |
| 0.20 – 0.40 | **Moderate** — circuit outperforms random but not dramatically |
| 0.10 – 0.20 | **Weak** — marginal improvement over random |
| < 0.10 | **Fails baseline** — circuit may not be meaningful |

**Circuit vs. magnitude:** If the magnitude baseline is close to the circuit score, the discovery algorithm may be largely selecting high-weight components — which magnitude selection also does. This suggests the algorithm is not providing much value beyond a simple weight-norm heuristic.

**Measured values.** The GPT-2 IOI row was re-measured on 2026-10-11: `gpt2`, `ioi`, `eap-ig`,
seed 42, 256 discovery and 256 evaluation examples, 30% pruned, `float32`, one Tesla T4, with

```python
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=256, sparsity=0.3, seed=42)
report = ck.faithfulness(model, circuit, "ioi", pillars=["patching", "ablation", "baselines"],
                         n_examples=256, seed=42)
```

| Model | Task | Circuit | Random | Magnitude | Status |
|-------|------|:-------:|:------:|:---------:|--------|
| GPT-2 | IOI | 0.945 | 0.290 (mean of 5 draws) | -0.089 | re-measured 2026-10-11 |
| GPT-2 | IOI | 0.83 | 0.31 | 0.48 | before the `ioi` scored position was corrected; superseded by the row above |
| GPT-2 | SVA | 0.79 | 0.28 | 0.45 | audit value; not affected by the correction |
| Llama-1B | MMLU | 0.71 | 0.33 | 0.50 | measured before the padding correction (`mmlu` pads left) and not re-measured |

In the re-measured row the raw logit differences are 4.044 (circuit), 1.172 (random) and -0.493
(magnitude), against a clean value of 4.288 and a corrupt value of -0.101. The five random draws
range from -0.37 to 3.09 (standard deviation 1.29), so one random draw is not a reliable baseline;
the gap to the mean is 0.65. The magnitude selection scores below the corrupt baseline here, where
the earlier number had it at 0.48.

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
