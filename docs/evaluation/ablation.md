# Pillar 2: Ablation

Ablation is the **sufficiency** test: *if we keep the circuit nodes and ablate everything else (set the out-of-circuit activations to zero/mean), does the model still produce the behavior?*

A circuit that survives this — behaving like the full model even with all other components stripped out — is sufficient on its own. Pillar 2 measures how much of the clean behavior the circuit alone recovers.

---

## The Measurement

**Setup:**
1. Keep the circuit nodes and ablate the out-of-circuit nodes (set their activation to zero, the dataset mean, or a position-specific mean)
2. Measure the logit-difference metric on the ablated model, $y_\text{circuit}$
3. Normalize against the clean and corrupt baselines

The headline score is the normalized faithfulness ratio:

$$F = \frac{y_\text{circuit} - y_\text{corrupt}}{y_\text{clean} - y_\text{corrupt}}$$

where $y_\text{clean}$ is the metric on the full (unmodified) model, $y_\text{corrupt}$ is the metric on the corrupt baseline, and $y_\text{circuit}$ is the metric on the ablated circuit. $F = 1.0$ means the ablated circuit fully recovers the clean behavior; $F = 0.0$ means it does no better than the corrupt baseline.

!!! note "Degenerate baselines"
    When the clean and corrupt baselines are near-identical ($|y_\text{clean} - y_\text{corrupt}|$ below a small epsilon), the denominator collapses and the ratio is undefined. In that case the pillar logs a warning and reports the sentinel score `0.0`.

---

## Running Pillar 2

```python
# Via Pipeline
pipe.evaluate(pillars=["ablation"], n_examples=256)
print(pipe.report.ablation_score)

# Via evaluate_circuit  
results = evaluate_circuit({..., "eval": {"pillars": ["ablation"]}})
```

Low-level ablation strategies:

```python
from circuitkit.evaluation.evaluate import evaluate_graph

# Zero ablation (fastest)
score = evaluate_graph(model, graph, dataloader, metrics=[metric_fn],
                       intervention="zero", quiet=True)

# Mean ablation (more principled — ablates to the dataset mean)
score = evaluate_graph(model, graph, dataloader, metrics=[metric_fn],
                       intervention="mean",
                       intervention_dataloader=dataloader,
                       quiet=True)

# Mean-positional ablation (mean conditioned on position)
score = evaluate_graph(model, graph, dataloader, metrics=[metric_fn],
                       intervention="mean-positional",
                       intervention_dataloader=dataloader,
                       quiet=True)
```

---

## Ablation Strategies

| Strategy | What it does | Use when |
|----------|-------------|---------|
| `"zero"` | Sets out-of-circuit activations to 0 | Fast baseline; may introduce artifacts for biased activations |
| `"mean"` | Sets out-of-circuit activations to the dataset mean (computed over `intervention_dataloader`) | Standard principled ablation |
| `"mean-positional"` | Mean conditioned on position in the sequence | When position matters (e.g., the indirect-object position in IOI) |

`"mean"` and `"mean-positional"` require an `intervention_dataloader` to compute the mean activations; `"zero"` needs no extra data. Any other value raises `ValueError`.

---

## Interpreting the Score

| Faithfulness Ratio $F$ | Interpretation |
|----------------|---------------|
| ≥ 0.80 | **Strong sufficiency.** The circuit alone recovers ≥ 80% of the clean behavior. |
| 0.50 – 0.80 | **Moderate sufficiency.** The circuit carries much of the behavior but misses some. |
| < 0.50 | **Low sufficiency.** The circuit alone is not enough — important components live outside it. |

A low score means ablating the out-of-circuit nodes destroys behavior that the circuit was supposed to carry on its own — a sign the circuit is incomplete.

---

## Pillar 1 vs. Pillar 2 Together

Both pillars probe sufficiency, from different angles: Pillar 1 patches the circuit *into* a corrupt run, while Pillar 2 keeps the circuit and ablates *everything else*.

| P1 (Patching) | P2 (Ablation) | Interpretation |
|:---:|:---:|----------------|
| High | High | **Faithful circuit** — sufficient under both interventions |
| High | Low | **Fragile under ablation** — recovers behavior when patched in, but not when isolated |
| Low | High | **Fragile under patching** — survives isolation, but does not transfer into a corrupt run |
| Low | Low | **Poor circuit** — insufficient under either intervention |

For a strong faithfulness claim, a circuit should score high on both Pillar 1 and Pillar 2.

---

## Next Steps

- [Stability (Pillar 3)](stability.md) — consistency across seeds
- [Baselines (Pillar 5)](baselines.md) — is this better than random?
- [Framework Overview](framework.md) — all 6 pillars
