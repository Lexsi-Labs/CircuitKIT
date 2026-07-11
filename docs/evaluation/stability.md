# Pillar 3: Stability

Stability tests whether the discovered circuit is consistent across re-discovery runs with different random seeds. If re-running discovery from the same model and task produces very different circuits each time, the circuit is not a reliable description of the model's computation.

**Cost:** Expensive — Pillar 3 re-runs the full discovery algorithm `n_stability_runs` times.

---

## The Measurement

CircuitKit runs discovery $N$ times with different random seeds and computes the **Spearman rank correlation** between the importance score vectors across seed pairs:

$$\text{stability} = \text{mean}_{i \neq j} \, \rho_S(\text{scores}_i, \text{scores}_j)$$

where $\rho_S$ is the Spearman rank correlation. A value of 1.0 means all seeds produce identical score rankings; 0.0 means the rankings are random.

Additionally, CircuitKit measures **node overlap** as the Jaccard similarity between the selected node sets of two seeds (reported as `mean_jaccard`), alongside the Dice coefficient (`mean_dice`):

$$\text{overlap} = \text{Jaccard}(A_i, A_j) = \frac{|A_i \cap A_j|}{|A_i \cup A_j|}$$

---

## Running Pillar 3

```python
# Via Pipeline
pipe.evaluate(pillars=["stability"], n_examples=256, n_stability_runs=5)
print(pipe.report.stability)
# {
#   "mean_spearman": 0.87,   # mean Spearman rho of score vectors across seed pairs
#   "std_spearman": 0.04,
#   "mean_jaccard": 0.82,    # mean top-K node overlap across seed pairs
#   "std_jaccard": 0.05,
#   "mean_dice": 0.90,
#   "n_stable_nodes": 24,    # nodes present in every run
#   "n_runs": 5,
# }

# Full audit with stability
pipe.evaluate(pillars=None, n_stability_runs=5)
```

---

## Cost Considerations

Pillar 3 is the most expensive pillar. With `n_stability_runs=5`:

| Model | Runtime per run | Total (5 runs) |
|-------|----------------|----------------|
| GPT-2 (128 examples) | ~2 min | ~10 min |
| Llama-1B (128 examples) | ~8 min | ~40 min |
| Gemma-4B (128 examples) | ~20 min | ~100 min |

**Optimization:** Reduce `n_examples` for the stability runs (they need fewer examples than initial discovery since you're measuring consistency, not accuracy):

```python
pipe.evaluate(
    pillars=["stability"],
    n_examples=64,      # fewer examples OK for stability check
    n_stability_runs=3, # 3 runs is often sufficient
)
```

---

## Interpreting the Results

| Spearman rho | Interpretation |
|:---:|----------------|
| ≥ 0.90 | **Highly stable** — algorithm consistently finds the same circuit |
| 0.75 – 0.90 | **Moderately stable** — core circuit is consistent; peripheral nodes vary |
| 0.60 – 0.75 | **Weakly stable** — significant variation across seeds |
| < 0.60 | **Unstable** — circuit is not reliably identifiable |

**Stability is algorithm-specific:** EAP-IG tends to be more stable than ACDC (which is sensitive to the order of edge removal). If stability is low, try increasing `n_examples` or switching to a more stable algorithm variant.

---

## What Low Stability Means

Low stability (`rho < 0.75`) can indicate:

1. **Too few examples** — attribution scores are noisy; increase `n_examples`
2. **Task ambiguity** — the task does not have a consistent circuit (may be distributed across many paths)
3. **Algorithm sensitivity** — some algorithms are inherently more variable; switch to `eap-ig`
4. **Redundant circuits** — the model has multiple equivalent circuits; any of them passes Pillar 1 but they're different per seed

---

## Next Steps

- [Robustness (Pillar 4)](robustness.md) — input corruption stress test
- [Baselines (Pillar 5)](baselines.md) — comparing to random selection
- [Framework Overview](framework.md) — all 6 pillars
