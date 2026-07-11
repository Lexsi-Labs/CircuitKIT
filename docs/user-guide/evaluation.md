# Evaluation

CircuitKit evaluates circuit faithfulness with a **6-pillar framework**. Each pillar tests a different property; together they show whether the discovered subgraph actually explains the model's behaviour.

## Quick start

```python
from circuitkit.api import discover_circuit, evaluate_circuit

discover_circuit({
    "model": {"name": "gpt2"},
    "discovery": {"algorithm": "eap-ig", "task": "ioi", "level": "node",
                  "data_params": {"num_examples": 128}},
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})

results = evaluate_circuit({
    "model": {"name": "gpt2"},
    "discovery": {"algorithm": "eap-ig", "task": "ioi", "level": "node",
                  "data_params": {"num_examples": 256}},
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
    "eval": {"pillars": ["patching", "ablation", "baselines"]},
})
# results is a FaithfulnessReport:
#   results.patching_score  -> 0.48   (Pillar 1)
#   results.ablation_score  -> 0.83   (Pillar 2)
#   results.baseline_comparison        (Pillar 5, incl. circuit-vs-random)
```

With Pipeline:
```python
pipe = Pipeline("gpt2", task="ioi")
pipe.discover(algorithm="eap-ig", n_examples=128, sparsity=0.3)
pipe.evaluate(pillars=["patching", "ablation", "baselines"], n_examples=256)     # subset
pipe.evaluate(pillars=None, n_examples=512,           # full audit
              n_stability_runs=5)
print(pipe.report)
```

## The 6 pillars

| # | Pillar | Question | Cost |
|---|---|---|---|
| 1 | **Causal Patching** | Does patching back only circuit nodes recover behaviour? | Fast |
| 2 | **Ablation** | Does ablating circuit nodes degrade behaviour? | Fast |
| 3 | **Stability** | Is the circuit consistent across re-discovery seeds? | Expensive |
| 4 | **Robustness** | Does the circuit hold under input corruptions? | Moderate |
| 5 | **Baselines** | Is the circuit better than random/magnitude selection? | Moderate |
| 6 | **Generalization** | Does the circuit transfer to a related task? | Expensive |

!!! warning "Pillar 6 is preliminary"
    Generalization is implemented but not yet validated at scale.

## Subset vs. full audit

**Fast subset** — for iteration during circuit development:
```python
pipe.evaluate(pillars=["patching", "ablation"], n_examples=128)
# ~2–5 minutes on GPU for GPT-2
```

**Standard audit** — before reporting results:
```python
pipe.evaluate(pillars=["patching", "ablation", "baselines"], n_examples=256)
```

**Full audit** — publication quality:
```python
pipe.evaluate(pillars=None, n_examples=512,
              n_stability_runs=5, target_task="sva")
```

## The FaithfulnessReport

```python
report.patching_score         # float: Pillar 1
report.ablation_score         # float: Pillar 2
report.stability              # Dict: Pillar 3
report.robustness             # Dict: Pillar 4
report.baseline_comparison    # Dict: Pillar 5
report.generalization         # Dict: Pillar 6
report.metadata               # Dict: run metadata
```

## Cost per pillar (GPT-2, 256 examples, A100)

Rows are **sorted by cost** (cheapest first), not by pillar number.

| Pillar | Time |
|---|---|
| 1 (Patching) | ~30s |
| 2 (Ablation) | ~30s |
| 5 (Baselines) | ~2 min |
| 4 (Robustness) | ~3 min |
| 3 (Stability) | ~10–15 min |
| 6 (Generalization) | ~5–10 min |

## Interpreting results

| Indicator | Weak | Strong |
|---|---|---|
| `report.patching_score` (Pillar 1) | < 0.70 | ≥ 0.85 |
| `report.ablation_score` (Pillar 2) | < 0.70 | ≥ 0.85 |
| circuit vs random (`report.baseline_comparison["circuit_vs_random"]`) | < 0.20 | ≥ 0.40 |
| Stability Spearman | < 0.70 | ≥ 0.85 |
| Better than magnitude (Pillar 5) | No | Yes by ≥ 0.10 |

**`report.patching_score`** (Pillar 1) = normalized faithfulness ratio under
patching, `(y_circuit − y_corrupt) / (y_clean − y_corrupt)` clamped at 1.0;
1.0 means the circuit fully recovers full-model behavior.  
**`report.ablation_score`** (Pillar 2) = the same normalized ratio computed on
the ablated circuit.

## Optional: Intervention reliability

This is **not** the same as Pillar 3 Stability. Pillar 3 Stability measures whether the *discovered circuit* (its node set / score ranking) is consistent across re-discovery seeds; intervention reliability is a separate auxiliary check that scores the reproducibility of the *intervention outcome* across seeds via a reliability index (harmonic mean of R1/R2/R3). Stability is part of the 6-pillar audit; intervention reliability is an optional add-on.

Cross-seed circuit reproducibility — does re-running discovery with different seeds produce the same circuit?

```python
from circuitkit.evaluation.pillars.intervention_reliability import run_intervention_reliability

result = run_intervention_reliability(
    model, graph, task_spec, discovery_cfg, pruning_cfg,
    device=device, metric_fn=metric_fn, dataloader=dataloader,  # required — no defaults
    n_seeds=3,
)
# result["reliability_index"] — harmonic mean of R1/R2/R3 in [0, 1]
```

## Downstream benchmarking

There is no `circuitkit.benchmarks.run_lm_eval_harness`. After pruning/quantization, benchmark the compressed checkpoint with the flat API:
```python
import circuitkit as ck

scores = ck.benchmark(
    "./output/pruned",
    tasks=["boolq", "winogrande"],
    fewshot=0,
    limit=100,
)
```

## Next steps

- [:octicons-arrow-right-24: Causal Patching](../evaluation/causal-patching.md)
- [:octicons-arrow-right-24: Ablation](../evaluation/ablation.md)
- [:octicons-arrow-right-24: Stability](../evaluation/stability.md)
- [:octicons-arrow-right-24: Robustness](../evaluation/robustness.md)
- [:octicons-arrow-right-24: Baselines](../evaluation/baselines.md)
- [:octicons-arrow-right-24: Generalization](../evaluation/generalization.md)
- [:octicons-arrow-right-24: Downstream Benchmarking](../evaluation/benchmarking.md)
