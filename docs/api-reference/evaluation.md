# Evaluation API

**Module**: `circuitkit.evaluation`

---

## `run_full_faithfulness`

**Module**: `circuitkit.evaluation.full`

Orchestrates the 6-pillar faithfulness framework end-to-end. Pillars run in cost order (fast first). Returns a `FaithfulnessReport`.

```python
from circuitkit.evaluation import run_full_faithfulness

report = run_full_faithfulness(
    model=model,
    graph=graph,
    task_spec=task_spec,
    discovery_cfg=discovery_cfg,
    # optional:
    pillars=None,           # List[str] or None (all)
    n_stability_runs=5,
    n_reliability_seeds=3,
    target_task_spec=None,  # required for Pillar 6
    target_dataloader=None,
    pruning_cfg=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HookedTransformer` | — | Loaded model |
| `graph` | circuit graph | — | The EAP graph with nodes selected via `apply_topn()` or `apply_threshold()` |
| `task_spec` | `TaskSpec` | — | Task specification |
| `discovery_cfg` | `Dict` | — | Discovery config block |
| `pillars` | `List[str]` or `None` | `None` (all) | Subset to run |
| `n_stability_runs` | `int` | 5 | Pillar 3 re-discovery runs |
| `n_reliability_seeds` | `int` | 3 | Intervention-reliability seeds |
| `target_task_spec` | optional | `None` | Required for Pillar 6; omitting skips it |
| `pruning_cfg` | `Dict` or `None` | `None` | Passed through to re-discovery pillars |

### Valid Pillar Keys

```python
pillars = [
    "patching",               # Pillar 1
    "ablation",               # Pillar 2
    "stability",              # Pillar 3
    "robustness",             # Pillar 4
    "baselines",              # Pillar 5
    "generalization",         # Pillar 6 (preliminary)
    "intervention_reliability",  # optional 7th pillar
]
```

!!! warning "Pillar 6 is preliminary"
    Generalization has not been validated at scale. Treat its scores as preliminary until a production sweep is completed.

### Example

```python
from circuitkit.evaluation import run_full_faithfulness

# Run only the fast pillars
report = run_full_faithfulness(
    model, graph, task_spec, discovery_cfg,
    pillars=["patching", "ablation", "baselines"],
)
print(report.patching_score)
print(report.ablation_score)
```

---

## `FaithfulnessReport`

**Module**: `circuitkit.evaluation.report`

Dataclass returned by `run_full_faithfulness` and `ck.faithfulness`.

```python
from circuitkit.evaluation.report import FaithfulnessReport

report.patching_score          # float or None
report.ablation_score          # float or None
report.stability               # Dict or None
report.robustness              # Dict or None
report.baseline_comparison     # Dict or None
report.generalization          # Dict or None
report.intervention_reliability  # Dict or None
report.metadata                # Dict with run metadata
```

All fields are `None` if the corresponding pillar was not run.

```python
# Check which pillars ran
ran = {k: v for k, v in report.__dict__.items() if v is not None}
print(ran.keys())
```

---

## `evaluate_graph`

**Module**: `circuitkit.evaluation.evaluate`

Low-level: score a circuit's faithfulness by running the model with out-of-circuit edges ablated.

```python
from circuitkit.evaluation.evaluate import evaluate_graph

score = evaluate_graph(
    model=model,
    graph=graph,
    dataloader=dataloader,
    metrics=metric_fn,         # callable or list of callables
    intervention="patching",   # "patching", "zero", "mean", "mean-positional"
    quiet=False,
    skip_clean=True,
)
```

### Metric Signature

```python
def my_metric(logits, clean_logits, input_lengths, labels) -> torch.Tensor:
    ...
```

### Intervention Options

| Mode | Description |
|------|-------------|
| `"patching"` | Default — patch in activations from clean run |
| `"zero"` | Zero out out-of-circuit activations |
| `"mean"` | Replace with mean activations |
| `"mean-positional"` | Replace with mean per position |

`"mean"` and `"mean-positional"` require `intervention_dataloader`.

Call `graph.apply_topn(n)` or `graph.apply_threshold(t)` before calling `evaluate_graph` to define the circuit boundary.

---

## `evaluate_baseline`

**Module**: `circuitkit.evaluation.evaluate`

Evaluate the unmodified model on a dataset to establish a performance baseline. No interventions are applied. (Random and magnitude baseline circuits are handled by `Pillar5_Baselines`.)

```python
from circuitkit.evaluation.evaluate import evaluate_baseline

score = evaluate_baseline(model, dataloader, metrics)
```

Pass `run_corrupted=True` to score the corrupted input instead of the clean input.

---

## Pillar Classes

For fine-grained control, run individual pillars. Each pillar exposes a static `run()` method (no instantiation):

```python
from circuitkit.evaluation.pillars import (
    Pillar1_CausalPatching,
    Pillar2_Ablation,
    Pillar3_Stability,
    Pillar4_Robustness,
    Pillar5_Baselines,
    Pillar6_Generalization,
)

result = Pillar1_CausalPatching.run(model, graph, dataloader, metric_fn)
score = result["score"]
```

---

## Intervention Reliability (Optional Pillar 7)

**Module**: `circuitkit.evaluation.pillars.intervention_reliability`

```python
from circuitkit.evaluation.pillars.intervention_reliability import run_intervention_reliability

result = run_intervention_reliability(
    model, graph, task_spec, discovery_cfg, pruning_cfg,
    device, metric_fn, dataloader, n_seeds=3,
)

result["r1_seed_consistency"]   # mean Spearman rho across seed pairs
result["r2_effect_magnitude"]
result["r3_effect_variance"]
result["reliability_index"]     # harmonic mean, [0, 1]
result["n_seeds"]
result["per_seed"]              # per-seed breakdown
```

---

## Next Steps

- [Evaluation Framework](../evaluation/framework.md) — the 6-pillar architecture
- [Flat Typed API: faithfulness](flat-api.md#faithfulness) — simplified entrypoint
- [User Guide: Evaluation](../user-guide/evaluation.md) — workflow and interpretation
