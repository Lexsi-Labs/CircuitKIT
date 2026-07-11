# API Reference Overview

CircuitKit exposes three API surfaces for the same underlying engine. Pick whichever fits your workflow.

---

## Three API Surfaces

| Surface | Entry Point | Best For |
|---------|------------|---------|
| **Dict-Config API** | `discover_circuit(config)` | Config files, scripting, CLI parity |
| **Flat Typed API** | `ck.discover(model, task, ...)` | Interactive notebooks, typed code |
| **Pipeline API** | `Pipeline(model_name).discover(...).evaluate(...)` | Chained multi-step workflows |

All three call the same discovery engine and produce the same artifacts.

---

## Quick-Reference Cheat Sheet

### Flat API (`circuitkit.*`)

```python
import circuitkit as ck

model   = ck.load_model("gpt2", dtype="bfloat16")
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=128)
report  = ck.faithfulness(model, circuit, "ioi")
pruned  = ck.prune(model, circuit, sparsity=0.3, scope="both")
ck.export_checkpoint(pruned, circuit, "./checkpoint")
ck.benchmark("./checkpoint", tasks=["boolq"])
scores  = ck.load_scores("./circuit.pt")
```

### Dict-Config API (`circuitkit.api`)

```python
from circuitkit import discover_circuit, evaluate_circuit, load_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2"},
    "discovery": {"algorithm": "eap-ig", "task": "ioi", "level": "node",
                  "data_params": {"num_examples": 128}},
    "pruning": {"target_sparsity": 0.3, "scope": "both"},
    "output_path": "./circuit.pt",
})
# Illustrative — pass the same config dict shape as discover_circuit above.
report = evaluate_circuit(
    {
        "model": {"name": "gpt2"},
        "discovery": {"task": "ioi", "level": "node"},
    },
    scores_path="./circuit_scores.pt",
)
```

### Pipeline API (`circuitkit.Pipeline`)

```python
from circuitkit import Pipeline

pipe = (
    Pipeline("gpt2", task="ioi")
    .discover(algorithm="eap-ig", n_examples=128)
    .evaluate()
    .prune(sparsity=0.3)
)
pipe.export("./checkpoint")  # returns the checkpoint path
pipe.summary()
```

---

## Module Map

| Module | Contents |
|--------|---------|
| `circuitkit.api` | `discover_circuit`, `evaluate_circuit`, `load_circuit` |
| `circuitkit.quick` | Flat API: `load_model`, `discover`, `faithfulness`, `prune`, `quantize`, `export_checkpoint`, `benchmark`, `load_scores`, `selective_finetune`, `visualize_circuit` |
| `circuitkit.pipeline` | `Pipeline` class |
| `circuitkit.backends` | `STABILITY`, `DISCOVERY_ALGORITHMS`, `is_stable`, `default_algorithm` |
| `circuitkit.evaluation` | `run_full_faithfulness`, `evaluate_graph`, `FaithfulnessReport` |
| `circuitkit.tasks` | `get_task`, `list_tasks`, `register_task` |
| `circuitkit.selection` | `get_selector`, `list_selectors`, `register` |
| `circuitkit.applications` | Pruning, quantization, editing, steering, finetuning |

---

## Stability Guarantee

The following are considered the **stable public API** (no breaking changes without a major version bump):

- `discover_circuit`, `evaluate_circuit`, `load_circuit`
- All 10 flat API functions in `circuitkit.quick`
- `Pipeline` class methods
- `circuitkit.backends` stability map
- `circuitkit.evaluation` 6-pillar surface

Experimental and research-tier backends may change without notice.

---

## Detailed Reference

- [Dict-Config API](dict-config.md) — `discover_circuit`, `evaluate_circuit`, `load_circuit`
- [Flat Typed API](flat-api.md) — `ck.*` function signatures
- [Pipeline Class](pipeline.md) — constructors and method chaining
- [Backends](backends.md) — stability tiers and algorithm registry
- [Evaluation](evaluation.md) — `run_full_faithfulness`, `FaithfulnessReport`
- [Tasks](tasks.md) — task registry and custom task registration
- [Selectors](selectors.md) — selector registry and custom selectors
- [Applications](applications.md) — pruning, quantization, steering, editing
