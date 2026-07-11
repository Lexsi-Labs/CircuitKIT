# Dict-Config API

**Module**: `circuitkit.api` (re-exported from `circuitkit`)

The dict-config API is the lowest-level public surface. Every flat API function and Pipeline method ultimately calls into it. Use it when you need precise control over any discovery parameter not exposed by the flat API, or when working with YAML config files.

---

## `discover_circuit`

```python
discover_circuit(config: Union[str, Dict]) -> Union[List[str], Dict]
```

Run a discovery algorithm and return a pruning artifact. Writes three files to disk: `circuit.pt`, `circuit_scores.json`, `circuit_scores.pt`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str` or `Dict` | YAML file path or config dict with required keys: `model`, `discovery`, `pruning` |

### Config Schema

```python
{
    "model": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",   # HF / TransformerLens model id
        "precision": "bfloat16",                        # torch dtype string
    },
    "discovery": {
        "algorithm": "eap-ig",      # one of 13; only eap/eap-ig are validated at scale
        "task": "mmlu",             # registered task name
        "level": "node",            # "node" or "neuron"
        "chat_template_mode": "auto",  # "auto", "on", "off" — optional
        "batch_size": 4,
        "data_params": {
            "num_examples": 128,
            "batch_size": 4,
        },
        # EAP-IG specific (optional):
        "ig_steps": 5,
    },
    "pruning": {
        "target_sparsity": 0.3,    # fraction of components to remove
        "scope": "both",           # "heads", "mlp", or "both"
    },
    "output_path": "./circuit.pt",  # optional; where to write the artifact
}
```

### Returns

- **Node-level** (`level="node"`): `List[str]` — node names to prune, e.g. `["A0.1", "MLP 3"]`
- **Neuron-level** (`level="neuron"`): `Dict` with `mlp`, `heads`, and `_meta` keys

### Warnings

Emits `UserWarning` when an experimental- or research-tier algorithm is requested.

### Example

```python
from circuitkit import discover_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "level": "node",
        "ig_steps": 5,
        "data_params": {"num_examples": 64, "batch_size": 4},
    },
    "pruning": {"target_sparsity": 0.3, "scope": "both"},
    "output_path": "./gpt2_ioi.pt",
})
print(circuit[:5])  # ['A0.1', 'A2.3', 'MLP 5', ...]
```

---

## `evaluate_circuit`

```python
evaluate_circuit(
    config: Union[str, Dict],
    pruned_artifact_path: Optional[str] = None,
    scores_path: Optional[str] = None,
) -> FaithfulnessReport
```

Evaluate circuit faithfulness with the 6-pillar framework. Thin wrapper around `run_full_faithfulness`; reconstructs the circuit graph from the saved scores file.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str` or `Dict` | Same shape as `discover_circuit` — model, discovery, and pruning keys |
| `pruned_artifact_path` | `str`, optional | Path to the `.pt` artifact; defaults to `config["output_path"]` |
| `scores_path` | `str`, optional | Path to the `_scores.pt` file; auto-derived if omitted |

### Returns

A `FaithfulnessReport` with `.patching_score` (Pillar 1, causal patching) and
`.ablation_score` (Pillar 2, ablation) — normalized faithfulness ratios in [0, 1] —
plus optional pillar fields and a `.metadata` dict. A random-circuit baseline, when
requested, is in `.metadata["random_avg"]`.

### Example

```python
from circuitkit import evaluate_circuit

report = evaluate_circuit(
    {"model": {"name": "gpt2"}, "discovery": {"task": "ioi"}, "pruning": {}},
    scores_path="./gpt2_ioi_scores.pt",
)
print(report.ablation_score)   # e.g. 0.83
```

---

## `load_circuit`

```python
load_circuit(circuit_path: str) -> Union[List[str], Dict]
```

Load a previously saved circuit artifact from disk. Returns the same type as `discover_circuit` (list for node-level, dict for neuron-level).

```python
from circuitkit import load_circuit

nodes = load_circuit("./gpt2_ioi.pt")
```

---

## YAML Config Files

All three functions accept a path to a YAML file instead of a dict:

```yaml
# circuit.yaml
model:
  name: gpt2
  precision: float32

discovery:
  algorithm: eap-ig
  task: ioi
  level: node
  ig_steps: 5
  data_params:
    num_examples: 128
    batch_size: 4

pruning:
  target_sparsity: 0.3
  scope: heads

output_path: ./circuit.pt
```

```python
circuit = discover_circuit("./circuit.yaml")
```

This is the same format the CLI uses internally.

---

## Package-Root Re-exports

These are available directly from `circuitkit`:

```python
import circuitkit

circuitkit.discover_circuit(config)
circuitkit.evaluate_circuit(config)
circuitkit.load_circuit(path)
circuitkit.get_task("ioi")
circuitkit.list_tasks()
circuitkit.register_task(spec)
circuitkit.__version__   # "1.0.0"
```

---

## Next Steps

- [Flat Typed API](flat-api.md) — same operations with typed kwargs
- [Configuration Guide](../getting-started/configuration.md) — all five interfaces compared
