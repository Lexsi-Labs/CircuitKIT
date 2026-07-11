# Circuit Artifacts

Every `discover_circuit` call writes three files to disk. This page describes the format, how to load them, and how to work with the `Circuit` object.

---

## The Three Files

| File | Format | Contents |
|------|--------|---------|
| `circuit.pt` | PyTorch serialized | Pruning artifact — list of node names (node-level) or neuron dict (neuron-level) |
| `circuit_scores.json` | JSON | Human-readable `CircuitScores` record — top-level `task` / `model` / `algorithm` / `level` / `timestamp` metadata plus `node_scores`, the `{"A0.1": 0.83, "MLP 5": 0.61, ...}` mapping. (A bare `Circuit.save()` writes just `{"node_scores": {...}}`.) |
| `circuit_scores.pt` | PyTorch serialized | Same scores as tensors — required by `selective_finetune` and `Pipeline.from_scores()` |

The `circuit.pt` file is the primary artifact. The `_scores.json` and `_scores.pt` side-cars are written next to it automatically (same directory, same stem with suffix).

---

## File Naming

`Pipeline.discover()` writes to:
```text
{output_dir}/{algorithm}_{model}_{task}_{level}.pt
```

For example: `./results/eap-ig_gpt2_ioi_node.pt`

`discover_circuit(config)` uses the `output_path` key in the config dict.

---

## Loading Artifacts

Which loader you use dictates which file you must pass. The pruning artifact (`circuit.pt`) and the tensor score side-car (`circuit_scores.pt`) are **not interchangeable** — passing the wrong one is the most common mistake here:

| Loader | Required file |
|---|---|
| `ck.load_scores(...)` | `circuit.pt` |
| `Pipeline.from_artifact(...)` | `circuit.pt` |
| `Pipeline.from_scores(...)` | `circuit_scores.pt` |
| `selective_finetune(...)` | `circuit_scores.pt` |

### Via `ck.load_scores`

```python
import circuitkit as ck

circuit = ck.load_scores("./circuit.pt")

# Inspect
print(circuit.nodes)          # list of node names: ["A0.1", "A2.3", "MLP 5", ...]
print(circuit.level)          # "node" or "neuron"
print(circuit.task)           # "ioi"
print(circuit.algorithm)      # "eap-ig"
print(circuit.model_name)     # "gpt2"
print(len(circuit))           # number of nodes in the circuit
print(circuit.top_nodes(5))   # top 5 nodes by score
```

### Via `Pipeline.from_artifact`

```python
from circuitkit import Pipeline

pipe = Pipeline.from_artifact(
    "./circuit.pt",
    model_name="gpt2",
    task="ioi",
)
pipe.prune(sparsity=0.3)     # can go straight to prune
```

### Via `Pipeline.from_scores`

```python
pipe = Pipeline.from_scores(
    "./circuit_scores.pt",   # must be the _scores.pt file
    model_name="gpt2",
)
result = pipe.selective_finetune(top_fraction=0.2)
```

### Loading the JSON Scores

```python
import json

with open("./circuit_scores.json") as f:
    scores = json.load(f)["node_scores"]  # unwrap the nested CircuitScores record

# Sort by score
sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for node, score in sorted_nodes[:10]:
    print(f"{node}: {score:.4f}")
```

---

## The Circuit Object

`Circuit` is returned by `ck.load_scores`, `ck.discover`, and stored in `pipe.circuit`.

```python
circuit.nodes          # List[str] (node-level) or Dict (neuron-level)
circuit.scores         # Dict[str, float]: {node_name: importance_score}
circuit.level          # "node" or "neuron"
circuit.task           # task name the circuit was discovered for
circuit.algorithm      # algorithm used for discovery
circuit.model_name     # model name
circuit.artifact_path  # path to the .pt file

# Convenience methods
circuit.top_nodes(n)   # top N nodes by score
len(circuit)           # number of nodes
```

---

## Artifact Metadata

`circuit.pt` is **not** a dict — for node-level circuits it is a plain `list[str]` of node names, saved directly via `torch.save(nodes_to_prune, output_path)`. Calling `.keys()` on it raises `AttributeError`. Metadata (level, task, algorithm, model name, scores) lives in the `_scores.json` / `_scores.pt` side-cars described above, not in `circuit.pt` itself.

```python
import torch

nodes = torch.load("./circuit.pt", map_location="cpu")
print(nodes[:10])  # list[str], e.g. ['A0.1', 'A2.3', 'MLP 5', ...]
```

---

## The CircuitArtifact Schema (Research)

For research use, CircuitKit also provides a `CircuitArtifact` class that supports JSON serialization, graph queries, and validation:

```python
from circuitkit.artifacts import eap_to_artifact, CircuitArtifact

# Convert EAP scores to CircuitArtifact
artifact = eap_to_artifact(
    node_scores={"A0.1": 0.83, "MLP 5": 0.61},
    model_id="gpt2",
    task="ioi",
    dataset="ioi_dataset",
    threshold=0.3,
    granularity="head",
)

# Save/load as JSON
artifact.save_json("./circuit.json")
artifact2 = CircuitArtifact.load_json("./circuit.json")

# Query
print(artifact.get_nodes_by_layer(0))
print(artifact.get_sparsity())
print(artifact.validate())
```

`CircuitArtifact` is for research workflows that need graph-level queries or cross-method comparison. For standard discovery/evaluation/application workflows, use the `Circuit` object and `.pt` artifacts.

---

## Normalizing Scores for Cross-Method Comparison

When comparing circuits from different algorithms, normalize scores to [0, 1]:

```python
from circuitkit.artifacts import normalize_importance_scores

scores_eap_ig = {"A0.1": 2.3, "MLP 5": 1.1, "A2.3": 0.4}
scores_acdc = {"A0.1": 0.8, "MLP 5": 0.3, "A2.3": 0.05}

norm_eap = normalize_importance_scores(scores_eap_ig, method="minmax")
norm_acdc = normalize_importance_scores(scores_acdc, method="minmax")

# Now directly comparable
for node in norm_eap:
    print(f"{node}: eap-ig={norm_eap[node]:.3f}, acdc={norm_acdc.get(node, 0):.3f}")
```

---

## Next Steps

- [API Reference: Flat API](../api-reference/flat-api.md) — `ck.load_scores` signature
- [User Guide: Selectors](../user-guide/selectors.md) — how scores are computed
- [User Guide: Applications](../user-guide/applications.md) — acting on circuit artifacts
