# CircuitArtifact: Unified Circuit Representation

## Overview

`CircuitArtifact` is the unified schema for representing circuits discovered by all CircuitKit discovery methods (ACDC, EAP, EAP-IG, IBCircuit). It provides:

- **Unified graph structure**: Nodes (circuit units) + Edges (connections)
- **JSON serialization**: Save/load circuits for reproducibility
- **Cross-method compatibility**: Convert from any discovery method
- **Intervention masks**: Automatic mask generation for pruning/quantization
- **Sparsity analysis**: Calculate compression ratios and importance distributions
- **Validation framework**: Ensure artifact consistency

## Architecture

### Core Components

```python
from circuitkit.artifacts import CircuitArtifact, Node, Edge, NodeType

# Node: Represents a unit in the circuit
node = Node(
    layer_idx=0,           # Layer in the model
    node_type=NodeType.ATTENTION_HEAD,  # Type of unit
    index=3,               # Index within layer (head/neuron index)
    importance=0.85,       # Normalized importance [0, 1]
    name="L0H3"           # Optional human-readable name
)

# Edge: Represents a connection between units
edge = Edge(
    src_id="L0H3",        # Source node ID
    dst_id="L1H5",        # Destination node ID
    weight=0.9,           # Edge importance [0, 1]
    attribution="direct"  # Type of attribution
)

# CircuitArtifact: Complete circuit representation
artifact = CircuitArtifact(
    model_id="gpt2",
    discovery_method="eap",  # "acdc", "eap", "eap_ig", "ibcircuit"
    task="ioi",
    dataset="ioi_dataset",
    granularity="head",      # "head", "neuron", or "layer"
    threshold=0.5            # Minimum importance for filtering
)

# Add nodes and edges
artifact.add_node("L0H3", node)
artifact.add_edge("E0", edge)
```

### Node Types

Supported node types represent different units in a Transformer:

- **ATTENTION_HEAD**: Individual attention head within a layer
- **MLP_NEURON**: Individual neuron in the MLP (feed-forward) layer
- **ATTENTION_LAYER**: Entire attention layer
- **MLP_LAYER**: Entire MLP layer
- **EMBEDDING**: Embedding layer
- **RESIDUAL**: Residual stream

### Schema Version

The artifact schema includes version tracking for backward compatibility:

```python
artifact.version  # "1.0" - allows future schema evolution
artifact.timestamp  # ISO 8601 creation time
artifact.algorithm_params  # Discovery hyperparameters
```

## Converting from Discovery Methods

### From ACDC

```python
from circuitkit.artifacts import acdc_to_artifact

# After running ACDC discovery
prune_scores = {...}  # Dict[str, Tensor] from ACDC

artifact = acdc_to_artifact(
    prune_scores=prune_scores,
    model_id="gpt2",
    task="ioi",
    dataset="ioi_dataset",
    threshold=0.0,
    granularity="head"
)
```

### From EAP

```python
from circuitkit.artifacts import eap_to_artifact

# After running EAP discovery
node_scores = {
    "A0.0": 0.92,     # Attention layer 0, head 0
    "A0.1": 0.15,     # Attention layer 0, head 1
    "MLP 0": 0.55,    # MLP layer 0
}

artifact = eap_to_artifact(
    node_scores=node_scores,
    model_id="gpt2",
    task="sva",
    dataset="counterfact",
    threshold=0.3,
    granularity="head"
)
```

### From IBCircuit

```python
from circuitkit.artifacts import ibcircuit_to_artifact

# After running IBCircuit discovery
node_scores = {
    "L0.N10": 0.75,   # Layer 0, neuron 10
    "L0.N25": 0.45,   # Layer 0, neuron 25
    "L1.N5": 0.85,    # Layer 1, neuron 5
}

artifact = ibcircuit_to_artifact(
    node_scores=node_scores,
    model_id="pythia-70m",
    task="greater_than",
    dataset="numeric",
    threshold=0.3,
    granularity="neuron"
)
```

## Serialization

### Save to JSON

```python
from pathlib import Path

# Save artifact
artifact.save_json(Path("circuits/my_circuit.json"))

# The JSON includes:
# - Metadata (model, task, discovery method, etc.)
# - All nodes with their properties
# - All edges with their properties
# - Algorithm parameters
# - Timestamp and schema version
```

### Load from JSON

```python
loaded_artifact = CircuitArtifact.load_json(Path("circuits/my_circuit.json"))
```

### Round-trip Serialization

Artifacts support complete round-trip serialization:

```python
# Export
artifact.save_json("circuit.json")

# Load back
artifact2 = CircuitArtifact.load_json("circuit.json")

# Verify equivalence
assert len(artifact.nodes) == len(artifact2.nodes)
assert artifact.model_id == artifact2.model_id
```

## Graph Operations

### Query Nodes

```python
# Get all nodes in a specific layer
layer_0_nodes = artifact.get_nodes_by_layer(0)

# Get all nodes of a specific type
attn_nodes = artifact.get_nodes_by_type(NodeType.ATTENTION_HEAD)
mlp_nodes = artifact.get_nodes_by_type(NodeType.MLP_NEURON)
```

### Query Edges

```python
# Get all edges pointing to a node
incoming = artifact.get_incoming_edges("L1H5")

# Get all edges from a node
outgoing = artifact.get_outgoing_edges("L0H3")
```

### Batch Operations

```python
# Add multiple nodes at once
nodes = {
    "L0H0": Node(0, NodeType.ATTENTION_HEAD, 0, 0.9),
    "L0H1": Node(0, NodeType.ATTENTION_HEAD, 1, 0.3),
}
artifact.add_node_batch(nodes)

# Add multiple edges at once
edges = {
    "E0": Edge("L0H0", "L1H2", 0.85),
    "E1": Edge("L0H1", "L1H2", 0.4),
}
artifact.add_edge_batch(edges)
```

## Conversion to Intervention Masks

Convert circuit to binary masks for pruning/quantization:

```python
from circuitkit.applications import get_arch_config

# Get architecture configuration
arch_cfg = get_arch_config("llama")

# Generate masks (1 = keep, 0 = remove)
masks = artifact.to_mask(model, arch_cfg)

# Masks dict contains binary tensors per layer:
# {
#     "layer_0_attn_heads": Tensor([1, 0, 1, 1, 0, ...]),
#     "layer_0_mlp_neurons": Tensor([1, 1, 0, 1, ...]),
#     ...
# }

# Use masks in interventions
for layer_idx, mask in masks.items():
    # Apply mask to model weights
    pass
```

## Sparsity Analysis

### Calculate Sparsity

```python
# Sparsity = fraction of nodes kept (above threshold)
sparsity = artifact.get_sparsity()
print(f"Circuit sparsity: {sparsity:.2%}")  # e.g., "32.5%"

# Compression = 1 - sparsity
compression = artifact.get_compression_ratio()
print(f"Compression: {compression:.2%}")  # e.g., "67.5%"
```

### Threshold Impact

```python
# Change threshold and recalculate
artifact.threshold = 0.7
new_sparsity = artifact.get_sparsity()

# Summary includes sparsity
print(artifact.summary())
# Output:
# CircuitArtifact Summary
#   Model: gpt2
#   Task: ioi
#   Discovery: eap
#   Nodes: 127 (85 attention, 42 MLP)
#   Edges: 203
#   Sparsity: 32.50%
#   ...
```

## Validation

### Validate Artifact

```python
checks = artifact.validate()

# Returns dict of validation checks:
# {
#     "has_model_id": True,
#     "valid_method": True,
#     "has_nodes": True,
#     "all_nodes_valid": True,
#     "all_edges_valid": True,
#     ...
# }

if all(checks.values()):
    print("Artifact is valid!")
else:
    failed = [k for k, v in checks.items() if not v]
    print(f"Validation failed: {failed}")
```

### Validation Checks

- **Metadata**: Model ID, discovery method, task, dataset present and valid
- **Granularity/threshold**: Granularity is one of head/neuron/layer; threshold is in [0, 1]
- **Nodes**: All nodes are `Node` instances with importance in [0, 1]
- **Edges**: All edges reference existing nodes, weights in [0, 1]
- **Timestamp**: Valid ISO 8601 format

## Normalization

### Normalize Importance Scores

```python
from circuitkit.artifacts import normalize_importance_scores

scores = {
    "n0": 10.0,
    "n1": 5.0,
    "n2": 20.0,
}

# Min-max normalization to [0, 1]
normalized = normalize_importance_scores(scores, method="minmax")
# {
#     "n0": 0.5,  # (10-5)/(20-5)
#     "n1": 0.0,  # (5-5)/(20-5)
#     "n2": 1.0,  # (20-5)/(20-5)
# }

# Z-score normalization (mean 0, std 1)
normalized = normalize_importance_scores(scores, method="zscore")
```

## Integration with Interventions

### Using Artifacts for Pruning

`StructuralPruner` has no `prune_model()` method — the method is `prune()`, and it takes a `CircuitScores` object (node-level scores), not a mask dict:

```python
from circuitkit.applications.pruning import StructuralPruner

pruner = StructuralPruner()

# circuit_scores is a CircuitScores instance with node-level scores
pruned_model = pruner.prune(
    model,
    circuit_scores,
    sparsity=0.3,      # fraction of nodes to mask
    scope="both",      # "heads", "mlp", or "both"
)
```

### Using Artifacts for Quantization

`circuit_quantize()` takes per-head and per-MLP score dicts and the layer count directly — there is no `circuit=`/`precision=`/`keep_mask=` interface:

```python
from circuitkit.applications.quantization import circuit_quantize

quantized_model = circuit_quantize(
    model,
    q_head_scores=q_head_scores,   # Dict[(layer, head), float]
    mlp_scores=mlp_scores,         # Dict[layer, Tensor]
    n_layers=n_layers,
    high_fraction=0.3,             # fraction of layers kept at high precision
)
```

## Best Practices

1. **Always validate**: Run `artifact.validate()` after construction
2. **Preserve metadata**: Keep discovery parameters in `algorithm_params`
3. **Use versioning**: Track artifact schema version for compatibility
4. **Normalize scores**: Use `normalize_importance_scores()` for cross-method comparison
5. **Document sources**: Save with clear model/task/dataset identifiers
6. **Test round-trips**: Verify save/load preserves all information

## Complete Example

```python
from circuitkit.artifacts import eap_to_artifact, CircuitArtifact
from pathlib import Path

# 1. Run EAP discovery
# (external code)
node_scores = {...}

# 2. Convert to artifact
artifact = eap_to_artifact(
    node_scores=node_scores,
    model_id="gpt2",
    task="ioi",
    dataset="ioi_dataset",
    threshold=0.3,
    granularity="head"
)

# 3. Validate
assert all(artifact.validate().values())

# 4. Analyze
print(artifact.summary())
print(f"Sparsity: {artifact.get_sparsity():.2%}")

# 5. Save
artifact.save_json(Path("circuits/gpt2_ioi.json"))

# 6. Use for interventions
from circuitkit.applications import get_arch_config
arch_cfg = get_arch_config("gpt2")
masks = artifact.to_mask(model, arch_cfg)

# 7. Load later
artifact2 = CircuitArtifact.load_json(Path("circuits/gpt2_ioi.json"))
```

## API Reference

See `circuitkit/artifacts/circuit_artifact.py` for full API documentation.

### Key Methods

- `add_node(node_id, node)`: Add single node
- `add_edge(edge_id, edge)`: Add single edge
- `add_node_batch(nodes)`: Add multiple nodes
- `add_edge_batch(edges)`: Add multiple edges
- `get_nodes_by_layer(layer_idx)`: Query nodes in layer
- `get_nodes_by_type(node_type)`: Query nodes by type
- `get_incoming_edges(node_id)`: Find incoming edges
- `get_outgoing_edges(node_id)`: Find outgoing edges
- `to_dict()`: Convert to dictionary
- `save_json(path)` / `load_json(path)`: Serialization
- `to_mask(model, arch_cfg)`: Generate intervention masks
- `get_sparsity()`: Calculate circuit sparsity
- `get_compression_ratio()`: Calculate compression
- `validate()`: Validate artifact consistency
- `summary()`: Human-readable summary
