# Activation Steering Module

## Overview

The Activation Steering module enables steering model behavior via activation patching at circuit-discovered nodes. Instead of fine-tuning weights, steering modifies activations at specific layers during inference to guide the model toward desired outputs.

**Key Capabilities:**
- Compute steering vectors from source/target example pairs
- Apply steering at different coefficient strengths
- Combine multiple steering vectors
- Optimize steering coefficients
- Analyze node importance via ablation
- Generate baseline vectors for comparison

## Core Concept

Steering vectors represent the direction to push activations:
```python
steering_vector = mean(target_activations) - mean(source_activations)
```

During inference, activations at steering nodes are modified:
```python
activation_steered = activation_original + coefficient * steering_vector
```

Where `coefficient` controls steering strength:
- `coefficient = 0.0`: No steering (original behavior)
- `coefficient = 1.0`: Full steering
- `coefficient = 0.5`: Half steering
- `coefficient > 1.0`: Over-steer (amplified effect)

## Quick Start

### 1. Initialize Steering

```python
from circuitkit.applications.steering import ActivationSteering
from circuitkit.artifacts.scores import CircuitScores
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2")

# Load circuit scores (from discovery)
scores = CircuitScores.from_json("circuits/gpt2_ioi.json")

# Initialize steering
steering = ActivationSteering(
    model, 
    scores.node_scores,
    score_threshold=0.5  # Only high-importance nodes
)
```

### 2. Compute Steering Vectors

```python
source_examples = [
    {"text": "When Alice and Bob went to the store, Bob gave"},
    {"text": "The book was given to both Charlie and David. David took"},
    ...
]

target_examples = [
    {"text": "When Charlie and Bob went to the store, Charlie gave"},
    {"text": "The book was given to both Alice and David. Alice took"},
    ...
]

# Compute steering vectors
steering_vectors = steering.compute_steering_vector(
    source_examples,
    target_examples,
    batch_size=32,
    return_all_positions=False  # Average across sequence positions
)
```

### 3. Apply Steering

```python
# Test input
test_input = "When Alice and Bob went to the store, Alice gave"

# Apply steering with coefficient 1.0
result = steering.steer(test_input, steering_vectors, coefficient=1.0)
output_logits = result['output']
output_probs = result['output_probs']

# Test with different coefficients
for coeff in [0.0, 0.5, 1.0, 1.5]:
    result = steering.steer(test_input, steering_vectors, coefficient=coeff)
    print(f"Coefficient {coeff}: {result['output'].shape}")
```

## Advanced Usage

### Multi-Vector Steering

Combine steering vectors from multiple sources:

```python
# Compute steering vectors for different concept pairs
steering_vectors_1 = steering.compute_steering_vector(
    source_1, target_1
)
steering_vectors_2 = steering.compute_steering_vector(
    source_2, target_2
)

# Apply combined steering
result = steering.steer_with_multi_vectors(
    test_input,
    [steering_vectors_1, steering_vectors_2],
    coefficients=[0.6, 0.4],
    aggregate="sum"  # or "mean", "weighted_sum"
)
```

Aggregation modes:
- `"sum"`: Scale and add vectors
- `"mean"`: Average vectors (equal weight)
- `"weighted_sum"`: Weighted combination

### Coefficient Optimization

Find the optimal steering coefficient:

```python
def metric_fn(logits):
    """Custom metric to maximize."""
    # E.g., logit difference between target and distractor
    top_logits, _ = torch.topk(logits[:, -1, :], 2)
    return (top_logits[:, 0] - top_logits[:, 1]).mean().item()

optimal_coeff = steering.find_steering_coefficient(
    test_input,
    metric_fn,
    coef_range=(0.0, 2.0),
    num_steps=11
)
```

### Node Importance Analysis

Measure which nodes have the largest steering effect:

```python
node_importance = steering.analyze_steering_importance(
    test_input,
    metric_fn,
    steering_vectors
)

# Sort by importance
sorted_importance = sorted(
    node_importance.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

for node_name, importance in sorted_importance[:10]:
    print(f"{node_name}: {importance:.4f}")
```

### Baseline Comparisons

Generate baseline vectors for comparison:

```python
# Random baseline (same statistics as actual vectors)
random_baseline = steering.get_random_baseline_vectors()

# Semantic baselines
opposite_baseline = steering.get_semantic_baseline_vectors("opposite")  # Negate vectors
zero_baseline = steering.get_semantic_baseline_vectors("zero")        # All zeros
half_baseline = steering.get_semantic_baseline_vectors("half")        # 0.5x strength
```

### Steering Effects Measurement

Measure how steering strength affects model output:

```python
effects = steering.measure_steering_effect(
    test_input,
    metric_fn,
    coefficients=[0.0, 0.25, 0.5, 0.75, 1.0],
    steering_vectors=steering_vectors
)

# Plot or analyze effects
for coeff, metric_value in sorted(effects.items()):
    print(f"{coeff:.2f}: {metric_value:.4f}")
```

## CLI Usage

### Basic Steering

```bash
circuitkit steer \
  --model gpt2 \
  --circuit-scores circuits/gpt2_ioi.json \
  --source-examples data/ioi_source.csv \
  --target-examples data/ioi_target.csv \
  --coefficient 1.0 \
  --output results/steering
```

### With Analysis

```bash
circuitkit steer \
  --model gpt2 \
  --circuit-scores circuits/gpt2_ioi.json \
  --source-examples data/ioi_source.csv \
  --target-examples data/ioi_target.csv \
  --coefficient 1.0 \
  --analyze \
  --threshold 0.5  # Only nodes with score >= 0.5
```

## API Reference

### ActivationSteering

```python
class ActivationSteering:
    def __init__(self, model, circuit_scores, score_threshold=0.0):
        """
        Initialize steering with model and circuit scores.
        
        Args:
            model: HookedTransformer model
            circuit_scores: Dict mapping node names to importance scores
            score_threshold: Only use nodes with score >= threshold
        """

    def compute_steering_vector(self, source_examples, target_examples,
                               batch_size=32, return_all_positions=False):
        """
        Compute steering vectors from source/target distributions.
        
        Args:
            source_examples: List of source examples
            target_examples: List of target examples
            batch_size: Batch size for activation collection
            return_all_positions: Keep or average sequence positions
        
        Returns:
            Dict mapping node names to steering vectors
        """

    def steer(self, inputs, steering_vectors=None, coefficient=1.0, 
              layer_weights=None):
        """
        Apply steering during forward pass.
        
        Args:
            inputs: Input text or tensor
            steering_vectors: Dict of steering vectors
            coefficient: Steering strength (float or dict)
            layer_weights: Optional per-node weights
        
        Returns:
            Dict with 'output', 'output_probs', 'steered_nodes'
        """

    def steer_with_multi_vectors(self, inputs, steering_vectors_list,
                                 coefficients=None, aggregate="sum"):
        """Apply multiple steering vectors with aggregation."""

    def find_steering_coefficient(self, inputs, target_metric,
                                  steering_vectors=None,
                                  coef_range=(0.0, 2.0), num_steps=11):
        """Find optimal coefficient via grid search."""

    def analyze_steering_importance(self, inputs, target_metric,
                                    steering_vectors=None):
        """Measure node importance via ablation."""

    def measure_steering_effect(self, inputs, metric_fn, coefficients=None,
                               steering_vectors=None):
        """Measure metric at different steering coefficients."""

    def get_steering_statistics(self):
        """Get statistics about computed steering vectors."""

    def get_random_baseline_vectors(self):
        """Generate random vectors with same statistics."""

    def get_semantic_baseline_vectors(self, baseline_type="opposite"):
        """Generate semantic baseline vectors."""
```

## Examples

### IOI Task Steering

Steer GPT-2 to change which person is predicted as indirect object:

```python
# Load circuit for IOI
scores = CircuitScores.from_json("circuits/gpt2_ioi.json")
steering = ActivationSteering(model, scores.node_scores, score_threshold=0.6)

# Source: Predict C (wrong, corrupted)
# Target: Predict B (correct, clean)
source = [{"text": "When Charlie and Bob went to the store, Charlie gave"}]
target = [{"text": "When Alice and Bob went to the store, Alice gave"}]

vectors = steering.compute_steering_vector(source, target)

# Apply to new example
new_input = "When John and Mary went to the store, John gave"
result = steering.steer(new_input, vectors, coefficient=1.0)
```

### Circuit Node vs Random Node Comparison

Verify that circuit nodes have larger steering effect:

```python
# Circuit nodes
circuit_vectors = {
    name: vectors[name]
    for name in list(vectors.keys())[:5]  # Top nodes
}

# Random nodes
random_vectors = steering.get_random_baseline_vectors()

# Compare effects
effect_circuit = steering.measure_steering_effect(
    test_input, metric_fn, coefficients=[1.0],
    steering_vectors=circuit_vectors
)[1.0]

effect_random = steering.measure_steering_effect(
    test_input, metric_fn, coefficients=[1.0],
    steering_vectors=random_vectors
)[1.0]

print(f"Circuit effect: {effect_circuit:.4f}")
print(f"Random effect: {effect_random:.4f}")
print(f"Ratio: {effect_circuit / effect_random:.2f}x")
```

## Testing

Run unit tests:

```bash
pytest tests/unit/test_steering.py -v
```

Run integration tests on IOI:

```bash
pytest tests/integration/test_steering_ioi.py -v
```

Run example:

```bash
python examples/10-steering.py
```

## Key Design Decisions

1. **Activation Patching**: Steer at hook points, not weight parameters
2. **Circuit-Guided**: Only steer at important nodes (circuit scores >= threshold)
3. **Differential Vectors**: Steering = target_mean - source_mean
4. **Flexible Aggregation**: Support sum, mean, weighted combinations
5. **Coefficient Search**: Grid search for optimal steering strength
6. **Importance Analysis**: Ablation-based node importance ranking

## Limitations

1. **Data Requirements**: Need good source/target example pairs
2. **Hook Point Mapping**: Limited to model's available hooks
3. **Position Averaging**: Current implementation averages across sequence positions
4. **Batch Processing**: Activations collected in batches, may affect aggregation
5. **Device Memory**: Large models may require small batch sizes

## Future Work

1. Position-specific steering (not averaged)
2. Causal steering analysis (trace effects through layers)
3. Adaptive coefficient scheduling
4. Steering with constraints
5. Steering for multiple objectives
6. GPU-accelerated vector search

## References

- Zou et al. (2023): "Representation Engineering: The Right Tool for AI Control"
- Turner et al. (2023): "Activation Addition: Steering Language Models Without Optimization" [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- See also: Soft Healing module for LoRA-based intervention

## Questions?

For issues or questions about steering:
1. Check examples in `examples/10-steering.py`
2. Review integration tests in `tests/integration/test_steering_ioi.py`
3. See full API docs above
