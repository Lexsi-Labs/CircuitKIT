# CircuitKit Analysis Module

Analysis tools for evaluating and understanding circuit behavior.

## Overview

The analysis module provides tools for:
- **Metrics Computation**: Calculate various evaluation metrics
- **Score Aggregation**: Combine scores across multiple dimensions
- **Statistical Analysis**: Analyze circuit properties
- **Attribution Analysis**: Understand feature and node importance

## Main Components

### metrics.py
Compute domain-specific metrics for circuit evaluation:
- Accuracy metrics (accuracy, F1, precision, recall)
- Ranking metrics (MRR, NDCG)
- QA metrics (exact match, span F1)
- Generation metrics (BLEU, ROUGE, BERTScore)
- Semantic metrics (perplexity, similarity)

**Usage:**
```python
from circuitkit.analysis import compute_metrics

metrics = compute_metrics(
    predictions=logits,
    targets=labels,
    metric_types=["accuracy", "f1"]
)
```

### scores.py
Aggregates and processes scores:
- Normalize scores to comparable ranges
- Combine scores using different strategies
- Track score evolution over iterations
- Statistical analysis of scores

**Usage:**
```python
from circuitkit.analysis import compute_scores

scores = compute_scores(
    circuit=circuit,
    samples=data,
    scoring_method="importance"
)
```

## Key Functions

### compute_metrics()
```python
def compute_metrics(predictions, targets, metric_types):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions or logits
        targets: Ground truth labels
        metric_types: List of metrics to compute
        
    Returns:
        dict: Computed metrics and their values
    """
```

### compute_scores()
```python
def compute_scores(circuit, samples, scoring_method):
    """
    Compute importance scores for circuit components.
    
    Args:
        circuit: Circuit to analyze
        samples: Sample data
        scoring_method: Method to use (importance, gradient, etc.)
        
    Returns:
        dict: Scores for each component
    """
```

## Metric Types

### Classification Metrics
- Accuracy
- F1 Score
- Precision
- Recall
- Macro/Micro averaged versions

### Ranking Metrics
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)

### Question Answering Metrics
- Exact Match (EM)
- Span F1
- BLEU (for generation-style QA)

### Generation Metrics
- BLEU Score
- ROUGE (ROUGE-L, ROUGE-1, ROUGE-2)
- BERTScore
- Perplexity

## Examples

### Computing Accuracy
```python
from circuitkit.analysis import compute_metrics

# Simple accuracy
metrics = compute_metrics(
    predictions=torch.argmax(logits, dim=-1),
    targets=labels,
    metric_types=["accuracy"]
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Computing Multiple Metrics
```python
metrics = compute_metrics(
    predictions=logits,
    targets=labels,
    metric_types=["accuracy", "f1", "precision", "recall"]
)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Analyzing Circuit Components
```python
from circuitkit.analysis import compute_scores

# Get importance scores for circuit
scores = compute_scores(
    circuit=my_circuit,
    samples=task.get_samples(n=100),
    scoring_method="importance"
)

# Rank components by importance
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for component, score in ranked[:10]:
    print(f"{component}: {score:.4f}")
```

## Integration with Evaluation Framework

The analysis module integrates with CircuitKit's main evaluation pipeline:

```python
from circuitkit import evaluate_circuit

# Evaluation automatically uses analysis metrics
metrics = evaluate_circuit(
    circuit=circuit,
    task=task,
    evaluation_pillars=["accuracy", "sufficiency"]
)
```

## Contributing

When adding new metrics:
1. Add computation function to `metrics.py`
2. Update metric registry
3. Add tests in `tests/analysis/`
4. Document in this README

## References

- [Metric Definitions](../../docs/CONCEPTS.md#metrics)
- [Evaluation Framework](../../docs/CONCEPTS.md#faithfulness-evaluation)
- [6-Pillar Evaluation](../../docs/v03/README.md#evaluation-framework-m70)
