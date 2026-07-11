# Hallucination Detection Guide

## Overview

The `HallucinationDetector` scores circuit activations for a piece of text to flag probable hallucinations. It uses trained linear probes to distinguish between factual and hallucinated outputs based on circuit activation patterns. Scoring is post-hoc over already-produced text — the current implementation runs the probes on activations extracted from the input rather than streaming them live during token generation.

## Motivation

Large language models frequently generate plausible-sounding but factually incorrect outputs ("hallucinations"). Detecting these requires:

1. **Activation-level scoring** of generated text
2. **Circuit-aware** - only monitor relevant model components
3. **Data-driven** - learn patterns from labeled examples
4. **Lightweight** - minimal computational overhead

The `HallucinationDetector` addresses all four requirements by:
- Training simple linear probes on circuit activations
- Using these probes to classify tokens during generation
- Aggregating per-token scores into generation-level confidence

## Architecture

### Components

```python
Model
  ├─> Extract circuit activations (hook into forward pass)
  │     └─> Pass through trained linear probes
  │           └─> Compute per-token hallucination probability
  │                 └─> Aggregate to generation probability
  └─> Detection results & explanations
```

### LinearProbe

Simple model: `activations → Linear(hidden_dim, 1) → Sigmoid → [0, 1]`

```python
class LinearProbe(nn.Module):
    """Map [batch, seq_len, hidden_dim] → [batch, seq_len, 1] probability"""
```

- **Input**: Activation tensor from a specific layer
- **Output**: Hallucination probability in [0, 1]
- **Training**: Binary cross-entropy loss on clean/corrupt pairs

### ProbeTrainer

Handles training loop with validation and early stopping:

```python
class ProbeTrainer:
    """Trainer for linear probes with AUROC monitoring"""
```

- **Optimization**: Adam with L2 regularization
- **Metrics**: Accuracy, AUROC (area under ROC curve), loss
- **Early stopping**: Stop if validation AUROC plateaus
- **Checkpointing**: Save best model state

### HallucinationDetector

Main interface for hallucination detection:

```python
class HallucinationDetector:
    """Monitor circuits for hallucination signals during generation"""
```

Key methods:
- `train_probes()` - Train on labeled examples
- `detect_hallucinations()` - Detect hallucinations in text
- `get_activation_profile()` - Extract circuit activations
- `save_probes()` / `load_probes()` - Serialize trained probes

## Usage

### 1. Initialize Detector

!!! warning "Use a HuggingFace model + a `CircuitArtifact` — not `ck.load_model` / `ck.load_scores`"
    `HallucinationDetector` reads `model.config.hidden_size` and each node's `.layer_idx`,
    so it needs a **HuggingFace** model (`AutoModelForCausalLM.from_pretrained(...)`) and a
    **`CircuitArtifact`** (whose nodes carry `.layer_idx`). The flat-API `ck.load_model()`
    returns a TransformerLens `HookedTransformer` (no `.config.hidden_size`), and
    `ck.load_scores()` returns a `Circuit` of plain node-name strings (no `.layer_idx`) —
    passing either directly raises `AttributeError`. Convert scores first with
    `eap_to_artifact(node_scores=circuit.scores, model_id="gpt2", ...)`.

```python
from circuitkit.applications.common_utils.hallucination_detection import HallucinationDetector
from circuitkit.artifacts import CircuitArtifact

# Load or discover a circuit
circuit = CircuitArtifact.load_json("path/to/circuit.json")

# Create detector
detector = HallucinationDetector(
    model=model,
    circuit=circuit,
    arch_cfg=arch_cfg,
    device="cuda"
)
```

### 2. Prepare Training Data

Data should be a list of dictionaries:

```python
train_data = [
    {
        "text": "Paris is the capital of France",
        "is_hallucination": False
    },
    {
        "text": "Paris is the capital of Germany",
        "is_hallucination": True
    },
    # ... more examples
]

val_data = [
    {
        "text": "2 + 2 equals 4",
        "is_hallucination": False
    },
    {
        "text": "2 + 2 equals 5",
        "is_hallucination": True
    },
]
```

**Data Requirements:**
- **Minimum examples**: 20-50 per class for reliable training
- **Balance**: Equal or near-equal number of hallucination vs. factual
- **Diversity**: Cover multiple domains/topics for robustness
- **Labels**: Must be carefully verified (critical for performance)

### 3. Train Probes

```python
results = detector.train_probes(
    train_data=train_data,
    val_data=val_data,
    batch_size=32,
    epochs=10,
    learning_rate=1e-3,
    patience=3  # Early stopping after 3 epochs without improvement
)

print(f"Best validation AUROC: {results['best_val_auroc']:.4f}")
print(f"Trained {results['num_probes']} probes")
```

**Training Parameters:**
- `batch_size`: 16-64 typical (adjust for memory)
- `epochs`: 5-20 usually sufficient
- `learning_rate`: 1e-3 to 1e-4 typical
- `patience`: 3-5 for early stopping

**Output:**
Returns dictionary with:
- `best_val_auroc`: Validation AUROC of best model
- `num_probes`: Number of trained probes (one per circuit layer)
- `probes`: Dict of trained LinearProbe objects
- `circuit_layers`: Layers where probes were trained

### 4. Detect Hallucinations

```python
result = detector.detect_hallucinations(
    text="The capital of France is Paris",
    threshold=0.5
)

# Result contains:
print(result["hallucination_prob"])  # e.g., 0.15 (likely factual)
print(result["is_hallucination"])    # e.g., False
print(result["explanation"])         # Human-readable explanation
print(result["flagged_tokens"])      # Indices of high-prob hallucination tokens
```

**Output:**
```python
{
    "text": "The capital of France is Paris",
    "is_hallucination": False,
    "hallucination_prob": 0.15,
    "per_token_probs": [0.2, 0.1, 0.3, ...],  # Per-token scores
    "flagged_tokens": [2, 5],  # Token indices > threshold
    "circuit_activations": {...},  # Raw activation data
    "explanation": "Low hallucination probability (15%). Likely factual.",
    "num_probes": 3
}
```

## Integration with Generation

### Monitor During Generation

```python
def generate_with_monitoring(model, prompt, detector, max_length=100):
    """Generate with hallucination monitoring."""
    generated = prompt
    all_results = []

    for step in range(max_length):
        # Generate next token
        token = model.generate_one(prompt)
        generated += token

        # Check for hallucination
        result = detector.detect_hallucinations(generated)
        all_results.append(result)

        if result["hallucination_prob"] > 0.7:
            print(f"WARNING: High hallucination probability at step {step}")
            print(f"  Current text: {generated}")
            break

    return generated, all_results
```

### Selective Generation

```python
# Only allow generation to continue if hallucination prob stays low
MAX_HALLUC_PROB = 0.6

def generate_safely(model, prompt, detector):
    generated = prompt
    history = []

    while len(generated) < max_tokens:
        result = detector.detect_hallucinations(generated)
        history.append(result)

        if result["hallucination_prob"] > MAX_HALLUC_PROB:
            return generated, "STOPPED (hallucination detected)"

        # Generate next token
        token = model.generate_one(generated)
        generated += token

    return generated, "COMPLETED"
```

## Best Practices

### 1. Data Quality
- **Verify labels manually** - Mislabeled data dramatically hurts performance
- **Use domain experts** - Especially for technical domains
- **Document label criteria** - What counts as hallucination?

### 2. Training
- **Monitor AUROC** - Target > 0.85 for reliable detection
- **Use validation set** - Prevents overfitting to training distribution
- **Try multiple seeds** - Probe training has stochasticity
- **Save best model** - Use trained state dict after convergence

### 3. Deployment
- **Set appropriate thresholds** - 0.5 is default but may need tuning
- **Monitor false positive/negative rates** - What's the cost of each?
- **Update probes periodically** - As model gets deployed/fine-tuned
- **Log detections** - Track where hallucinations are happening

### 4. Evaluation
```python
def evaluate_detector(detector, test_data):
    """Evaluate hallucination detection accuracy."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
    
    predictions = []
    labels = []
    
    for example in test_data:
        result = detector.detect_hallucinations(example["text"])
        predictions.append(result["hallucination_prob"])
        labels.append(example["is_hallucination"])
    
    # Compute metrics
    auroc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, [p > 0.5 for p in predictions])
    precision, recall, _ = precision_recall_curve(labels, predictions)
    
    return {"auroc": auroc, "accuracy": acc, "precision": precision, "recall": recall}
```

## Troubleshooting

### Problem: AUROC Below 0.7
**Causes:**
- Insufficient training data
- Poor data quality / mislabeled examples
- Threshold too high/low
- Circuit doesn't capture hallucination signals

**Solutions:**
- Collect more high-quality data (50+ examples per class minimum)
- Review and verify labels
- Try different thresholds on validation set
- Try different circuit (rediscover or use different task)

### Problem: All Probes Predict High Probability
**Cause:** Class imbalance or training instability

**Solution:**
- Balance training data (equal hallucination/factual)
- Reduce learning rate (1e-4 instead of 1e-3)
- Use early stopping with higher patience

### Problem: Out of Memory During Training
**Cause:** Batch size too large or hidden dimension too large

**Solutions:**
- Reduce batch_size (16 → 8)
- Use gradient accumulation
- Extract activations offline before training

## API Reference

### LinearProbe

```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0)
    def forward(x: Tensor) -> Tensor:  # [..., input_dim] → [..., 1]
    def get_logits(x: Tensor) -> Tensor:  # Raw logits before sigmoid
```

### ProbeTrainer

```python
class ProbeTrainer:
    def __init__(probe, device="cuda", learning_rate=1e-3, weight_decay=1e-4)
    def train_epoch(train_loader, val_loader) -> (train_loss, train_auroc, val_loss, val_auroc)
    def train(train_loader, val_loader, epochs=10, patience=5, verbose=True) -> history_dict
    def get_probe() -> LinearProbe
    def get_metrics() -> {"best_val_auroc": float}
```

### HallucinationDetector

```python
class HallucinationDetector:
    def __init__(model, circuit, arch_cfg, device="cuda")
    
    # Training
    def train_probes(
        train_data,        # List[Dict[str, Any]]
        val_data,          # List[Dict[str, Any]]
        batch_size=32,
        epochs=10,
        learning_rate=1e-3,
        patience=3
    ) -> Dict[str, Any]
    
    # Detection
    def detect_hallucinations(
        text: str,
        generate_fn: Optional[Callable] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]
    
    # Analysis
    def get_activation_profile(text: str) -> Dict[int, Tensor]
    def get_model_probes() -> Dict[int, LinearProbe]
    def get_probe_stats() -> Dict[str, Any]
    
    # Persistence
    def save_probes(path: str) -> None
    def load_probes(path: str) -> None
```

## Complete Example

```python
from circuitkit.applications.common_utils.hallucination_detection import HallucinationDetector
from circuitkit.artifacts import CircuitArtifact

# 1. Load circuit
circuit = CircuitArtifact.load_json("circuits/gpt2_ioi.json")

# 2. Create detector
detector = HallucinationDetector(model, circuit, arch_cfg, device="cuda")

# 3. Prepare training data
train_data = [
    {"text": "Paris is the capital of France", "is_hallucination": False},
    {"text": "Paris is the capital of Germany", "is_hallucination": True},
    # ... more examples ...
]

val_data = [
    {"text": "2 + 2 equals 4", "is_hallucination": False},
    {"text": "2 + 2 equals 5", "is_hallucination": True},
]

# 4. Train probes
results = detector.train_probes(
    train_data=train_data,
    val_data=val_data,
    batch_size=32,
    epochs=10,
    patience=3
)
print(f"Trained {results['num_probes']} probes, best AUROC: {results['best_val_auroc']:.4f}")

# 5. Save probes
detector.save_probes("models/hallucination_probes.pt")

# 6. Detect hallucinations
result = detector.detect_hallucinations("The capital of France is London")
print(f"Hallucination probability: {result['hallucination_prob']:.2%}")
print(f"Explanation: {result['explanation']}")

# 7. Load probes later
detector2 = HallucinationDetector(model, circuit, arch_cfg)
detector2.load_probes("models/hallucination_probes.pt")
```

## See Also

- `CircuitArtifact` - Unified circuit representation
- `LinearProbe` - Individual probe model
- `ProbeTrainer` - Training infrastructure
- Examples in `examples/validation/applications/15_hallucination_gpt2.py` and `examples/validation/applications/27_hallucination_truthfulqa_llama.py`
