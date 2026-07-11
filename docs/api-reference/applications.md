# Applications API

**Module**: `circuitkit.applications`

The applications module contains all post-discovery interventions: structural pruning, circuit-aware quantization, activation steering, knowledge editing, and selective fine-tuning.

---

## Architecture Registry

See the [Architecture Registry reference](../advanced/architecture-registry.md) for supported model families.

```python
from circuitkit.applications import (
    MODEL_ARCH_REGISTRY,
    SUPPORTED_FAMILIES,
    PRODUCTION_FAMILIES,
    READY_FAMILIES,
    get_model_family,
    detect_model_architecture,
    get_arch_config,
    get_layers, get_attn_proj, get_mlp_proj, get_head_dim,
    UnsupportedArchitectureError,
    ArchitectureValidationError,
)
```

---

## Pruning

**Module**: `circuitkit.applications.pruning`

### `StructuralPruner`

Structured masking of attention heads and MLP layers: it zeroes their weights in place without resizing tensors. Physical parameter removal and smaller checkpoints happen separately at export time via `save_pruned_checkpoint`.

```python
from circuitkit.applications.pruning import StructuralPruner

pruner = StructuralPruner()
pruned_model = pruner.prune(model, circuit.scores, sparsity=0.3, scope="heads")
```

### `NodePruner`

Lower-level pruner operating at individual node granularity.

```python
from circuitkit.applications.pruning import NodePruner, get_nodes_to_prune

nodes_to_prune = get_nodes_to_prune(circuit.scores, target_sparsity=0.3, pruning_scope="heads")
pruner = NodePruner()
nodes_to_prune = pruner.prune(circuit.scores, target_sparsity=0.3, scope="heads")
```

### Other Pruning Utilities

```python
from circuitkit.applications.pruning import (
    zero_attention_head_weights,     # zero a specific head's weight matrices
    get_attention_architecture_info, # inspect attention shape for a model
)
```

---

## Quantization

**Module**: `circuitkit.applications.quantization`

Circuit-aware mixed-precision quantization: circuit nodes receive higher precision, out-of-circuit components are quantized more aggressively.

```python
import circuitkit as ck

quantized = ck.quantize(model, circuit, protect_layers=[0, -1])
```

Quantization selectors (`awq`, `tacq`, `gptq`) are accessed through the quantization sub-package. `wanda` is a pruning selector, not a quantization one. See [Selectors API](selectors.md) for details.

---

## Activation Steering

**Module**: `circuitkit.applications.steering`

Three distinct steering methods, not interchangeable:

### `ActivationSteering`

Per-head, per-position activation steering via runtime hooks. Weights are untouched and the effect is fully reversible.

```python
from circuitkit.applications.steering import ActivationSteering

steerer = ActivationSteering(model, circuit_scores=circuit.scores)
result = steerer.steer(
    "The quick brown fox",
    coefficient=1.5,
)
```

This is the standard literature baseline.

### `CircuitWeightSteering`

Contrastive weight steering (C-ΔΘ): permanently edits per-head W_Q/K/V/O slices via `θ_pos − θ_neg`. The paper-faithful method. Not reversible without reloading.

```python
from circuitkit.applications.steering import CircuitWeightSteering

steerer = CircuitWeightSteering(model, circuit.scores, top_k_frac=0.01)
steerer.fine_tune_positive(pos_dataloader, loss_fn)
steerer.fine_tune_negative(neg_dataloader, loss_fn)
steerer.compute_steering_vector()
steered_model = steerer.apply_steering(k=2.0)
```

### `SteeringComposer`

Compose multiple steering operations or synthesize safety datasets.

```python
from circuitkit.applications.steering import SteeringComposer, SafetyDatasetSynthesis

composer = SteeringComposer()
composer.add_steering("safety", safety_vectors, coefficient=0.7)
composer.add_steering("style", style_vectors, coefficient=0.3)
composed = composer.get_composed_vectors(aggregate="sum")
```

---

## Knowledge Editing

**Module**: `circuitkit.applications.editing`

Circuit-guided knowledge editing using ROME / MEMIT under the hood.

```python
from circuitkit.applications.editing import CircuitKnowledgeEditor, EditResult

editor = CircuitKnowledgeEditor(model)
result: EditResult = editor.edit_via_circuit(
    prompt="The Eiffel Tower is located in",
    subject="Eiffel Tower",
    target="Berlin",
    circuit=circuit,
    method="rome",
)
print(result.success)
print(result.confidence_after)
```

### Editing Classes

| Class | Description |
|-------|-------------|
| `CircuitKnowledgeEditor` | Main editor — uses circuit to select MLP targets |
| `BatchKnowledgeEditor` | Edit multiple facts in one pass |
| `CircuitGuidedEditor` | Lower-level circuit-guided editing |
| `MCircKEEditor` | Multi-hop circuit knowledge editing |
| `CaKEEditor` | Circuit-aware knowledge editing |
| `RomeHandler` / `RomeWrapper` | ROME backend wrappers |
| `MemitHandler` | MEMIT backend wrapper |
| `UnlearningReport` | Result class for unlearning/removal |
| `UnlearningVerifier` | Verify that facts were successfully removed |

---

## Selective Fine-tuning

**Module**: `circuitkit.applications.finetuning`

!!! warning "Access path"
    `selective_finetune` is **NOT** in `applications.__all__`. Access it via `ck.selective_finetune()` or `Pipeline.selective_finetune()` only.

```python
import circuitkit as ck

result = ck.selective_finetune(circuit, top_fraction=0.2)
```

### Finetuning Classes

| Class | Description |
|-------|-------------|
| `CircuitTuner` | LoRA fine-tuning restricted to circuit-identified MLP layers |
| `CircuitLoRA` | LoRA adapter targeting circuit nodes |
| `LoRALayer` | Individual LoRA layer |
| `CircuitPEFT` | PEFT-compatible circuit tuner |
| `PEFTComposer` | Compose multiple PEFT adapters |
| `HealingMetrics` | Metrics for post-intervention recovery |
| `HealingEvaluator` | Evaluate recovery after editing/pruning |
| `compute_recovery_metrics` | Compute recovery score |

---

## PEFT Benchmarking

**Module**: `circuitkit.applications.finetuning.benchmark_peft`

```python
from circuitkit.applications.finetuning.benchmark_peft import (
    PEFTBenchmark,
    CrossArchitectureBenchmark,
    BenchmarkMetrics,
)

bench = PEFTBenchmark(model, method="lora", rank=8, device="cuda")
metrics: BenchmarkMetrics = bench.run(num_batches=10, batch_size=4)
print(metrics.param_efficiency)
print(metrics.batches_per_second)

cross = CrossArchitectureBenchmark(models_dict={"gpt2": gpt2_model, "qwen": qwen_model}, device="cuda")
cross.run_all(num_batches=5)
report = cross.generate_report()
```

---

## Hallucination Detection

**Module**: `circuitkit.applications.common_utils.hallucination_detection`

See [Hallucination Detection](../guides/hallucination_detection_guide.md) for the full guide.

```python
from circuitkit.applications.common_utils.hallucination_detection import HallucinationDetector

detector = HallucinationDetector(model, circuit, arch_cfg, device="cuda")
detector.train_probes(train_data, val_data)
result = detector.detect_hallucinations("The capital of France is London")
print(result["hallucination_prob"])
```

---

## Next Steps

- [User Guide: Applications](../user-guide/applications.md) — workflow guide for each application
- [Architecture Registry](../advanced/architecture-registry.md) — supported model families
- [Hallucination Detection](../guides/hallucination_detection_guide.md) — probe training and detection
