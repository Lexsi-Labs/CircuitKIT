# Applications

After discovering and evaluating a circuit, CircuitKit provides five ways to act on it. Pruning, quantization, and selective fine-tuning are available through the flat `ck.*` API and `Pipeline`; steering and knowledge editing are used via their `circuitkit.applications.*` classes. Pruning, quantization, and steering also have CLI commands.

<div class="grid cards" markdown>

-   :material-content-cut:{ .lg .middle } **Structural Pruning**

    ---

    Remove lowest-scoring components. Writes a real HuggingFace checkpoint.

    [:octicons-arrow-right-24: Details below](#1-structural-pruning)

-   :material-tune:{ .lg .middle } **Circuit-Aware Quantization**

    ---

    Mixed-precision, protecting high-importance circuit components.

    [:octicons-arrow-right-24: Details below](#2-circuit-aware-quantization)

-   :material-pencil:{ .lg .middle } **Selective Fine-tuning**

    ---

    Circuit-restricted LoRA on only the components that matter.

    [:octicons-arrow-right-24: Details below](#3-selective-fine-tuning-circuit-restricted-lora)

-   :material-steering:{ .lg .middle } **Activation Steering**

    ---

    Modify activations at runtime. No weights changed.

    [:octicons-arrow-right-24: Details below](#4-activation-steering)

-   :material-book-edit:{ .lg .middle } **Knowledge Editing**

    ---

    ROME/MEMIT at circuit-identified MLP layers.

    [:octicons-arrow-right-24: Details below](#5-circuit-guided-knowledge-editing)

</div>

---

## 1. Structural Pruning

Remove the lowest-scoring components and write a reloadable HuggingFace checkpoint.

```python
import circuitkit as ck

model = ck.load_model("gpt2", dtype="float32")
circuit = ck.load_scores("./circuit.pt")

pruned = ck.prune(model, circuit, sparsity=0.3, scope="both")
ck.export_checkpoint(pruned, circuit, "./output/pruned_checkpoint")
```

| Parameter | Default | Description |
|---|---|---|
| `sparsity` | `0.3` | Fraction of components to remove (0.0–1.0) |
| `scope` | `"both"` | `"heads"`, `"mlp"`, or `"both"` |
| `protect_layers` | `None` | Layer indices never to prune |
| `inplace` | `False` | Modify model in place |

**What pruning does:** Zero-masks weight matrices of lowest-scoring components. The model retains its architecture — no layer deletion. `export_checkpoint` writes a HuggingFace-format checkpoint reloadable with `transformers.AutoModelForCausalLM.from_pretrained`.

Via Pipeline:
```python
pipe.prune(sparsity=0.3, scope="both")
pipe.export("./output/checkpoint")
```

Via CLI:
```bash
circuitkit prune --model gpt2 --artifact ./circuit.pt --sparsity 0.3 --output ./output/pruned
```

---

## 2. Circuit-Aware Quantization

Apply mixed-precision quantization, protecting high-importance circuit components.

```python
import circuitkit as ck
from transformers import AutoModelForCausalLM

hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
circuit = ck.load_scores("./circuit.pt")

plan = ck.quantize(
    hf_model, circuit,
    bits=4,
    high_fraction=0.3,   # top 30% of layers at full precision
    backend="quanto",
)
```

Requires: `pip install -e ".[quantization]"`.

**How it works:** Ranks layers by circuit importance. The top `high_fraction` of layers by circuit importance stay at full precision; the rest are quantized. This protects circuit-critical computations while compressing everything else.

!!! note "`bits` only applies to the `llmcompressor` backend"
    The default `backend="quanto"` assigns integer precision tiers internally and **ignores the `bits` argument**. To pin an explicit weight bit-width, use `backend="llmcompressor"`, which honours `bits` ∈ `{3, 4, 8}` (GPTQ-calibrated, vLLM-compatible).

Via Pipeline:
```python
pipe.quantize(bits=4, high_fraction=0.3, backend="quanto")
pipe.export("./output/quantized", intervention="quantization")
```

---

## 3. Selective Fine-tuning (Circuit-Restricted LoRA)

Identify which components should receive LoRA adapters.

```python
import circuitkit as ck

circuit = ck.load_scores("./circuit.pt")

result = ck.selective_finetune(
    circuit,
    model_name="gpt2",
    top_fraction=0.2,
    scope="both",
)

print(result.attn)   # {"attn_0": {"q": [...], "k": [...], "v": [...], "o": [...]}, ...}
print(result.mlp)    # {"mlp_2": [col indices] or None, ...}
```

Use the result to configure a PEFT LoRA trainer:
```python
from peft import LoraConfig, get_peft_model

target_modules = []
for attn_key in result.attn:                 # attn_key is e.g. "attn_4"
    layer = int(attn_key.split("_")[1])
    target_modules.append(f"transformer.h.{layer}.attn.c_attn")

config = LoraConfig(r=8, lora_alpha=16,
                    target_modules=target_modules, lora_dropout=0.1)
peft_model = get_peft_model(model, config)
```

Via Pipeline:
```python
result = pipe.selective_finetune(top_fraction=0.2, scope="attn")
```

---

## 4. Activation Steering

Modify activations at runtime without changing weights.

```python
from circuitkit.applications.steering import ActivationSteering

circuit = ck.load_scores("./circuit.pt")
model = ck.load_model("gpt2")

steering = ActivationSteering(model, circuit_scores=circuit.scores)
output = steering.steer("When Mary and John went to the store,", ...)
```

Three steering methods:

| Class | Method | Reversible? |
|---|---|---|
| `ActivationSteering` | Add/subtract direction at circuit heads | Yes (hook-based) |
| `CircuitWeightSteering` | Contrastive weight steering (C-DTheta) | Yes (applies to copy) |
| `SteeringComposer` | Compose multiple steering vectors | Yes |

Via CLI:
```bash
circuitkit steer --model gpt2 --circuit-scores ./circuit_scores.json \
                 --source-examples ./source.csv --target-examples ./target.csv \
                 --coefficient 1.5
```

---

## 5. Circuit-Guided Knowledge Editing

Surgically rewrite facts at circuit-identified MLP layers.

```python
from circuitkit.applications.editing import CircuitKnowledgeEditor

editor = CircuitKnowledgeEditor(model)
editor.edit_via_circuit(
    prompt="The capital of France is",
    subject="France",
    target="Lyon",
    circuit=circuit,
    method="rome",       # "rome" or "memit"
)
```

**Methods:**

| Method | What it does |
|---|---|
| ROME | Single-layer update at the most circuit-important MLP layer |
| MEMIT | Multi-layer update across top-K circuit-identified layers |

!!! note "Context token auto-inference"
    The context token is auto-inferred from the subject string. Verify the inferred token matches the target.

## Next steps

- [:octicons-arrow-right-24: Pipeline Overview](pipeline-overview.md)
- [:octicons-arrow-right-24: Visualization](visualization.md)
- [:octicons-arrow-right-24: Selectors](selectors.md)
