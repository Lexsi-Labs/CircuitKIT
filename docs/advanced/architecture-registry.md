# Architecture Registry

CircuitKit's **architecture registry** provides a unified interface for pruning and quantization across different Transformer families. Discovery always runs through TransformerLens (which abstracts the architecture), but the application layer (pruning, quantization) needs to know the specific module paths and projection names for each family.

---

## Supported Architectures

| Family | Status | Examples |
|--------|--------|---------|
| `llama` | Production | `meta-llama/Llama-3.2-1B`, `Llama-3.2-3B` |
| `qwen` | Production | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen3-7B` |
| `gemma` | Production | `google/gemma-2-2b-it` |
| `gemma3` | Production | `google/gemma-3-4b-it` |
| `mistral` | Ready | `mistralai/Mistral-7B-v0.1` |
| `phi` | Ready | `microsoft/Phi-3-mini-4k-instruct` |
| `falcon` | Ready | `tiiuae/falcon-7b` |
| `gpt2` | Ready | `gpt2`, `gpt2-xl` |

**Production** = validated in the CircuitKit paper audit. **Ready** = registry entry exists, high confidence, not in the audit.

---

## Using the Registry

### Auto-Detection

```python
from circuitkit.applications import detect_model_architecture

from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
arch_type = detect_model_architecture(hf_model)  # "qwen"
```

### Getting Architecture Config

```python
from circuitkit.applications import get_arch_config, get_layers, get_attn_proj, get_mlp_proj

arch_cfg = get_arch_config("gemma")

# Access layers
layers = get_layers(hf_model, arch_cfg)

# Access specific projections
for layer in layers:
    q_proj = get_attn_proj(layer, arch_cfg, "q_proj")
    gate_proj = get_mlp_proj(layer, arch_cfg, "gate_proj")
```

### Registry Queries

```python
from circuitkit.applications import (
    MODEL_ARCH_REGISTRY,
    SUPPORTED_FAMILIES,
    PRODUCTION_FAMILIES,
    READY_FAMILIES,
    get_model_family,
    get_head_dim,
)

print(SUPPORTED_FAMILIES)    # All registered families
print(PRODUCTION_FAMILIES)   # Production-validated families
print(READY_FAMILIES)        # Ready-to-use families

family = get_model_family("qwen2")  # "qwen" — maps an HF model_type, not a repo path
head_dim = get_head_dim(layer, arch_cfg)  # first arg is a single decoder layer, not the whole model
```

---

## Registry Entry Format

Each entry in `MODEL_ARCH_REGISTRY` follows this structure:

```python
{
    "name": "LLaMA / Llama-2 / Llama-3 / Llama-3.1 / CodeLlama",
    "models": ["llama", "llama2", "llama3", "codellama"],  # HF model_type values
    "layers_path": ["model.layers"],

    "attn": {
        "module": "self_attn",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "q_proj": "q_proj",
        "o_proj": "o_proj",
        "head_dim": "head_dim",
    },

    "mlp": {
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    },

    "gqa_capable": True,   # Group Query Attention — note: "gqa_capable", not "gqa"
    "transformer_lens_support": "full",
    "status": "PRODUCTION",
    "priority": 1,
    "notes": "Meta's flagship model, well-tested",
}
```

There is no `"norm"` key anywhere in the registry, and the MLP block has no `"module"` key (only `"attn"` does). Both were fabricated in earlier drafts of this doc.

---

## Adding a New Architecture

To add support for a model family not in the registry:

```python
from circuitkit.applications.arch_registry import MODEL_ARCH_REGISTRY

MODEL_ARCH_REGISTRY["mymodel"] = {
    "name": "MyModel Family",
    "models": ["mymodel-7b", "mymodel-13b"],  # HF model_type values

    "layers_path": ["model.layers"],

    "attn": {
        "module": "attention",           # your model's attention module name
        "k_proj": "key_projection",      # adjust to your model's names
        "v_proj": "value_projection",
        "q_proj": "query_projection",
        "o_proj": "output_projection",
        "head_dim": "head_size",
    },

    "mlp": {
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
    },

    "gqa_capable": False,
    "transformer_lens_support": "none",
    "status": "NOT_STARTED",
    "priority": 3,
    "notes": "",
}
```

Then test that the registry can find the model family:

```python
from transformers import AutoModelForCausalLM
from circuitkit.applications import detect_model_architecture, get_arch_config

hf_model = AutoModelForCausalLM.from_pretrained("my-model")
family = detect_model_architecture(hf_model)   # should return "mymodel"
cfg = get_arch_config("mymodel")
print(cfg)
```

---

## Error Handling

If an architecture is not registered, pruning and quantization raise:

```python
from circuitkit.applications import UnsupportedArchitectureError, ArchitectureValidationError
```

`UnsupportedArchitectureError` — the model family is not in the registry.
`ArchitectureValidationError` — the model is registered but the projection paths don't exist (e.g., your config is wrong).

---

## GQA Support

Models with Grouped Query Attention (GQA) — Llama-3, Gemma-2, Gemma-3, Mistral-large — require special handling in pruning because attention heads are grouped. The pruner detects GQA at runtime by comparing head counts (`n_kv_heads != n_heads`) and only zeros a KV head once every query head in its group has been pruned. The `"gqa_capable"` flag in each registry entry is descriptive metadata; the runtime head-count check is what actually drives the handling. Set `gqa_capable=True` in your entry for any model where `n_kv_heads < n_heads`.

---

## Next Steps

- [User Guide: Applications](../user-guide/applications.md) — pruning and quantization
- [API Reference: Applications](../api-reference/applications.md) — `StructuralPruner` and `circuit_quantize`
