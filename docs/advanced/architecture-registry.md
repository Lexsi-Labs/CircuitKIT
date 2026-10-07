# Architecture Registry

CircuitKIT's **architecture registry** provides a unified interface for pruning and quantization across different Transformer families. Discovery always runs through TransformerLens (which abstracts the architecture), but the application layer (pruning, quantization) needs to know the specific module paths and projection names for each family.

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
| `cohere` | Experimental | `CohereLabs/tiny-aya-base`, `CohereLabs/c4ai-command-r7b-12-2024` (`cohere2`); `CohereLabs/aya-expanse-8b`, `CohereLabs/aya-expanse-32b` (`cohere1`) |
| `smollm3` | Experimental | `HuggingFaceTB/SmolLM3-3B` |

**Production** = validated in the CircuitKIT paper audit. **Ready** = registry entry exists, high confidence, not in the audit. **Experimental** = discovery, evaluation, and interventions supported via the circuitkit TransformerLens port (`circuitkit.backends._tl_compat`), validated on real weights (see below for exactly what "validated" means for each surface — Aya Expanse 32B is a partial exception, noted below). The `cohere` family now covers four checkpoints across two HF architectures — Tiny Aya and Command R7B (`Cohere2ForCausalLM`, `model_type="cohere2"`) and Aya Expanse 8B / 32B (`CohereForCausalLM`, `model_type="cohere"`) — sharing one weight converter and one registry entry, since the pruning/quantization/steering module layout (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`) is identical across all four. `smollm3` is a separate, Llama-shaped family (new in this stage). See [Tiny Aya (Cohere2)](tiny-aya.md) and [Experimental Models](experimental-models.md).

**Aya Expanse 32B is only partially validated.** Registration, real-weight
config conversion (at a truncated depth), and — of the registry-entry
resolution this page documents — **pruning score extraction and
quantization pattern matching** are confirmed on the real 32B checkpoint;
see [Experimental Models](experimental-models.md#tests) for the actual
numbers. **Weight-steering slices** have not: the full checkpoint
downloaded and verified completely (it is not a download-time issue), but
exercising weight steering needs TransformerLens's own multi-GPU
(`n_devices`) support to fit the full 40-layer model in memory, and that
support is confirmed broken in transformer-lens==3.8.0 — see the warning in
[Experimental Models](experimental-models.md#tests) for the exact mechanism
and how it was reproduced. Everywhere below that says "all four real loaded
models" for the `cohere` family, that means Tiny Aya, Command R7B, Aya
Expanse 8B and SmolLM3-3B — Aya Expanse 32B's coverage is pruning and
quantization only, not weight steering.

### Two honest caveats (apply to `cohere` and `smollm3` alike)

- **Quantization is validated for target-module resolution only.**
  `applications/quantization/quant_utils.build_patterns`'s fnmatch patterns
  were confirmed to match real `nn.Linear` submodules by name on all four
  real loaded models (Tiny Aya, Command R7B, Aya Expanse, SmolLM3-3B) — but
  `optimum-quanto`/`llmcompressor`/`compressed-tensors` are optional
  dependencies **not installed** in the validation environment, so the
  actual `quantize()`/GPTQ compression call itself was never run for any of
  the four. This is not a full compression run; it proves the registry
  entry resolves the right layers, nothing more.
- **`CircuitWeightSteering`'s circuit-score format does not match
  `discover_circuit`'s node-name format**, for any model, not just these
  four. `weight_steering.py`'s own contract is `"A{layer}.{head}"`
  (uppercase `A`, no `.h`); `discover_circuit`'s `Graph` node names are
  `"a{layer}.h{head}"` (lowercase `a`, `.h` before the head index). A
  discovered circuit's `node_scores` dict cannot be passed to
  `CircuitWeightSteering` as-is — it needs a small regex conversion first.
  This is a pre-existing property of the steering module (the existing
  offline `tests/apply/test_weight_steering.py` already builds its scores in
  the module's own format on GPT-2), not something introduced by or specific
  to the `cohere`/`smollm3` families.

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
    EXPERIMENTAL_FAMILIES,
    get_model_family,
    get_head_dim,
)

print(SUPPORTED_FAMILIES)     # All registered families
print(PRODUCTION_FAMILIES)    # Production-validated families
print(READY_FAMILIES)         # Ready-to-use families
print(EXPERIMENTAL_FAMILIES)  # Experimental (e.g. cohere, smollm3 via the TL port)

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
