# Chat Templates

CircuitKit runs discovery on pairs of (clean, corrupt) prompts. Whether those prompts are wrapped in a model's chat template affects what activations the circuit sees, and therefore what circuit gets discovered. This page explains the chat template policy, when to change it, and the underlying helpers.

---

## The Three Modes

| Mode | Behavior |
|------|----------|
| `"auto"` | Apply template if and only if the model is a chat model (tokenizer has a `chat_template`) |
| `"on"` | Always apply the chat template, regardless of model type |
| `"off"` | Always use raw text — no template applied |

---

## Default Per Task Type

The task type sets the default mode; you can override it per run.

| Task type | Default mode | Examples |
|-----------|-------------|---------|
| Downstream-behavior tasks | `"auto"` | `boolq`, `glue`, `mmlu`, `truthfulqa`, `ifeval`, `wmdp`, `gsm8k`, `winogrande_mc` |
| Diagnostic minimal-pair tasks | `"off"` | `ioi`, `sva`, `greater_than`, `gender_bias`, `capital_country`, `hypernymy`, `double_io` |
| Cloze tasks | `"off"` | `winogrande` |
| Custom `GenericTaskSpec` / `NormalizedTaskSpec` | `"auto"` | User-defined tasks |

The rationale: diagnostic tasks like IOI are designed around raw token completions, where adding a system prompt or `[INST]` wrapper would change the task semantics. Downstream-behavior tasks like MMLU are typically evaluated on chat models where the template matters.

---

## Overriding the Mode

### Python API

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "meta-llama/Llama-3.2-1B-Instruct"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "capital_country",
        "chat_template_mode": "on",   # force template even on diagnostic task
        "level": "node",
        "data_params": {"num_examples": 64},
    },
    "output_path": "./circuit.pt",
})
```

### CLI

```bash
circuitkit discover \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --task capital_country \
    --chat-template-mode on \
    --output ./circuit.pt
```

### YAML task definition

```yaml
task: capital_country
chat_template_mode: on
```

---

## Helper Functions

The `circuitkit.tasks._chat` module provides the policy implementation:

### `resolve_chat_template(mode, model) -> bool`

Collapses a declared mode string into a single boolean for a given model.

```python
from circuitkit.tasks._chat import resolve_chat_template

apply = resolve_chat_template("auto", tl_model)   # True if chat model, else False
apply = resolve_chat_template("on", tl_model)     # always True
apply = resolve_chat_template("off", tl_model)    # always False
```

Raises `ValueError` if `mode` is not one of `"auto"`, `"on"`, `"off"`.

### `model_is_chat(model) -> bool`

Returns `True` if the model's tokenizer has a `chat_template` attribute.

```python
from circuitkit.tasks._chat import model_is_chat

if model_is_chat(tl_model):
    print("Chat model — 'auto' will apply the template")
```

### `wrap_prompt(model, user_text, assistant_prefix="", *, apply) -> str`

Formats one task prompt, wrapping it in the chat template when `apply=True`.

```python
from circuitkit.tasks._chat import wrap_prompt

raw = "The capital of France is"
wrapped = wrap_prompt(tl_model, raw, apply=True)
# "[INST] The capital of France is [/INST]"
```

### `to_tokens(model, text, *, templated) -> torch.Tensor`

Tokenizes text with BOS handling adjusted for whether the text is already templated. Templated text already carries its own BOS token, so `prepend_bos=False` is used automatically.

```python
from circuitkit.tasks._chat import to_tokens

tokens = to_tokens(tl_model, wrapped, templated=True)
```

---

## Common Pitfalls

**Discovering on a chat model with `"off"`**: The circuit may look correct but will capture raw-text activations that don't reflect how the model is used in practice. For chat models running downstream tasks, `"auto"` is almost always the right choice.

**Discovering with `"on"` and evaluating with a base model**: If you discover a circuit on an instruct model with `chat_template_mode="on"` and then try to evaluate it on a base model without a chat template, evaluation will error since the base model's tokenizer has no `chat_template` to resolve against. `chat_template_mode` is **not** persisted into the artifact — it is resolved fresh on every call from the mode you pass and the model in hand, so this mismatch only surfaces at evaluation time, not via stored metadata. Use the same model type for discovery and evaluation.

**Different template versions**: Some models (Llama-3.x, Qwen-3) have multiple template variants. CircuitKit uses the tokenizer's default `chat_template`. If your inference stack uses a custom template, set `chat_template_mode="off"` and pre-format prompts yourself.

---

## Diagnosing Template Issues

There is no `chat_template_mode` field stored in the `.pt` artifact and no `UserWarning` for a mode mismatch — `discover_circuit` never writes `chat_template_mode` into artifact metadata. If discovery and evaluation appear to disagree on chat formatting, check what each call actually resolved to:

```python
from circuitkit.tasks._chat import model_is_chat
import transformer_lens as tl
model = tl.HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print("Model is chat:", model_is_chat(model))
```

---

## Next Steps

- [User Guide: Tasks](../user-guide/tasks.md) — `chat_template_mode` defaults per task
- [User Guide: Troubleshooting](../user-guide/troubleshooting.md) — template mismatch errors
- [Algorithms: EAP](../algorithms/eap.md) — instruction-tuned model guidance
