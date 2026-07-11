# Tasks API

**Module**: `circuitkit.tasks`

---

## Built-in Tasks

16 built-in tasks are registered on first use:

| Task | Type | Metric | Default template |
|------|------|--------|-----------------|
| `ioi` | Diagnostic | logit-diff | `"off"` |
| `sva` | Diagnostic | logit-diff | `"off"` |
| `gender_bias` | Diagnostic | logit-diff | `"off"` |
| `capital_country` | Diagnostic | logit-diff | `"off"` |
| `hypernymy` | Diagnostic | logit-diff | `"off"` |
| `greater_than` | Diagnostic | logit-diff | `"off"` |
| `double_io` | Diagnostic | logit-diff | `"off"` |
| `winogrande` | Cloze | suffix log-likelihood | `"off"` |
| `boolq` | MCQ | logit-diff | `"auto"` |
| `glue` | MCQ | logit-diff | `"auto"` |
| `mmlu` | MCQ | logit-diff | `"auto"` |
| `winogrande_mc` | MCQ | logit-diff | `"auto"` |
| `truthfulqa` | MCQ | logit-diff | `"auto"` |
| `ifeval` | Instruction | — (collateral-eval only, no discovery) | `"auto"` |
| `wmdp` | MCQ | logit-diff | `"auto"` |
| `gsm8k` | Generation | NLL on answer span | `"auto"` |

---

## Functions

### `list_tasks`

```python
list_tasks() -> List[str]
```

Return all registered task names.

```python
import circuitkit
print(circuitkit.list_tasks())
# sorted list: ['boolq', 'capital_country', 'double_io', ...]
```

### `get_task`

```python
get_task(name: str) -> TaskSpec
```

Return the task spec for a registered task name. Raises `ValueError` if not found.

```python
import circuitkit
spec = circuitkit.get_task("ioi")
print(spec.name)
print(spec.chat_template_mode)
```

### `register_task`

```python
register_task(spec: TaskSpec) -> None
```

Register a custom task. After registration, the task name is accepted by `discover_circuit`, `ck.discover`, and the CLI.

```python
import circuitkit
from circuitkit.data.normalized_task import NormalizedTaskSpec

circuitkit.register_task(NormalizedTaskSpec(my_dataset, name="my_task"))
```

---

## Custom Task Specs

### `NormalizedTaskSpec`

The standard task spec for custom datasets. Wraps a dataset with paired (clean, corrupt) records.

```python
from circuitkit.data.normalized_task import NormalizedTaskSpec, validate_token_alignment

spec = NormalizedTaskSpec(dataset, name="my_task")
report = validate_token_alignment(spec, model=tl_model)
```

### `GenericTaskSpec`

A lower-level spec for tasks that need custom metric functions or non-standard corruption strategies.

---

## Registering from a YAML File

```yaml
# task.yaml
name: my_task
source:
  type: csv
  path: ./data.csv
schema:
  prompt: prompt
  answer: answer
  corrupted_prompt: corrupt_prompt
corruption:
  strategy: entity_swap
metric: logit_diff
chat_template_mode: auto
```

```bash
circuitkit discover-yaml --task-yaml ./task.yaml --algorithm eap-ig --model gpt2
```

---

## Registering from a HuggingFace Dataset

```python
from datasets import load_dataset
from circuitkit.data.adapters.mcq import MCQAdapter
from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.tasks.registry import register_task

raw = list(load_dataset("cais/mmlu", "high_school_world_history",
                        split="test", streaming=True).take(24))
ds = MCQAdapter().adapt(raw, name="mmlu_hist", max_records=20)
ds.records = [MCQChoiceSwap().apply(r) for r in ds.records]
ds.records = [r for r in ds.records if r.is_paired]
register_task(NormalizedTaskSpec(ds, name="mmlu_hist"))

# Now use it:
import circuitkit as ck
circuit = ck.discover(model, "mmlu_hist", n_examples=20)
```

---

## Token Alignment Audit

```python
from circuitkit.data.normalized_task import validate_token_alignment

report = validate_token_alignment(task_spec, model=tl_model)

print(report["total"])                    # total records
print(report["same_prompt_frac"])         # fraction with identical clean/corrupt
print(report["multi_token_answer_frac"])  # fraction where answer is multi-token
print(report["dropped_frac"])             # fraction that will be silently dropped
print(report["records_ok"])               # count of records surviving the drop filter
```

---

## WMDP Task Factory

```python
from circuitkit.tasks.builtins.wmdp import build_wmdp_spec

spec = build_wmdp_spec(
    subset="wmdp-bio",   # "wmdp-bio", "wmdp-chem", "wmdp-cyber"
    split="test",
    max_records=200,
)
register_task(spec)
```

---

## Next Steps

- [Built-in tasks](../user-guide/tasks.md) — all 16 tasks with metric details
- [Bring your own data](../user-guide/custom-data.md) — full custom data guide
- [Data model](../user-guide/data.md) — what discovery needs
- [Chat Templates](../advanced/chat-templates.md) — `chat_template_mode` reference
