# Bring your own data

You don't need to write Python to run discovery on your own dataset. Point a task YAML at a CSV, JSONL, or HuggingFace dataset, map the columns, and run. This page covers that path, plus the inline `data` config and the Python `register_task` route for when you need more control.

First, decide what kind of data you have. If you're running an EAP-family algorithm, you need a contrastive pair — either supply the corrupt half in explicit columns, or configure a [corruption strategy](data-corruption.md) to generate it. If you're running IBCircuit or CD-T, the clean prompt and answer are enough. See [the data model](data.md) for the full picture.

!!! warning "Prefer explicit contrastive columns for anything that isn't a syntactic template"
    The two ways to get the corrupt half are **not** equally robust:

    - **Explicit `corrupted_prompt` / `corrupted_answer` columns (recommended for arbitrary data).** You author the counterfactual. The loader checks the corrupt prompt actually differs from the clean one; the inline `data` / normalized path (below) additionally validates token-length alignment and a discriminative answer token, and drops degenerate pairs. Use this for instruction-tuned, factual, or safety-style data.
    - **A `corruption:` strategy (auto-generated).** The five strategies were built for **syntactic templates** — IOI-style prompts with named entities and subject–verb–object structure. On prompts without that structure a strategy may find nothing to change and quietly produce a `clean == corrupt` pair, and length-changing edits (a distractor sentence, a multi-token entity swap) can misalign the pair position-for-position. `paraphrase` and `distractor` deliberately keep the answer correct, so they are **not** counterfactual for discovery. Treat auto-corruption as valid for syntactic tasks; for anything else, supply explicit pairs.

## Task YAML

A task YAML has three required top-level keys — `name`, `source`, and `schema` — and a few optional ones. The `schema` block maps CircuitKit's field names (the keys) onto your dataset's column names (the values).

### A complete, working example

```yaml
# my_task.yaml
name: capitals                 # required: registered task name

source:
  type: csv                    # required: "csv", "jsonl", or "hf"
  path: ./data/capitals.csv    # required for csv/jsonl (use dataset_id for hf)

schema:
  prompt: question             # required: column holding the clean prompt
  answer: answer               # required: column holding the clean answer
  corrupted_prompt: counterfactual   # the corrupt prompt (paired algorithms)
  corrupted_answer: wrong_answer      # the corrupt answer (paired algorithms)

corruption:
  strategy: entity_swap        # optional: used only if no corrupted column exists

metric: logit_diff             # optional: logit_diff (default) | kl | accuracy

chat_template_mode: auto       # optional: auto (default) | on | off
```

Run it:

```bash
circuitkit discover-yaml --model gpt2 \
    --task-yaml my_task.yaml \
    --algorithm eap-ig \
    --level node \
    --sparsity 0.3 \
    --output ./circuit.pt
```

### Schema keys

The required keys are `prompt` and `answer`. Everything else is optional and depends on your algorithm and data.

| Schema key | Purpose | Required |
|---|---|---|
| `prompt` | Clean prompt column | Yes |
| `answer` | Clean answer column | Yes |
| `corrupted` **or** `corrupted_prompt` | Explicit corrupt-prompt column | For paired discovery, if no strategy |
| `corrupted_answer` **or** `corrupt_answer` | Explicit corrupt-answer column | Optional, pairs with the corrupt prompt |
| `context` | Passage/background text, prepended to the prompt | Optional |
| `choices` | Multiple-choice options column | Optional |

!!! warning "Get the corrupt-column key right"
    The schema key for the corrupt prompt is `corrupted` or `corrupted_prompt` — **not** `corrupt_prompt`. The corrupt-answer key is `corrupted_answer` or `corrupt_answer`. If you write `corrupt_prompt` as a schema key it is silently ignored, and discovery falls back to the corruption strategy (or, with no strategy, to clean == corrupt with no contrastive signal). The *column name* you map onto these keys can be anything; only the key on the left has to match.

CircuitKit also picks up literal `corrupted_prompt` and `corrupted_answer` columns even when the schema doesn't declare them, so data that already uses those column names works without extra mapping.

Required fields and either/or keys are enforced at load time — omit `name`, `source`, `schema`, or the `prompt`/`answer` schema keys and the loader raises a `ValueError`.

## Source formats

### CSV

```csv
question,answer,counterfactual,wrong_answer
"The capital of France is"," Paris","The capital of Italy is"," Rome"
```

### JSONL

One JSON object per line:

```jsonl
{"question": "The capital of France is", "answer": " Paris", "counterfactual": "The capital of Italy is", "wrong_answer": " Rome"}
```

### HuggingFace

Use `dataset_id` (not `path`), and optionally a `split` — it defaults to `test`:

```yaml
source:
  type: hf
  dataset_id: google/boolq    # HF dataset identifier
  split: validation           # optional; defaults to "test"
```

## Inline `data` config

If you drive discovery from a config dict or YAML instead of the `discover-yaml` command, you can embed a `data` block. This path uses a different set of keys — `data.type` is `template`, `auto`, or `clean_only` — and it builds a normalized contrastive dataset directly. `data.path` is required for every type. The built task auto-registers as `custom:<file-stem>` (e.g. `custom:capitals` from `capitals.csv`); any `discovery.task` you set on this path is overwritten, so you don't need to supply one.

```yaml
model:
  name: gpt2

discovery:
  algorithm: eap-ig

data:
  type: template               # "template" | "auto" | "clean_only"
  path: ./data/capitals.csv
  template:
    clean_prompt: question
    clean_answer: answer
    corrupt_prompt: counterfactual
    corrupt_answer: wrong_answer

pruning:
  target_sparsity: 0.3
  scope: both
```

- **`template`** pairs each clean record with a corrupt one using the four `clean_prompt` / `corrupt_prompt` / `clean_answer` / `corrupt_answer` keys, then runs a token-alignment pass so the pair stays position-matched. For a clean-only algorithm (IBCircuit, CD-T) you may supply just `clean_prompt` (and `clean_answer`) and skip the corrupt keys.
- **`auto`** infers the mapping from column names and applies a default corruption strategy.
- **`clean_only`** loads clean prompts and answers with no corrupt partner — compatible with IBCircuit and CD-T only. Set `prompt_column` and `answer_column` if your columns aren't named `prompt`/`answer`.

Note the key mismatch between the two paths: the task-YAML `schema` uses `corrupted_prompt`, while the inline `data.template` block uses `corrupt_prompt`. They are separate loaders. Use whichever entry point fits your workflow, but don't copy keys between them.

## Validate before you run

Check a dataset without spending a discovery run on it:

```bash
# Validate and summarize a CSV (prints a verdict and per-check report)
circuitkit data check data.csv

# Write the report as JSON
circuitkit data check data.csv --output ./report.json

# Build a normalized (clean, corrupt) dataset and save it
circuitkit data prepare data.csv --output ./normalized.json
```

## Via Python: `register_task`

For full control, build and register a spec directly. `register_task` and `NormalizedTaskSpec` are imported from the package root and `circuitkit.data.normalized_task`. `NormalizedTaskSpec` wraps a `NormalizedDataset` of `ContrastiveRecord`s — it takes the dataset positionally and an optional `name`, not `metric=` / `templates=` / `data=` keywords. The metric is selected later via `.metric_fn(metric_type=...)`.

```python
from circuitkit import register_task, discover_circuit
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape

records = [
    ContrastiveRecord(
        record_id="0",
        clean_prompt="John gave a drink to",
        clean_answer=" Mary",
        corrupt_prompt="Susan gave a drink to",
        corrupt_answer=" Bob",
    ),
]
ds = NormalizedDataset(name="my_task", shape=DatasetShape.QA, records=records)

spec = NormalizedTaskSpec(ds, name="my_task")
register_task(spec)

circuit = discover_circuit({
    "model": {"name": "gpt2"},
    "discovery": {"algorithm": "eap-ig", "task": "my_task"},
    "pruning": {"target_sparsity": 0.3, "scope": "both"},
})
```

`NormalizedTaskSpec` takes the dataset (positional), an optional `name` (defaults to `f"normalized:{ds.name}"`), and an optional `cache_dir` (defaults to `"./cache/normalized"`). That's it — the prompts come from the wrapped records' `clean_prompt` / `clean_answer` / `corrupt_prompt` / `corrupt_answer` fields.

## Next steps

- [Corruption strategies](data-corruption.md) — generating the corrupt half when you don't supply it
- [YAML configuration](../cli/yaml-config.md) — full task-YAML and discovery-config reference
- [Data model](data.md) — what each algorithm needs
