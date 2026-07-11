# YAML Configuration

CircuitKit's dict-config API and CLI both accept YAML files. This page covers the full YAML schema for discovery configs and task files.

---

## Discovery Config YAML

A full YAML config for `discover_circuit("config.yaml")` or `circuitkit discover-yaml`:

```yaml
# config.yaml — full discovery config
model:
  name: meta-llama/Llama-3.2-1B-Instruct   # required
  precision: bfloat16                        # optional, default: bfloat16

discovery:
  algorithm: eap-ig            # required (one of 13 discovery algorithms)
  task: mmlu                   # required (built-in or registered task name)
  level: node                  # "node" or "neuron", default: node
  chat_template_mode: auto     # "auto", "on", "off" — optional
  batch_size: 4
  ig_steps: 5                  # EAP-IG only
  data_params:
    num_examples: 128
    batch_size: 4

pruning:
  target_sparsity: 0.3         # fraction [0.0, 1.0]
  scope: heads                 # "heads", "mlp", "both"

output_path: ./results/circuit.pt   # optional; auto-named if omitted
```

---

## Task YAML Schema

A task YAML is used with `circuitkit discover-yaml --task-yaml task.yaml`.

### Minimal Task YAML

```yaml
name: my_task
source:
  type: csv
  path: ./data.csv
schema:
  prompt: prompt
  answer: answer
  corrupted_prompt: corrupt_prompt
```

### Full Task YAML

```yaml
# task.yaml

# --- Name (required) ---
name: my_task

# --- Source (required) ---
source:
  type: csv          # "csv", "jsonl", or "hf"
  path: ./data.csv   # for csv/jsonl

# For HuggingFace dataset:
# source:
#   type: hf
#   dataset_id: cais/mmlu
#   split: test

# --- Schema (required) ---
schema:
  prompt: prompt              # column name for the clean prompt
  answer: answer              # column name for the expected answer
  corrupted_prompt: corrupt   # column name for the corrupted prompt

# --- Corruption (optional) ---
corruption:
  strategy: entity_swap       # strategy name (see below)
  # or:
  # strategy: token_swap
  # strategy: paraphrase
  # strategy: distractor
  # strategy: role_swap
  config: {}                  # strategy-specific keyword arguments

# --- Metric (optional, default: logit_diff) ---
metric: logit_diff
# or:
# metric: kl                  # differentiable KL divergence
# metric: accuracy            # non-differentiable; reporting only, not for discovery

# --- Chat template (optional) ---
chat_template_mode: auto      # "auto", "on", "off"
```

---

## Source Types

| `type` | Description | Required fields |
|--------|-------------|-----------------|
| `csv` | Local CSV file | `path` |
| `jsonl` | Local JSONL file | `path` |
| `hf` | HuggingFace `datasets` | `dataset_id`, optionally `split` |

---

## Corruption Strategies

These are the only strategy names the task-YAML loader accepts under `corruption.strategy`. Any other name raises `Unknown corruption strategy`.

| Strategy | Class | Description |
|----------|-------|-------------|
| `entity_swap` | `EntitySwapCorruption` | Swap named entities |
| `token_swap` | `TokenSwapCorruption` | Swap tokens at specific positions |
| `paraphrase` | `ParaphraseCorruption` | Paraphrase the prompt |
| `distractor` | `DistractorInjectionCorruption` | Inject a distracting element |
| `role_swap` | `RoleSwapCorruption` | Swap subject/object roles |

See [Corruption strategies](../user-guide/data-corruption.md) for when these apply and how to write your own.

---

## Schema Column Names

The `schema` block maps your CSV/JSONL column names to CircuitKit's expected fields:

| CircuitKit field | Description | Required |
|-----------------|-------------|----------|
| `prompt` | The clean (factual) prompt | Yes |
| `answer` | The expected answer token/string | Yes |
| `corrupted_prompt` | The corrupted (counterfactual) prompt | Yes for discovery |

If your CSV has different column names, remap them:

```yaml
schema:
  prompt: my_clean_text_column
  answer: correct_label
  corrupted_prompt: my_corrupt_text_column
```

---

## Pipeline YAML

The `circuitkit run` command reads a flat top-level config. `model` and `task` sit at the top; each stage (`discovery`, `evaluate`, `applications`, `export`) is its own block:

```yaml
# pipeline.yaml
model: gpt2
task: ioi
precision: bfloat16
output_dir: ./results

discovery:
  algorithm: eap-ig
  level: node
  sparsity: 0.3
  n_examples: 128
  batch_size: 4
  scope: both

evaluate:
  pillars: [patching, ablation, baselines]

applications:
  - type: prune
    sparsity: 0.3
    scope: heads

export:
  path: ./checkpoints/pruned
```

`Pipeline` has no `from_yaml()` classmethod or `.run()` method. Run a pipeline YAML file via the CLI instead:

```bash
circuitkit run pipeline.yaml
```

---

## Next Steps

- [Discovery Commands](discovery.md) — using YAML with the CLI
- [Data model](../user-guide/data.md) — what discovery needs
- [Bring your own data](../user-guide/custom-data.md) — creating custom tasks
- [Corruption strategies](../user-guide/data-corruption.md) — generating the corrupt half
- [API Reference: Dict-Config](../api-reference/dict-config.md) — Python equivalent
