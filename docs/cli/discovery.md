# Discovery Commands

---

## `circuitkit discover`

Run circuit discovery with a built-in or registered task.

```bash
circuitkit discover [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | required | Model name or HF path (e.g. `gpt2`, `meta-llama/Llama-3.2-1B-Instruct`) |
| `--algorithm` | `-a` | `eap-ig` | Discovery algorithm (restricted to 13 discovery algorithms) |
| `--task` | `-t` | `ioi` | Built-in task name |
| `--output` | `-o` | auto | Output `.pt` file path |
| `--sparsity` | `-s` | `0.3` | Target sparsity (0.0–1.0) |
| `--level` | `-l` | `node` | `node` or `neuron` |
| `--batch-size` | `-b` | `4` | Discovery batch size |
| `--ig-steps` | | `5` | Integrated Gradients steps (EAP-IG only) |
| `--scope` | | `both` | Pruning scope: `heads`, `mlp`, `both` |
| `--num-examples` | | `128` | Number of examples to attribute over |
| `--chat-template-mode` | | task default | `auto`, `on`, or `off` |
| `--evaluate` | | `False` | Run 6-pillar evaluation after discovery |
| `--random` | | `False` | Also evaluate a random baseline |
| `--mlp1` | | `False` | Score intermediate d_mlp layer for neurons (neuron-level only) |
| `--verbose` / `-v` | | `False` | Verbose output |

### Default Output Path

When `--output` is omitted, the artifact is written to:
```bash
results/{algorithm}_{model}_{task}_{level}.pt
```

For example: `results/eap-ig_gpt2_ioi_node.pt`

### Examples

```bash
# Minimal: GPT-2 IOI with EAP-IG
circuitkit discover -m gpt2 -t ioi

# Llama-3.2-1B with custom sparsity and evaluation
circuitkit discover \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --algorithm eap-ig \
    --task mmlu \
    --sparsity 0.3 \
    --num-examples 128 \
    --evaluate

# Neuron-level discovery
circuitkit discover -m gpt2 -t ioi --level neuron

# Force chat template for instruction-tuned model
circuitkit discover \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --task capital_country \
    --chat-template-mode on

# Memory-optimized for large model
circuitkit discover \
    --model google/gemma-3-4b-it \
    --algorithm eap-ig \
    --task mmlu \
    --ig-steps 3 \
    --batch-size 1 \
    --num-examples 64
```

---

## `circuitkit discover-yaml`

Run discovery with a custom task defined in a YAML file.

```bash
circuitkit discover-yaml [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | required | Model name or HF path |
| `--task-yaml` | `-t` | required | Path to YAML task file (must exist) |
| `--algorithm` | `-a` | `eap-ig` | Discovery algorithm |
| `--output` | `-o` | auto | Output `.pt` file path |
| `--sparsity` | `-s` | `0.3` | Target sparsity |
| `--level` | `-l` | `node` | `node` or `neuron` |
| `--batch-size` | `-b` | `4` | Batch size |
| `--num-examples` | | `128` | Number of examples |
| `--evaluate` | | `False` | Run evaluation after discovery |
| `--random` | | `False` | Evaluate random baseline |

### YAML Task File

See [YAML Configuration](yaml-config.md) for the full task YAML schema. Minimal example:

```yaml
# task.yaml
name: my_task          # required
source:
  type: csv
  path: ./my_data.csv
schema:
  prompt: prompt
  answer: answer
  corrupted_prompt: corrupted_prompt
metric: logit_diff
```

### Example

```bash
circuitkit discover-yaml \
    --model gpt2 \
    --task-yaml ./tasks/my_factual_task.yaml \
    --algorithm eap-ig \
    --sparsity 0.25 \
    --num-examples 64
```

---

## `circuitkit discover-smart`

Run discovery with automatic memory requirement checks before loading the model.

```bash
circuitkit discover-smart [OPTIONS]
```

The `--check-memory` flag reports available memory and checks whether the model fits before running. If it does not fit, it suggests smaller alternative models.

### Additional Options

Adds `--check-memory` and auto-selects a smaller model when the requested one does not fit. Unlike `discover`, this command does not accept `--batch-size`, `--ig-steps`, or `--chat-template-mode`; batch size is chosen automatically from the memory-efficient config.

| Option | Default | Description |
|--------|---------|-------------|
| `--check-memory` | `False` | Estimate memory requirements before running |

### Example

```bash
# Pre-flight memory check for Gemma-4B
circuitkit discover-smart \
    --model google/gemma-3-4b-it \
    --algorithm eap-ig \
    --task mmlu \
    --check-memory

# Proceed with reduced settings if check passes
circuitkit discover-smart \
    --model google/gemma-3-4b-it \
    --task mmlu \
    --num-examples 64 \
    --sparsity 0.1
```

---

## `circuitkit list-models`

List TransformerLens-supported models.

```bash
circuitkit list-models [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | `50` | Max models to show |
| `--type` | — | Filter by model type (e.g. `Llama`, `GPT`) |
| `--size` | — | Filter by size (e.g. `Small`, `Large`) |

```bash
circuitkit list-models --type Llama --limit 20
```

---

## Next Steps

- [Evaluation Commands](evaluation.md) — `evaluate`, `transfer-matrix`
- [YAML Configuration](yaml-config.md) — task YAML schema
- [Advanced: Memory Optimization](../advanced/memory-optimization.md) — reduce VRAM for large models
