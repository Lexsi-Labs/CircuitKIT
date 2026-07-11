# Application Commands

The CLI exposes a subset of the application operations.

---

## `circuitkit prune`

```bash
circuitkit prune --model gpt2 --artifact ./circuit.pt --sparsity 0.3 --scope both --output ./pruned
```

Structurally prune a model's weights down to the discovered circuit.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Model name or path |
| `--artifact` | required | Path to circuit artifact (`.pt`) |
| `--sparsity` | `0.3` | Fraction of nodes to prune |
| `--scope` | `both` | `heads`, `mlp`, or `both` |
| `--output` / `-o` | required | Output checkpoint directory |
| `--precision` | `bfloat16` | Torch dtype for model loading |

---

## `circuitkit quantize`

```bash
circuitkit quantize --model gpt2 --artifact ./circuit.pt --bits 4 --output ./quantized
```

Apply circuit-aware mixed-precision quantization.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Model name or path |
| `--artifact` | required | Path to circuit artifact |
| `--bits` | `4` | Target bit width (3, 4, or 8) |
| `--high-fraction` | `0.3` | Fraction of high-importance layers to protect |
| `--backend` | `quanto` | Quantization backend (`quanto`, `llmcompressor`) |
| `--output` / `-o` | required | Output checkpoint directory |
| `--precision` | `bfloat16` | Torch dtype for model loading |

---

## `circuitkit export`

```bash
circuitkit export --model gpt2 --artifact ./circuit.pt --sparsity 0.3 --output ./checkpoint
```

Export a pruned or quantized model as a reloadable HuggingFace checkpoint.

---

## `circuitkit heal`

```bash
circuitkit heal --model gpt2 --pruned-model ./pruned.pt --circuit-scores ./circuit_scores.pt --task ioi --epochs 3
```

Post-pruning model recovery via circuit-restricted LoRA fine-tuning.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` / `-m` | required | Model name (e.g. `gpt2`) |
| `--pruned-model` / `-p` | required | Path to pruned model weights |
| `--circuit-scores` / `-c` | required | Path to circuit scores artifact (`.pt`) |
| `--task` / `-t` | `ioi` | Task name for training data |
| `--lora-rank` | `8` | LoRA rank |
| `--epochs` | `3` | Training epochs |
| `--learning-rate` | `1e-4` | Learning rate |
| `--batch-size` / `-b` | `4` | Batch size |
| `--score-threshold` | `0.0` | Only heal nodes with score >= threshold |
| `--output` / `-o` | — | Output path for healed model |
| `--device` | `cuda` | Device to use (`cuda`/`cpu`) |

There is no `--artifact` flag on `heal` — use `--pruned-model` and `--circuit-scores`.

---

## `circuitkit steer`

```bash
circuitkit steer --model gpt2 --circuit-scores circuits/gpt2_ioi_scores.json \
    --source-examples data/ioi_source.csv --target-examples data/ioi_target.csv \
    --coefficient 1.0
```

Apply activation steering, learning steering vectors from source/target example pairs.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` / `-m` | required | Model name or path |
| `--circuit-scores` / `-cs` | required | Path to circuit scores JSON |
| `--task` / `-t` | `ioi` | Task name |
| `--source-examples` / `-se` | required | Path to source examples CSV |
| `--target-examples` / `-te` | required | Path to target examples CSV |
| `--coefficient` / `-c` | `1.0` | Steering strength (0.0–2.0) |
| `--output` / `-o` | — | Output directory for results |
| `--batch-size` / `-b` | `32` | Batch size for steering |
| `--metric` | `logit_diff` | `logit_diff`, `accuracy`, or `top_k` |
| `--top-k` | `5` | Top-k for metric calculation |
| `--threshold` | `0.0` | Minimum circuit score threshold |
| `--analyze` | `False` | Run detailed analysis |

There are no `--vector`, `--layers`, or `--coeff` flags on `steer` — use `--circuit-scores`, `--source-examples`, `--target-examples`, and `--coefficient`.

---

## `circuitkit benchmark`

```bash
circuitkit benchmark --models gpt2 --tasks ioi --algorithms eap-ig --interventions prune --baselines magnitude
```

Runs a full discovery + intervention + baseline comparison sweep and produces a report — this is **not** an lm-eval-harness checkpoint runner. There are no `--checkpoint` or `--num_fewshot` flags. For lm-eval-harness benchmarking of an exported checkpoint, use the Python API instead: `ck.benchmark(checkpoint_path, tasks=[...])` (see [Flat API](../api-reference/flat-api.md#benchmark)).

| Option | Default | Description |
|--------|---------|-------------|
| `--models` / `-m` | `gpt2` | Model names to benchmark |
| `--tasks` / `-t` | `ioi` | Task names |
| `--algorithms` / `-a` | `eap`, `eap-ig` | Discovery algorithms |
| `--interventions` / `-i` | `prune`, `heal` | Interventions to benchmark |
| `--baselines` / `-b` | `magnitude`, `wanda`, `random` | Baselines to compare |
| `--sparsity-levels` | `0.1`, `0.3`, `0.5` | Sparsity levels to test |
| `--output-dir` / `-o` | `./benchmark_results` | Output directory for results |
| `--num-examples` | `100` | Number of examples per task |
| `--report-format` | `html` | `html`, `markdown`, `json`, or `latex` |

---

## `circuitkit inspect`

```bash
circuitkit inspect ./circuit.pt
```

Inspect a circuit artifact and print its contents: algorithm, scores, metadata.

---

## `circuitkit validate-config`

```bash
circuitkit validate-config --config ./pipeline.yaml
```

Validate a discovery/evaluation config YAML without running it. The path is passed via `--config` / `-c`, not as a positional argument.

---

## `circuitkit data`

`SOURCE` is a required positional argument on every `data` subcommand below — not a `--source` flag.

```bash
# Check dataset compatibility (SOURCE positional)
circuitkit data check boolq --shape qa

# Prepare dataset for discovery (SOURCE positional, --output required)
circuitkit data prepare my_data.csv --strategy entity_swap --output ./prepared.json

# Build a NormalizedDataset from a CSV via template substitution
# (SOURCE positional; --clean-prompt, --clean-answer, --output required)
circuitkit data template my_data.csv \
    --clean-prompt "{question}" --clean-answer "{answer}" \
    --output my_task.json

# Extract clean-only records from a CSV (SOURCE is a file path, --output required)
circuitkit data clean-only my_data.csv --output ./clean.json

# List supported shapes and strategies
circuitkit data shapes
circuitkit data strategies
```

Dataset inspection and preparation utilities for bringing custom data into the discovery pipeline.

---

## `circuitkit run`

```bash
circuitkit run ./pipeline.yaml
```

Execute a full discover → evaluate → intervene pipeline from a single YAML config.

---

## Verbose Mode

All CLI commands support `--verbose` / `-v` for detailed progress output:

```bash
circuitkit -v discover --model gpt2 --task ioi
```

---

## Next Steps

- [YAML Configuration](yaml-config.md) — full YAML schema for scripted workflows
- [User Guide: Applications](../user-guide/applications.md) — pruning, quantization, steering
- [Advanced: vLLM Evaluation](../advanced/vllm-evaluation.md) — fast benchmarking
