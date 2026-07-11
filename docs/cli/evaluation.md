# Evaluation Commands

---

## `circuitkit evaluate`

Evaluate circuit faithfulness for a discovered artifact using the 6-pillar framework.

```bash
circuitkit evaluate [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | required | Model name or HF path |
| `--artifact` | `-a` | required | Path to `.pt` circuit artifact |
| `--task` | `-t` | auto | Task name; auto-derived from `_scores.json` side-car if omitted |
| `--num-examples` | `-n` | 256 | Examples for faithfulness evaluation |
| `--precision` | | `bfloat16` | Torch dtype for model loading |
| `--report-path` | `-r` | auto | Path to write the JSON evaluation report |

### Auto-Derived Task

`evaluate` reads the `_scores.json` side-car that `discover` writes next to the artifact to get the task name, algorithm, and level automatically. You only need `--task` if the side-car is missing.

```bash
results/
├── eap-ig_gpt2_ioi_node.pt           ← --artifact
└── eap-ig_gpt2_ioi_node_scores.json  ← auto-read for task/algorithm/level
```

### Output

The command writes a JSON report:

```json
{
    "patching_score": 0.71,
    "ablation_score": 0.83,
    "stability": null,
    "robustness": null,
    "baseline_comparison": null,
    "generalization": null,
    "intervention_reliability": null,
    "metadata": {"random_avg": 0.52}
}
```

The top-level keys are the `FaithfulnessReport` fields (`report.to_json()` writes
`dataclasses.asdict(report)`). Pillar fields not run are `null`; a random-circuit
baseline, when requested, appears under `metadata.random_avg`.

### Examples

```bash
# Minimal — task auto-derived from side-car
circuitkit evaluate \
    --model gpt2 \
    --artifact ./results/eap-ig_gpt2_ioi_node.pt

# Explicit task (when side-car is missing)
circuitkit evaluate \
    --model gpt2 \
    --artifact ./results/gpt2_circuit.pt \
    --task ioi

# Custom report path
circuitkit evaluate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --artifact ./results/llama_mmlu.pt \
    --report-path ./reports/llama_mmlu_eval.json \
    --num-examples 128
```

---

## `circuitkit transfer-matrix`

Build and analyze a cross-task transfer matrix. Discovers circuits on each source task and evaluates them on all target tasks, producing an N×N matrix showing transfer coverage.

```bash
circuitkit transfer-matrix [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | required | Model name or HF path |
| `--tasks` | `-t` | required | Comma-separated task names |
| `--algorithm` | `-a` | `eap-ig` | Discovery algorithm |
| `--output` | `-o` | auto | Output directory |
| `--sparsity` | `-s` | `0.3` | Target sparsity |
| `--level` | `-l` | `node` | `node` or `neuron` |
| `--num-examples` | | `128` | Examples per task |
| `--visualize` | | `True` | Generate visualizations |
| `--analyze` | | `True` | Run statistical analysis |

### Cost

Transfer matrix requires N discovery runs + N² evaluation runs. For 3 tasks:
- 3 discovery runs
- 9 evaluation runs

For 5 tasks on Llama-1B: plan for 30–60 minutes on a single GPU.

### Examples

```bash
# Basic 3-task matrix
circuitkit transfer-matrix \
    --model gpt2 \
    --tasks ioi,sva,greater_than

# Larger matrix with custom output directory
circuitkit transfer-matrix \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tasks ioi,sva,capital_country,mmlu \
    --algorithm eap-ig \
    --output ./transfer_results
```

### Output

Results are written to the output directory:
```bash
transfer_results/
├── transfer_matrix.npy            # Raw N×N transfer scores
├── transfer_matrix_analysis.json  # Human-readable scores and analysis
├── circuit_<task>_discovery.pt    # Discovered circuit per source task
└── transfer_matrix_heatmap.png    # Heatmap + other plots (if --visualize)
```

---

## Using Evaluation with CI

For automated evaluation in CI:

```bash
#!/bin/bash
# ci_eval.sh

circuitkit discover \
    --model gpt2 --task ioi --algorithm eap-ig \
    --output ./ci/circuit.pt

circuitkit evaluate \
    --model gpt2 --artifact ./ci/circuit.pt \
    --report-path ./ci/eval_report.json

# Check ablation_score threshold
python3 -c "
import json
with open('./ci/eval_report.json') as f:
    r = json.load(f)
assert r['ablation_score'] > 0.70, f'Circuit quality below threshold: {r[\"ablation_score\"]}'
print('Circuit quality check passed:', r['ablation_score'])
"
```

---

## Next Steps

- [Application Commands](applications.md) — `benchmark`, `prune`
- [Evaluation Framework](../evaluation/framework.md) — what the scores mean
- [API Reference: Evaluation](../api-reference/evaluation.md) — Python evaluation API
