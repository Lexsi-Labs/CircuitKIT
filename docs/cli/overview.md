# CLI Reference

CircuitKit ships a `circuitkit` command-line interface built with Click and Rich. The CLI is the primary interface for scripted workflows, CI pipelines, and YAML-driven experiments.

---

## Installation

The CLI is installed with the package:

```bash
pip install -e ".[gpu-cu126]"   # development install
circuitkit --help
```

---

## Command Overview

| Command | Description |
|---------|-------------|
| `circuitkit discover` | Run circuit discovery with a built-in task |
| `circuitkit discover-yaml` | Run discovery with a custom YAML task |
| `circuitkit discover-smart` | Run discovery with automatic memory checks |
| `circuitkit evaluate` | Evaluate a discovered circuit's faithfulness |
| `circuitkit transfer-matrix` | Build a cross-task transfer matrix |
| `circuitkit list-models` | List TransformerLens-supported models |
| `circuitkit prune` | Structurally prune a model to the circuit |
| `circuitkit quantize` | Circuit-aware mixed-precision quantization |
| `circuitkit export` | Export a pruned/quantized model as HF checkpoint |
| `circuitkit heal` | Post-pruning model recovery via LoRA |
| `circuitkit steer` | Activation steering at inference |
| `circuitkit benchmark` | Compare circuit methods and baselines across tasks |
| `circuitkit inspect` | Inspect a circuit artifact's contents |
| `circuitkit validate-config` | Validate a discovery config YAML |
| `circuitkit run` | Run a full pipeline from a YAML config |
| `circuitkit data check` | Check a dataset for EAP compatibility |
| `circuitkit data prepare` | Prepare a dataset for discovery |
| `circuitkit data template` | Generate a YAML task template |
| `circuitkit data clean-only` | Extract clean-only records from a dataset |
| `circuitkit data shapes` | List supported dataset shapes |
| `circuitkit data strategies` | List supported corruption strategies |

---

## Global Options

```bash
circuitkit [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose    Enable verbose output
  -c, --config     Path to configuration file
  --help           Show this message and exit
```

---

## Quick Examples

### Discover a circuit

```bash
# Basic discovery
circuitkit discover --model gpt2 --algorithm eap-ig --task ioi --sparsity 0.3

# With output path
circuitkit discover -m gpt2 -a eap-ig -t ioi -s 0.3 -o ./results/gpt2_ioi.pt

# Custom task from YAML
circuitkit discover-yaml -m gpt2 -t ./my_task.yaml -a eap-ig

# With memory check
circuitkit discover-smart --model google/gemma-3-4b-it --task mmlu --check-memory
```

### Evaluate a circuit

```bash
circuitkit evaluate --model gpt2 --artifact ./results/gpt2_ioi.pt
```

### Transfer matrix

```bash
circuitkit transfer-matrix -m gpt2 -t ioi,sva,greater_than
```

---

## Output Files

Discovery commands write three files by default to `results/`:

```bash
results/
├── eap-ig_gpt2_ioi_node.pt          # Circuit artifact
├── eap-ig_gpt2_ioi_node_scores.json # Human-readable scores
└── eap-ig_gpt2_ioi_node_scores.pt   # Machine-readable scores
```

The `evaluate` command writes a JSON evaluation report:

```bash
results/evaluation_report_gpt2.json
```

---

## Algorithm Restriction

The `--algorithm` option of discovery commands accepts only the 13 **discovery algorithms**. It does not accept compression selector names like `wanda`, `gptq`, or `magnitude` — those are accessed via the Python API.

Valid values: `acdc`, `atp-gd`, `cdt`, `eap`, `eap-clean-corrupted`, `eap-exact`, `eap-gp`, `eap-ifr`, `eap-ig`, `eap-ig-activations`, `ibcircuit`, `peap`, `relp`

---

## Detailed Reference

- [Discovery Commands](discovery.md) — `discover`, `discover-yaml`, `discover-smart`
- [Evaluation Commands](evaluation.md) — `evaluate`, `transfer-matrix`
- [Application Commands](applications.md) — full reference for `prune`, `quantize`, `export`, `heal`, `steer`, `benchmark`, `inspect`, `run`, and the `data` subcommands
- [YAML Configuration](yaml-config.md) — full YAML schema reference
