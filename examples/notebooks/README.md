# CircuitKit Notebooks

9 notebooks covering the full CircuitKit v1.0 API — Pipeline, flat `ck.*` functions, and CLI.

## Start Here

1. **[00_colab_setup](00_colab_setup.ipynb)** — Environment setup reference (GPU and CPU tracks)
2. **[01_quickstart_pipeline](01_quickstart_pipeline.ipynb)** — Your first circuit in 10 minutes

## Notebook Index

| # | Notebook | GPU? | Runtime | What You'll Learn |
|---|----------|------|---------|-------------------|
| 00 | [Colab Setup](00_colab_setup.ipynb) | — | — | Environment setup, dependency install, HF login |
| 01 | [Quickstart Pipeline](01_quickstart_pipeline.ipynb) | No | ~5 min | Pipeline E2E: discover → evaluate → prune → export (GPT-2/IOI) |
| 02 | [Algorithm Comparison](02_algorithm_comparison.ipynb) | Yes | ~25 min | 6 algorithms head-to-head on Gemma 2B |
| 03 | [Custom Data](03_custom_data_jailbreak.ipynb) | Yes | ~20 min | Bring-your-own CSV, paired vs. clean-only (Qwen 1.5B) |
| 04 | [Evaluation Deep Dive](04_evaluation_deep_dive.ipynb) | Yes | ~30 min | All 6 faithfulness pillars explained (Llama 1B) |
| 05 | [Visualization Gallery](05_visualization_gallery.ipynb) | No | ~5 min | Graph viz, comparison dashboard, score analysis |
| 06 | [Applications](06_applications.ipynb) | Yes | ~20 min | Pruning, quantization, selective finetuning (Gemma 2B) |
| 07 | [CLI and YAML](07_cli_and_yaml.ipynb) | No | ~5 min | All CLI commands, YAML task configs, full YAML pipeline |
| 08 | [Advanced Research](08_advanced_research_tools.ipynb) | No | ~10 min | Selector registry, MasterGrid, IF metric, artifact reuse |

## Models Used

The notebooks showcase CircuitKit across multiple model families:

| Model | Size | Notebooks | Why |
|-------|------|-----------|-----|
| `gpt2` | 124M | 01, 05, 07, 08 | CPU-friendly, fast iteration |
| `google/gemma-2-2b-it` | 2B | 02, 06 | Instruction-tuned, real-world scale |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 03 | Different model family, custom data demo |
| `meta-llama/Llama-3.2-1B` | 1B | 04 | Llama family, evaluation focus |

## Prerequisites

- **GPU notebooks** (02, 03, 04, 06): Colab T4 or better
- **CPU notebooks** (01, 05, 07, 08): Any runtime
- **Gated models** (Gemma, Llama): Requires HF token — see notebook 00

## Related Resources

- **`examples/`** — Python scripts (`.py`) for the same workflows, CI-testable
- **`docs/`** — Guides for Pipeline, custom data, applications, evaluation, CLI
