<!-- circuitkit-logo -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/circuitkit-logo-white.png">
    <img src="../../docs/assets/circuitkit-logo-black.png" width="200" alt="CircuitKIT">
  </picture>
</p>

# CircuitKIT Notebooks

8 notebooks covering the full CircuitKIT v1.0 API — Pipeline, flat `ck.*` functions, and CLI.

## Start Here

1. **[00_colab_setup](00_colab_setup.ipynb)** — Environment setup reference (GPU and CPU tracks)
2. **[01_quickstart_pipeline](01_quickstart_pipeline.ipynb)** — Your first circuit in 10 minutes

## Notebook Index

| # | Notebook | GPU? | Runtime | What You'll Learn |
|---|----------|------|---------|-------------------|
| 00 | [Colab Setup](00_colab_setup.ipynb) | — | — | Environment setup, dependency install, HF login |
| 01 | [Quickstart Pipeline](01_quickstart_pipeline.ipynb) | No | ~5 min | Pipeline E2E: discover → evaluate → prune → export (GPT-2/IOI) |
| 02 | [Algorithm Comparison](02_algorithm_comparison.ipynb) | Yes | ~25 min | 6 algorithms head-to-head on Gemma 2B |
| 03 | [Evaluation Deep Dive](03_evaluation_deep_dive.ipynb) | Yes | ~30 min | All 6 faithfulness pillars explained (Llama 1B) |
| 04 | [Visualization Gallery](04_visualization_gallery.ipynb) | No | ~5 min | Graph viz, comparison dashboard, score analysis |
| 05 | [Applications](05_applications.ipynb) | Yes | ~20 min | Pruning, quantization, selective finetuning (Gemma 2B) |
| 06 | [CLI and YAML](06_cli_and_yaml.ipynb) | No | ~5 min | All CLI commands, YAML task configs, full YAML pipeline |
| 07 | [Advanced Research](07_advanced_research_tools.ipynb) | No | ~10 min | Selector registry, MasterGrid, IF metric, artifact reuse |

Bringing your own CSV to discover a circuit on custom data is covered end-to-end
in the [jailbreak refusal case study](../case-studies/23-jailbreak-refusal-multi-model.ipynb).

## Models Used

The notebooks showcase CircuitKIT across multiple model families:

| Model | Size | Notebooks | Why |
|-------|------|-----------|-----|
| `gpt2` | 124M | 01, 04, 06, 07 | CPU-friendly, fast iteration |
| `google/gemma-2-2b-it` | 2B | 02, 05 | Instruction-tuned, real-world scale |
| `meta-llama/Llama-3.2-1B` | 1B | 03 | Llama family, evaluation focus |

## Prerequisites

- **GPU notebooks** (02, 03, 05): Colab T4 or better
- **CPU notebooks** (01, 04, 06, 07): Any runtime
- **Gated models** (Gemma, Llama): Requires HF token — see notebook 00

## Related Resources

- **`examples/`** — Python scripts (`.py`) for the same workflows, CI-testable
- **`examples/case-studies/`** — Applied end-to-end studies, including
  [jailbreak refusal localization across three models](../case-studies/23-jailbreak-refusal-multi-model.ipynb)
- **`docs/`** — Guides for Pipeline, custom data, applications, evaluation, CLI
