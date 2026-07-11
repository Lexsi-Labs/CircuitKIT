# Notebooks

9 Colab-ready notebooks covering the full CircuitKit v1.0 API. CPU notebooks run on any Colab instance; GPU notebooks require a T4 or better.

---

## Notebook Index

| # | Title | GPU | Runtime | Key Topics |
|---|-------|-----|---------|-----------|
| 00 | `00_colab_setup.ipynb` | — | — | Dependency install, HF login, GPU check |
| 01 | `01_quickstart_pipeline.ipynb` | No | ~5 min | Pipeline E2E on GPT-2/IOI |
| 02 | `02_algorithm_comparison.ipynb` | Yes | ~25 min | 6 algorithms head-to-head on Gemma 2B |
| 03 | `03_custom_data_jailbreak.ipynb` | Yes | ~20 min | Bring-your-own CSV, Qwen 1.5B |
| 04 | `04_evaluation_deep_dive.ipynb` | Yes | ~30 min | All 6 faithfulness pillars on Llama 1B |
| 05 | `05_visualization_gallery.ipynb` | No | ~5 min | Graph viz, comparison dashboard |
| 06 | `06_applications.ipynb` | Yes | ~20 min | Pruning, quantization, finetuning on Gemma 2B |
| 07 | `07_cli_and_yaml.ipynb` | No | ~5 min | CLI commands, YAML configs |
| 08 | `08_advanced_research_tools.ipynb` | No | ~10 min | Selector registry, artifact reuse |

---

## Notebook Descriptions

### 00 — Colab Setup

Environment setup reference for both GPU and CPU tracks. Install CircuitKit, authenticate with Hugging Face (required for gated models like Gemma and Llama), and verify GPU availability.

```bash
# The notebook runs this for you:
pip install "circuitkit[gpu-cu126] @ git+https://..."
```

**Run first** if you're new to Colab.

---

### 01 — Quickstart Pipeline (CPU)

Your first complete circuit in about 5 minutes. Walks through the full Pipeline workflow on GPT-2 IOI — no GPU required.

**Models**: `gpt2`  
**You'll learn**: `Pipeline`, `discover`, `evaluate`, `prune`, `export`, `summary`

---

### 02 — Algorithm Comparison (GPU: T4+)

Side-by-side comparison of 6 discovery algorithms on Gemma 2B with the IOI task. Includes circuit quality metrics, runtime, and memory usage for each algorithm.

**Models**: `google/gemma-2-2b-it`  
**You'll learn**: Algorithm selection trade-offs, stability tiers in practice

---

### 03 — Custom Data (GPU: T4+)

Bring your own CSV dataset. Demonstrates both paired (clean/corrupt) and clean-only paths. Uses Qwen 1.5B on a custom jailbreak detection task.

**Models**: `Qwen/Qwen2.5-1.5B-Instruct`  
**You'll learn**: `MCQAdapter`, `NormalizedTaskSpec`, `validate_token_alignment`, YAML task format

---

### 04 — Evaluation Deep Dive (GPU: T4+)

All 6 faithfulness pillars explained with real scores. Runs the full evaluation suite on Llama 1B and interprets each pillar result.

**Models**: `meta-llama/Llama-3.2-1B`  
**You'll learn**: Pillar selection, `FaithfulnessReport`, what scores mean

---

### 05 — Visualization Gallery (CPU)

All three visualization modes. No GPU needed — uses a pre-saved circuit artifact.

**Models**: `gpt2` (pre-saved)  
**You'll learn**: `visualize(mode="graph")`, `CircuitGraphVisualizer`, `ComparisonDashboard`, `JupyterWidgetSuite`, Streamlit

---

### 06 — Applications (GPU: T4+)

Structural pruning, circuit-aware quantization, and selective fine-tuning on Gemma 2B.

**Models**: `google/gemma-2-2b-it`  
**You'll learn**: `ck.prune`, `ck.quantize`, `ck.selective_finetune`, `ck.export_checkpoint`, `ck.benchmark`

---

### 07 — CLI and YAML (CPU)

All CLI commands and YAML configuration formats, run from within Colab using `!` shell cells.

**Models**: `gpt2`  
**You'll learn**: `circuitkit discover`, `discover-yaml`, `evaluate`, full YAML pipeline schema

---

### 08 — Advanced Research Tools (CPU)

Selector registry, MasterGrid comparison, IF metric, and artifact reuse patterns.

**Models**: `gpt2`  
**You'll learn**: `register("my_selector")`, `Pipeline.from_artifact`, `normalize_importance_scores`

---

## Models Used

| Model | Size | Notebooks |
|-------|------|-----------|
| `gpt2` | 124M | 01, 05, 07, 08 |
| `google/gemma-2-2b-it` | 2B | 02, 06 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 03 |
| `meta-llama/Llama-3.2-1B` | 1B | 04 |

Gated models (Gemma, Llama) require HF token authentication — see notebook 00.

---

## Next Steps

- [Python Scripts](scripts.md) — CI-testable, CPU-friendly versions of the same workflows
- [Case Studies](case-studies.md) — three more notebooks (21–23: permanent unlearning, bias audit & mitigation, jailbreak steering) framed around real scenarios
- [Getting Started: Quick Start](../getting-started/quickstart.md) — the fastest path to a first circuit
