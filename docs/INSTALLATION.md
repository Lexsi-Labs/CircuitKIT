# CircuitKit Installation Guide

> **Reproducing the paper experiments?** Use the fixed environment in
> [`../ENVIRONMENT.md`](../ENVIRONMENT.md) — the NVIDIA NeMo 25.09 container
> (`nvcr.io/nvidia/nemo:25.09`; CUDA 13 / torch 2.9 / vLLM 0.10). That
> container is the version pin. The scenarios below are for general library
> use on other machines; the `cu126` option is **not** the verified path.

## Quick Start

### Minimal Installation (Recommended for CPU)

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd CircuitKit
pip install -e .
```

### GPU Installation (CUDA 12.6)

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd CircuitKit
pip install -e ".[gpu-cu126]" --extra-index-url https://download.pytorch.org/whl/cu126
```

---

## Dependency Groups

CircuitKit uses optional dependency groups for different use cases:

### Core Dependencies

Required for all features:
```
torch, einops, plotly, tqdm, numpy, huggingface-hub,
transformers>=4.52.3, ordered-set, pyyaml, networkx,
matplotlib, ipywidgets, scikit-learn, evaluate,
sacrebleu, rouge_score, bert-score, accelerate,
sentence-transformers, click, rich, torch-pruning>=1.0,
transformer-lens>=2.18,<3 (PyPI, verified against 2.18.x — adds Gemma-3 support)
```

Install core only:
```bash
pip install -e .
```

### Corruption Pipeline

spaCy (`spacy>=3.8`) backs the corruption strategies and ships with the base
install — no extra is required. The strategies also need the small English
model, a one-time download:
```bash
python -m spacy download en_core_web_sm
```

Use cases:
- Paraphrase corruption
- Custom text transformations
- Semantic corruption variants

### Benchmarking Suite

Required for comprehensive evaluation:
```bash
pip install -e .[benchmarks]
```

Includes: `lm-eval` (from GitHub), `datasets>=2.20.0`

Use cases:
- GSM8K, MMLU, TruthfulQA, HumanEval, HellaSwag
- 100+ benchmark tasks via lm-evaluation-harness
- Automated performance comparison

### GPU CUDA 12.6

Pinned CUDA 12.6 wheels:
```bash
pip install -e .[gpu-cu126] --extra-index-url https://download.pytorch.org/whl/cu126
```

Includes: `torch==2.6.0+cu126`, `torchvision==0.21.0+cu126`

Alternative CUDA versions available via PyTorch's wheel index:
- CUDA 12.1: `cu121`
- CUDA 11.8: `cu118`

### Development & Testing

For contributing to CircuitKit:
```bash
pip install -e .[dev]
```

Includes: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `isort`, `pre-commit`

### Documentation

For building docs locally:
```bash
pip install -e .[docs]
```

Includes: `mkdocs`, `mkdocs-material`, `pymdown-extensions`, `mkdocstrings[python]`, `mkdocs-jupyter`, `pygments`

---

## Installation Scenarios

### Scenario 1: Local Development (Recommended)

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd CircuitKit
pip install -e .[dev,benchmarks,gpu-cu126] --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Enables: Full development, all features, GPU support, testing

### Scenario 2: Server Deployment

```bash
pip install -e .[benchmarks]
```

Enables: Circuit discovery, all evaluation features, minimal overhead

### Scenario 3: Research-Only (No Benchmarks)

```bash
pip install -e .
```

Enables: Circuit discovery, corruption robustness, faithfulness evaluation
(spaCy-backed corruption ships with the base install)

### Scenario 4: CPU-Only

```bash
pip install -e .
```

Enables: Basic circuit discovery, limited to CPU (slower)

### Scenario 5: Install from a Git ref

```bash
pip install "git+https://github.com/Lexsi-Labs/circuitkit.git"
```

Installs CircuitKit directly from GitHub (no local clone). CircuitKit is not yet
published to PyPI; install from source or from a Git ref.

---

## System Requirements

### Python Version

- **Required**: Python 3.10 or later (`requires-python = ">=3.10"`)
- **Recommended**: Python 3.11+

Check your version:
```bash
python --version
```

### GPU Requirements

**For GPU Acceleration**:
- NVIDIA GPU (Recommended: RTX 4090, A100, or similar)
- CUDA Toolkit 12.1+ (or 12.6 for best performance)
- cuDNN 8.0+
- Driver: Latest stable

**Memory**:
- GPT-2: 4GB VRAM minimum
- GPT-2-XL: 8GB VRAM minimum
- LLaMA-7B: 16GB VRAM minimum
- LLaMA-70B: 80GB VRAM minimum (requires distributed setup)

**For CPU-Only**:
- 32GB RAM minimum for GPT-2-XL
- 64GB+ for larger models
- Much slower (10-50x slower than GPU)

### Platform Support

| OS | Status | Notes |
|----|--------|-------|
| Linux | ✅ Full | Recommended, all features |
| macOS | ✅ Full | CPU or Metal acceleration |
| Windows | ⚠️ Partial | WSL2 recommended, GPU support requires CUDA |

---

## Troubleshooting Installation

### Issue: `ImportError: No module named 'transformer_lens'`

**Solution**: CircuitKit pins `transformer-lens>=2.18,<3` from PyPI. Ensure pip has internet access:

```bash
pip install -e . --no-cache-dir
```

### Issue: `CUDA out of memory` during installation

**Solution**: Use CPU-only install temporarily, then add GPU:

```bash
pip install -e .
pip install torch==2.6.0+cu126 -f https://download.pytorch.org/whl/cu126
```

### Issue: `pygraphviz` compilation fails

**Solution**: Install system dependencies first:

**Ubuntu/Debian**:
```bash
sudo apt-get install graphviz graphviz-dev
```

**macOS**:
```bash
brew install graphviz
```

**Windows**:
Download from [Graphviz website](https://graphviz.org/download/)

Then retry:
```bash
pip install pygraphviz
```

### Issue: `spacy` model not found

**Solution**: Download spacy language model:

```bash
python -m spacy download en_core_web_sm
```

### Issue: `lm-eval` not working

**Solution**: Ensure lm-eval is properly installed from GitHub:

```bash
pip install --upgrade lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

### Issue: `ModuleNotFoundError: No module named 'circuitkit'`

**Solution**: Ensure editable install worked:

```bash
pip install -e . --force-reinstall --no-deps
python -c "import circuitkit; print(circuitkit.__version__)"
```

---

## Verification

### Verify Installation

```python
import circuitkit
print(circuitkit.__version__)  # Should be 1.0.0 or later
```

### Test CLI

```bash
circuitkit --help
```

Should display help menu without errors.

### Test Core Import

```python
from circuitkit.api import discover_circuit, evaluate_circuit
from circuitkit.evaluation import evaluate_graph
from circuitkit.corruption import CorruptionPipeline
from circuitkit.applications.pruning import StructuralPruner
print("All imports successful!")
```

### Run Basic Discovery

```bash
circuitkit discover-smart --model gpt2 --algorithm eap-ig --task ioi --check-memory
```

Should complete without errors and output memory estimates.

---

## Uninstallation

### Remove CircuitKit

```bash
pip uninstall circuitkit
```

### Remove with Dependencies

To safely remove only CircuitKit and optional dependencies:

```bash
pip uninstall circuitkit spacy lm-eval torch-pruning
```

---

## Advanced Configuration

### Custom PyTorch Build

For specific CUDA/cuDNN versions:

```bash
pip install torch==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Install from Specific Commit

```bash
pip install git+https://github.com/Lexsi-Labs/circuitkit.git@<commit_hash>
```

### Offline Installation

Download wheels and install locally:

```bash
pip download -e . -d ./wheels
# Transfer wheels to offline environment
pip install --no-index --find-links ./wheels circuitkit
```

---

## Getting Help

- **Documentation**: [GitHub Wiki](https://github.com/Lexsi-Labs/circuitkit/wiki)
- **Issues**: [GitHub Issues](https://github.com/Lexsi-Labs/circuitkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lexsi-Labs/circuitkit/discussions)
- **Email**: circuitkit@example.com

---

## Next Steps

After installation:

1. Read the [Quickstart](../README.md#quickstart) in the README
2. Run the tutorial notebooks in `docs/tutorials/`
3. Check examples in `examples/`
4. Review the [feature reference](reference/FEATURES.md)
