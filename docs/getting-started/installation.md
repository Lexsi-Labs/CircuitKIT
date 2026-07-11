# Installation

CircuitKit requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Standard install (CPU)

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd circuitkit
pip install -e .
```

## GPU install (CUDA 12.6)

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd circuitkit
pip install -e ".[gpu-cu126]" --extra-index-url https://download.pytorch.org/whl/cu126
```

For other CUDA versions, install core first, then your preferred torch wheel separately.

## Install without cloning

```bash
pip install "git+https://github.com/Lexsi-Labs/circuitkit.git"
```

## Optional extras

```bash
pip install -e ".[benchmarks]"   # lm-evaluation-harness integration
pip install -e ".[quantization]" # optimum-quanto quantisation backend
pip install -e ".[cdt]"          # captum/lime/shap (CD-T research backend)
pip install -e ".[dev]"          # pytest, black, flake8, mypy
pip install -e ".[docs]"         # MkDocs documentation build
```

spaCy ships with the base install (it backs the Pillar 4 corruption
strategies). Those strategies also need the small English model, which is a
one-time download:

```bash
python -m spacy download en_core_web_sm
```

Combine extras:
```bash
pip install -e ".[gpu-cu126,benchmarks]"
```

| Extra | Enables |
|---|---|
| *(core)* | Discovery, evaluation, pruning, visualisation, robustness pillar (Pillar 4) with semantic/entity corruptions |
| `benchmarks` | GSM8K, MMLU, BoolQ, WinoGrande via lm-eval |
| `quantization` | Circuit-aware mixed-precision via optimum-quanto |
| `cdt` | CD-T research-tier discovery backend |

## Verify

```python
import circuitkit
print(circuitkit.__version__)   # 1.0.0
```

```bash
circuitkit --help
```

```bash
circuitkit discover-smart --model gpt2 --algorithm eap-ig --task ioi --check-memory
```

## Platform notes

| Platform | Status |
|---|---|
| Linux | Full — recommended |
| macOS | Full — CPU or Metal |
| Windows | Partial — WSL2 recommended |

## Troubleshooting

??? warning "ImportError: No module named 'transformer_lens'"
    ```bash
    pip install -e . --no-cache-dir
    ```

??? warning "ModuleNotFoundError: No module named 'circuitkit'"
    ```bash
    pip install -e . --force-reinstall --no-deps
    ```

??? warning "spacy model not found"
    ```bash
    python -m spacy download en_core_web_sm
    ```

??? warning "CUDA out of memory during install"
    Install CPU-only first, then swap in the GPU torch wheel:
    ```bash
    pip install -e .
    pip install torch==2.6.0+cu126 -f https://download.pytorch.org/whl/cu126
    ```

## Next steps

- [Quick Start](quickstart.md) — run your first circuit in 5 minutes
- [Core Concepts](core-concepts.md) — understand circuits, tasks, and faithfulness
