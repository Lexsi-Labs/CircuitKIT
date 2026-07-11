## CircuitKit Testing Guidelines

These guidelines summarize the current capabilities, configuration options, test focus areas, and environment setup required to validate CircuitKit across local servers, Lightning, and Colab.

### Supported features (alpha)
- Unified API for discovery and evaluation via `circuitkit.api`:
  - `discover_circuit(config)` supports 13 discovery algorithms with explicit stability tiers (see `circuitkit.backends`): **Stable** — `eap`, `eap-ig`; **Experimental** — `acdc`, `ibcircuit`; **Research** (GPT-2 IOI only) — `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`, `eap-ifr`, `cdt`. Experimental/research algorithms emit a `UserWarning`.
  - `evaluate_circuit(...)` for LM evaluation harness integration
  - LM evaluation harness integration with popular benchmarks (GSM8K, MMLU, TruthfulQA, HumanEval, HellaSwag)
- Pruning modes:
  - Node-level: attention heads and MLP blocks (`scope`: `heads` | `mlp` | `both`)
  - Neuron-level pruning
- CLI commands in `circuitkit`:
  - `discover` with `--model`, `--algorithm`, `--data-path`, `--sparsity`, `--level`, `--batch-size`, `--ig-steps`, `--scope`
  - `discover-smart` for memory-aware defaults and checks
  - `evaluate` with `--model`, `--artifact`, `--task`, `--num-examples`, `--precision`, `--report-path`. There are no `--enable-lm-eval` / `--disable-lm-eval` / `--lm-eval-*` flags on this command — lm-eval-harness benchmarking is a separate `circuitkit benchmark` command (see [CLI Reference](../cli/applications.md))
  - `list-models`, `--help`
- Memory optimization helpers (internal, used by `discover-smart`): `get_memory_efficient_config`, `check_memory_requirements`, `optimize_memory_usage`
- Visualization and reporting utilities

### Configuration overview
Configuration can be provided as a dict or YAML. Defaults are merged automatically; required keys are validated.

- `model`
  - `name` (required): e.g., `gpt2`, `meta-llama/Meta-Llama-3-8B`
  - `precision` (optional): default `bfloat16` (`float16`|`float32` supported)
- `discovery`
  - `algorithm` (required): any of the 13 supported algorithms (see Supported features above)
  - `level`: `node` (default) | `neuron`
  - For `eap`/`eap-ig` and most algorithms:
    - `data_path` (required): path to CSV used by EAP datasets
    - `batch_size` (default `4`)
    - `ig_steps` (default `5`, for `eap-ig`)
    - `method` (default `EAP-IG-inputs`)
- `pruning`
  - `target_sparsity` (required): `0.0–1.0`
  - `scope`: `heads` | `mlp` | `both` (default)
- `output_path`: file where results are written (default `./circuit_discovery_results.pt`)

### LM Evaluation (downstream benchmarking)

There is no `qa_params.lm_eval` config key. lm-eval-harness benchmarking is a separate step, run after an intervention via `ck.benchmark(checkpoint_path, tasks, *, backend="hf", limit=None, fewshot=0, device=None, dtype="float32")` or the `circuitkit benchmark` CLI command — see [Flat API: benchmark](../api-reference/flat-api.md#benchmark) and [CLI Reference](../cli/applications.md).

### Test Suite Overview

CircuitKit includes a comprehensive test suite located in the `tests/` directory:

#### Unit Tests (`tests/unit/`)
- **`test_api.py`**: Tests for core API functions
  - `discover_circuit()` with various configurations
  - `evaluate_circuit()` functionality
  - Configuration validation and error handling
- **`test_cli.py`**: Tests for command-line interface
  - CLI command parsing and execution
  - Parameter validation and error handling

#### Integration Tests (`tests/integration/`)
- End-to-end functionality tests covering chat templates, steering, soft healing, and transfer-matrix workflows — e.g. `test_chat_template_instruct.py`, `test_steering_ioi.py`, `test_transfer_matrix.py`. There is no single `test_core_functionality.py`.

#### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=circuitkit --cov-report=html
```

### Areas to test
- Discovery correctness and stability
  - `acdc` node-level runs on small models (e.g., `gpt2`) with `task=ioi`
  - `eap` and `eap-ig` with a small CSV and small `ig_steps` (e.g., `3–5`)
  - Neuron-level mode for EAP/EAP-IG (`discovery.level=neuron`)
- Pruning behavior
  - Verify `target_sparsity` and `scope` are respected
  - Confirm artifact formats (node lists vs neuron dicts) and downstream evaluation compatibility
- CLI parity vs API
  - Cross-check CLI `discover`/`discover-smart` vs `api.discover_circuit`
  - `evaluate` flow generates metrics without errors
  - LM evaluation CLI options match API configuration
  - Test both enabled and disabled LM evaluation modes
- Memory efficiency
  - `discover-smart --check-memory` and `utils.memory` functions
  - Behavior without GPU (CPU fallback)
- Visualizations and reports
  - Generated visualizations render without errors on headless environments
- LM evaluation functionality
  - Test with different task combinations and limits
  - Verify original vs pruned model comparison
  - Test few-shot settings and generation token limits
  - Ensure graceful handling of lm-eval failures

### Environment setup

`requirements.txt` does not pin a Torch/CUDA stack — it is just `-e .` (see [Installation](../getting-started/installation.md) for the real extras-based install flow):

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd circuitkit
pip install -e ".[dev,docs]"   # or "[dev,docs,gpu-cu126]" for GPU
```

There is no `scripts/validate_environment.py` and no `dlbactrace` dependency in this project. Validate your install with:

```bash
python -c "import circuitkit; print(circuitkit.__version__)"
circuitkit --help
```

#### Lightning / Server / Colab validation
- Lightning/Server
  - Create a fresh virtualenv and run the install + verification commands above
  - Execute `python examples/01-quickstart.py` and one EAP-IG example with `batch_size=2`
- Colab
  - Use the notebooks in `examples/notebooks/` (e.g. `00_colab_setup.ipynb`, `01_quickstart_pipeline.ipynb`) — they are numbered, not prefixed `Colab_`. The setup notebook installs CircuitKit and its dependencies.
  - If GPU is unavailable on your Colab runtime, the CPU install path works for GPT-2-scale examples.

### Sample visualizations

There is no `docs/illustrations/` directory. Circuit visualizations are produced via the Python API (`ck.visualize_circuit()` / `Pipeline.visualize()`) — see [Visualization](../user-guide/visualization.md).

### Colab notebooks

Notebooks are available under `examples/notebooks/`, numbered `00`–`08` (e.g. `00_colab_setup.ipynb`, `01_quickstart_pipeline.ipynb`, `02_algorithm_comparison.ipynb`) — see [Notebooks](../examples/notebooks.md) for the full list. There is no `Colab_EAP_IG_Node.ipynb` / `Colab_EAP_Neuron.ipynb` / `Colab_ACDC_Node.ipynb`.

Each notebook contains: environment setup, a minimal example config, a short discovery run on `gpt2`, and pointers to evaluation.

### LM Evaluation Testing

Test the LM evaluation functionality with these commands:

```bash
# Test basic LM evaluation (enabled by default)
circuitkit evaluate --model gpt2 --artifact results.pt --num-examples 5

# Test the faithfulness report path
circuitkit evaluate --model gpt2 --artifact results.pt \
  --num-examples 5 \
  --report-path ./report.json

# Test API configuration
python -c "
from circuitkit.api import evaluate_circuit
results = evaluate_circuit('results.pt', pruned_artifact_path='results.pt')
print(results)
"
```

There are no `--enable-lm-eval` / `--disable-lm-eval` / `--lm-eval-*` flags on `evaluate`, and `evaluate_circuit` does not take a `qa_params` positional argument — it takes `config` (path or dict) plus optional `pruned_artifact_path` and `scores_path`. Downstream lm-eval-harness benchmarking is tested separately via `ck.benchmark()` / `circuitkit benchmark` (see [CLI Reference](../cli/applications.md)).

Expected behavior:
- Evaluation runs without errors and produces a `FaithfulnessReport`
- CLI options correctly configure the evaluation
- `--report-path` writes a JSON report if specified

### Reporting issues
Please capture the following in bug reports:
- Environment (OS, Python, CUDA, GPU)
- Exact command/config (CLI flags or YAML)
- Logs and stack traces
- If memory-related, include output of `discover-smart --check-memory`



