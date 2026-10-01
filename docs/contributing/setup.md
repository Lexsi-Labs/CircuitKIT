# Development Setup

---

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ or 12.x (for GPU testing)
- Git

---

## Clone and Install

```bash
git clone https://github.com/Lexsi-Labs/circuitkit.git
cd circuitkit

# Development install (CPU/default torch)
pip install -e ".[dev,docs]"
```

For GPU development:

```bash
pip install -e ".[dev,docs,gpu-cu126]"
```

The `dev` extra installs `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`, and `pre-commit`. There is no `ruff` dependency — CircuitKIT's toolchain is black + isort + flake8 + mypy. The `docs` extra installs MkDocs and plugins.

---

## Verify Installation

```bash
python -c "import circuitkit; print(circuitkit.__version__)"
circuitkit --help
```

---

## Running Tests

```bash
# All tests
pytest tests/

# Fast tests only (skip slow tests)
pytest tests/ -m "not slow"

# Single module
pytest tests/unit/test_selector.py -v

# With coverage
pytest tests/ --cov=src/circuitkit --cov-report=term-missing
```

Test markers:
- `@pytest.mark.slow` — takes >30 seconds
- `@pytest.mark.integration` — requires full model load

---

## Linting and Type Checking

Ruff is not part of this toolchain — CircuitKIT uses black, isort, flake8, and mypy:

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/circuitkit --ignore-missing-imports
```

There is currently no CI job that runs lint, type-check, or tests — the only GitHub Actions workflow (`.github/workflows/docs.yml`) builds and deploys the documentation site. Run the commands above locally before submitting a PR.

---

## Building the Docs

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally (auto-reloads)
mkdocs serve

# Build static site
mkdocs build

# Strict build (fails on warnings — CI standard)
mkdocs build --strict
```

---

## Project Structure

```bash
circuitkit/
├── src/circuitkit/         # Main package source
│   ├── api.py              # discover_circuit, evaluate_circuit, load_circuit
│   ├── quick.py            # Flat typed API (ck.discover, ck.prune, ...)
│   ├── pipeline.py         # Pipeline class
│   ├── backends/           # Discovery algorithm backends
│   ├── evaluation/         # 6-pillar faithfulness framework
│   ├── tasks/              # Task registry and built-in tasks
│   ├── selection/          # Selector registry
│   ├── applications/       # Pruning, quantization, steering, editing, finetuning
│   └── cli/                # Click CLI commands
├── tests/                  # Pytest test suite
├── examples/               # Runnable Python scripts
├── examples/notebooks/              # Colab notebooks
└── docs/                   # MkDocs documentation source
```

The `experiments/` directory at the repo root holds paper-resubmission scripts and is not part of the installable `circuitkit` package.

---

## Common Dev Workflows

### Adding a new selector

1. Create the selector function in `src/circuitkit/selection/`
2. Register it with `@register("my_selector")`
3. Add it to `ALGORITHMS` in `src/circuitkit/backends/__init__.py` (the `STABILITY` map is derived from `ALGORITHMS` automatically — don't edit it directly)
4. Add tests in `tests/unit/test_selector.py`
5. Document it in `docs/user-guide/selectors.md`

### Adding a new task

1. Create the task spec in `src/circuitkit/tasks/builtins/`
2. Register it in `_bootstrap_builtin_tasks()` in `src/circuitkit/tasks/bootstrap.py` (the single source of truth for built-in task registration)
3. Add it to the 16-task table in `docs/user-guide/tasks.md`

---

## Next Steps

- [Code Standards](standards.md) — style guide and commit conventions
- [Adding Algorithms](new-algorithms.md) — how to add a discovery backend
- [Documentation Guide](documentation.md) — contributing to these docs
