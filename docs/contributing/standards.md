# Code Standards

---

## Python Style

- **Python 3.10+** with full type annotations on all public functions.
- **black** for formatting, **isort** for import ordering, **flake8** for linting. There is no Ruff dependency in this project.
- **Line length**: 100 characters.

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

---

## Type Annotations

All public functions must have complete type annotations:

```python
# Good
def discover(
    model: HookedTransformer,
    task: str,
    *,
    algorithm: str = "eap-ig",
    n_examples: int = 128,
) -> Circuit:
    ...

# Bad — missing annotations
def discover(model, task, algorithm="eap-ig"):
    ...
```

Run `mypy` before submitting:

```bash
mypy src/circuitkit --ignore-missing-imports
```

---

## Docstrings

Public functions get Google-style docstrings. Private helpers don't need them.

```python
def discover(model: HookedTransformer, task: str, *, algorithm: str = "eap-ig") -> Circuit:
    """Run circuit discovery and return a Circuit.

    Args:
        model: A configured HookedTransformer from load_model.
        task: Registered task name, e.g. "ioi", "mmlu".
        algorithm: Discovery algorithm. Defaults to "eap-ig".

    Returns:
        A Circuit wrapping the discovered nodes and their scores.

    Raises:
        ValueError: If algorithm is not a known discovery algorithm.
    """
```

---

## Testing

- **Pytest** with markers: `@pytest.mark.slow`, `@pytest.mark.integration`.
- All new functions need at least one test in `tests/`.
- Tests that require a model load must use `@pytest.mark.integration` or mock the model.
- Target: ≥80% coverage for new code.

```python
import pytest

@pytest.mark.integration
def test_discover_returns_circuit():
    import circuitkit as ck
    model = ck.load_model("gpt2", dtype="float32")
    circuit = ck.discover(model, "ioi", n_examples=8)
    assert len(circuit) > 0
```

---

## Commit Conventions

Conventional Commits format:

```bash
type(scope): short description

Body (optional): explain the WHY, not the WHAT.

Co-Authored-By: Name <email>
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

**Examples**:
```bash
feat(selectors): add Taylor selector with gradient-product weighting
fix(backends): resolve OOM in IBCircuit for >3B parameter models
docs(user-guide): add winogrande metric details to tasks page
test(evaluation): add integration test for Pillar 3 stability
```

---

## Pull Request Process

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Write code + tests + docs
3. Run `black`, `isort`, `flake8`, `mypy`, `pytest -m "not slow"` locally — there is no CI job that runs these for you yet, only a docs-build workflow
4. Open a PR against `main`
5. CI runs: docs build only (`.github/workflows/docs.yml`)
6. Wait for review

---

## Deprecation Policy

For deprecating public API:

1. Add a `DeprecationWarning` in the function: `warnings.warn("...", DeprecationWarning, stacklevel=2)`
2. Keep the old function for one minor version
3. Remove it in the next minor version
4. Document the change in `docs/about/release-notes.md`

---

## Next Steps

- [Development Setup](setup.md) — environment and test setup
- [Adding Algorithms](new-algorithms.md) — how to contribute a new discovery backend
