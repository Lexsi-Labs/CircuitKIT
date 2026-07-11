# Adding Discovery Algorithms

This guide shows how to add a new discovery algorithm backend to CircuitKit.

---

## Architecture Overview

A discovery backend is a Python module that:
1. Accepts a loaded model, task spec, and config
2. Computes per-node importance scores
3. Returns them in a format the pruning stage can consume

Backends live in `src/circuitkit/backends/`. The stability map in `src/circuitkit/backends/__init__.py` is the single source of truth for algorithm names and tiers.

---

## Step 1: Create the Backend Module

Create `src/circuitkit/backends/myalgo/`:

```bash
src/circuitkit/backends/myalgo/
├── __init__.py
├── adapter.py      # main entry point
└── core.py         # algorithm implementation
```

### `adapter.py`

```python
"""MyAlgo discovery backend adapter."""

from __future__ import annotations
from typing import Any, Dict, Union, List

from transformer_lens import HookedTransformer


def run_myalgo_discovery(
    model: HookedTransformer,
    task_spec,
    config: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, float]:
    """Run MyAlgo discovery and return per-node importance scores.

    Args:
        model: Loaded HookedTransformer with hook flags set.
        task_spec: Task specification from the task registry.
        config: Discovery config block (from the full circuit config dict).
        device: Target device.

    Returns:
        Dict mapping node names to importance scores: {"A0.1": 0.83, "MLP 5": 0.61}
    """
    from .core import MyAlgoCore

    n_examples = config.get("data_params", {}).get("num_examples", 128)
    level = config.get("level", "node")

    algo = MyAlgoCore(model, device=device)
    scores = algo.compute_scores(task_spec, n_examples=n_examples, level=level)

    return scores
```

### `core.py`

Implement your algorithm logic here. The key invariant: return `Dict[str, float]` where keys are node names in CircuitKit's convention:
- Attention heads: `"A{layer}.{head}"` — e.g. `"A0.1"`, `"A11.5"`
- MLP layers: `"MLP {layer}"` — e.g. `"MLP 5"`

---

## Step 2: Wire It Into `discover_circuit`

There is no `_BACKEND_DISPATCH` table. `discover_circuit` (in `src/circuitkit/api.py`) dispatches on the `algorithm` string via a sequence of `if`/`elif` branches — add yours alongside the existing ones:

```python
# In discover_circuit(), alongside the existing algorithm branches
elif algo == "myalgo":
    from .backends.myalgo.adapter import run_myalgo_discovery
    node_scores = run_myalgo_discovery(model, task_spec, discovery_cfg, device=device)
```

Unknown `algorithm` values raise `AlgorithmError` listing `DISCOVERY_ALGORITHMS` — adding the branch above (plus the `DISCOVERY_ALGORITHMS` entry in Step 3) is what makes `"myalgo"` a valid value.

---

## Step 3: Add to the `ALGORITHMS` Registry

In `src/circuitkit/backends/__init__.py`, add one entry to `ALGORITHMS`. This dict maps each name to a `(category, stability)` tuple and is the single source of truth. `STABILITY`, `DISCOVERY_ALGORITHMS`, and the other registries are derived from it, so do not edit them directly — they are rebuilt from `ALGORITHMS` on import:

```python
ALGORITHMS: dict[str, tuple[str, str]] = {
    # ── Discovery: EAP family ──
    "eap": ("discovery", "stable"),
    "eap-ig": ("discovery", "stable"),
    # ...
    # Your new algorithm — category "discovery", tier "experimental"
    # (or "research" until validated)
    "myalgo": ("discovery", "experimental"),
}
```

Because you gave it the `"discovery"` category, `myalgo` is automatically included in the derived `DISCOVERY_ALGORITHMS` frozenset (and its `"experimental"` tier flows into `STABILITY` and `EXPERIMENTAL_ALGORITHMS`).

---

## Step 4: Register as a Selector (optional, for direct/manual use)

Optionally register your algorithm in the selector registry (`src/circuitkit/selection/`) so it can be invoked directly via `get_selector("myalgo")`, independent of `discover_circuit`:

```python
# src/circuitkit/selection/myalgo_selector.py
from circuitkit.selection import register
from circuitkit.backends.myalgo.adapter import run_myalgo_discovery

@register("myalgo")
def myalgo_selector(model, task_name: str, config: dict) -> dict:
    from circuitkit.tasks import get_task
    task_spec = get_task(task_name)
    return run_myalgo_discovery(model, task_spec, config)
```

```python
# src/circuitkit/selection/__init__.py — add the import so registration runs on package import
from . import myalgo_selector  # noqa: F401
```

This registry entry is independent of Step 2 — `discover_circuit`'s `algorithm` dispatch does **not** read from it. Registering here only makes `get_selector("myalgo")` work for direct invocation; it does not, by itself, make `"myalgo"` a valid `discover_circuit` algorithm value.

---

## Step 5: Add Tests

Create `tests/test_myalgo.py`:

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_myalgo_returns_scores():
    import circuitkit as ck

    model = ck.load_model("gpt2", dtype="float32")
    circuit = ck.discover(model, "ioi", algorithm="myalgo", n_examples=8)

    assert len(circuit) > 0
    assert all(isinstance(v, float) for v in circuit.scores.values())

def test_myalgo_in_discovery_algorithms():
    from circuitkit.backends import DISCOVERY_ALGORITHMS, STABILITY
    assert "myalgo" in DISCOVERY_ALGORITHMS
    assert "myalgo" in STABILITY
```

---

## Step 6: Documentation

1. Add a row to the algorithms table in `docs/algorithms/overview.md`
2. Add an entry to `docs/algorithms/stability-tiers.md`
3. Create `docs/algorithms/myalgo.md` (optional, for complex algorithms)

---

## Stability Promotion

Algorithms start at **Research** tier. Promote to:

- **Experimental**: validated on GPT-2 IOI, basic OOM/crash testing done
- **Stable**: validated on GPT-2, Llama, and Gemma; tested at 1B+ scale; CI-passing

Submit a PR with the tier change in `STABILITY` and evidence of validation.

---

## Next Steps

- [Code Standards](standards.md) — style requirements
- [Algorithms: Overview](../algorithms/overview.md) — where your algorithm will appear
- [API Reference: Backends](../api-reference/backends.md) — STABILITY and DISCOVERY_ALGORITHMS
