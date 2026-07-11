# Backends

**Module**: `circuitkit.backends`

The backends module provides the stability tier registry, algorithm enumeration, and the tier-query helpers. It is the single source of truth for which algorithms are production-ready.

---

## Stability Tiers

CircuitKit ships **13 discovery algorithms** across 4 backends, each with an explicit stability tier. Only 2 are validated at production scale (`eap`, `eap-ig`); `acdc` and `ibcircuit` are experimental (GPT-2 scale; `ibcircuit` OOMs above ~3B) and the other 9 are research (GPT-2 IOI only).

| Tier | Algorithms |
|------|-----------|
| **Stable** | `eap`, `eap-ig` |
| **Experimental** | `acdc`, `ibcircuit` |
| **Research** | `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`, `eap-ifr`, `cdt` |

---

## Exports

```python
from circuitkit.backends import (
    STABILITY,
    DISCOVERY_ALGORITHMS,
    STABLE_ALGORITHMS,
    EXPERIMENTAL_ALGORITHMS,
    RESEARCH_ALGORITHMS,
    is_stable,
    is_experimental,
    is_research,
    default_algorithm,
)
```

### `STABILITY`

`Dict[str, str]` — maps algorithm/selector name → tier string (`"stable"`, `"experimental"`, `"research"`).

!!! note
    `STABILITY` includes both the 13 discovery algorithms AND the 8 compression selector keys (`random`, `magnitude`, `taylor`, `wanda`, `multi_granular`, `gptq`, `awq`, `tacq`). Use `DISCOVERY_ALGORITHMS` when you want only the discovery algorithms.

```python
from circuitkit.backends import STABILITY

print(STABILITY["eap-ig"])   # "stable"
print(STABILITY["acdc"])     # "experimental"
print(STABILITY["cdt"])      # "research"
```

### `DISCOVERY_ALGORITHMS`

`frozenset[str]` — exactly the 13 discovery algorithm names.

```python
from circuitkit.backends import DISCOVERY_ALGORITHMS

print(sorted(DISCOVERY_ALGORITHMS))
# ['acdc', 'atp-gd', 'cdt', 'eap', 'eap-clean-corrupted', 'eap-exact',
#  'eap-gp', 'eap-ifr', 'eap-ig', 'eap-ig-activations', 'ibcircuit', 'peap', 'relp']
```

### `STABLE_ALGORITHMS`, `EXPERIMENTAL_ALGORITHMS`, `RESEARCH_ALGORITHMS`

Tier subsets derived from the full `ALGORITHMS`/`STABILITY` map (not just `DISCOVERY_ALGORITHMS`), so `STABLE_ALGORITHMS` also includes the stable-tier compression selectors.

```python
from circuitkit.backends import STABLE_ALGORITHMS, EXPERIMENTAL_ALGORITHMS

print(sorted(STABLE_ALGORITHMS))
# ['awq', 'eap', 'eap-ig', 'gptq', 'magnitude',
#  'multi_granular', 'random', 'tacq', 'taylor', 'wanda']
```

---

## Functions

### `is_stable(algo: str) -> bool`

```python
from circuitkit.backends import is_stable

is_stable("eap-ig")   # True
is_stable("acdc")     # False
```

### `is_experimental(algo: str) -> bool`

```python
from circuitkit.backends import is_experimental
is_experimental("acdc")     # True
is_experimental("eap-ig")   # False
```

### `is_research(algo: str) -> bool`

```python
from circuitkit.backends import is_research
is_research("cdt")      # True
is_research("eap-ig")   # False
```

### `default_algorithm() -> str`

Returns the default discovery algorithm (`"eap-ig"`).

```python
from circuitkit.backends import default_algorithm
default_algorithm()   # "eap-ig"
```

---

## Tier Warnings

`discover_circuit` automatically emits a `UserWarning` when you request a non-stable algorithm:

```text
UserWarning: Algorithm 'acdc' is experimental. May fail on larger models or non-IOI tasks. Use 'eap-ig' for production.
```

Research-tier algorithms get a similar warning:

```text
UserWarning: Algorithm 'cdt' is research-quality (only validated on GPT-2 IOI). Use 'eap-ig' for production.
```

To suppress:

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="circuitkit")
```

---

## Backend Modules

Each backend is also directly importable if you need lower-level access:

| Backend | Module |
|---------|--------|
| EAP family | `circuitkit.backends.eap` |
| ACDC | `circuitkit.backends.acdc` |
| IBCircuit | `circuitkit.backends.ibcircuit` |
| CD-T | `circuitkit.backends.cdt` |

For normal usage, call them through `discover_circuit` rather than directly.

---

## Next Steps

- [Algorithms: Overview](../algorithms/overview.md) — algorithm selection guide
- [Algorithms: Stability Tiers](../algorithms/stability-tiers.md) — tier explanation and flowchart
- [Algorithms: EAP Family](../algorithms/eap.md) — the EAP-family algorithms in depth
