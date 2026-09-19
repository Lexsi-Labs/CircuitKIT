# Backends

**Module**: `circuitkit.backends`

The backends module provides the stability tier registry, algorithm enumeration, and the tier-query helpers. It is the single source of truth for which algorithms are production-ready.

---

## Stability Tiers

CircuitKit ships **13 discovery algorithms** across 4 backends, each with an explicit stability tier. Six are stable and have been tested across the GPT-2, Llama, Gemma, and Qwen families. The other seven are research: implemented and validated on GPT-2/IOI but not yet exercised at scale or across architectures. No discovery algorithm is in the experimental tier at the moment.

| Tier | Algorithms |
|------|-----------|
| **Stable** | `eap`, `eap-ig`, `eap-gp`, `acdc`, `ibcircuit`, `cdt` |
| **Experimental** | none currently |
| **Research** | `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `relp`, `peap`, `eap-ifr` |

Caveats on stable-tier algorithms:

- `acdc` is node-only by construction (edge search) and slow.
- `ibcircuit` has a memory ceiling on multi-billion-parameter models at aggressive settings.
- `cdt` works from clean inputs only. It uses a frozen-RoPE attention approximation (Q/K are not decomposed) and a 50/50 gated-MLP cross-term split, so its scores on RoPE models are approximate.

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
print(STABILITY["acdc"])     # "stable"
print(STABILITY["relp"])     # "research"
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
# ['acdc', 'awq', 'cdt', 'eap', 'eap-gp', 'eap-ig', 'gptq', 'ibcircuit',
#  'magnitude', 'multi_granular', 'random', 'tacq', 'taylor', 'wanda']

print(sorted(EXPERIMENTAL_ALGORITHMS))
# []
```

---

## Functions

### `is_stable(algo: str) -> bool`

```python
from circuitkit.backends import is_stable

is_stable("eap-ig")   # True
is_stable("relp")     # False
```

### `is_experimental(algo: str) -> bool`

```python
from circuitkit.backends import is_experimental
is_experimental("acdc")     # False (no discovery algorithm is experimental currently)
is_experimental("eap-ig")   # False
```

### `is_research(algo: str) -> bool`

```python
from circuitkit.backends import is_research
is_research("relp")     # True
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
UserWarning: Algorithm 'relp' is research-quality (only validated on GPT-2 IOI). Use 'eap-ig' for production.
```

Experimental-tier algorithms get a similar warning. No discovery algorithm is in that tier currently.

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
