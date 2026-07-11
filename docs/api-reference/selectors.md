# Selectors API

**Module**: `circuitkit.selection`

Selectors are named callables that compute per-node importance scores. CircuitKit ships 14 registered selectors — 6 for circuit discovery, 7 for compression (pruning/quantization), plus a `random` baseline.

---

## Registry Functions

### `list_selectors`

```python
list_selectors() -> List[str]
```

Return all registered selector names.

```python
from circuitkit.selection import list_selectors
print(list_selectors())
# ['awq', 'cdt', 'eap', 'eap-gp', 'eap-ig', 'gptq', 'ibcircuit',
#  'magnitude', 'multi_granular', 'random', 'relp', 'tacq', 'taylor', 'wanda']
```

### `get_selector`

```python
get_selector(name: str) -> Callable
```

Return a registered selector callable.

```python
from circuitkit.selection import get_selector

selector = get_selector("eap-ig")
scores = selector(model, "ioi", config)
```

### `register` (decorator)

```python
from circuitkit.selection import register

@register("my_selector")
def my_selector(model, task_name: str, config: dict) -> dict:
    # Returns {node_name: importance_score} dict
    ...
    return {"A0.1": 0.9, "MLP 3": 0.5}
```

---

## All 14 Selectors

| Selector | Category | Description |
|----------|----------|-------------|
| `eap` | Discovery | Edge Attribution Patching (stable) |
| `eap-ig` | Discovery | EAP with Integrated Gradients (stable, default) |
| `eap-gp` | Discovery | EAP-GP / GradPath — adaptive integration path (research) |
| `ibcircuit` | Discovery | Information Bottleneck Circuit (experimental) |
| `cdt` | Discovery | Contextual Decomposition for Transformers (research) |
| `relp` | Discovery | Relevance Patching via LRP-style hooks (research) |
| `random` | Baseline | Uniform random scores |
| `magnitude` | Compression | L2 norm of weights |
| `taylor` | Compression | First-order Taylor expansion |
| `wanda` | Compression | Weight × Activation magnitude |
| `gptq` | Compression | GPTQ quantization-style |
| `awq` | Compression | Activation-aware Weight Quantization |
| `tacq` | Compression | Task-Circuit Quantization |
| `multi_granular` | Compression | Multi-granular selector (head + neuron) |

ACDC is not a registered selector. It lives in `DISCOVERY_ALGORITHMS` and is invoked as `algorithm="acdc"` in the discovery config, not via the selector registry.

---

## Selector Function Signature

All selectors share the same call signature:

```python
def selector(model: HookedTransformer, task_name: str, config: dict) -> Dict[str, float]:
```

- `model` — loaded `HookedTransformer`
- `task_name` — registered task name
- `config` — discovery config dict (same shape as `discover_circuit`)
- **Returns** — `{node_name: importance_score}` where node names follow the convention:
  - Attention heads: `A{layer}.{head}` (e.g. `"A0.1"`, `"A11.5"`)
  - MLP layers: `MLP {layer}` (e.g. `"MLP 5"`)

---

## Selector vs Algorithm vs Backend

These three terms refer to the same underlying code at different levels of abstraction:

| Term | Scope | Example |
|------|-------|---------|
| **Backend** | Full module (discovery + output) | `circuitkit.backends.eap` |
| **Algorithm** | Named method in `discover_circuit` config | `"eap-ig"` |
| **Selector** | Named callable in `circuitkit.selection` | `get_selector("eap-ig")` |

The selector registry is the thin wrapper that makes algorithms addressable by name. For compression selectors (`magnitude`, `taylor`, etc.), there is no corresponding discovery backend — they operate on weight magnitudes directly.

---

## Writing a Custom Selector

```python
from circuitkit.selection import register

@register("my_gradient_selector")
def my_gradient_selector(model, task_name: str, config: dict) -> dict:
    from circuitkit.tasks import get_task

    task = get_task(task_name)
    # ... compute importance scores using model and task ...
    scores = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            scores[f"A{layer}.{head}"] = float(compute_head_score(model, layer, head))
        scores[f"MLP {layer}"] = float(compute_mlp_score(model, layer))
    return scores
```

Registering a selector does **not** make it usable as `discover_circuit`'s `algorithm` value — that dispatch is a fixed set of branches limited to `DISCOVERY_ALGORITHMS` and does not consult this registry. Call the registered selector directly instead:

```python
from circuitkit.selection import get_selector

selector_fn = get_selector("my_gradient_selector")
scores = selector_fn(model, "ioi", {"level": "node"})
```

---

## Next Steps

- [User Guide: Selectors](../user-guide/selectors.md) — all 14 selectors with use-case guidance
- [Algorithms: Overview](../algorithms/overview.md) — algorithm selection flowchart
- [Backends](backends.md) — stability tiers for discovery algorithms
