# selectors

A registry of component-scoring functions (attribution methods and pruning/quantization baselines) that map a loaded model to per-component importance scores.

## Key modules

- `__init__.py` — the registry: `register` decorator, `get_selector`, `list_selectors`, and imports that register every built-in selector (including application pruning/quantization selectors).
- `eap_selector.py` — registers `eap` and `eap-ig` attribution via CircuitKit's `Graph` + `attribute_node`.
- `eap_gp_selector.py` — registers `eap-gp` (GradPath adaptive integration path).
- `relp_selector.py` — registers `relp` (Relevance Patching via LRP-style detach hooks).
- `cdt_selector.py` — registers `cdt` (Contextual Decomposition for Transformers, gradient-free).
- `ibcircuit_selector.py` — registers `ibcircuit` (Information-Bottleneck circuit discovery; includes 3B/4B OOM fixes).
- `magnitude_selector.py` — registers `magnitude` (RMS weight-norm baseline).
- `random_selector.py` — registers `random` (seeded random baseline).
- `wanda_selector.py` — registers `wanda` (real per-weight Wanda metric aggregated to component granularity).
- `gptq_selector.py` — registers `gptq` (GPTQ-derived diagonal-Hessian saliency *proxy*, explicitly not the GPTQ algorithm).

## Public API / entry points

`register`, `get_selector(name)`, `list_selectors()`, and the type alias `SelectorFn = Callable[[model, task_name, config], Dict[str, float]]`.

## How it fits

A scoring interface used by both CircuitKit applications (pruning/quantization) and the discovery/experiment framework, so attribution methods and baselines are interchangeable by name.
