# selectors

Pluggable importance selectors that score model components for circuit-guided pruning.

## Key modules

- `taylor_selector.py` — `taylor_selector`: Taylor saliency, using `|grad(W) * W|` at each weight matrix as per-component importance (single-input, gradient*activation).
- `multi_granular_selector.py` — `multi_granular_selector`: multi-granular importance scored at block, head, and neuron levels (single-input, no clean-vs-patched contrast).

## Public API / entry points

Each module exposes a `@register`-decorated selector function; there is no `__all__` (`__init__.py` is an empty namespace file). Selectors are resolved through the registry rather than imported by name.

## How it fits

Provides the scoring strategies that the `pruning/` package uses to decide which components to prune.
