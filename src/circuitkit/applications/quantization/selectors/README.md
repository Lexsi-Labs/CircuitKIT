# selectors

Pluggable selectors that score components to guide mixed-precision quantization.

## Key modules

- `awq_selector.py` — `awq_selector`: AWQ-derived activation-salience scoring, ranking each component by the mean absolute magnitude of its output activations.
- `tacq_selector.py` — `tacq_selector`: TaCQ-style task-circuit quantization scoring, using the absolute weight delta times weight for a positive-definite score (with a 4-bit quantization simulation helper).

## Public API / entry points

Each module exposes a registered selector function; there is no `__all__` (`__init__.py` is an empty namespace file). Selectors are resolved through the registry rather than imported by name.

## How it fits

Provides the scoring strategies the `quantization/` package uses to assign per-layer precision tiers.
