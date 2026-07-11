# applications

The "Intervene" layer of CircuitKit: model surgery and deployment on discovered circuits.

## Key modules

- `arch_registry.py` — `MODEL_ARCH_REGISTRY`, a family-based map from HF model types to layer structure and module names, covering the TransformerLens model set (237+).
- `arch_utils.py` — architecture detection and validation helpers: `detect_model_architecture`, `get_arch_config`, `get_layers`, `get_attn_proj`, `get_mlp_proj`, `get_head_dim`, `validate_model_paths`, plus `UnsupportedArchitectureError` / `ArchitectureValidationError`.

## Public API / entry points

Re-exported from `__init__.py`: the architecture registry constants (`MODEL_ARCH_REGISTRY`, `SUPPORTED_MODELS`, `SUPPORTED_FAMILIES`, `PRODUCTION_FAMILIES`, `READY_FAMILIES`, `get_model_family`); the arch utilities above; and the subpackages `common`, `pruning`, `quantization` (with `selective_finetuning` imported when available). The structural `pruner` module is also re-exported here for convenience.

## Subpackages

- `pruning/`, `quantization/`, `selective_finetuning/`, `common/` — public API (v1.0).
- `steering/`, `editing/`, `finetuning/` — usable via `circuitkit.applications.*`, not part of the flat public API.

## How it fits

This package takes circuits discovered and evaluated elsewhere in CircuitKit and applies them to real models: pruning, quantization, selective finetuning, and related interventions, all sharing the architecture registry/utilities defined here.
