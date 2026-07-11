"""
CircuitKit Applications — model surgery and deployment.

Public API (v1.0):
  pruning/            — weight removal, structural pruning, selectors
  quantization/       — mixed-precision quantization, selectors
  selective_finetuning/ — circuit-guided selective finetuning
  common_utils/             — linear probe, hallucination detection, unlearning, benchmarks

Available under circuitkit.applications.* (not part of the flat public API):
  steering/           — activation and weight steering
  editing/            — knowledge editing (ROME, MEMIT, circuit-guided)
  finetuning/         — LoRA healing, circuit tuning, PEFT
"""

# Subdirectory imports
from . import common_utils, pruning, quantization

try:
    from . import selective_finetuning  # noqa: F401
except ImportError:
    pass
# steering, editing, finetuning — usable via circuitkit.applications.*, not part of the flat public API

# Architecture registry and utilities
from .arch_registry import (
    MODEL_ARCH_REGISTRY,
    PRODUCTION_FAMILIES,
    READY_FAMILIES,
    SUPPORTED_FAMILIES,
    SUPPORTED_MODELS,
    get_model_family,
)
from .arch_utils import (
    ArchitectureValidationError,
    UnsupportedArchitectureError,
    detect_model_architecture,
    get_arch_config,
    get_attn_proj,
    get_head_dim,
    get_layers,
    get_mlp_proj,
    validate_model_paths,
)

# Convenience re-export: the structural pruner module is addressed both as
# ``circuitkit.applications.pruning.pruner`` and as ``circuitkit.applications.pruner``.
from .pruning import pruner

__all__ = [
    "MODEL_ARCH_REGISTRY",
    "SUPPORTED_MODELS",
    "SUPPORTED_FAMILIES",
    "PRODUCTION_FAMILIES",
    "READY_FAMILIES",
    "get_model_family",
    "detect_model_architecture",
    "get_arch_config",
    "validate_model_paths",
    "get_layers",
    "get_attn_proj",
    "get_mlp_proj",
    "get_head_dim",
    "UnsupportedArchitectureError",
    "ArchitectureValidationError",
    "pruning",
    "quantization",
    "common_utils",
    "pruner",
]