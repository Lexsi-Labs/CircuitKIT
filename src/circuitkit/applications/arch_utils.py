"""
Architecture Detection & Validation Utilities

Helper functions for detecting model architecture, validating layer paths,
and accessing architecture-specific layer modules.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from .arch_registry import MODEL_ARCH_REGISTRY, SUPPORTED_MODELS, get_model_family

logger = logging.getLogger(__name__)


class UnsupportedArchitectureError(Exception):
    """Raised when model architecture is not supported."""


class ArchitectureValidationError(Exception):
    """Raised when model layers don't match expected architecture."""


def detect_model_architecture(hf_model: nn.Module) -> str:
    """
    Auto-detect model architecture family from HuggingFace model config.

    Maps model.config.model_type to an architecture family key in
    MODEL_ARCH_REGISTRY (e.g., "llama", "qwen", "gemma").

    Args:
        hf_model: HuggingFace model instance

    Returns:
        str: Architecture family key (e.g., "llama", "gemma", "gpt2")

    Raises:
        UnsupportedArchitectureError: If model type not in registry
    """
    model_type = hf_model.config.model_type

    try:
        family = get_model_family(model_type)
        logger.info(f"Detected architecture: {model_type} → {family} family")
        return family
    except KeyError as e:
        raise UnsupportedArchitectureError(
            f"\n{'='*70}\n"
            f"Model type '{model_type}' is not yet supported in CircuitKit.\n\n"
            f"Supported model types: {', '.join(SUPPORTED_MODELS)}\n\n"
            f"To add support for '{model_type}':\n"
            f"  1. Identify which architecture family it belongs to\n"
            f"  2. Add '{model_type}' to the 'models' list for that family\n"
            f"     in MODEL_ARCH_REGISTRY (arch_registry.py)\n"
            f"  3. Or create a new family entry if it's a new architecture\n"
            f"  4. Run tests to validate\n"
            f"{'='*70}\n"
        ) from e


def get_arch_config(model_type: str) -> Dict[str, Any]:
    """
    Get architecture configuration for a model type.

    Args:
        model_type: Model type string (e.g., "llama")

    Returns:
        dict: Architecture configuration

    Raises:
        UnsupportedArchitectureError: If model type not supported
    """
    if model_type not in MODEL_ARCH_REGISTRY:
        raise UnsupportedArchitectureError(f"Model type '{model_type}' not supported")
    return MODEL_ARCH_REGISTRY[model_type]


def getattr_recursive(obj: Any, path: str) -> Any:
    """
    Get nested attribute using dot-separated path.

    Args:
        obj: Object to traverse
        path: Dot-separated path (e.g., "model.layers")

    Returns:
        Nested attribute value

    Raises:
        AttributeError: If path doesn't exist
    """
    attrs = path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def hasattr_recursive(obj: Any, path: str) -> bool:
    """
    Check if nested attribute exists using dot-separated path.

    Args:
        obj: Object to check
        path: Dot-separated path (e.g., "model.layers")

    Returns:
        bool: True if path exists, False otherwise
    """
    try:
        getattr_recursive(obj, path)
        return True
    except AttributeError:
        return False


def validate_model_paths(hf_model: nn.Module, arch_cfg: Dict[str, Any]) -> None:
    """
    Validate that model has expected layer paths from architecture.

    Args:
        hf_model: HuggingFace model
        arch_cfg: Architecture configuration dict

    Raises:
        ArchitectureValidationError: If required paths not found
    """
    missing_paths = []

    for path in arch_cfg["layers_path"]:
        if not hasattr_recursive(hf_model, path):
            missing_paths.append(path)

    if missing_paths:
        raise ArchitectureValidationError(
            f"\nModel {hf_model.config.model_type} is missing expected layer paths:\n"
            f"  Missing: {missing_paths}\n"
            f"  Expected: {arch_cfg['layers_path']}\n"
            f"This may indicate a model variant with different structure.\n"
            f"Please verify model config and architecture.\n"
        )

    logger.info(f"✓ Validated {hf_model.config.model_type} architecture")


def get_layers(hf_model: nn.Module, arch_cfg: Dict[str, Any]) -> nn.ModuleList:
    """
    Get the layers module from model using architecture config.

    Args:
        hf_model: HuggingFace model
        arch_cfg: Architecture configuration dict

    Returns:
        nn.ModuleList: The layers module

    Raises:
        ArchitectureValidationError: If layers not found
    """
    for path in arch_cfg["layers_path"]:
        if hasattr_recursive(hf_model, path):
            layers = getattr_recursive(hf_model, path)
            logger.info(f"✓ Found layers at path '{path}'")
            return layers

    raise ArchitectureValidationError(
        f"Could not find layers module for {hf_model.config.model_type}\n"
        f"Expected one of: {arch_cfg['layers_path']}"
    )


def get_attn_proj(
    layer: nn.Module, arch_cfg: Dict[str, Any], proj_name: str = "k_proj"
) -> nn.Module:
    """
    Get attention projection module from layer.

    Args:
        layer: Model layer module
        arch_cfg: Architecture configuration dict
        proj_name: Projection name ("k_proj", "v_proj", "q_proj", "o_proj", "c_attn", etc.)

    Returns:
        nn.Module: The projection module

    Raises:
        AttributeError: If projection not found
    """
    attn_cfg = arch_cfg["attn"]
    attn_module_name = attn_cfg.get("module")

    if attn_module_name:
        attn_module = getattr(layer, attn_module_name)
        if proj_name in attn_cfg:
            proj_path = attn_cfg[proj_name]
            return getattr(attn_module, proj_path)
        else:
            raise KeyError(f"Projection '{proj_name}' not in architecture config")
    else:
        raise ValueError("No attn.module specified in architecture config")


def get_mlp_proj(
    layer: nn.Module, arch_cfg: Dict[str, Any], proj_name: str = "gate_proj"
) -> Optional[nn.Module]:
    """
    Get MLP projection module from layer.

    Args:
        layer: Model layer module
        arch_cfg: Architecture configuration dict
        proj_name: Projection name ("gate_proj", "up_proj", "down_proj", etc.)

    Returns:
        nn.Module: The projection module, or None if not found

    Raises:
        AttributeError: If projection path incorrect
    """
    mlp_cfg = arch_cfg.get("mlp", {})

    # Some models don't nest MLP, some do
    for mlp_module_name in ["mlp", None]:
        if mlp_module_name:
            if hasattr(layer, mlp_module_name):
                mlp_module = getattr(layer, mlp_module_name)
            else:
                continue
        else:
            mlp_module = layer

        if proj_name in mlp_cfg:
            proj_path = mlp_cfg[proj_name]
            if hasattr(mlp_module, proj_path):
                return getattr(mlp_module, proj_path)

    return None


def get_head_dim(
    layer: nn.Module, arch_cfg: Dict[str, Any], num_heads: Optional[int] = None
) -> int:
    """
    Get attention head dimension for a layer.

    Args:
        layer: Model layer module
        arch_cfg: Architecture configuration dict
        num_heads: Total number of attention heads (for fallback calculation)

    Returns:
        int: Head dimension

    Raises:
        ValueError: If head_dim cannot be determined
    """
    attn_cfg = arch_cfg["attn"]
    head_dim_attr = attn_cfg.get("head_dim")

    if head_dim_attr:
        # Direct attribute path
        try:
            attn_module = getattr(layer, attn_cfg["module"])
            return int(getattr(attn_module, head_dim_attr))
        except AttributeError:
            pass

    # Fallback: compute from config
    if num_heads and hasattr(layer, "self_attn"):
        hidden_dim = layer.self_attn.q_proj.out_features
        return hidden_dim // num_heads

    raise ValueError(
        f"Cannot determine head_dim for layer. " f"Expected attribute: {head_dim_attr}"
    )


def validate_architecture_support(model_type: str) -> Tuple[bool, str]:
    """
    Check architecture support level and return status message.

    Args:
        model_type: Model architecture name

    Returns:
        tuple: (is_supported, status_message)
    """
    if model_type not in MODEL_ARCH_REGISTRY:
        return False, f"Architecture '{model_type}' not supported"

    arch_cfg = MODEL_ARCH_REGISTRY[model_type]
    status = arch_cfg.get("status", "UNKNOWN")

    messages = {
        "PRODUCTION": f"✓ {model_type} fully supported and tested",
        "READY": f"⚠ {model_type} ready but not extensively tested",
        "NOT_STARTED": f"✗ {model_type} not yet implemented",
    }

    return True, messages.get(status, f"? {model_type} - unknown status")
