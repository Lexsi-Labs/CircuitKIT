"""
Memory optimization utilities for CircuitKit.
Provides memory-efficient configurations and helpers.
"""

import gc
from typing import Any, Dict

import torch

from circuitkit.utils.logging import get_logger

logger = get_logger(__name__)


def _estimate_model_params(model_name: str) -> int:
    """Estimate parameter count using HuggingFace AutoConfig (no weight loading)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
        n_layers = getattr(cfg, "num_hidden_layers", 12)
        d_model = getattr(cfg, "hidden_size", 768)
        d_ffn = getattr(cfg, "intermediate_size", d_model * 4)
        getattr(cfg, "num_attention_heads", 12)
        vocab_size = getattr(cfg, "vocab_size", 50257)
        # rough estimate: embed + n_layers*(attn + mlp) + lm_head
        n_params = (
            vocab_size * d_model  # embedding
            + n_layers * (4 * d_model * d_model + 2 * d_model * d_ffn)  # attn + mlp
            + vocab_size * d_model  # lm_head
        )
        return n_params
    except Exception:
        return 0


def get_memory_efficient_config(model_name: str, algorithm: str = "eap-ig") -> Dict[str, Any]:
    """Get memory-efficient configuration for any TL-supported model.

    Uses HuggingFace AutoConfig to estimate model size without loading weights,
    so this works for any architecture -- not just GPT-2 / Llama.

    Args:
        model_name: HuggingFace model ID or TransformerLens model name.
        algorithm: Discovery algorithm.

    Returns:
        Memory-optimized configuration dictionary.
    """
    n_params = _estimate_model_params(model_name)
    n_params_b = n_params / 1e9  # billions

    # Scale settings by estimated size: <1B small, 1-10B medium, >10B large
    if n_params_b >= 10:
        precision = "bfloat16"
        batch_size = 1
        ig_steps = 1
        sparsity = 0.05
        mem_opt = {"gradient_checkpointing": True, "low_memory_mode": True, "max_memory_usage": 0.8}
    elif n_params_b >= 1:
        precision = "bfloat16"
        batch_size = 1
        ig_steps = 2
        sparsity = 0.08
        mem_opt = {
            "gradient_checkpointing": False,
            "low_memory_mode": False,
            "max_memory_usage": 0.9,
        }
    else:
        precision = "float32"
        batch_size = 2
        ig_steps = 3
        sparsity = 0.1
        mem_opt = {}

    config: Dict[str, Any] = {
        "model": {"name": model_name, "precision": precision},
        "discovery": {
            "algorithm": algorithm,
            "level": "node",
            "task": "ioi",
            "batch_size": batch_size,
            "ig_steps": ig_steps,
        },
        "pruning": {"target_sparsity": sparsity, "scope": "heads"},
        "batch_size": batch_size,
    }
    if mem_opt:
        config["memory_optimization"] = mem_opt

    logger.info(
        f"Generated memory-efficient config for {model_name} " f"(~{n_params_b:.1f}B params)",
        context={"algorithm": algorithm, "batch_size": batch_size, "precision": precision},
    )
    return config


def optimize_memory_usage():
    """Apply memory optimization settings."""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    # Set memory fraction if needed
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)

    logger.info("Applied memory optimizations")


def get_available_memory() -> Dict[str, float]:
    """Get current memory usage information."""
    memory_info = {}

    if torch.cuda.is_available():
        memory_info["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_info["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info["cuda_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        memory_info["cuda_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        memory_info["cuda_free"] = memory_info["cuda_total"] - memory_info["cuda_allocated"]

    return memory_info


def check_memory_requirements(model_name: str) -> bool:
    """
    Check if there's enough memory for the model.

    Args:
        model_name: Name of the model

    Returns:
        True if sufficient memory, False otherwise
    """
    memory_info = get_available_memory()

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot check memory requirements")
        return False

    # Estimate memory requirements from actual parameter count via HF config.
    # Assume float32 (4 bytes/param) + 2x overhead for gradients/activations.
    n_params = _estimate_model_params(model_name)
    if n_params > 0:
        required_memory = (n_params * 4 / 1024**3) * 2  # float32 * overhead
    else:
        required_memory = 8.0  # conservative default if config unavailable

    available_memory = memory_info.get("cuda_free", 0)

    logger.info(
        "Memory check",
        context={
            "model": model_name,
            "required": f"{required_memory}GB",
            "available": f"{available_memory:.1f}GB",
            "sufficient": available_memory >= required_memory,
        },
    )

    return available_memory >= required_memory


def suggest_alternatives(model_name: str) -> list:
    """
    Suggest alternative models if the current one is too large.

    Args:
        model_name: Name of the model

    Returns:
        List of alternative model suggestions
    """
    model_name_lower = model_name.lower()

    if "llama-3-8b" in model_name_lower or "llama-2-7b" in model_name_lower:
        return ["gpt2", "gpt2-medium", "opt-125m", "opt-350m"]
    elif "llama-3-70b" in model_name_lower or "llama-2-13b" in model_name_lower:
        return ["gpt2-large", "gpt2-xl", "llama-2-7b", "llama-3-8b"]
    elif "gpt2-xl" in model_name_lower:
        return ["gpt2-large", "gpt2-medium", "gpt2"]
    elif "gpt2-large" in model_name_lower:
        return ["gpt2-medium", "gpt2", "opt-350m"]
    else:
        return ["gpt2", "gpt2-medium", "opt-125m"]
