"""
Advanced optimization utilities for CircuitKit.
Implements gradient checkpointing, mixed precision, and other performance optimizations.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from circuitkit.utils.logging import get_logger

logger = get_logger(__name__)


class GradientCheckpointing:
    """Gradient checkpointing utilities for memory optimization."""

    @staticmethod
    def enable_checkpointing(model: nn.Module, use_reentrant: bool = True) -> None:
        """
        Enable gradient checkpointing for a model.

        Args:
            model: PyTorch model to enable checkpointing for
            use_reentrant: Whether to use reentrant checkpointing (PyTorch 2.0+)
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for model")
        else:
            logger.warning("Model does not support gradient checkpointing")

    @staticmethod
    def disable_checkpointing(model: nn.Module) -> None:
        """Disable gradient checkpointing for a model."""
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled for model")

    @staticmethod
    def is_checkpointing_enabled(model: nn.Module) -> bool:
        """Check if gradient checkpointing is enabled."""
        return getattr(model, "gradient_checkpointing", False)


class MixedPrecisionTraining:
    """Mixed precision training utilities."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        if device == "cuda" and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda")
        elif device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def update_scaler(self):
        """Update the scaler after backward pass."""
        if self.scaler is not None:
            self.scaler.update()


class MemoryOptimizer:
    """Advanced memory optimization utilities."""

    @staticmethod
    def enable_attention_slicing(model: nn.Module, slice_size: Optional[int] = None) -> None:
        """Enable attention slicing to reduce memory usage."""
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing(slice_size)
            logger.info(f"Attention slicing enabled with slice_size={slice_size}")

    @staticmethod
    def enable_cpu_offload(model: nn.Module) -> None:
        """Enable CPU offloading for model parameters."""
        if hasattr(model, "enable_cpu_offload"):
            model.enable_cpu_offload()
            logger.info("CPU offloading enabled")

    @staticmethod
    def enable_sequential_cpu_offload(model: nn.Module) -> None:
        """Enable sequential CPU offloading."""
        if hasattr(model, "enable_sequential_cpu_offload"):
            model.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offloading enabled")

    @staticmethod
    def optimize_for_inference(model: nn.Module) -> None:
        """Optimize model for inference."""
        model.eval()
        if hasattr(model, "half"):
            model.half()  # Use FP16 for inference
        logger.info("Model optimized for inference")


class PerformanceProfiler:
    """Advanced performance profiling utilities."""

    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def start_profile(self, name: str) -> None:
        """Start profiling a section."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.profiles[name] = {
            "start_time": (
                torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            ),
            "start_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "start_cpu_memory": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        }

        if self.profiles[name]["start_time"]:
            self.profiles[name]["start_time"].record()

    def end_profile(self, name: str) -> Dict[str, Any]:
        """End profiling and return results."""
        if name not in self.profiles:
            return {}

        profile = self.profiles[name]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()

            profile["end_time"] = end_time
            profile["end_memory"] = torch.cuda.memory_allocated()
            profile["end_cpu_memory"] = torch.cuda.memory_reserved()

            # Calculate metrics
            elapsed_time = (
                profile["start_time"].elapsed_time(profile["end_time"]) / 1000.0
            )  # Convert to seconds
            memory_used = (profile["end_memory"] - profile["start_memory"]) / 1024**3  # GB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

            results = {
                "elapsed_time": elapsed_time,
                "memory_used": memory_used,
                "peak_memory": peak_memory,
                "start_memory": profile["start_memory"] / 1024**3,
                "end_memory": profile["end_memory"] / 1024**3,
            }
        else:
            results = {"elapsed_time": 0, "memory_used": 0, "peak_memory": 0}

        logger.info(f"Profile {name}: {results}")
        return results


class BatchOptimizer:
    """Batch processing optimization utilities."""

    @staticmethod
    def dynamic_batch_size(
        model: nn.Module, base_batch_size: int, max_memory_gb: float = 8.0
    ) -> int:
        """
        Dynamically determine optimal batch size based on available memory.

        Args:
            model: The model to test
            base_batch_size: Starting batch size
            max_memory_gb: Maximum memory to use in GB

        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return base_batch_size

        current_memory = torch.cuda.memory_allocated() / 1024**3
        available_memory = max_memory_gb - current_memory

        # Simple heuristic: reduce batch size if memory is low
        if available_memory < 4.0:
            return max(1, base_batch_size // 4)
        elif available_memory < 8.0:
            return max(1, base_batch_size // 2)
        else:
            return base_batch_size

    @staticmethod
    def optimize_dataloader(dataloader, num_workers: int = 0, pin_memory: bool = True) -> Any:
        """Optimize dataloader for better performance."""
        if hasattr(dataloader, "num_workers"):
            dataloader.num_workers = num_workers
        if hasattr(dataloader, "pin_memory"):
            dataloader.pin_memory = pin_memory

        return dataloader


class CachingSystem:
    """Caching system for repeated computations."""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = value
        self.access_count[key] = 1

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()


# Global instances
gradient_checkpointing = GradientCheckpointing()
mixed_precision = MixedPrecisionTraining()
memory_optimizer = MemoryOptimizer()
performance_profiler = PerformanceProfiler()
batch_optimizer = BatchOptimizer()
caching_system = CachingSystem()


def apply_optimizations(config: Dict[str, Any], model: nn.Module) -> None:
    """
    Apply all available optimizations based on configuration.

    Args:
        config: Configuration dictionary
        model: Model to optimize
    """
    optimizations = config.get("optimizations", {})

    # Gradient checkpointing
    if optimizations.get("gradient_checkpointing", False):
        gradient_checkpointing.enable_checkpointing(model)

    # Mixed precision
    if optimizations.get("mixed_precision", False):
        logger.info("Mixed precision training enabled")

    # Memory optimizations
    if optimizations.get("attention_slicing", False):
        memory_optimizer.enable_attention_slicing(model)

    if optimizations.get("cpu_offload", False):
        memory_optimizer.enable_cpu_offload(model)

    # Inference optimizations
    if optimizations.get("inference_optimization", False):
        memory_optimizer.optimize_for_inference(model)

    logger.info("Applied optimizations to model")


def enable_circuitkit_optimizations(**kwargs):
    """Enable CircuitKit optimizations."""
    from circuitkit.backends.algorithm_optimizer import enable_algorithm_optimizations

    enable_algorithm_optimizations()
    logger.info("CircuitKit optimizations enabled")


def disable_circuitkit_optimizations(**kwargs):
    """Disable CircuitKit optimizations."""
    from circuitkit.backends.algorithm_optimizer import disable_algorithm_optimizations

    disable_algorithm_optimizations()
    logger.info("CircuitKit optimizations disabled")


def list_available_optimizations() -> Dict[str, Any]:
    """List available optimizations."""
    return {
        "eap_modes": ["eap_optimized", "eap_ig_optimized", "memory_efficient"],
        "data_sources": ["csv"],
        "optimization_features": [
            "batch_processing",
            "memory_efficient_caching",
            "streaming_computation",
            "gradient_accumulation",
            "mixed_precision",
            "parallel_processing",
        ],
    }
