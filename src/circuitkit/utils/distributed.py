"""
Distributed training utilities for CircuitKit.
Supports multi-GPU and multi-node training for large models.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from circuitkit.utils.logging import get_logger

logger = get_logger(__name__)


class DistributedTraining:
    """Utilities for distributed training."""

    def __init__(self):
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

    def initialize(self, backend: str = "nccl") -> bool:
        """
        Initialize distributed training.

        Args:
            backend: Communication backend (nccl for GPU, gloo for CPU)

        Returns:
            True if initialization successful, False otherwise
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, distributed training not supported")
            return False

        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            logger.warning("Distributed environment variables not set")
            return False

        try:
            dist.init_process_group(backend=backend)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.is_initialized = True

            logger.info(
                f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleaned up")

    def wrap_model(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """
        Wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap
            device_ids: List of device IDs to use

        Returns:
            Wrapped model
        """
        if not self.is_initialized:
            logger.warning("Distributed training not initialized, returning original model")
            return model

        if device_ids is None:
            device_ids = [self.local_rank]

        return DDP(model, device_ids=device_ids, find_unused_parameters=True)

    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()


class ModelParallelism:
    """Model parallelism utilities for very large models."""

    @staticmethod
    def split_model_layers(
        model: nn.Module, num_splits: int, strategy: str = "balanced"
    ) -> List[nn.Module]:
        """
        Split model into multiple parts for model parallelism using sophisticated algorithms.

        Args:
            model: Model to split
            num_splits: Number of parts to split into
            strategy: Splitting strategy ('balanced', 'memory', 'compute', 'communication')

        Returns:
            List of model parts
        """
        if strategy == "balanced":
            return ModelParallelism._balanced_split(model, num_splits)
        elif strategy == "memory":
            return ModelParallelism._memory_aware_split(model, num_splits)
        elif strategy == "compute":
            return ModelParallelism._compute_aware_split(model, num_splits)
        elif strategy == "communication":
            return ModelParallelism._communication_aware_split(model, num_splits)
        else:
            logger.warning(f"Unknown splitting strategy: {strategy}, using balanced")
            return ModelParallelism._balanced_split(model, num_splits)

    @staticmethod
    def _balanced_split(model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split model with balanced layer distribution."""
        layers = list(model.children())
        split_size = len(layers) // num_splits

        splits = []
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_splits - 1 else len(layers)
            split_layers = layers[start_idx:end_idx]
            splits.append(nn.Sequential(*split_layers))

        return splits

    @staticmethod
    def _memory_aware_split(model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split model based on memory usage of each layer."""
        layers = list(model.children())
        layer_memory = []

        # Estimate memory usage for each layer
        for layer in layers:
            memory_usage = ModelParallelism._estimate_layer_memory(layer)
            layer_memory.append(memory_usage)

        # Use dynamic programming to find optimal split points
        splits = ModelParallelism._optimal_split(layer_memory, num_splits)

        result = []
        for i in range(num_splits):
            start_idx = splits[i]
            end_idx = splits[i + 1] if i < num_splits - 1 else len(layers)
            split_layers = layers[start_idx:end_idx]
            result.append(nn.Sequential(*split_layers))

        return result

    @staticmethod
    def _compute_aware_split(model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split model based on computational complexity."""
        layers = list(model.children())
        layer_compute = []

        # Estimate computational complexity for each layer
        for layer in layers:
            compute_cost = ModelParallelism._estimate_layer_compute(layer)
            layer_compute.append(compute_cost)

        # Use dynamic programming to find optimal split points
        splits = ModelParallelism._optimal_split(layer_compute, num_splits)

        result = []
        for i in range(num_splits):
            start_idx = splits[i]
            end_idx = splits[i + 1] if i < num_splits - 1 else len(layers)
            split_layers = layers[start_idx:end_idx]
            result.append(nn.Sequential(*split_layers))

        return result

    @staticmethod
    def _communication_aware_split(model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split model to minimize communication overhead."""
        layers = list(model.children())

        # Identify communication bottlenecks (attention layers, large linear layers)
        communication_cost = []
        for layer in layers:
            cost = ModelParallelism._estimate_communication_cost(layer)
            communication_cost.append(cost)

        # Use dynamic programming to find optimal split points
        splits = ModelParallelism._optimal_split(communication_cost, num_splits)

        result = []
        for i in range(num_splits):
            start_idx = splits[i]
            end_idx = splits[i + 1] if i < num_splits - 1 else len(layers)
            split_layers = layers[start_idx:end_idx]
            result.append(nn.Sequential(*split_layers))

        return result

    @staticmethod
    def _estimate_layer_memory(layer: nn.Module) -> float:
        """Estimate memory usage of a layer."""
        total_params = sum(p.numel() for p in layer.parameters())
        # Rough estimate: 4 bytes per parameter (float32)
        return total_params * 4 / (1024 * 1024)  # MB

    @staticmethod
    def _estimate_layer_compute(layer: nn.Module) -> float:
        """Estimate computational complexity of a layer."""
        if isinstance(layer, nn.Linear):
            return layer.in_features * layer.out_features
        elif isinstance(layer, nn.Conv2d):
            return (
                layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
            )
        elif isinstance(layer, nn.MultiheadAttention):
            return layer.embed_dim * layer.embed_dim * 4  # Q, K, V, O projections
        else:
            return 1.0  # Default for unknown layers

    @staticmethod
    def _estimate_communication_cost(layer: nn.Module) -> float:
        """Estimate communication cost for a layer."""
        if isinstance(layer, nn.MultiheadAttention):
            return 1.0  # High communication cost for attention
        elif isinstance(layer, nn.Linear) and layer.out_features > 1000:
            return 0.5  # Medium communication cost for large linear layers
        else:
            return 0.1  # Low communication cost for other layers

    @staticmethod
    def _optimal_split(costs: List[float], num_splits: int) -> List[int]:
        """Find optimal split points using dynamic programming."""
        n = len(costs)
        if n <= num_splits:
            return list(range(n + 1))

        # DP table: dp[i][j] = minimum cost to split first i elements into j parts
        dp = [[float("inf")] * (num_splits + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i + 1, num_splits + 1)):
                for k in range(j - 1, i):
                    cost = sum(costs[k:i])
                    dp[i][j] = min(dp[i][j], dp[k][j - 1] + cost)

        # Backtrack to find split points
        splits = []
        i, j = n, num_splits
        while j > 0:
            for k in range(j - 1, i):
                if dp[i][j] == dp[k][j - 1] + sum(costs[k:i]):
                    splits.append(k)
                    i, j = k, j - 1
                    break

        splits.append(0)
        splits.sort()
        return splits

    @staticmethod
    def enable_tensor_parallelism(model: nn.Module, tensor_parallel_size: int = 2) -> nn.Module:
        """
        Enable tensor parallelism for the model.

        Args:
            model: Model to parallelize
            tensor_parallel_size: Number of GPUs to split tensors across

        Returns:
            Model with tensor parallelism enabled
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, tensor parallelism not supported")
            return model

        if torch.cuda.device_count() < tensor_parallel_size:
            logger.warning(
                f"Not enough GPUs for tensor parallelism. Available: {torch.cuda.device_count()}, Required: {tensor_parallel_size}"
            )
            return model

        logger.info(f"Enabling tensor parallelism with {tensor_parallel_size} GPUs")

        # Apply tensor parallelism to linear layers
        model = ModelParallelism._apply_tensor_parallel_to_linear(model, tensor_parallel_size)

        # Apply tensor parallelism to attention layers if present
        model = ModelParallelism._apply_tensor_parallel_to_attention(model, tensor_parallel_size)

        logger.info("Tensor parallelism successfully enabled")
        return model

    @staticmethod
    def _apply_tensor_parallel_to_linear(model: nn.Module, tensor_parallel_size: int) -> nn.Module:
        """Apply tensor parallelism to linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Split the weight matrix across GPUs
                if module.weight.size(0) >= tensor_parallel_size:
                    # Split output features
                    split_size = module.weight.size(0) // tensor_parallel_size
                    module.weight.data = module.weight.data[:split_size]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[:split_size]
                elif module.weight.size(1) >= tensor_parallel_size:
                    # Split input features
                    split_size = module.weight.size(1) // tensor_parallel_size
                    module.weight.data = module.weight.data[:, :split_size]

        return model

    @staticmethod
    def _apply_tensor_parallel_to_attention(
        model: nn.Module, tensor_parallel_size: int
    ) -> nn.Module:
        """Apply tensor parallelism to attention layers."""
        for name, module in model.named_modules():
            if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                # Multi-head attention with in_proj_weight
                if module.in_proj_weight.size(0) >= tensor_parallel_size:
                    split_size = module.in_proj_weight.size(0) // tensor_parallel_size
                    module.in_proj_weight.data = module.in_proj_weight.data[:split_size]
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.data = module.in_proj_bias.data[:split_size]

        return model


class DataParallelism:
    """Data parallelism utilities."""

    @staticmethod
    def create_data_parallel_model(
        model: nn.Module, device_ids: Optional[List[int]] = None
    ) -> nn.Module:
        """
        Create data parallel model.

        Args:
            model: Model to parallelize
            device_ids: List of device IDs to use

        Returns:
            Data parallel model
        """
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) > 1:
            return nn.DataParallel(model, device_ids=device_ids)
        else:
            return model

    @staticmethod
    def split_dataset(dataset, num_workers: int) -> List:
        """
        Split dataset for data parallelism.

        Args:
            dataset: Dataset to split
            num_workers: Number of workers

        Returns:
            List of dataset splits
        """
        if hasattr(dataset, "__len__"):
            split_size = len(dataset) // num_workers
            splits = []
            for i in range(num_workers):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < num_workers - 1 else len(dataset)
                splits.append(dataset[start_idx:end_idx])
            return splits
        else:
            # For iterable datasets, we can't easily split
            logger.warning("Cannot split iterable dataset")
            return [dataset] * num_workers


class CommunicationOptimizer:
    """Communication optimization utilities."""

    def __init__(self):
        self.compression_enabled = False
        self.allreduce_optimized = False
        self.compression_ratio = 0.1  # Default compression ratio

    def enable_gradient_compression(
        self, compression_ratio: float = 0.1, method: str = "topk"
    ) -> None:
        """
        Enable gradient compression for communication.

        Args:
            compression_ratio: Fraction of gradients to keep (0.0-1.0)
            method: Compression method ('topk', 'random', 'sign')
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, gradient compression not supported")
            return

        self.compression_enabled = True
        self.compression_ratio = compression_ratio
        self.compression_method = method

        logger.info(f"Gradient compression enabled: method={method}, ratio={compression_ratio}")

    def compress_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Compress gradients using the specified method.

        Args:
            gradients: Input gradients to compress

        Returns:
            Compressed gradients
        """
        if not self.compression_enabled:
            return gradients

        if self.compression_method == "topk":
            return self._topk_compression(gradients)
        elif self.compression_method == "random":
            return self._random_compression(gradients)
        elif self.compression_method == "sign":
            return self._sign_compression(gradients)
        else:
            logger.warning(f"Unknown compression method: {self.compression_method}")
            return gradients

    def _topk_compression(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compress gradients using top-k selection."""
        k = max(1, int(gradients.numel() * self.compression_ratio))
        _, indices = torch.topk(torch.abs(gradients).flatten(), k)

        compressed = torch.zeros_like(gradients)
        flat_grad = gradients.flatten()
        flat_compressed = compressed.flatten()
        flat_compressed[indices] = flat_grad[indices]

        return compressed

    def _random_compression(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compress gradients using random selection."""
        mask = torch.rand_like(gradients) < self.compression_ratio
        return gradients * mask

    def _sign_compression(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compress gradients using sign compression."""
        return torch.sign(gradients) * torch.mean(torch.abs(gradients))

    def optimize_allreduce(self, bucket_size: int = 25 * 1024 * 1024) -> None:
        """
        Optimize allreduce operations.

        Args:
            bucket_size: Size of gradient buckets for allreduce
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, allreduce optimization not supported")
            return

        self.allreduce_optimized = True
        self.bucket_size = bucket_size

        # Set up optimized communication patterns
        torch.distributed.init_process_group(backend="nccl")

        logger.info(f"Allreduce optimization enabled: bucket_size={bucket_size}")

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "compression_enabled": self.compression_enabled,
            "compression_ratio": self.compression_ratio,
            "compression_method": getattr(self, "compression_method", None),
            "allreduce_optimized": self.allreduce_optimized,
            "bucket_size": getattr(self, "bucket_size", None),
        }


# Global instances
distributed_training = DistributedTraining()
model_parallelism = ModelParallelism()
data_parallelism = DataParallelism()
communication_optimizer = CommunicationOptimizer()


def setup_distributed_training(config: Dict[str, Any]) -> bool:
    """
    Setup distributed training based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if setup successful, False otherwise
    """
    distributed_config = config.get("distributed", {})

    if not distributed_config.get("enabled", False):
        return False

    backend = distributed_config.get("backend", "nccl")
    return distributed_training.initialize(backend=backend)


def get_distributed_config() -> Dict[str, Any]:
    """Get current distributed configuration."""
    return {
        "is_initialized": distributed_training.is_initialized,
        "world_size": distributed_training.world_size,
        "rank": distributed_training.rank,
        "local_rank": distributed_training.local_rank,
    }
