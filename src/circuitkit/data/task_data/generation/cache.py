"""
ACDC Data Generation Cache

Intelligent caching system for ACDC data generation with file management.
"""

import time
from typing import Any, Dict, Optional

import torch

from .utils import FileManager, GenerationConfig, create_data_summary, validate_data_integrity


import logging

logger = logging.getLogger(__name__)

class ACDCCache:
    """Intelligent cache for ACDC data generation."""

    def __init__(self, base_storage_dir: str = None, max_cache_size: int = 1000):
        self.file_manager = FileManager(base_storage_dir)
        self.max_cache_size = max_cache_size
        self.cache_stats = {"hits": 0, "misses": 0, "generations": 0, "subsamples": 0}

    def get_cached_data(self, config: GenerationConfig) -> Optional[Dict[str, Any]]:
        """
        Get cached data for configuration.

        Args:
            config: Generation configuration

        Returns:
            Cached data if available and valid, None otherwise
        """
        # Check if file exists
        if not self.file_manager.file_exists(config):
            self.cache_stats["misses"] += 1
            return None

        # Load data
        data = self.file_manager.load_data(config)
        if data is None:
            self.cache_stats["misses"] += 1
            return None

        # Validate data integrity
        if not validate_data_integrity(data, config.num_examples):
            logger.warning(f"Warning: Cached data for {config.task_name} failed validation, regenerating...")
            self.cache_stats["misses"] += 1
            return None

        # Check if we need to subsample
        if self._needs_subsampling(data, config.num_examples):
            logger.info(
                f"Subsampling cached data from {len(data[next(iter(data.keys()))])} to {config.num_examples} samples"
            )
            data = self._subsample_data(data, config.num_examples, config.seed)
            self.cache_stats["subsamples"] += 1

        self.cache_stats["hits"] += 1
        return data

    def cache_data(
        self, config: GenerationConfig, data: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Cache generated data.

        Args:
            config: Generation configuration
            data: Generated data
            metadata: Optional metadata

        Returns:
            True if caching successful, False otherwise
        """
        # Validate data before caching
        if not validate_data_integrity(data, config.num_examples):
            logger.warning(f"Warning: Generated data for {config.task_name} failed validation, not caching")
            return False

        # Save data
        success = self.file_manager.save_data(config, data)
        if not success:
            return False

        # Save metadata
        if metadata is None:
            metadata = create_data_summary(data)

        metadata.update(
            {
                "generation_time": time.time(),
                "config_hash": config.to_hash(),
                "config": config.__dict__,
            }
        )

        self.file_manager.save_metadata(config, metadata)
        self.cache_stats["generations"] += 1

        return True

    def _needs_subsampling(self, data: Dict[str, Any], target_size: int) -> bool:
        """Check if data needs subsampling."""
        if not data:
            return False

        first_key = next(iter(data.keys()))
        first_value = data[first_key]

        if isinstance(first_value, torch.Tensor):
            current_size = len(first_value)
        elif isinstance(first_value, list):
            current_size = len(first_value)
        else:
            return False

        return current_size > target_size

    def _subsample_data(self, data: Dict[str, Any], target_size: int, seed: int) -> Dict[str, Any]:
        """Subsample data to target size."""
        from .utils import subsample_data


        return subsample_data(data, target_size, seed)

    def get_cache_info(self, task_name: str = None) -> Dict[str, Any]:
        """
        Get cache information.

        Args:
            task_name: Optional task name to filter by

        Returns:
            Cache information dictionary
        """
        info = {"stats": self.cache_stats.copy(), "hit_rate": 0.0, "files": {}, "total_size": 0}

        # Calculate hit rate
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests > 0:
            info["hit_rate"] = self.cache_stats["hits"] / total_requests

        # Get file information
        if task_name:
            task_dir = self.file_manager.get_task_storage_dir(task_name)
            if task_dir.exists():
                files = list(task_dir.glob("*.pkl"))
                for file_path in files:
                    file_info = {
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "path": str(file_path),
                    }
                    info["files"][file_path.name] = file_info
                    info["total_size"] += file_info["size"]
        else:
            # Get info for all tasks
            base_dir = self.file_manager.base_storage_dir
            if base_dir.exists():
                for task_dir in base_dir.iterdir():
                    if task_dir.is_dir():
                        task_files = list(task_dir.glob("*.pkl"))
                        for file_path in task_files:
                            file_info = {
                                "size": file_path.stat().st_size,
                                "modified": file_path.stat().st_mtime,
                                "path": str(file_path),
                            }
                            info["files"][f"{task_dir.name}/{file_path.name}"] = file_info
                            info["total_size"] += file_info["size"]

        return info

    def cleanup_cache(self, task_name: str = None, max_files: int = 10) -> int:
        """
        Clean up cache files.

        Args:
            task_name: Optional task name to clean up
            max_files: Maximum number of files to keep per task

        Returns:
            Number of files removed
        """
        if task_name:
            return self.file_manager.cleanup_old_files(task_name, max_files)
        else:
            # Clean up all tasks
            total_removed = 0
            base_dir = self.file_manager.base_storage_dir
            if base_dir.exists():
                for task_dir in base_dir.iterdir():
                    if task_dir.is_dir():
                        total_removed += self.file_manager.cleanup_old_files(
                            task_dir.name, max_files
                        )
            return total_removed

    def clear_cache(self, task_name: str = None) -> int:
        """
        Clear all cache files.

        Args:
            task_name: Optional task name to clear

        Returns:
            Number of files removed
        """
        if task_name:
            task_dir = self.file_manager.get_task_storage_dir(task_name)
            if not task_dir.exists():
                return 0

            removed_count = 0
            for file_path in task_dir.glob("*"):
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Error removing {file_path}: {e}")

            return removed_count
        else:
            # Clear all cache
            base_dir = self.file_manager.base_storage_dir
            if not base_dir.exists():
                return 0

            removed_count = 0
            for file_path in base_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing {file_path}: {e}")

            return removed_count

    def is_cache_valid(self, config: GenerationConfig) -> bool:
        """
        Check if cache is valid for configuration.

        Args:
            config: Generation configuration

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.file_manager.file_exists(config):
            return False

        # Load metadata to check configuration
        metadata = self.file_manager.load_metadata(config)
        if metadata is None:
            return False

        # Check if configuration matches
        cached_config_hash = metadata.get("config_hash")
        if cached_config_hash != config.to_hash():
            return False

        return True
