"""
ACDC Data Generation Utilities

Utility classes and functions for data generation, caching, and file management.
"""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


import logging

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for ACDC data generation."""

    task_name: str
    num_examples: int
    prompt_type: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    metric_name: str = "logit_diff"
    template_type: Optional[str] = None
    corruption_strategy: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

    def to_hash(self) -> str:
        """Generate a hash for this configuration."""
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def to_filename(self) -> str:
        """Generate filename for this configuration."""
        hash_str = self.to_hash()[:8]
        return f"{self.task_name}_{self.num_examples}_{hash_str}.pkl"


class FileManager:
    """Manages file operations for generated data."""

    def __init__(self, base_storage_dir: str = None):
        if base_storage_dir is None:
            # Default to CircuitKit storage directory
            base_storage_dir = Path(__file__).parent.parent / "storage"

        self.base_storage_dir = Path(base_storage_dir)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

    def get_task_storage_dir(self, task_name: str) -> Path:
        """Get storage directory for a specific task."""
        task_dir = self.base_storage_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def get_file_path(self, config: GenerationConfig) -> Path:
        """Get file path for a configuration."""
        task_dir = self.get_task_storage_dir(config.task_name)
        return task_dir / config.to_filename()

    def file_exists(self, config: GenerationConfig) -> bool:
        """Check if file exists for configuration."""
        file_path = self.get_file_path(config)
        return file_path.exists()

    def get_file_size(self, config: GenerationConfig) -> int:
        """Get file size in bytes."""
        file_path = self.get_file_path(config)
        if file_path.exists():
            return file_path.stat().st_size
        return 0

    def load_data(self, config: GenerationConfig) -> Optional[Dict[str, Any]]:
        """Load generated data from file."""
        file_path = self.get_file_path(config)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Error loading data from {file_path}: {e}")
            return None

    def save_data(self, config: GenerationConfig, data: Dict[str, Any]) -> bool:
        """Save generated data to file.

        Some task generators (e.g. IOI / greater-than via ACDC) return metric
        *closures* (``validation_metric``, ``test_metrics``) alongside the
        tensors. Local functions cannot be pickled, so any value that fails to
        pickle is dropped from the cached copy rather than aborting the whole
        save. Metric closures are model-bound and recomputed on demand, so
        dropping them only affects the cached blob, not live results.
        """
        file_path = self.get_file_path(config)

        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, dict):
                picklable: Dict[str, Any] = {}
                dropped = []
                for key, value in data.items():
                    try:
                        pickle.dumps(value)
                        picklable[key] = value
                    except Exception:
                        dropped.append(key)
                if dropped:
                    logger.info(
                        f"Note: skipped unpicklable keys when caching "
                        f"{file_path.name}: {dropped}"
                    )
                to_dump = picklable
            else:
                to_dump = data

            with open(file_path, "wb") as f:
                pickle.dump(to_dump, f)
            return True
        except Exception as e:
            logger.warning(f"Error saving data to {file_path}: {e}")
            return False

    def get_metadata_path(self, config: GenerationConfig) -> Path:
        """Get metadata file path for configuration."""
        file_path = self.get_file_path(config)
        return file_path.with_suffix(".json")

    def save_metadata(self, config: GenerationConfig, metadata: Dict[str, Any]) -> bool:
        """Save metadata for generated data."""
        metadata_path = self.get_metadata_path(config)

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Error saving metadata to {metadata_path}: {e}")
            return False

    def load_metadata(self, config: GenerationConfig) -> Optional[Dict[str, Any]]:
        """Load metadata for generated data."""
        metadata_path = self.get_metadata_path(config)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.warning(f"Error loading metadata from {metadata_path}: {e}")
            return None

    def cleanup_old_files(self, task_name: str, max_files: int = 10) -> int:
        """Clean up old files, keeping only the most recent ones."""
        task_dir = self.get_task_storage_dir(task_name)

        if not task_dir.exists():
            return 0

        # Get all files sorted by modification time
        files = list(task_dir.glob("*.pkl"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files
        removed_count = 0
        for file_path in files[max_files:]:
            try:
                file_path.unlink()
                # Also remove metadata file
                metadata_path = file_path.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Error removing file {file_path}: {e}")

        return removed_count


def subsample_data(data: Dict[str, Any], target_size: int, seed: int = 42) -> Dict[str, Any]:
    """
    Subsample data to target size.

    Args:
        data: Dictionary containing data arrays
        target_size: Target number of samples
        seed: Random seed for reproducibility

    Returns:
        Subsampled data dictionary
    """
    if target_size <= 0:
        return data

    # Set random seed
    torch.manual_seed(seed)

    # Get the length of the first data array to determine current size
    first_key = next(iter(data.keys()))
    current_size = len(data[first_key])

    if current_size <= target_size:
        return data

    # Generate random indices
    indices = torch.randperm(current_size)[:target_size]

    # Subsample all arrays
    subsampled_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            subsampled_data[key] = value[indices]
        elif isinstance(value, list):
            subsampled_data[key] = [value[i] for i in indices]
        elif isinstance(value, (int, float, str)):
            # Keep scalar values as is
            subsampled_data[key] = value
        else:
            # For other types, try to index if possible
            try:
                subsampled_data[key] = value[indices]
            except (TypeError, IndexError):
                # If indexing fails, keep original value
                subsampled_data[key] = value

    return subsampled_data


def validate_data_integrity(data: Dict[str, Any], expected_size: int) -> bool:
    """
    Validate that data has the expected structure and size.

    Args:
        data: Data dictionary to validate
        expected_size: Expected number of samples

    Returns:
        True if data is valid, False otherwise
    """
    if not isinstance(data, dict):
        return False

    if not data:
        return False

    # Check that all arrays have the same length
    lengths = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            lengths.append(len(value))
        elif isinstance(value, list):
            lengths.append(len(value))

    # If no arrays found, consider it valid (basic data structure)
    if not lengths:
        return True

    # All arrays should have the same length
    if not all(length == lengths[0] for length in lengths):
        return False

    # Check if size matches expected (allow some tolerance)
    actual_size = lengths[0]
    if abs(actual_size - expected_size) > max(1, expected_size * 0.1):
        return False

    return True


def create_data_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of the generated data.

    Args:
        data: Data dictionary

    Returns:
        Summary dictionary
    """
    summary = {"num_samples": 0, "data_types": {}, "tensor_shapes": {}, "memory_usage": 0}

    if not data:
        return summary

    # Get sample count from first array
    first_key = next(iter(data.keys()))
    first_value = data[first_key]

    if isinstance(first_value, torch.Tensor):
        summary["num_samples"] = len(first_value)
    elif isinstance(first_value, list):
        summary["num_samples"] = len(first_value)

    # Analyze each data component
    for key, value in data.items():
        summary["data_types"][key] = type(value).__name__

        if isinstance(value, torch.Tensor):
            summary["tensor_shapes"][key] = list(value.shape)
            summary["memory_usage"] += value.numel() * value.element_size()
        elif isinstance(value, list):
            summary["tensor_shapes"][key] = [len(value)]

    return summary
