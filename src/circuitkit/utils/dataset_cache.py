"""
Dataset caching with model compatibility validation.

This module provides utilities for caching datasets with tokenizer metadata
to ensure model compatibility when loading cached data, as well as caching
corruption results to avoid regenerating expensive corruptions.
"""

import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .token_utils import TokenIDGenerator


class DatasetCache:
    """Cache datasets with model compatibility validation."""

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize dataset cache.

        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_with_metadata(
        self, data: Any, filepath: str, token_generator: TokenIDGenerator, format: str = "pickle"
    ) -> str:
        """
        Save dataset with tokenizer metadata.

        Args:
            data: Dataset to save
            filepath: Path to save the dataset
            token_generator: TokenIDGenerator for metadata
            format: Save format ("pickle" or "json")

        Returns:
            Path where data was saved

        Raises:
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get metadata
        metadata = token_generator.get_metadata()

        # Save data
        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        elif format == "json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")

        # Save metadata
        metadata_path = filepath.with_suffix(filepath.suffix + ".metadata")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(filepath)

    def load_and_validate(
        self, filepath: str, token_generator: TokenIDGenerator, format: str = "pickle"
    ) -> Any:
        """
        Load dataset and validate it matches current model.

        Args:
            filepath: Path to load the dataset from
            token_generator: TokenIDGenerator for validation
            format: Load format ("pickle" or "json")

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If dataset or metadata file doesn't exist
            ValueError: If dataset is incompatible with current model
        """
        filepath = Path(filepath)
        metadata_path = filepath.with_suffix(filepath.suffix + ".metadata")

        # Check if files exist
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                f"Cannot validate model compatibility."
            )

        # Load metadata
        with open(metadata_path, "r") as f:
            cached_metadata = json.load(f)

        # Validate compatibility
        if not token_generator.validate_compatibility(cached_metadata):
            current_metadata = token_generator.get_metadata()
            raise ValueError(
                f"Dataset incompatible with current model. "
                f"Cached model: {cached_metadata['model_name']}, "
                f"Current model: {current_metadata['model_name']}. "
                f"Cached vocab size: {cached_metadata['vocab_size']}, "
                f"Current vocab size: {current_metadata['vocab_size']}. "
                f"Please regenerate dataset for current model."
            )

        # Load data
        if format == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        elif format == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")

        return data

    def is_cached_and_compatible(self, filepath: str, token_generator: TokenIDGenerator) -> bool:
        """
        Check if dataset is cached and compatible with current model.

        Args:
            filepath: Path to check
            token_generator: TokenIDGenerator for validation

        Returns:
            True if cached and compatible, False otherwise
        """
        try:
            self.load_and_validate(filepath, token_generator)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def clear_cache(self, pattern: str = "*") -> None:
        """
        Clear cached datasets.

        Args:
            pattern: Pattern to match files for deletion
        """
        import glob

        # Find all cache files matching pattern
        cache_files = glob.glob(str(self.cache_dir / pattern))
        metadata_files = glob.glob(str(self.cache_dir / f"{pattern}.metadata"))

        # Delete files
        for file_path in cache_files + metadata_files:
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore errors (file might not exist)


class CorruptionCache:
    """Cache corruption results to avoid regenerating expensive corruptions.

    Uses SHA256 hashing of (clean_data + strategy_config + seed) to create
    cache keys. Stores corruptions as JSON with metadata (timestamp, strategy,
    version) for easy inspection and validation.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize corruption cache.

        Args:
            cache_dir: Directory to store cached corruptions. Defaults to
                      ./cache/corruptions. Directory is created automatically.
        """
        if cache_dir is None:
            cache_dir = Path("./cache/corruptions")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(self, clean_data: str, strategy_config: Dict[str, Any], seed: int) -> str:
        """
        Generate a cache key from clean data, strategy config, and seed.

        Uses SHA256 hash of:
        - clean_data (the original prompt/text)
        - strategy_config (JSON-serialized dict of strategy parameters)
        - seed (random seed for reproducibility)

        Args:
            clean_data: Original clean text/prompt
            strategy_config: Configuration dict for the corruption strategy
            seed: Random seed used for corruption

        Returns:
            SHA256 hash as hexadecimal string (64 chars)
        """
        if not isinstance(clean_data, str):
            clean_data = str(clean_data)

        # Serialize config to JSON with sorted keys for deterministic ordering
        config_str = json.dumps(strategy_config, sort_keys=True, default=str)
        seed_str = str(seed)

        # Combine and hash
        combined = clean_data + config_str + seed_str
        hash_obj = hashlib.sha256(combined.encode("utf-8"))
        return hash_obj.hexdigest()

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load a corruption from cache.

        Args:
            key: Cache key (from get_key())

        Returns:
            Dictionary with 'corrupted' (the corrupted example) and 'metadata'
            (including timestamp, strategy_name, version), or None if not found.
        """
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    def save(
        self,
        key: str,
        corrupted_example: Dict[str, Any],
        strategy_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a corruption to cache.

        Args:
            key: Cache key (from get_key())
            corrupted_example: The corrupted example dict
            strategy_name: Name of the corruption strategy used
            metadata: Optional additional metadata to store. Will be merged
                     with auto-generated metadata (timestamp, strategy_name, version)
        """
        cache_file = self.cache_dir / f"{key}.json"

        # Build metadata
        cache_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy_name": strategy_name,
            "version": "1.0",  # Version for future compatibility
        }

        if metadata:
            cache_metadata.update(metadata)

        # Build cache entry
        cache_entry = {
            "corrupted": corrupted_example,
            "metadata": cache_metadata,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_entry, f, indent=2, default=str)
        except IOError as e:
            # Log warning but don't fail - cache is optional
            import warnings

            warnings.warn(f"Failed to save corruption to cache: {e}")

    def clear_cache(self, pattern: str = "*") -> None:
        """
        Clear cached corruptions.

        Args:
            pattern: Glob pattern to match files for deletion. Defaults to "*"
                    to clear all cache files.
        """
        import glob

        cache_files = glob.glob(str(self.cache_dir / pattern))
        for file_path in cache_files:
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore errors (file might not exist)
