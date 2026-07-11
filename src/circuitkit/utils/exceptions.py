"""
Custom exceptions for CircuitKit with enhanced error handling.
"""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


class CircuitKitError(Exception):
    """Base exception for CircuitKit."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        self.traceback = traceback.format_exc()

    def __str__(self):
        base_msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        if self.suggestion:
            base_msg += f" Suggestion: {self.suggestion}"
        return base_msg


class ConfigurationError(CircuitKitError):
    """Raised when there are configuration issues."""


class ModelError(CircuitKitError):
    """Raised when there are model-related issues."""


class DataError(CircuitKitError):
    """Raised when there are data-related issues."""


class AlgorithmError(CircuitKitError):
    """Raised when there are algorithm-related issues."""


class EvaluationError(CircuitKitError):
    """Raised when there are evaluation-related issues."""


class FileError(CircuitKitError):
    """Raised when there are file I/O issues."""


class ValidationError(CircuitKitError):
    """Raised when validation fails."""


class ResourceError(CircuitKitError):
    """Raised when there are resource-related issues (memory, GPU, etc.)."""


class TimeoutError(CircuitKitError):
    """Raised when operations timeout."""


class DependencyError(CircuitKitError):
    """Raised when there are dependency-related issues."""


# Specific error classes for common scenarios
class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found or loaded."""


class InvalidModelError(ModelError):
    """Raised when a model is invalid or incompatible."""


class InsufficientMemoryError(ResourceError):
    """Raised when there's insufficient memory for an operation."""


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is required but not available."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""


class DataFileNotFoundError(DataError):
    """Raised when a required data file is not found."""


class InvalidDataFormatError(DataError):
    """Raised when data format is invalid."""


class AlgorithmNotSupportedError(AlgorithmError):
    """Raised when an algorithm is not supported."""


class EvaluationFailedError(EvaluationError):
    """Raised when evaluation fails."""


class FileNotFoundError(FileError):
    """Raised when a required file is not found."""


class FileWriteError(FileError):
    """Raised when file writing fails."""


class FileReadError(FileError):
    """Raised when file reading fails."""


class ValidationFailedError(ValidationError):
    """Raised when validation fails."""


class DependencyNotFoundError(DependencyError):
    """Raised when a required dependency is not found."""


class DependencyVersionError(DependencyError):
    """Raised when a dependency version is incompatible."""


# Error handling utilities
def handle_exception(
    exception: Exception, context: Optional[Dict[str, Any]] = None
) -> CircuitKitError:
    """Convert a generic exception to a CircuitKitError with context."""
    if isinstance(exception, CircuitKitError):
        return exception

    # Map common exceptions to CircuitKit errors
    if isinstance(exception, FileNotFoundError):
        return FileNotFoundError(str(exception), context)
    elif isinstance(exception, MemoryError):
        return InsufficientMemoryError(str(exception), context)
    elif isinstance(exception, ValueError):
        return ValidationError(str(exception), context)
    elif isinstance(exception, TypeError):
        return ValidationError(str(exception), context)
    elif isinstance(exception, KeyError):
        return MissingConfigurationError(str(exception), context)
    else:
        return CircuitKitError(str(exception), context)


def validate_file_exists(file_path: str, description: str = "file") -> None:
    """Validate that a file exists, raise FileNotFoundError if not."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{description.capitalize()} not found: {file_path}",
            context={"file_path": file_path, "description": description},
            suggestion=f"Check that the {description} exists and the path is correct",
        )


def validate_directory_exists(dir_path: str, description: str = "directory") -> None:
    """Validate that a directory exists, raise FileNotFoundError if not."""
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{description.capitalize()} not found: {dir_path}",
            context={"directory_path": dir_path, "description": description},
            suggestion=f"Create the {description} or check the path is correct",
        )


def validate_positive_number(value: Any, name: str, min_value: float = 0.0) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__}",
            context={"value": value, "name": name, "type": type(value).__name__},
        )

    if value < min_value:
        raise ValidationError(
            f"{name} must be >= {min_value}, got {value}",
            context={"value": value, "name": name, "min_value": min_value},
        )


def validate_in_range(value: Any, name: str, min_value: float, max_value: float) -> None:
    """Validate that a value is within a range."""
    validate_positive_number(value, name, min_value)

    if value > max_value:
        raise ValidationError(
            f"{name} must be <= {max_value}, got {value}",
            context={"value": value, "name": name, "max_value": max_value},
        )


def validate_choice(value: Any, name: str, choices: List[Any]) -> None:
    """Validate that a value is one of the allowed choices."""
    if value not in choices:
        raise ValidationError(
            f"{name} must be one of {choices}, got {value}",
            context={"value": value, "name": name, "choices": choices},
        )


def validate_required_keys(
    config: Dict[str, Any], required_keys: List[str], section: str = ""
) -> None:
    """Validate that required keys exist in a configuration dictionary."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        section_prefix = f"{section}." if section else ""
        raise MissingConfigurationError(
            f"Missing required configuration keys: {missing_keys}",
            context={"missing_keys": missing_keys, "section": section},
            suggestion=f"Add the missing keys to the {section_prefix}configuration",
        )


def validate_model_name(model_name: str) -> None:
    """Validate model name format."""
    if not model_name or len(model_name) < 2:
        raise ValidationError(
            "Model name must be at least 2 characters long", context={"model_name": model_name}
        )

    # Check for invalid characters
    invalid_chars = ["<", ">", '"', "|", "?", "*"]
    found_invalid = [char for char in invalid_chars if char in model_name]
    if found_invalid:
        raise ValidationError(
            f"Model name contains invalid characters: {found_invalid}",
            context={"model_name": model_name, "invalid_chars": found_invalid},
        )

    # Note: We don't validate against a specific list since TransformerLens
    # supports many models and new ones are added frequently. The actual
    # validation happens when trying to load the model.


# ── Algorithm category registries ────────────────────────────────────────
# Single source of truth lives in circuitkit.backends.ALGORITHMS; these names
# are re-exported here so validation helpers (and the CLI) keep one import.
from ..backends import (  # noqa: E402
    DISCOVERY_ALGORITHMS,
    PRUNING_ALGORITHMS,
    QUANTIZATION_ALGORITHMS,
    SUPPORTED_ALGORITHMS,
)


def validate_algorithm(algorithm: str) -> None:
    """Validate algorithm name against all known algorithms."""
    validate_choice(algorithm.lower(), "algorithm", SUPPORTED_ALGORITHMS)


def validate_discovery_algorithm(algorithm: str) -> None:
    """Validate algorithm is a known discovery algorithm."""
    validate_choice(algorithm.lower(), "algorithm", sorted(DISCOVERY_ALGORITHMS))


def validate_pruning_algorithm(algorithm: str) -> None:
    """Validate algorithm is a known pruning algorithm."""
    validate_choice(algorithm.lower(), "algorithm", sorted(PRUNING_ALGORITHMS))


def validate_quantization_algorithm(algorithm: str) -> None:
    """Validate algorithm is a known quantization algorithm."""
    validate_choice(algorithm.lower(), "algorithm", sorted(QUANTIZATION_ALGORITHMS))


def validate_sparsity(sparsity: float) -> None:
    """Validate sparsity value."""
    validate_in_range(sparsity, "target_sparsity", 0.0, 1.0)


def validate_batch_size(batch_size: int) -> None:
    """Validate batch size."""
    validate_positive_number(batch_size, "batch_size", 1)


def validate_device(device: str) -> None:
    """Validate device specification."""
    valid_devices = ["cpu", "cuda", "auto"]
    validate_choice(device.lower(), "device", valid_devices)


# Error context manager
class ErrorContext:
    """Context manager for adding context to exceptions."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.original_context = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and isinstance(exc_val, CircuitKitError):
            # Add context to existing CircuitKit error
            exc_val.context.update(self.context)
        elif exc_val:
            # Convert to CircuitKit error with context
            raise handle_exception(exc_val, self.context) from exc_val


# Decorator for error handling
def handle_errors(
    context: Optional[Dict[str, Any]] = None, reraise: bool = True, log_error: bool = True
):
    """Decorator to handle errors with context."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or {}
                error_context.update(
                    {
                        "function": func.__name__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200],
                    }
                )

                if log_error:
                    from .logging import get_logger

                    logger = get_logger()
                    logger.log_error_with_traceback(f"Error in {func.__name__}", e)

                if reraise:
                    raise handle_exception(e, error_context) from e
                else:
                    return None

        return wrapper

    return decorator
