"""
CircuitKit Tasks Package

This package provides the TaskSpec abstraction and registry for multi-task
circuit discovery and analysis.
"""

from .auto_schema import SchemaAnalyzer, TaskType, TaskTypeDetection
from .builtins.ioi import IOITaskSpec
from .generic import GenericTaskSpec
from .hf_factory import (
    SchemaPreview,
    auto_task_from_hf,
    list_compatible_datasets,
    preview_schema,
    validate_hf_dataset,
)
from .registry import get_task, list_tasks, register_task
from .specs import TaskSpec
from .validator import DatasetValidator, ValidationResult, validate_dataset

__all__ = [
    "TaskSpec",
    "GenericTaskSpec",
    "register_task",
    "get_task",
    "list_tasks",
    "DatasetValidator",
    "ValidationResult",
    "validate_dataset",
    "SchemaAnalyzer",
    "TaskType",
    "TaskTypeDetection",
    "auto_task_from_hf",
    "preview_schema",
    "list_compatible_datasets",
    "validate_hf_dataset",
    "SchemaPreview",
    "IOITaskSpec",
]
