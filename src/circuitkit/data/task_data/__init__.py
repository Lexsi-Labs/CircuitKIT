"""
ACDC Data Generation Integration

This module provides ACDC data generation capabilities for CircuitKit,
including intelligent caching, file management, and task-specific data generation.
"""

from .generation.cache import ACDCCache
from .generation.manager import ACDCDataManager

__all__ = ["ACDCDataManager", "ACDCCache"]
