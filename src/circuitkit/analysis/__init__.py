"""
CircuitKit Analysis Module

Provides analysis tools for circuits including metrics, scoring, and statistical analysis.
"""

from .cross_method_jaccard import CrossMethodJaccardResult, cross_method_jaccard
from .metrics import *  # noqa: F401,F403 - intentional API re-export
from .scores import *  # noqa: F401,F403 - intentional API re-export

__all__ = [  # noqa: F405 - names provided via star imports above
    # Metrics
    "compute_metrics",
    # Scores
    "compute_scores",
    # Cross-method comparison (EMNLP 2026, Section 5)
    "cross_method_jaccard",
    "CrossMethodJaccardResult",
]
