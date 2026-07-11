"""
Baseline implementations for pruning and other interventions.

These provide simple heuristic baselines for comparing against
circuit-guided interventions.
"""

from .gptq import GptqBaseline
from .magnitude import MagnitudeBaseline
from .random import RandomBaseline
from .sparsegpt import SparseGPTBaseline
from .wanda import WandaBaseline

__all__ = [
    "MagnitudeBaseline",
    "WandaBaseline",
    "GptqBaseline",
    "SparseGPTBaseline",
    "RandomBaseline",
]
