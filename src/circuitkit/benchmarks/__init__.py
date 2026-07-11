"""
Workstream K: Benchmarking & Baselines

Comprehensive benchmarking suite for circuit-guided interventions.
Compares circuit-guided methods (prune, heal, steer, quantize) against
strong baselines (magnitude, WANDA, GPTQ, SparseGPT, random).

Main Classes:
- CircuitBenchmark: Unified benchmark orchestrator
- BenchmarkResult: Result container
- BenchmarkAggregator: Results aggregation and analysis
- BenchmarkReporter: Publication-quality report generation

Baseline Classes:
- MagnitudeBaseline: Weight magnitude heuristic
- WandaBaseline: Weight × Activation heuristic
- GptqBaseline: Post-training quantization
- SparseGPTBaseline: Structured pruning with Hessian
- RandomBaseline: Random selection baseline
"""

from .baselines import (
    GptqBaseline,
    MagnitudeBaseline,
    RandomBaseline,
    SparseGPTBaseline,
    WandaBaseline,
)
from .benchmark import BenchmarkResult, CircuitBenchmark
from .reporting import BenchmarkAggregator, BenchmarkReporter

__all__ = [
    # Core classes
    "CircuitBenchmark",
    "BenchmarkResult",
    # Reporting
    "BenchmarkAggregator",
    "BenchmarkReporter",
    # Baselines
    "MagnitudeBaseline",
    "WandaBaseline",
    "GptqBaseline",
    "SparseGPTBaseline",
    "RandomBaseline",
]

__version__ = "0.4.0"
