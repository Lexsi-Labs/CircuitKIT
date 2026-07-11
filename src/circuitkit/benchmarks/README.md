# benchmarks

The benchmarking suite. It compares circuit-guided interventions (prune, heal, steer, quantize) against baselines.

## Key modules

- `benchmark.py` — `CircuitBenchmark` orchestrator and `BenchmarkResult` container; runs interventions and baselines over a benchmark grid.
- `reporting.py` — `BenchmarkAggregator` (multi-dimensional results aggregation and analysis) and `BenchmarkReporter` (publication-quality report generation).

## Public API / entry points

`CircuitBenchmark`, `BenchmarkResult`, `BenchmarkAggregator`, `BenchmarkReporter`, and the baseline classes (`MagnitudeBaseline`, `WandaBaseline`, `GptqBaseline`, `SparseGPTBaseline`, `RandomBaseline`).

## How it fits

The benchmark runner applies circuit-guided methods alongside the heuristic baselines in `baselines/`, then aggregates and reports the comparison. This is how the suite measures discovered circuits against those baselines.
