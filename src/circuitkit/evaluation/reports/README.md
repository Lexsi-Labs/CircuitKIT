# reports

JSON-serializable report classes that hold stability and robustness evaluation results.

## Key modules

- `stability_report.py` — `StabilityReport`: Pillar 3 output (Jaccard/Dice overlap, layer-wise breakdown, bootstrap metrics).
- `robustness_report.py` — `RobustnessReport`: Pillar 4 output (per-corruption performance, relative degradation, baseline comparison).
- `aggregator.py` — `StabilityRobustnessReport` (combined Pillar 3 + 4) and `ComprehensiveEvaluationReport` (all pillars); handles JSON serialization and summary statistics.

## Public API / entry points

`StabilityReport`, `RobustnessReport`, `StabilityRobustnessReport`, `ComprehensiveEvaluationReport`.

## How it fits

The pillar classes in `evaluate/pillars/` and the `run_full_faithfulness()` orchestrator populate these dataclasses; they are the serializable outputs written to disk and consumed by downstream reporting. Distinct from the top-level `evaluate/report.py` `FaithfulnessReport`, which these can be aggregated alongside.
