# evaluate

Faithfulness evaluation for discovered circuits. It runs the 6-pillar suite and adds checkpoint export and cross-task transfer analysis.

## Key modules

- `evaluate.py` — core `evaluate_graph()` / `evaluate_baseline()`; `intervention="patching"` runs Pillar 1, `intervention="zero"` runs Pillar 2.
- `full.py` — `run_full_faithfulness()` orchestrator that runs all 6 pillars in cost-optimal order.
- `report.py` — `FaithfulnessReport` dataclass capturing results across all six pillars (JSON-serializable).
- `intervention_faithfulness.py` — `IF` / `IFFold` / `IFResult`: the Intervention-Faithfulness metric predicting downstream-intervention outcomes.
- `master_grid.py` — `MasterGrid` / `MasterGridCell`: runs the (method × wrapper) master grid as a library-level artifact.
- `checkpoint_benchmark.py` — export an intervened model as an HF checkpoint and run lm-eval on it (`export_and_benchmark`, `compare_base_vs_intervened`, `run_lm_eval`).
- `hf_checkpoint.py` — save/load pruned, quantized, and compressed-tensors HF checkpoints for lm-eval.
- `lm_harness.py` — lm-eval harness `LM` wrapper with hook-based node/neuron interventions on a live model.
- `lm_eval_simple.py` — simplified lm-eval wrapper reusing the harness hook builders.
- `mmlu_eval.py` — `evaluate_mmlu()`: lightweight 5-shot MMLU accuracy for any HookedTransformer.
- `weight_based_eval.py` — lm-eval on weight-pruned models (persisted weights, not live hooks).
- `stability_discovery.py` — internal helper that re-runs discovery across seeds/resamples for Pillar 3.
- `transfer.py` — `TransferMatrix`: build an N×N cross-task circuit transfer matrix.
- `transfer_analysis.py` — `TransferMatrixAnalyzer`: correlation, clustering, and transferability rankings.
- `transfer_visualizer.py` — `TransferMatrixVisualizer`: heatmaps and comparison plots.

## Public API / entry points

`evaluate_graph`, `evaluate_baseline`, `run_full_faithfulness`, `FaithfulnessReport`, `Pillar1_CausalPatching`, `Pillar2_Ablation`, `IF`/`IFFold`/`IFResult`, `MasterGrid`/`MasterGridCell`, the `StabilityReport`/`RobustnessReport`/`ComprehensiveEvaluationReport` classes, `TransferMatrix`/`TransferMatrixAnalyzer`/`TransferMatrixVisualizer`, and the HF-checkpoint helpers.

## How it fits

This package consumes discovered circuits (`Graph` objects) and a `TaskSpec`/dataloader and scores how faithfully the circuit explains model behavior. The pillar implementations live in `pillars/`; the structured stability/robustness reports live in `reports/`.
