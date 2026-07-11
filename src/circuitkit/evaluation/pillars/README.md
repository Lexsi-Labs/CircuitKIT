# pillars

The 6-pillar faithfulness framework, with a supplementary seventh reliability pillar, for evaluating circuit quality.

## Key modules

- `causal_patching.py` — Pillar 1 (Causal Patching): `Pillar1_CausalPatching`; interchange-intervention faithfulness ratio F = (y_circuit − y_corrupt) / (y_clean − y_corrupt).
- `ablation.py` — Pillar 2 (Ablation): `Pillar2_Ablation`; keeps circuit nodes, ablates the rest, measuring circuit sufficiency via the same faithfulness ratio.
- `stability.py` — Pillar 3 (Stability): `Pillar3_Stability`; re-runs discovery across seeds and computes Jaccard/Dice overlap.
- `robustness.py` — Pillar 4 (Robustness): `Pillar4_Robustness`; evaluates faithfulness under paraphrase, entity-swap, and other corruptions (generate-or-skip contract).
- `baselines.py` — Pillar 5 (Baseline Comparison): `Pillar5_Baselines`; compares the circuit against random, magnitude, and Wanda baselines.
- `generalization.py` — Pillar 6 (Generalization): `Pillar6_Generalization`; scores a source-task circuit on related target tasks.
- `intervention_reliability.py` — Pillar 7 (Intervention Reliability, supplementary): consistency of a circuit-guided intervention across seeds and prompt variations (not exported in `__all__`).

## Public API / entry points

`Pillar1_CausalPatching`, `Pillar2_Ablation`, `Pillar3_Stability`, `Pillar4_Robustness`, `Pillar5_Baselines`, `Pillar6_Generalization`.

## How it fits

These pillar classes are composed by `evaluate/full.py`'s `run_full_faithfulness()` orchestrator; each returns metrics that feed the `FaithfulnessReport` and the structured reports in `evaluate/reports/`.
