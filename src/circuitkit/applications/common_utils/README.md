# common_utils

Shared application utilities: linear probes, hallucination detection, circuit-restricted unlearning, PEFT benchmark analysis, and editing corpus statistics.

## Key modules

- `linear_probe.py` — `LinearProbe` and `ProbeTrainer`: simple linear probes trained on circuit activations to detect hallucination signals during generation.
- `hallucination_detection.py` — `HallucinationDetector` (and `HallucinationDataset`): monitors circuit activations during generation, training probes on clean/corrupted activation pairs.
- `cure_clue.py` — `CureClueUnlearner` (config `CureClueConfig`): CURE/CLUE circuit-restricted unlearning via gradient ascent on the forget loss constrained to circuit MLP layers, with an optional retain-set KL loss (CLUE).
- `benchmark_analysis.py` — `BenchmarkAnalysis` and `MethodRecommendation`: analyzes PEFT benchmark results into performance comparisons, trade-off analysis, and method recommendations.
- `_covariance.py` — internal ROME/MEMIT corpus-statistic estimation (`get_covariance`, `solve_with_C`).
- `_metrics.py` — internal paper-canonical knowledge-editing metrics (efficacy, paraphrase, neighborhood, generation entropy, editing score).
- `_tokenization.py` — internal single source of truth for tokenizer-aware operations (target formatting, teacher forcing, subject location, scoring).

## Public API / entry points

`__all__`: `LinearProbe`, `ProbeTrainer`, `HallucinationDetector`, `CureClueUnlearner`, `BenchmarkAnalysis`, `MethodRecommendation`.

## How it fits

Helpers used by the intervention applications. The `_`-prefixed modules back the editing and unlearning workflows.
