# finetuning

Circuit-guided fine-tuning: LoRA healing of pruned models, circuit tuning, and PEFT methods with benchmarks (usable via `circuitkit.applications.finetuning`, not part of the flat public API).

## Key modules

- `soft_healing.py` — `CircuitLoRA` and `LoRALayer`: soft healing that applies Low-Rank Adaptation to circuit-relevant modules to recover pruned-model performance.
- `circuit_tuning.py` — `CircuitTuner` (config `CircuitTunerConfig`): LoRA fine-tuning restricted to weight matrices of high-scoring circuit nodes.
- `peft_methods.py` — `CircuitPEFT` and `PEFTComposer`: circuit-aware parameter-efficient fine-tuning (LoRA, Adapter, Prefix-tuning, BitFit) targeted to specific circuit neurons.
- `healing_metrics.py` — `HealingMetrics`, `HealingEvaluator`, `compute_recovery_metrics`: measure recovery of pruned-model performance after soft healing.
- `benchmark_peft.py` — `BenchmarkMetrics`: benchmarks PEFT methods across multiple architectures.

## Public API / entry points

`__all__`: `CircuitLoRA`, `LoRALayer`, `HealingMetrics`, `HealingEvaluator`, `compute_recovery_metrics`, `CircuitTuner`, `CircuitPEFT`, `PEFTComposer`.

## How it fits

One of the intervention applications. It heals or adapts models by concentrating parameter-efficient fine-tuning on circuit-relevant weights, typically after pruning.
