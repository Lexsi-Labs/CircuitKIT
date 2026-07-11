# baselines

Heuristic baselines for pruning and quantization. They are the comparison points for circuit-guided interventions.

## Key modules

- `magnitude.py` — `MagnitudeBaseline`: prunes weights with smallest absolute value.
- `wanda.py` — `WandaBaseline`: selects weights by weight × activation magnitude (LLM pruning heuristic).
- `random.py` — `RandomBaseline`: randomly prunes to a target sparsity (sanity-check baseline).
- `gptq.py` — `GptqBaseline`: GPTQ post-training quantization using second-order information.
- `sparsegpt.py` — `SparseGPTBaseline`: SparseGPT one-shot structured pruning with Hessian information.

## Public API / entry points

`MagnitudeBaseline`, `WandaBaseline`, `RandomBaseline`, `GptqBaseline`, `SparseGPTBaseline`.

## How it fits

These baseline classes are driven by `benchmarks/benchmark.py`'s `CircuitBenchmark` to establish the reference performance that circuit-guided interventions are measured against.
