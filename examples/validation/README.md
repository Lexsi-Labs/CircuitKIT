<!-- circuitkit-logo -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/circuitkit-logo-white.png">
    <img src="../../docs/assets/circuitkit-logo-black.png" width="200" alt="CircuitKIT">
  </picture>
</p>

# Real validation suite

End-to-end validation scripts that run actual CircuitKIT pipelines on real
models. Not unit tests — these verify features work on real hardware.

The scripts use deliberately small `num_examples` (8 to 24) so they finish quickly. That is below
the sensible minimum of 32, so CircuitKIT prints a `HyperparameterWarning` for them; this is expected
and does not affect the result.

## Structure

```
examples/validation/
├── README.md
├── _common.py              # shared fixture (GPT-2 IOI discovery, cached)
├── _runner.py              # runs all validations, aggregates results
├── algos/                  # 12 algorithm validation scripts
├── applications/           # 28 application validation scripts
├── benchmark/              # 18 benchmark validation scripts
├── data/                   # 12 data pipeline validation scripts
├── visualizations/         # 11 visualization validation scripts
└── master_aggregator.py    # results aggregator
```

## Run

```bash
# Single validation
python examples/validation/visualizations/01_circuit_graph.py

# All validations
python examples/validation/_runner.py
```
