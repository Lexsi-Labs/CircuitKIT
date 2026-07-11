# Real validation suite

End-to-end validation scripts that run actual CircuitKit pipelines on real
models. Not unit tests — these verify features work on real hardware.

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
