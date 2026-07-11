# circuitkit

A Discover → Evaluate → Intervene toolkit for mechanistic interpretability of transformer models.

## Entry points

- `api.py` — core dict-config API: `discover_circuit`, `evaluate_circuit`, `load_circuit`.
- `quick.py` — flat, typed front-door API (`load_model`, `discover`, `faithfulness`, `prune`, `quantize`, `export_checkpoint`, `benchmark`, `load_scores`, `selective_finetune`, `visualize_circuit`).
- `pipeline.py` — `Pipeline`, a stateful orchestrator carrying model/circuit/eval state across the Discover→Evaluate→Intervene workflow.
- `circuit.py` — `Circuit`, the typed result wrapper returned by the flat API.

The top-level `__all__` re-exports the above plus task helpers (`get_task`, `list_tasks`, `register_task`) and version metadata; heavy symbols are lazily imported.

## Subpackages

- `backends/` — discovery-algorithm registry and stability tiers; holds the canonical algorithm names (EAP, EAP-IG, ACDC, IBCircuit, CD-T, etc.).
- `applications/` — model surgery and deployment: pruning, quantization, selective finetuning, and common probes/benchmarks.
- `evaluate/` — circuit evaluation: causal patching, ablation, and full-faithfulness (6-pillar) entry points.
- `tasks/` — `TaskSpec` abstraction, task registry, built-in tasks (e.g. IOI), and HuggingFace dataset factory.
- `data/` — data loading and preprocessing for circuit discovery.
- `corruption/` — corruption strategy classes (negation, role/entity/voice swap, paraphrase, etc.) with validators and pipeline.
- `cli/` — the `circuitkit` command-line entry point.
- `visualize/` — circuit graph, saliency, comparison, editor, gallery, Jupyter, and Streamlit visualizers.
- `artifacts/` — the `CircuitScores` / `CircuitArtifact` schemas and converters shared across backends.
- `utils/` — shared helpers: device, config, logging, memory, caching, profiling, exceptions.
- `benchmarks/` — benchmarking suite comparing circuit-guided methods against baselines (magnitude, WANDA, GPTQ, SparseGPT, random).
- `selectors/` — registry of component-scoring functions (attribution methods plus pruning/quantization baselines).
- `datasets/` — invariance-grouped dataset infrastructure for circuit evaluation.
- `analysis/` — circuit analysis: metrics, scoring, and cross-method comparison.
- `scripts/` — utility scripts for batch processing, benchmarking, and admin tasks.

## Layout

`api` / `quick` / `pipeline` are the entry points; they build dict-configs and delegate to `backends` (discovery), `evaluate` (faithfulness), and `applications` (intervention), with `artifacts` as the shared data contract and `utils` as the cross-cutting helper layer.
