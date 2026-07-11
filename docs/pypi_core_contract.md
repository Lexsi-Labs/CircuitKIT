# PyPI Core Engine Architectural Contract — Deprecated

!!! danger "This document is deprecated and describes a fictional API"
    Everything below described an API surface that was never implemented: a `QuickCircuit` class, `circuitkit.seed_everything()`, eager top-level imports of `CircuitArtifact`/`FaithfulnessReport`/`CircuitScores`, and `discover_circuit`/`evaluate_circuit`/`prune` signatures that do not match the real functions. None of it exists in `circuitkit` v1.0.0. This page is excluded from the published documentation site (see `exclude_docs` in `mkdocs.yml`) and is kept only as a removal marker for anyone browsing the repository source.

    For the real, verified public API, use:

    - [Flat Typed API](api-reference/flat-api.md) — `ck.load_model`, `ck.discover`, `ck.faithfulness`, `ck.prune`, `ck.quantize`, `ck.export_checkpoint`, `ck.benchmark`, `ck.load_scores`, `ck.selective_finetune`, `ck.visualize_circuit`
    - [Dict-Config API](api-reference/dict-config.md) — `discover_circuit(config)`, `evaluate_circuit(config, ...)`, `load_circuit(path)`
    - [Pipeline Class](api-reference/pipeline.md) — stateful method-chaining interface
    - [Backends](api-reference/backends.md) — stability tiers and algorithm registry

    There is no reproducibility-seed function (`seed_everything`), no `ModelLoadError` / `GPUMemoryError` exception types, and no config-hash audit-metadata embedding in artifacts. Discovery errors are raised as `AlgorithmError`, `ValueError`, or `RuntimeError` — see the pages linked above for the actual error contracts.
