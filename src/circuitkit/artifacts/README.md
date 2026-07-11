# artifacts

Score and circuit representations shared across all discovery backends, with converters and serialization.

## Key modules

- `scores.py` — `CircuitScores`: the node-level importance-score artifact (JSON serialization, normalization, algorithm/task/model metadata) emitted by every backend.
- `circuit_artifact.py` — `CircuitArtifact` schema (v1.0) plus `Node`, `Edge`, `NodeType`: standardized graph structure with metadata, validation, and conversion to intervention masks across ACDC/EAP/EAP-IG/IBCircuit.
- `converters.py` — `acdc_to_artifact`, `eap_to_artifact`, `ibcircuit_to_artifact`, `normalize_importance_scores`: build `CircuitArtifact`s from method-specific outputs.

## Public API

`CircuitScores`, `CircuitArtifact`, `Node`, `Edge`, `NodeType`, `acdc_to_artifact`, `eap_to_artifact`, `ibcircuit_to_artifact`, `normalize_importance_scores`.

## How it fits

The data contract between discovery backends and downstream consumers (evaluate, applications, visualize), so every algorithm's output can be handled the same way.
