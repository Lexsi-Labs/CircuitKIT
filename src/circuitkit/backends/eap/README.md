# eap

Backend for the EAP (Edge Attribution Patching) family. This is the discovery
pipeline used in production. It computes edge/node attribution scores over a
TransformerLens computation graph and supports EAP-IG plus several research variants.

## Key modules

- `graph.py` — `Graph` and `Node` classes (`InputNode`, `AttentionNode`, `MLPNode`,
  `LogitNode`) representing the transformer computation graph and its scores.
- `attribute.py` — edge-level scoring; `attribute` dispatches over methods (`EAP`,
  `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`,
  `information-flow-routes`, `exact`) via `get_scores_*` functions.
- `attribute_node.py` — node-level scoring and research variants (`get_scores_ifr`,
  `get_scores_peap`, `get_scores_relp`, `get_scores_eap_gp`,
  `get_scores_atp_grad_drop`, ...); entry point `attribute_node`.
- `evaluate.py` — `evaluate_baseline` / `evaluate_graph` for measuring circuit
  faithfulness under a metric.
- `metrics.py` — multi-token-aware metrics (logit-diff, KL-div, accuracy,
  perplexity) via answer spans.
- `eap_utils.py` — tokenization, mean-activation caching, hook/matrix construction,
  and the `collate_EAP` batcher.
- `circuit_kit_adapter.py` — converts an EAP `Graph` into CircuitKit node/edge
  score dicts; perplexity helper.
- `visualization.py` — color helpers for rendering edges by QKV type / score.
- `artifact_export.py` — `export_circuit_artifact`: EAP node scores → `CircuitArtifact`.

## How it fits

Dispatched from `api.discover_circuit` for `eap`/`eap-ig` (stable) and the research
variants; `__init__.py` is a placeholder (no `__all__`) — entry points are imported
by module path.
