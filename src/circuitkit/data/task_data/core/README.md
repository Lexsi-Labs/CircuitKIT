# core

ACDC core components — the computational-graph primitives that back an ACDC
circuit-discovery experiment, ported from the Automatic-Circuit-Discovery repo.

## Key modules

- `TLACDCEdge.py` — `EdgeType` (ADDITION / DIRECT_COMPUTATION / PLACEHOLDER),
  `Edge`, and `TorchIndex` (indexing into hooked activation tensors).
- `TLACDCInterpNode.py` — `TLACDCInterpNode`: one node `(hook_name, index)` in
  the graph, tracking parents/children and its incoming edge type.
- `TLACDCCorrespondence.py` — `TLACDCCorrespondence`: the full computational
  graph, with efficient node/edge lookup maps.
- `TLACDCExperiment.py` — `TLACDCExperiment`: drives an ACDC run over a model,
  graph, and data; always minimizes its metric.
- `acdc_utils.py` — metric and container helpers (`kl_divergence`,
  `OrderedDefaultdict`, `make_nd_dict`, `extract_info`, `shuffle_tensor`, …);
  `wandb` is optional.

## Public API / entry points

`__init__.py` re-exports everything (`*`) from all five modules; the main
classes are `TLACDCExperiment`, `TLACDCCorrespondence`, `TLACDCInterpNode`,
`Edge`, `EdgeType`, `TorchIndex`.

## How it fits

The ACDC discovery algorithm builds on this graph and experiment machinery. It
is consumed by the discovery layer and by `task_data/generation`.
