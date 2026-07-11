# model_utils

Builds the factorized computation-graph nodes (source and destination nodes of
each edge) that the ACDC backend patches over.

## Key modules

- `transformer_lens_utils.py` — `factorized_src_nodes` / `factorized_dest_nodes`
  for a `HookedTransformer`; enumerates residual, attention-head, and MLP nodes,
  handling separate QKV and Grouped-Query Attention.
- `micro_model_utils.py` — a tiny toy `MicroModel` (two multiply "heads" per
  layer) with matching node builders, used for testing.

## How it fits

Consumed by `acdc/utils/graph_utils.py` when wrapping a model into a
`PatchableModel` for the ACDC discovery backend.
