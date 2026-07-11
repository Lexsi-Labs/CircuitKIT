# acdc

Backend for the ACDC (Automatic Circuit DisCovery, Conmy et al. 2023) discovery
algorithm. It is adapted from the `auto_circuit` codebase and built on an
edge-patchable TransformerLens model.

## Key modules

- `types.py` — core dataclasses/enums: `PromptPair`, `PromptPairBatch`, `Edge`,
  `Node`, `SrcNode`/`DestNode`, `PruneScores`, `PatchType`, `AblationType`,
  `PatchWrapper`, and the color palette.
- `data.py` — `PromptDataset` / `PromptDataLoader` and a device-aware
  `collate_fn_factory` for clean/corrupt prompt batches.
- `prune.py` — `run_circuits`: runs the patchable model while keeping only the
  top-scored edges for each requested edge count.
- `visualize.py` — Sankey-style circuit rendering (`net_viz`, `draw_seq_graph`).
- `artifact_export.py` — `export_circuit_artifact`: converts ACDC prune_scores +
  edges into the unified `CircuitArtifact` schema.

## Subpackages

- `prune_algos/` — scoring algorithms (ACDC threshold sweep, mask-gradient/EAP).
- `model_utils/` — factorized graph node builders for TransformerLens / a toy model.
- `tasks/` — IOI, docstring, and induction task datasets and metrics.
- `utils/` — patchable-model machinery, ablation activations, graph/tensor ops.

## How it fits

Dispatched from `api.discover_circuit` (`acdc`, experimental tier); its ACDC
prune-score entry point is imported from `prune_algos.ACDC`.
