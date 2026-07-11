# pyfunctions

Vendored CD-T reference implementation (contextual decomposition propagation ops).
Adapted from the CD_Circuit reference repo; modules use top-level
`from pyfunctions.X import Y` imports.

## Key modules

- `cdt_core.py` — core decomposition ops (`normalize_rel_irrel`, `prop_linear`,
  attention/layer-norm propagation) and model utilities.
- `cdt_basic.py` — vanilla contextual decomposition for an HF BERT model (worked
  example, no patching).
- `cdt_ablations.py` — rel/irrel patching to decompose intermediate-layer nodes.
- `cdt_from_source_nodes.py` — BERT decomposition patched at source nodes.
- `cdt_source_to_target.py` — the general source→target decomposition method
  (`batch_run`, `prop_GPT`, etc.).
- `wrappers.py` — `GPTAttentionWrapper` / `GPTLayerNormWrapper` and helper types
  (`Node`, `AblationSet`, `TargetNodeDecompositionList`) adapting GPT modules to
  the BERT-style code path.
- `faithfulness_ablations.py` — IOI-circuit faithfulness ablation hooks (ARENA 3.0).
- `toy_model.py` — 4-layer attention-only transformer example for the docstring task.
- `ioi_dataset.py` — IOI dataset (from EasyTransformer).
- `local_importance.py` — LIME / SHAP / integrated-gradients baselines for comparison.
- `pathology.py` — pathology-report dataset field definitions (BERT demo).
- `general.py` — misc I/O and list/string utilities.

## How it fits

Internal implementation invoked through `backends/cdt/adapter.py` and the CD-T
`__init__` re-exports; not part of the public CircuitKit API.
