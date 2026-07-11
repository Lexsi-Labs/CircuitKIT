# utils

Core machinery for the ACDC backend: turning a model into an edge-patchable graph
and running/scoring patches over it.

## Key modules

- `patchable_model.py` — `PatchableModel`: a `HookedTransformer`-like wrapper whose
  computation graph can be ablated along individual edges.
- `patch_wrapper.py` — `PatchWrapperImpl`: per-module wrapper that intercepts the
  forward pass to cache source outputs and interpolate destination inputs via a mask.
- `graph_utils.py` — builds the patchable graph and provides `patch_mode`,
  `set_all_masks`, `train_mask_mode`, and model-architecture analysis.
- `ablation_activations.py` — computes source ablations (resample / zero / mean
  variants) via `src_ablations` / `batch_src_ablations`.
- `tensor_ops.py` — tensor helpers: hard-concrete sampling, prune-score ordering
  and thresholding, KL-div, answer-diff metrics.
- `task_utils.py` — `AllDataThings` dataclass and small data helpers (e.g.
  `shuffle_tensor`).
- `misc.py` — path resolution, hook-removal context manager, and module-by-name
  getters/setters.
- `custom_tqdm.py` — project-default `tqdm` wrapper.

## How it fits

Imported across `prune.py`, `prune_algos/`, and `model_utils/` in the ACDC
discovery backend.
