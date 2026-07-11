# IOI (Indirect Object Identification)

The IOI task: in sentences like "When John and Mary went to the store, John gave a drink to ___", the model must predict the indirect object (Mary) — the name that appeared once rather than twice.

## Key modules

- `ioi_dataset.py` — the `IOIDataset` class plus prompt generators (`gen_prompt_uniform`, `gen_flipped_prompts`) and token-index helpers. Adapted from Redwood's Easy-Transformer.
- `utils.py` — GPT-2-small loaders, the `AllDataThings` bundle, IOI metrics (logit diff, KL), the ground-truth circuit (`get_ioi_true_edges`, `Conn`), and colorscheme.
- `__init__.py` — exports `IOIDataset` and re-exports the utils API.

## Public API / entry points

- `IOIDataset` — clean/corrupt dataset class (also reused by `double_io` and `greaterthan`).
- `get_all_ioi_things(...)` — returns the `AllDataThings` bundle.
- `get_gpt2_small(...)` / `get_ioi_true_edges(...)`: model loader and reference circuit.

## How it fits

Registered in the task registry. Supplies the IOI dataset, model, metrics, and reference circuit for discovery and evaluation.
