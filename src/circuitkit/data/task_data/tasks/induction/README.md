# Induction

The induction task: on sequences with a repeated random prefix, the model must copy the token that followed the earlier occurrence. This is the canonical behavior of induction heads.

## Key modules

- `utils.py` — loads Redwood's `redwood_attn_2l` model, downloads validation data from the Hub, builds repeat/induction candidate masks, and assembles the `AllDataThings` bundle and metrics.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `get_all_induction_things(...)` — returns the `AllDataThings` dataset/model/metric bundle.
- `get_model(...)` — loads the 2-layer attention-only reference model.
- `get_good_induction_candidates(...)` / `get_mask_repeat_candidates(...)`: position masks for induction.

## How it fits

Registered in the task registry. Supplies the induction dataset, model, and metrics for circuit discovery and evaluation.
