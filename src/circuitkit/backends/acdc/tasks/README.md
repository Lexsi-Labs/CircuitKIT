# tasks

Task datasets and metric constructors for the ACDC backend. Each `get_all_*_things`
builder returns an `AllDataThings` bundle (clean/corrupt data, labels, metrics).

## Key modules

- `ioi_dataset.py` — `IOIDataset`: generates Indirect Object Identification prompts
  from name/template banks.
- `ioi_utils.py` — `get_all_ioi_things`: builds clean/corrupt IOI data and metrics
  for a given model.
- `docstring_prompts.py` — random Python-docstring prompt generation
  (`docstring_prompt_gen`, `BatchedPrompts`) for the docstring task.
- `docstring_utils.py` — `get_all_docstring_things`: assembles the docstring task's
  data and metrics.
- `induction_utils.py` — `get_all_induction_things` plus KL-divergence and related
  metric functions for the induction task.

## Public API / entry points

`__init__.py` re-exports the submodules `docstring_utils`, `induction_utils`,
`ioi_utils` (via `__all__`).

## How it fits

Provides the evaluation data/metrics that the ACDC prune algorithms optimize
faithfulness against.
