# generation

ACDC data-generation system: produce task data on demand with intelligent
caching and on-disk file management.

## Key modules

- `manager.py` — `ACDCDataManager`: main interface; `get_task_data(task_name,
  num_examples, model, …)` generates (or reuses cached) data for ioi,
  greaterthan, induction, docstring, and tracks generation stats.
- `cache.py` — `ACDCCache`: config-keyed cache that checks for existing files,
  validates integrity, and records hit/miss/generation stats.
- `utils.py` — `GenerationConfig` (hashable config dataclass), `FileManager`
  (path layout under `storage/`), and helpers `create_data_summary` /
  `validate_data_integrity`.

## Public API / entry points

From `__init__.py` `__all__`: `ACDCDataManager`, `ACDCCache`,
`GenerationConfig`, `FileManager`.

## How it fits

Sits between the per-task builders in `tasks/` and the on-disk `storage/`
directory: it configures a generation run, caches the result, and writes the
`.pkl` data plus a `.json` metadata sidecar per task/config.
