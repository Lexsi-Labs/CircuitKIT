# storage

On-disk cache of generated ACDC task data, organized into one subdirectory per
task and populated by `task_data/generation`.

## Data files

Files are named `<task>_<num_examples>_<config_hash>.{pkl,json}`:

- `.pkl` — the generated tensors/prompts (written at runtime; not all present in
  the repo).
- `.json` — a metadata sidecar summarizing a generation run, with keys:
  `num_samples`, `data_types`, `tensor_shapes`, `memory_usage`,
  `generation_time`, `config_hash`, and the full `config`
  (`GenerationConfig` fields: task_name, num_examples, prompt_type, seed,
  device, metric_name, …).

Present subdirectories:

- `greaterthan/` — one config (32 examples).
- `ioi/` — several configs (8 / 16 / 32 / 64 / 500 examples, ABBA prompt type).

## How it fits

`ACDCCache` / `FileManager` read and write here to avoid regenerating identical
task data; the directory is otherwise inert (`__init__.py` only documents the
layout, e.g. planned `induction/` and `docstring/` folders).
