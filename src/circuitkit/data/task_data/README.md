# task_data

Built-in task datasets and the ACDC data-generation system (core graph
components, cached generation, and stored data).

## Key modules

This folder is mostly a namespace over its subpackages; `__init__.py` re-exports
the generation entry points.

## Public API / entry points

From `__init__.py` `__all__`: `ACDCDataManager`, `ACDCCache` (re-exported from
`generation/`).

## How it fits

Groups everything task-specific for circuit discovery. Subpackages:

- `core/` — ACDC computational-graph primitives (correspondence, edges, nodes,
  experiment, utils), copied from the Automatic-Circuit-Discovery repo.
- `generation/` — config-driven data generation with intelligent caching and
  file management (`ACDCDataManager`, `ACDCCache`, `GenerationConfig`).
- `storage/` — on-disk generated data organized by task (e.g. `ioi/`,
  `greaterthan/`).
- `tasks/` — per-task dataset builders (ioi, greaterthan, induction, docstring,
  sva, hypernymy, gender_bias, capital_country, double_io, binary_align, wmdp).
