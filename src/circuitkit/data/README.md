# data

Data ingestion, normalization, worthiness grading, and the shared schema
raw datasets are converted into before circuit discovery.

## Key modules

- `normalized.py` — core schema: `DatasetShape` / `ContrastSource` enums,
  `ContrastiveRecord`, and `NormalizedDataset`, the shared representation
  adapters and corruption strategies both use.
- `normalized_task.py` — `NormalizedTaskSpec`, bridges a paired
  `NormalizedDataset` into the `TaskSpec` interface for `discover_circuit`
  (EAP / EAP-IG / ACDC); tokenizes clean/corrupt answers into EAP-CSV form.
- `auto_detect.py` — `auto_normalize()`: sniffs a raw dataset's shape, picks
  the right adapter (and a default corruption strategy) in priority order.
- `clean_only.py` — `clean_only_normalize()`: loads clean-only data (no
  corrupt partner) for IBCircuit / CD-T discovery.
- `template.py` — `template_normalize()`: builds fully-paired datasets from
  user-defined placeholder templates over a CSV / DataFrame / records.
- `dataset_schema.py` — `SpanDef` / `DatasetSchema`: names semantically
  equivalent token spans across examples for PEAP (position-aware EAP).
- `worthiness.py` — `DataWorthinessReport`: grades a dataset (GREEN / YELLOW /
  RED) against ~8 core checks before discovery, with suggested fixes.
- `wikitext_calibration.py` — `wikitext_calibration_batches()`: general-text
  WikiText-2 calibration corpus for Wanda / GPTQ / AWQ selectors.

## Public API / entry points

`__init__.py` declares `__all__ = []`; import helpers directly, e.g.
`auto_normalize`, `template_normalize`, `clean_only_normalize`,
`NormalizedDataset`, `NormalizedTaskSpec`, `DataWorthinessReport`.

## How it fits

This package normalizes raw datasets into contrastive pairs, grades them, and
hands them to the discovery algorithms. Subpackages: `adapters/`
(shape → records), `corruption/` (records → contrastive pairs), and
`task_data/` (built-in task datasets and ACDC generation).
