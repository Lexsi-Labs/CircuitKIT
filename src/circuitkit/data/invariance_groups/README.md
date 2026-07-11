# invariance_groups

Data structures and a builder that turn task examples plus corruption transforms into typed, contracted invariance groups for circuit evaluation.

## Key modules

- `schema.py` — invariance-contract data structures: `VariantType` (transformation families), `InvarianceContract` (label/position/length invariance), `InvarianceVariant`, `InvarianceGroup`, `DEFAULT_CONTRACTS`, and `new_group_id`; all serialize to/from plain dicts (HuggingFace/Croissant-compatible).
- `builder.py` — `InvarianceGroupBuilder` (`from_task_examples`) wraps CircuitKit corruption transforms to produce contracted groups with length-delta annotation; `register_paraphrase_transform` registers a custom paraphrase function.

## Public API

`VariantType`, `InvarianceContract`, `InvarianceVariant`, `InvarianceGroup`, `DEFAULT_CONTRACTS`, `new_group_id`, `InvarianceGroupBuilder`, `register_paraphrase_transform`.

## How it fits

Builds on `corruption/` transforms, adding the `InvarianceContract` wrapper and QC so `datasets` can supply base and variant groups that test whether a discovered circuit preserves the specified properties.
