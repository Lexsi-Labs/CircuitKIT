# tasks

The `TaskSpec` abstraction and its registry, used for multi-task circuit discovery. It also handles dataset validation and auto-schema detection for arbitrary HuggingFace datasets.

## Key modules

- `specs.py` — `TaskSpec` protocol: the interface every task specification implements.
- `generic.py` — `GenericTaskSpec`: bring-your-own-dataset spec for CSV/JSONL/HF sources with schema mapping and corruption.
- `registry.py` — central `register_task` / `get_task` / `list_tasks` registry.
- `bootstrap.py` — `_bootstrap_builtin_tasks()`: single source of truth registering all built-in tasks (idempotent).
- `auto_schema.py` — `SchemaAnalyzer`, `TaskType`, `TaskTypeDetection`: infer task type from HF dataset columns.
- `hf_factory.py` — `auto_task_from_hf`, `preview_schema`, `list_compatible_datasets`, `validate_hf_dataset`, `SchemaPreview`.
- `validator.py` — `DatasetValidator` / `validate_dataset` / `ValidationResult`: schema, type, and format validation.
- `yaml_loader.py` — declarative task definition from YAML (no Python required).
- `safety_datasets.py` — lazy convenience registrations for safety / red-team datasets (AdvBench, etc.).
- `inspect.py` — CLI (`python -m circuitkit.tasks.inspect`) to inspect a dataset and report the auto-selected type spec.
- `_chat.py` — chat-template handling (`chat_template_mode`) so discovery and downstream stages share prompt formatting.
- `_algorithm_families.py` — shared per-task algorithm whitelists (EAP / ACDC / IB families).

## Public API / entry points

`TaskSpec`, `GenericTaskSpec`, `register_task`/`get_task`/`list_tasks`, `DatasetValidator`/`validate_dataset`/`ValidationResult`, `SchemaAnalyzer`/`TaskType`/`TaskTypeDetection`, `auto_task_from_hf`/`preview_schema`/`list_compatible_datasets`/`validate_hf_dataset`/`SchemaPreview`, `IOITaskSpec`.

## How it fits

Task specs produce the contrastive dataloaders and metrics that circuit discovery and evaluation consume. Concrete built-in tasks live in `builtins/`; reusable per-task-type base classes live in `type_specs/`.
