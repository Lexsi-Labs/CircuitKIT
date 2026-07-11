# Double IO

An IOI variant where both the subject and the indirect object appear twice in the prompt. The naive "remove duplicate tokens" IOI algorithm should fail on it even though GPT-2 still succeeds.

## Key modules

- `double_io_dataset.py` — generates DoubleIO clean and corrupted prompts by reusing IOI's name/place/object pools, with templates that insert a clause making the indirect object appear a second time. References "Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability" (arXiv 2411.16105).
- `__init__.py` — package marker.

## Public API / entry points

- `gen_double_io_prompts(...)` — build clean DoubleIO prompts.
- `gen_double_io_corrupted_prompts(...)` — build the corrupt counterparts.
- `get_double_io_data_only(...)` — assemble the tokenized dataset.

## How it fits

Registered in the task registry. Supplies a generalization-probing IOI variant for circuit discovery and evaluation.
