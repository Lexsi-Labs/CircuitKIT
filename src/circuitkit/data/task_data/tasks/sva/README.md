# SVA (Subject-Verb Agreement)

A subject-verb number-agreement task: given a prompt ending at a singular or plural subject ("The dog" vs. "The dogs"), the model must predict a number-matching verb (e.g. " is" vs. " are"). The number-contrastive structure follows Linzen et al. (2016) and Lakretz et al. (2021).

## Key modules

- `utils.py` — singular/plural subject pairs, single-token verb pairs, templates, verb-token resolution, and clean/corrupt data generation.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `generate_sva_data(...)` — dataset builder producing number-contrastive clean/corrupt pairs.
- `SUBJECT_PAIRS` / `VERB_PAIRS` / `TEMPLATES`: source data.

## How it fits

Registered in the task registry. Supplies subject-verb agreement data for circuit discovery and evaluation.
