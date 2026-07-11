# Gender bias

Probes gender-stereotype behavior via occupation-to-pronoun coreference. Prompts end just before a subject pronoun ("The {occupation} said that"), with the occupation as the only gender cue (Winogender schema, Rudinger et al. 2018).

## Key modules

- `utils2.py` — the active generator. Stereotyped male/female occupation lists, pronoun templates, and clean/corrupt data generation (imported by `__init__.py`).
- `utils.py` — earlier name-based gender-bias templates and dataset classes (`GenderBiasData`, `GenderBiasDataset`, `generate_gender_bias_data`).
- `__init__.py` — re-exports the `utils2` API.

## Public API / entry points

- `generate_gender_bias_data(...)` (in `utils2.py`): dataset builder for the occupation/pronoun task.
- `FEMALE_OCCUPATIONS` / `MALE_OCCUPATIONS` / `TEMPLATES`: source data.

## How it fits

Registered in the task registry. Supplies occupation/pronoun bias data for circuit discovery and evaluation.
