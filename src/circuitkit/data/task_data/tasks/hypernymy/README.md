# Hypernymy

A taxonomic (is-a) knowledge task: given a hyponym, the model must predict its hypernym / category (e.g. "dog" -> "animal", "car" -> "vehicle").

## Key modules

- `utils.py` — (hyponym, hypernym) pairs grouped by category, prompt construction, and clean/corrupt dataset generation.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `HypernymyData` (dataclass) and `HypernymyDataset`: data containers.
- `generate_hypernymy_data(...)` — dataset builder.
- `HYPERNYMY_PAIRS` — the (hyponym, hypernym) source list.

## How it fits

Registered in the task registry. Supplies paired hypernymy data for circuit discovery and evaluation.
