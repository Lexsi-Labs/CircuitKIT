# Greater-than

The ACDC greater-than task: given "The {noun} lasted from the year 17XX to 17__", the model must predict a two-digit year end strictly greater than the given start year.

## Key modules

- `utils.py` — noun pool, year-prompt generation, GPT-2-small loader, the greater-than metric, the `AllDataThings` bundle, `GreaterThanConstants`, and the ground-truth circuit edges/colorscheme.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `get_all_greaterthan_things(...)` — returns the `AllDataThings` dataset/model/metric bundle.
- `get_year_data(...)` — build the year prompts.
- `greaterthan_metric(...)` and `get_greaterthan_true_edges(...)`: metric and reference circuit.

## How it fits

Registered in the task registry. Supplies the greater-than dataset, metric, and reference circuit for discovery and evaluation.
