# Docstring

The ACDC docstring task: given a Python function signature and a partially written docstring, the model must predict the next argument name in the docstring. This is a variable-binding/induction-style task.

## Key modules

- `prompts.py` — prompt templates and generators (`docstring_prompt_gen`, `docstring_induction_prompt_generator`) plus `Prompt` / `BatchedPrompts` containers. Builds argument-name docstring examples from single-token noun pools.
- `utils.py` — loads the `attn-only-4l` model, assembles the `AllDataThings` bundle, defines metrics, and exposes the ground-truth circuit edges.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `get_all_docstring_things(...)` — returns the `AllDataThings` dataset/model/metric bundle.
- `get_docstring_model(...)` — loads the reference model.
- `get_docstring_subgraph_true_edges()` — canonical ground-truth circuit.

## How it fits

Registered in the task registry. Supplies the docstring dataset, model, metric, and reference circuit for discovery and evaluation.
