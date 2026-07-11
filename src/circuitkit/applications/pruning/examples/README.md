# examples

Runnable end-to-end circuit-guided pruning pipelines for specific model families.

## Key modules

- `prune.py` — universal circuit-guided pruning pipeline for any HuggingFace CausalLM (architecture auto-detected).
- `prune_llama.py` — full circuit-guided plus random-baseline pruning pipeline for LLaMA / Llama-2 / Llama-3.
- `prune_qwen.py` — full circuit-guided plus random-baseline pruning pipeline for Qwen3.

## Public API / entry points

None — these are command-line scripts (argparse-driven) rather than an importable API. `__init__.py` is an empty namespace file.

## How it fits

Reference drivers that wire together the `pruning/` package (score extraction, selectors, `StructuralPruner`, evaluation) into complete workflows you can run against real models.
