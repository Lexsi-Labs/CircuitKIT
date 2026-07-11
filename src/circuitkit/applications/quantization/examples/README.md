# examples

Runnable end-to-end circuit-guided quantization pipelines for specific model families.

## Key modules

- `quantize_llama.py` — full circuit-guided quantization pipeline for LLaMA / Llama-2 / Llama-3.
- `quantize_qwen.py` — full circuit-guided quantization pipeline for Qwen3.

## Public API / entry points

None — these are command-line scripts rather than an importable API. `__init__.py` is an empty namespace file.

## How it fits

Reference drivers that combine the `quantization/` package (score extraction, selectors, quantization backends) into complete workflows you can run against real models.
