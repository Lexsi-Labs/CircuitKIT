# examples

Runnable end-to-end selective fine-tuning pipelines for specific model families.

## Key modules

- `finetune_llama.py` — selective fine-tuning pipeline for LLaMA-family models (load scores, read config, load model, evaluate base, build dataloader, run circuit/random/baseline finetuning).
- `finetune_qwen.py` — selective fine-tuning pipeline for Qwen-family models; structurally identical to `finetune_llama.py`, differing in model loading and config reading.

## Public API / entry points

None — these are command-line scripts rather than an importable API. `__init__.py` is an empty namespace file.

## How it fits

Reference drivers that wire together `score_loader`, `selector`, and `finetune_utils` from the `selective_finetuning/` package into complete workflows.
