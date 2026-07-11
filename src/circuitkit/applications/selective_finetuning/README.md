# selective_finetuning

Circuit-guided selective fine-tuning: train only the most important components (heads / MLP layers / neurons) via gradient masking.

## Key modules

- `selector.py` — `select_components`, `SelectionResult`, `random_selection`, `build_baseline_selection`, `print_selection_summary`: selects the top-X% important components and resolves them to concrete weight-matrix index ranges.
- `score_loader.py` — loads and normalises circuit-discovery scores into a consistent form, handling node-level (any algorithm), neuron-level EAP/EAP-IG, and neuron-level IBCircuit formats.
- `finetune_utils.py` — `setup_selective_training`, `verify_gradient_masking`, `LanguageModelingDataset`, `build_finetune_dataloader`, `run_finetuning`: consume a `SelectionResult` and apply gradient-masked training to a HuggingFace causal LM.

## Public API / entry points

No `__all__` — `__init__.py` is an empty namespace file. Import the functions/types directly from the modules above (e.g. `selector.select_components`, `finetune_utils.run_finetuning`). Runnable pipelines live in `examples/`.

## How it fits

Circuit scores pick which components to train; only those receive gradients during fine-tuning.
