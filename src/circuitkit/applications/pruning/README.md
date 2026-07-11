# pruning

Circuit-guided structural pruning of transformer models, plus scoring, selectors, and post-pruning evaluation/fine-tuning utilities.

## Key modules

- `pruner.py` — `StructuralPruner`: structured *masking* of attention heads and MLP layers (zeroes whole heads/MLP output projections in place; does not physically remove parameters).
- `node_pruner.py` — `NodePruner` and `get_nodes_to_prune`: node-level pruning selection.
- `weight_pruner.py` — `zero_attention_head_weights` and `get_attention_architecture_info`: zero out per-head attention weights.
- `neuron_pruner.py` — `zero_neuron_hook` / `zero_attn_neuron_hook`: forward hooks for neuron-level zeroing.
- `importance.py` — `CircuitKitImportance`: bridges circuit-discovery scores into the LLM-Pruner importance interface.
- `score_extractor.py` — circuit-guided pruning score extraction (`run_discovery`, `extract_*`, `build_importance_dict`).
- `eval_utils.py` — task-level evaluation for HF models (`eval_hf_model_on_task`, `measure_latency`, `full_eval`).
- `finetune_utils.py` — LoRA-based post-pruning fine-tuning (Alpaca instruction-following or task-data modes).

## Public API / entry points

`__all__`: `StructuralPruner`, `zero_attention_head_weights`, `get_attention_architecture_info`, `NodePruner`, `get_nodes_to_prune`.

## How it fits

Circuit scores drive which heads, MLPs, and neurons to prune. Selectors live in `selectors/` and runnable pipelines in `examples/`.
