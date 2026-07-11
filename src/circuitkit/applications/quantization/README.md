# quantization

Circuit-guided mixed-precision quantization of transformer models, with two backends and node-level score extraction.

## Key modules

- `quant_utils.py` — `circuit_quantize`, `build_random_quantization_plan`, `compute_ppl`: circuit-guided mixed-precision quantization via optimum-quanto, assigning each layer to a quantization tier from circuit scores.
- `llmcompressor_quantize.py` — `llmcompressor_circuit_quantize`, `build_ignore_patterns`, `is_llmcompressor_quantized`, `SUPPORTED_BITS`: true low-bit, circuit-aware mixed-precision quantization via llm-compressor + compressed-tensors.
- `score_extractor.py` — node-level circuit discovery for quantization (`run_discovery`, `extract_node_head_scores`, `extract_node_mlp_scores`, `aggregate_attn_layer_scores`, `save_scores`, `load_scores`).

## Public API / entry points

`__all__`: `circuit_quantize`, `build_random_quantization_plan`, `compute_ppl`, `llmcompressor_circuit_quantize`, `build_ignore_patterns`, `is_llmcompressor_quantized`, `SUPPORTED_BITS`.

## How it fits

Circuit scores decide per-layer precision. Selectors live in `selectors/` and runnable pipelines in `examples/`.
