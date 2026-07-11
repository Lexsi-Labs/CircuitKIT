# Troubleshooting

Common issues and their solutions.

## Installation

### ImportError: No module named 'transformer_lens'

```bash
pip install -e . --no-cache-dir
```

### ModuleNotFoundError: No module named 'circuitkit'

```bash
pip install -e . --force-reinstall --no-deps
python -c "import circuitkit; print(circuitkit.__version__)"
```

### spacy model not found

```bash
python -m spacy download en_core_web_sm
```

### lm-eval not working

```bash
pip install -e ".[benchmarks]"
```

## GPU / Memory

### CUDA out of memory during discovery

Reduce example count or batch size:
```python
discover_circuit({
    "discovery": {"data_params": {"num_examples": 32, "batch_size": 2}, ...},
    ...
})
```

Or enable memory-efficient discovery:
```bash
circuitkit discover-smart --model gpt2 --algorithm eap-ig --task ioi --check-memory
```

### CUDA out of memory during install

Install CPU-only first, then swap in the GPU torch wheel:
```bash
pip install -e .
pip install torch==2.6.0+cu126 -f https://download.pytorch.org/whl/cu126
```

### Model too large for GPU

- Reduce to `float16` or `bfloat16` precision
- Use CPU fallback for small models (GPT-2): `device="cpu"`
- Split discovery across layers (not yet supported — use smaller models)

## Discovery

### Algorithm produces empty circuit

- Increase `target_sparsity` (you may be pruning too aggressively)
- Check `n_examples` — too few examples can produce noisy scores
- Verify the task metric: run `circuitkit debug test --model gpt2` to test

### Algorithm emits UserWarning

This is expected for non-Stable algorithms:
```bash
UserWarning: Algorithm 'acdc' is experimental. May fail on larger models or non-IOI tasks. Use 'eap-ig' for production.
```
Suppress with `warnings.filterwarnings("ignore")` only after verification.

### Discovery is very slow

- Use `eap` instead of `eap-ig` (30% faster)
- Reduce `ig_steps` (for `eap-ig`)
- Reduce `n_examples`
- Increase `batch_size` (if VRAM allows)

## Evaluation

### Pillar 4 (Robustness) fails

Requires the spaCy `en_core_web_sm` model (spaCy itself ships with the base install):
```bash
python -m spacy download en_core_web_sm
```

### Pillar 6 (Generalization) is slow

Generalization re-discovers the circuit on the target task. Limit `n_examples` or skip if not needed.

### Pillar 6 returns None

`target_task` was not supplied. Pass it explicitly:
```python
pipe.evaluate(pillars=None, target_task="sva")
```

## Pruning / Export

### Export checkpoint is much larger than expected

`export_checkpoint` writes the full model with zero-masked weights, not a sparse format. Use `torch.save` on the state dict for a smaller archive.

### Pruned model produces nonsense output

- The circuit may not be faithful at this sparsity level. Run evaluation first.
- Try lower sparsity (keep more components).

### Reloaded checkpoint has different behaviour

Ensure the reloaded model uses the same dtype and device as the original:
```python
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
```

## Visualization

### Graph is empty

The circuit has no node scores to plot. Re-run discovery with an `output_path` so scores are captured, then render:
```python
ck.visualize_circuit(circuit, mode="graph", output="circuit.html")
```

### Plotly not rendering in Jupyter

```bash
pip install -U plotly jupyterlab "ipywidgets>=7.6"
```

## Next steps

- [:octicons-arrow-right-24: FAQ](../about/faq.md)
- [:octicons-arrow-right-24: Memory Optimization](../advanced/memory-optimization.md)
