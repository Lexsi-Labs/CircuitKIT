# Memory Optimization

EAP-IG can be memory-intensive, especially for large models. This page covers all available memory optimizations and their trade-offs.

---

## Memory Budget by Model Size

| Model | Base VRAM | EAP-IG default | With all optimizations |
|-------|-----------|----------------|------------------------|
| GPT-2 (124M) | ~1 GB | ~2 GB | ~0.5 GB |
| Llama-1B | ~2.5 GB | ~4 GB | ~1.5 GB |
| Llama-3B | ~7 GB | ~12 GB | ~4 GB |
| Gemma-4B | ~10 GB | ~16 GB | ~6 GB |

All estimates at `ig_steps=5`, `batch_size=4`, `num_examples=128`. (The config key is `num_examples`, under `data_params`; prose sometimes calls it `n_examples` — it is the same knob.)

---

## Optimization Techniques

### 1. Model precision (bfloat16 is the default)

CircuitKit already loads models in bfloat16 — it's the default, and the backend warns if you set anything else, so you get this saving out of the box:

```python
discover_circuit({
    "model": {"name": "gpt2", "precision": "bfloat16"},  # the default, shown explicitly
    ...
})
```

**Why it's listed:** the float32 baseline in the table below is a what-if. Forcing float32 would roughly double model VRAM for no accuracy benefit, which is exactly why bfloat16 is the default rather than an opt-in.

### 2. Reduce `ig_steps` (linear savings)

`ig_steps` is the number of Integrated Gradients integration steps. The config default (`utils/config.py`) is 5. The EAP-IG backend falls back to 30 only when `ig_steps` is left unset (`ig_steps is None`), but the default config always supplies 5, so that fallback does not apply through `discover_circuit`.

```python
"discovery": {"algorithm": "eap-ig", "ig_steps": 3, ...}  # 3 is minimum
```

**Savings:** Linear with step reduction. `ig_steps=3` is ~40% faster and ~30% less memory than `ig_steps=5`.

### 3. Reduce `batch_size` (linear savings)

`batch_size` is read at the top level of the `discovery` block, not nested under `data_params`:

```python
"discovery": {"batch_size": 1, "data_params": {"num_examples": 128}}
```

**Savings:** Linear. `batch_size=1` is ~50-60% less peak memory than `batch_size=4`. Slower (more batches).

---

## Combined Strategy for Large Models

For Gemma-4B or Llama-3B (targeting ~6 GB peak VRAM):

```python
discover_circuit({
    "model": {
        "name": "google/gemma-3-4b-it",
        "precision": "bfloat16",         # 1. half-precision model
    },
    "discovery": {
        "algorithm": "eap-ig",
        "task": "mmlu",
        "level": "node",
        "ig_steps": 3,                   # 2. fewer IG steps
        "batch_size": 1,                 # 3. batch size 1 (top-level, not under data_params)
        "data_params": {
            "num_examples": 64,          # 4. fewer examples
        },
    },
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

---

## Memory Savings Summary

| Configuration | Peak VRAM reduction |
|--------------|---------------------|
| Baseline — float32 (reference only; not the default) | 0% |
| Model bfloat16 | ~50% |
| + Batch size 1 | ~70–75% |
| + Reduced ig_steps (3) | ~75–80% |
| **All combined** | **~80%** |

There are no `memory_efficient` or `use_half_precision_activations` config keys — neither is read anywhere in the discovery backend, so setting them silently does nothing. The levers above (model precision, `batch_size`, `ig_steps`, `n_examples`) are the only ones that actually affect memory.

---

## Monitoring Memory Usage

```python
import torch

if torch.cuda.is_available():
    print(f"Before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ... run discovery ...

if torch.cuda.is_available():
    print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
```

## Pre-Flight Memory Check

Before running discovery, check if the model will fit:

```bash
circuitkit discover-smart \
    --model google/gemma-3-4b-it \
    --algorithm eap-ig \
    --task mmlu \
    --check-memory
```

This runs a dry-run estimate without loading the full dataset.

---

## Performance Impact

All optimizations have negligible impact on circuit quality:

| Optimization | Speed impact | Circuit quality impact |
|-------------|-------------|----------------------|
| `ig_steps=3` | ~40% faster | < 0.5% score diff |
| `batch_size=1` | ~50% slower | None |
| `bfloat16` model | Same or faster | < 0.1% score diff |

---

## Troubleshooting OOM

If you still hit OOM with all optimizations:

1. **Reduce `num_examples` to 32** — enough for a reasonable circuit estimate
2. **Try `ig_steps=1`** — equivalent to vanilla `eap`; slightly less accurate but same memory as non-IG methods
3. **Use CPU** — very slow but no VRAM constraint. There is no `"device"` model-config key — device selection is automatic via `get_device()` (CUDA > MPS > CPU). To force CPU, make CUDA (and MPS) unavailable to the process (e.g. `CUDA_VISIBLE_DEVICES=""`) rather than setting a config key.
4. **Consider a smaller model** — GPT-2 or Pythia-70M for algorithm prototyping

---

## Next Steps

- [Algorithms: Capability Matrix](../algorithms/capability-matrix.md) — GPU requirements by model size
- [Algorithms: EAP Family](../algorithms/eap.md) — `ig_steps` tuning guidance
- [Troubleshooting](../user-guide/troubleshooting.md) — common OOM errors
