# EAP-IG Memory Optimization Guide

This document explains the memory optimizations that actually exist for EAP-IG discovery in CircuitKit. See [Advanced: Memory Optimization](../advanced/memory-optimization.md) for the canonical reference; this page focuses specifically on EAP-IG.

!!! warning "No `memory_efficient` or `use_half_precision_activations` config keys"
    Earlier drafts of this guide documented `discovery.memory_efficient` and `discovery.use_half_precision_activations` config keys with specific memory-savings figures. **Neither key is read anywhere in the EAP-IG backend or `api.py` — setting them silently does nothing.** They have been removed below. The real levers are model precision, `batch_size`, `ig_steps`, and (for MMLU/WMDP) `samples_per_subject`.

## Available Optimizations

### 1. Reduced IG Steps

**What it does:**
- Fewer integrated gradient steps = less memory accumulation
- Standard recommendation: 5 steps (the current CircuitKit default)

**Usage:**
```python
config = {
    'discovery': {
        'ig_steps': 3,  # minimum for reasonable results; default is 5
        ...
    }
}
```

**Trade-offs:**
- Slightly less smooth gradients (usually negligible for circuit discovery)
- Savings are roughly linear with step reduction

### 2. Smaller Batch Sizes

**What it does:**
- Process fewer items simultaneously
- Reduces peak memory during forward/backward passes

**Usage:**
```python
config = {
    'discovery': {
        'batch_size': 1,  # top-level key, not nested under data_params
        ...
    }
}
```

**Trade-offs:**
- Slower processing (more batches), but peak memory drops roughly linearly

### 3. Model Half Precision

**What it does:**
- Loads the model in bfloat16 or float16 instead of float32
- Reduces model weight memory by ~50%

**Usage:**
```python
config = {
    'model': {
        'precision': 'bfloat16',  # or 'float16'
        ...
    }
}
```

**Trade-offs:**
- Minimal accuracy impact for most models

### 4. Reduced Sample Counts (MMLU / WMDP only)

**What it does:**
- For the `mmlu` and `wmdp` built-in tasks, caps how many examples are drawn per subject/category
- Directly reduces batch memory requirements

**Usage:**
```python
config = {
    'discovery': {
        'samples_per_subject': 5,  # default is 20 for these tasks
        ...
    }
}
```

This key is read only by the `mmlu` and `wmdp` task implementations — it has no effect on other tasks.

## Combined Optimization Strategy

```python
config = {
    'model': {
        'name': 'Qwen/Qwen3-4B',
        'precision': 'bfloat16',  # half precision model
    },
    'discovery': {
        'task': 'mmlu',
        'algorithm': 'eap-ig',
        'level': 'node',
        'ig_steps': 5,              # reduced steps
        'batch_size': 1,            # small batches, top-level key
        'samples_per_subject': 10,  # fewer samples (mmlu/wmdp only)
    },
    'pruning': {
        'target_sparsity': 0.1,
    },
}

pruned_nodes = discover_circuit(config)
```

## Monitoring Memory Usage

```python
import torch

# Before discovery
if torch.cuda.is_available():
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# During discovery - can add to code
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# After discovery
if torch.cuda.is_available():
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
```

## Troubleshooting Out-of-Memory Errors

If you still encounter OOM errors with all optimizations:

1. **Reduce batch size to 1** (if not already)
2. **Reduce IG steps to 3** (minimum for reasonable results)
3. **Use `samples_per_subject`** to cap examples per subject (MMLU/WMDP only)
4. **Use a smaller model** for algorithm prototyping (GPT-2, Pythia-70M)
5. **Use CPU** for very large models (much slower; device selection is automatic via `get_device()` — there is no `device` config key to force this, so make CUDA unavailable to the process instead, e.g. `CUDA_VISIBLE_DEVICES=""`)

## Best Practices

1. **Start with `bfloat16` model precision** — safe for most models, saves significant memory
2. **Reduce batch size before reducing IG steps** (better accuracy/speed trade-off)
3. **Monitor memory** — track usage to find optimal settings for your model/task

## Next Steps

- [Advanced: Memory Optimization](../advanced/memory-optimization.md) — full memory budget tables and combined-strategy guidance
- [Algorithms: EAP Family](../algorithms/eap.md) — `ig_steps` tuning guidance
