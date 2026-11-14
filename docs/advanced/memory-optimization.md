# Memory Optimization

EAP-IG can be memory-intensive, especially for large models. This page covers all available memory optimizations and their trade-offs.

---

## Memory Budget by Model Size

Peak VRAM after the `perf/vram-optimizations` fixes (qkv-flag lifecycle,
per-batch buffer release, embedding-only gradients — see the changelog):

| Model | Previous default (measured) | With all optimizations (measured) |
|-------|----------------------------|-------------------------------------|
| GPT-2 (124M) | 1.0 GB | 0.5 GB |
| Llama-3.2-1B-Instruct | 8.7 GB | 4.1 GB |
| Llama-3B | not re-measured — order-of-magnitude only | not re-measured |
| Gemma-4B | not re-measured — order-of-magnitude only | not re-measured |

"Previous default" is `ig_steps=5`, `batch_size=4`, `num_examples=128` (`ig_steps` now defaults to 3). "With
all optimizations" is `ig_steps=3`, `batch_size=1`, `num_examples=64` (the
combined strategy below), both bf16. (The config key is `num_examples`, under
`data_params`; prose sometimes calls it `n_examples` — it is the same knob.)
The Llama-3B/Gemma-4B rows are not re-verified against the current code —
expect the same relative reduction as the two measured rows, not these exact
figures.

---

## Optimization Techniques

### 1. Model precision (bfloat16 is the default)

CircuitKIT already loads models in bfloat16 — it's the default, and the backend warns if you set anything else, so you get this saving out of the box:

```python
discover_circuit({
    "model": {"name": "gpt2", "precision": "bfloat16"},  # the default, shown explicitly
    ...
})
```

**Why it's listed:** the float32 baseline in the table below is a what-if. Forcing float32 would roughly double model VRAM for no accuracy benefit, which is exactly why bfloat16 is the default rather than an opt-in.

### 2. Reduce `ig_steps` (linear savings)

`ig_steps` is the number of Integrated Gradients integration steps. The default is 3 (`utils/config.py`), which is enough for EAP-IG: the EAP-IG paper (Hanna et al., 2024) finds every value above 2 similarly faithful, and 2 unfaithful on some tasks. Cost grows linearly with the step count, so the previous default of 5 takes about 1.7x as long. The backends fall back to 3 as well when `ig_steps` is left unset (`None`).

```python
"discovery": {"algorithm": "eap-ig", "ig_steps": 3, ...}  # the default; 2 is faster but unfaithful on some tasks
```

**Savings:** Linear with step reduction. `ig_steps=3` (the default) is ~40% faster and ~30% less memory than the previous default of `ig_steps=5`.

### 3. Reduce `batch_size` (linear savings)

`batch_size` is read at the top level of the `discovery` block, not nested under `data_params`:

```python
"discovery": {"batch_size": 1, "data_params": {"num_examples": 128}}
```

**Savings:** Linear. `batch_size=1` is ~50-60% less peak memory than `batch_size=4`. Slower (more batches).

### 4. Skip TransformerLens weight copies (opt-in, `CIRCUITKIT_TL_MEMORY_PATCH=1`)

CircuitKIT ships a TransformerLens patch that removes two large temporaries from every forward pass: attention
output is computed with one `einsum` over `W_O` instead of broadcasting a `[batch, pos, head, d_head, d_model]`
tensor, and gated MLPs (Llama, Qwen, Gemma, ...) and the unembedding use transposed weight views instead of a fresh
`.T.contiguous()` copy of each matrix. Set the variable before CircuitKIT is imported:

```bash
CIRCUITKIT_TL_MEMORY_PATCH=1 python run_discovery.py
```

- **Off by default.** Without the contiguous copies, bf16 matmuls accumulate in a different order, so logits differ
  from Hugging Face's in the last bits and scores can differ slightly between runs with and without the patch.
  Keep it on or off for every run you compare. The measurements below say how large the change is.
- **Import order.** The patch is applied when `circuitkit` is imported. If `transformer_lens` was imported first,
  CircuitKIT raises an `ImportError`; import `circuitkit` first (or restart the kernel).
- **Savings** grow with model width: the removed temporaries scale with `d_model × d_mlp` per layer and with
  `batch × pos × n_heads × d_head × d_model` in attention. They were the out-of-memory sites for Gemma-4 31B on an
  80 GB card. See `src/circuitkit/_tl_patches/README.md` for the patch series.

#### Measured: patch on vs off

Node-level EAP-IG on the `ioi` task: 64 examples, 3 IG steps, 3 data seeds (0, 1, 2), batch size 4 (1 for the
Cohere-architecture model). Every on/off pair ran in separate processes; the on and off runs of one model use the same
seeds and data. RTX 6000 Ada (48 GB), torch 2.13.0+cu126, transformer-lens 3.8.0, CircuitKIT `79f4b1b`. The GPU was
shared with other jobs, so wall times are indicative only. Peak is `torch.cuda.max_memory_allocated()` during
discovery and includes the weights. The Cohere-architecture row is a public fine-tune of `tiny-aya-global`
(`itssaharsh/tiny-aya-global-unlearn-bio-default-layers`, same `Cohere2ForCausalLM` architecture, 3.35B parameters)
loaded from a local checkpoint, because Tiny Aya itself is gated. It was run in bf16 only: fp32 weights alone
(13 GB) were more than the shared card could spare.

**Peak memory.**

| Model | dtype | Weights | Peak, patch off | Peak, patch on | Saving of peak | Saving beyond weights | Wall s, off → on |
|---|---|---|---|---|---|---|---|
| gpt2 | fp32 | 0.64 GiB | 1.13 GiB | 0.99 GiB | 12 % | 28 % | 3 → 3 |
| gpt2 | bf16 | 0.33 GiB | 0.62 GiB | 0.55 GiB | 11 % | 23 % | 3 → 3 |
| Qwen2.5-0.5B | fp32 | 2.41 GiB | 5.29 GiB | 3.61 GiB | 32 % | 58 % | 11 → 10 |
| Qwen2.5-0.5B | bf16 | 1.23 GiB | 2.77 GiB | 1.94 GiB | 30 % | 54 % | 11 → 10 |
| Qwen3-0.6B | fp32 | 2.90 GiB | 6.21 GiB | 4.65 GiB | 25 % | 47 % | 15 → 10 |
| Qwen3-0.6B | bf16 | 1.47 GiB | 3.31 GiB | 2.53 GiB | 24 % | 42 % | 11 → 10 |
| Tiny Aya architecture (3.35B) | bf16 | 7.34 GiB | 14.86 GiB | 9.42 GiB | 37 % | 72 % | 92 → 41 |

The saving is real and grows with model width. The wall-time differences are within what a shared GPU gives, except
that the patch cannot make discovery slower; do not read 92 → 41 s as a speed-up.

**Scores.** Same seed, patch on vs off, over the 3 seeds. The last column is the Spearman correlation between two
different seeds with the patch off, the run-to-run variation to compare against.

| Model | dtype | Spearman on/off, per seed | Top-k overlap, worst seed, k = 10 / 25 / 50 % of nodes | Seed vs seed, patch off |
|---|---|---|---|---|
| gpt2 | fp32 | 1.0000, 1.0000, 1.0000 | 1.00 / 1.00 / 1.00 | 0.867 |
| gpt2 | bf16 | 0.9968, 0.9981, 0.9989 | 1.00 / 0.97 / 0.97 | 0.870 |
| Qwen2.5-0.5B | fp32 | 1.0000, 1.0000, 1.0000 | 1.00 / 1.00 / 1.00 | 0.863 |
| Qwen2.5-0.5B | bf16 | 0.9906, 0.9941, 0.9935 | 0.97 / 0.99 / 0.98 | 0.859 |
| Qwen3-0.6B | fp32 | 1.0000, 1.0000, 1.0000 | 1.00 / 1.00 / 1.00 | 0.841 |
| Qwen3-0.6B | bf16 | 0.9941, 0.9944, **0.9866** | 0.96 / 0.98 / 0.97 | 0.842 |
| Tiny Aya architecture (3.35B) | bf16 | 0.9984, 0.9986, 0.9986 | 0.98 / 0.99 / 0.98 | 0.684 |

**Logits against Hugging Face** (8 short prompts, same token ids, same dtype; max |Δlogit| and top-1 agreement,
patch off / patch on). The last column is the largest on-vs-off difference at the last position of each prompt.

| Model | dtype | Max abs Δ, off / on | Top-1 agreement, off / on | On vs off, last position |
|---|---|---|---|---|
| gpt2 | fp32 | 7.3e-4 / 4.7e-4 | 1.000 / 1.000 | 2.4e-4 |
| gpt2 | bf16 | 9 / 10 | 0.936 / 0.917 | 1 |
| Qwen2.5-0.5B | fp32 | 9.1e-5 / 1.1e-4 | 1.000 / 1.000 | 3.9e-5 |
| Qwen2.5-0.5B | bf16 | 2.1 / 2.2 | 0.969 / 0.947 | 0.62 |
| Qwen3-0.6B | fp32 | 1.0e-4 / 7.6e-5 | 1.000 / 1.000 | 3.5e-5 |
| Qwen3-0.6B | bf16 | 0.78 / 1.3 | 0.985 / 0.977 | 0.75 |
| Tiny Aya architecture (3.35B) | bf16 | 1.3 / 1.2 | 0.965 / 0.979 | 0.31 |

In bf16 both settings are far from Hugging Face in absolute terms; the patch moves the logits by about as much as the
bf16 noise already present. In fp32 the difference is numerically tiny (about 1e-4 or less).

**Faithfulness** (patching score of the circuit that keeps the top 10 / 25 / 50 % of nodes, mean ± sd over the 3
seeds; each seed's circuit is scored on that seed's data; patch off | patch on). The ablation pillar is floored at 0
and was 0 for most circuits, so it is not shown.

| Model | dtype | Keep 10 % | Keep 25 % | Keep 50 % |
|---|---|---|---|---|
| gpt2 | fp32 | 0.151 ± 0.022 &#124; 0.151 ± 0.022 | 0.749 ± 0.047 &#124; 0.749 ± 0.047 | 0.969 ± 0.054 &#124; 0.969 ± 0.054 |
| gpt2 | bf16 | 0.146 ± 0.030 &#124; 0.145 ± 0.028 | 0.769 ± 0.037 &#124; 0.757 ± 0.029 | 0.953 ± 0.081 &#124; 0.957 ± 0.074 |
| Qwen2.5-0.5B | fp32 | 0.292 ± 0.095 &#124; 0.292 ± 0.095 | 0.725 ± 0.117 &#124; 0.725 ± 0.117 | 0.895 ± 0.068 &#124; 0.895 ± 0.068 |
| Qwen2.5-0.5B | bf16 | 0.317 ± 0.077 &#124; 0.311 ± 0.078 | 0.749 ± 0.136 &#124; 0.777 ± 0.074 | 0.909 ± 0.048 &#124; 0.897 ± 0.062 |
| Qwen3-0.6B | fp32 | 0.287 ± 0.009 &#124; 0.287 ± 0.009 | 0.640 ± 0.048 &#124; 0.640 ± 0.048 | 0.841 ± 0.025 &#124; 0.841 ± 0.025 |
| Qwen3-0.6B | bf16 | 0.288 ± 0.011 &#124; 0.289 ± 0.010 | 0.611 ± 0.045 &#124; 0.624 ± 0.053 | 0.810 ± 0.037 &#124; **0.855 ± 0.058** |
| Tiny Aya architecture (3.35B) | bf16 | 0.105 ± 0.075 &#124; 0.104 ± 0.075 | 0.539 ± 0.136 &#124; 0.539 ± 0.132 | 0.975 ± 0.021 &#124; 0.970 ± 0.027 |

**Why it stays opt-in.** The rule applied before the runs was: make the patch the default only if, in bf16, every
seed has on/off Spearman ≥ 0.99 and top-k overlap ≥ 0.95 at every k, faithfulness differences stay within one sd, the
fp32 differences are numerically tiny, and the saving is at least 15 % on the larger models. Everything holds except
two marginal cases, both on Qwen3-0.6B in bf16: one seed has Spearman 0.9866 (the mean over seeds is 0.9917), and
faithfulness at 50 % of nodes is 0.045 higher with the patch on, which is larger than the sd of the runs without it
(0.037) and smaller than the sd with it (0.058). With three seeds that is not enough to call the change noise, and a
default changes every bf16 user's scores, so the patch stays opt-in. The differences that do appear are much smaller
than the seed-to-seed variation, and the memory saving is large (24-37 % of peak on the models above). If your
model is wide or tight on memory, turning it on is a reasonable choice; keep it the same for every run you compare.

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
        "ig_steps": 3,                   # 2. IG steps (the default)
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
| + ig_steps 3 (now the default; was 5) | ~75–80% |
| **All combined** | **~80%** |

There are no `memory_efficient` or `use_half_precision_activations` config keys — neither is read anywhere in the discovery backend, so setting them silently does nothing. The levers above (model precision, `batch_size`, `ig_steps`, `n_examples`, and the opt-in `CIRCUITKIT_TL_MEMORY_PATCH=1`) are the only ones that actually affect memory.

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
| `ig_steps=3` (now the default; was 5) | ~40% faster | < 0.5% score diff |
| `batch_size=1` | ~50% slower | None |
| `bfloat16` model | Same or faster | < 0.1% score diff |

---

## Troubleshooting OOM

If you still hit OOM with all optimizations:

1. **Reduce `num_examples` to 32** — enough for a reasonable circuit estimate
2. **Try `ig_steps=1`** — equivalent to vanilla `eap`; slightly less accurate but same memory as non-IG methods
3. **Use CPU** — very slow but no VRAM constraint. Set `model: {device: cpu}` in the config (without it, `get_device()` picks CUDA > MPS > CPU), or `ck.load_model(..., device="cpu")` for the flat API.
4. **Consider a smaller model** — GPT-2 or Pythia-70M for algorithm prototyping

---

## Next Steps

- [Algorithms: Capability Matrix](../algorithms/capability-matrix.md) — GPU requirements by model size
- [Algorithms: EAP Family](../algorithms/eap.md) — `ig_steps` tuning guidance
- [Troubleshooting](../user-guide/troubleshooting.md) — common OOM errors
