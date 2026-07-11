# Testing on a CUDA machine

Some of CircuitKit's validation can't run on a CPU-only or Apple-Silicon box. The
`gpu-cu126` extra pins `torch==2.6.0+cu126` / `torchvision==0.21.0+cu126`, and those
`+cu126` wheels only exist on PyTorch's CUDA index — not on PyPI — so a plain
`uv`/`pip` resolve fails off a CUDA host with *"no version of torch==2.6.0+cu126"*.
Real-model discovery, quantization, vLLM evaluation, and the GPU path of the
corruption/pairing guards therefore need to be checked on an actual NVIDIA GPU.

Device selection is automatic (`get_device()`: CUDA → MPS → CPU), so you don't set
any config key — on a CUDA box the same code runs on the GPU.

## 1. Install on the CUDA box

Pick the toolchain you use. Both install CUDA 12.6 torch from PyTorch's index first,
then CircuitKit with the extras you need.

=== "uv"

    ```bash
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
        -e ".[gpu-cu126,dev,benchmarks,quantization]"
    ```

=== "pip"

    ```bash
    pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126
    pip install -e ".[dev,benchmarks,quantization]"
    ```

For a different CUDA version, swap `cu126` for your toolkit (e.g. `cu121`) in both
the index URL and the torch pins.

## 2. Confirm the GPU is visible

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `... True <your GPU>`. If `cuda.is_available()` is `False`, the wrong torch
build is installed — reinstall from the CUDA index above.

## 3. Run the test suite

```bash
pytest tests/ -q
```

### Corruption / pairing guards (PR #105)

These guards fail loud when auto-corruption or normalized pairing produces identical
clean/corrupt halves. Exercise both the `GenericTaskSpec` and normalized/template
paths:

```bash
pytest tests/tasks/test_generic_task.py tests/test_custom_data.py -q
```

## 4. Run the safety-refusal case study (PR #106)

End-to-end run of the **custom-data path with explicit contrastive pairs** on an
instruction-style (non-syntactic) task. It uses GPT-2 and the benign, illustrative
`24_safety_sample.csv`:

```bash
python examples/case-studies/24-safety-refusal-custom-data.py
```

On a CUDA host this exercises the discover → evaluate loop on the GPU.

## 5. GPU validation across model families

!!! note "Why the rest of the docs use GPT-2 — and why not here"
    GPT-2 is the default throughout the docs and examples because the IOI circuit is
    defined on it, it's tiny, ungated, and CPU-runnable, so tutorials and CI run
    anywhere. That's also why it's a *bad* CUDA test: GPT-2 fits comfortably on CPU,
    isn't GQA, and never touches the memory, quantization, or vLLM paths you came to
    the GPU for. On a CUDA box, validate real models — and across **families**, since
    the architecture registry handles each differently (GQA attention, RoPE, gemma's
    gated MLP). A single family only proves that family.

Pick small models (1B or less where the family offers it) so the sweep is cheap but
still crosses the architecture boundaries that matter. Suggested matrix, drawn from the
[architecture registry](../advanced/architecture-registry.md):

| Family | Small model | ~Size | Status | GQA | Gated? |
|--------|-------------|-------|--------|-----|--------|
| `gpt2` | `gpt2` | 124M | Ready | no | no (CPU baseline) |
| `qwen` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Production | yes | **no** |
| `llama` | `meta-llama/Llama-3.2-1B` | 1B | Production | yes | yes |
| `gemma3` | `google/gemma-3-1b-it` | 1B | Production | yes | yes |
| `gemma` | `google/gemma-2-2b` | 2B | Production | yes | yes |
| `phi` | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Ready | no | no |
| `mistral` | `mistralai/Mistral-7B-v0.1` | 7B | Ready | yes | no |
| `falcon` | `tiiuae/falcon-7b` | 7B | Ready | yes | no |

Start with the ungated ≤1B entries (`Qwen2.5-0.5B`, plus `Llama-3.2-1B` /
`gemma-3-1b-it` once you've `huggingface-cli login`'d) to cover GQA + non-GQA + gated-MLP
cheaply, then scale up as VRAM allows.

```python
from circuitkit import Pipeline

# Sweep a few families; each exercises different registry arch-handling.
for model in [
    "gpt2",                          # non-GQA baseline
    "Qwen/Qwen2.5-0.5B-Instruct",    # GQA, ungated
    "meta-llama/Llama-3.2-1B",       # GQA, gated
    "google/gemma-3-1b-it",          # GQA + gated MLP
]:
    pipe = Pipeline(model, task="ioi")
    pipe.discover(algorithm="eap-ig", n_examples=64)
    pipe.evaluate(pillars=["patching"])
    print(model, pipe.summary())
```

Watch `nvidia-smi` while it runs to confirm the work lands on the GPU. Then exercise the
paths a CPU box can't reach:

- **Per-family pruning** — GQA families (llama/qwen/gemma/mistral/falcon) exercise the
  `n_kv_heads != n_heads` grouped-head path; non-GQA (gpt2/phi) the plain path. See the
  [architecture registry](../advanced/architecture-registry.md).
- **Memory optimization** — see [Memory Optimization](../advanced/memory-optimization.md);
  push `num_examples` / `batch_size` / model size until you feel the VRAM ceiling.
- **Quantization** — `ck.quantize(...)` with `backend="llmcompressor"` (honours `bits`);
  requires `pip install -e ".[quantization]"`.
- **vLLM evaluation** — the `VLLM` lm-eval path in [vLLM Evaluation](../advanced/vllm-evaluation.md).
- **IBCircuit at scale** — validates the OOM boundary (OOMs at ~3B); try Llama-3.2-1B vs 3B.

A GPT-2 run is fine as a 10-second "does anything work" check, but it proves nothing
about GQA handling or the GPU-only code paths.
