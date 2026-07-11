# Fast Evaluation with vLLM

CircuitKit supports fast benchmark evaluation through [vLLM](https://docs.vllm.ai/),
**via lm-evaluation-harness's native vLLM backend**. CircuitKit does *not* reimplement
vLLM or wrap it directly — it simply hands a checkpoint to `lm-eval`, which knows how
to load and serve that checkpoint with vLLM.

This means:

- vLLM is an **optional, user-installed dependency**. CircuitKit never installs it for
  you and has no hard dependency on it.
- All vLLM behaviour, flags, and performance characteristics come from `lm-eval` + vLLM
  upstream — CircuitKit does not intercept or modify them.
- The same evaluation works with the default HuggingFace backend; vLLM is purely a
  speed optimisation for the throughput-bound parts of benchmark evaluation.

## When to use it

vLLM accelerates generative and multiple-choice benchmark evaluation by batching and
paged-attention serving. Use it when:

- You are evaluating a **pruned / quantized / edited checkpoint** on standard benchmarks
  (`boolq`, `winogrande`, `mmlu`, `gsm8k`, ...).
- The model is large enough that the default HuggingFace backend is throughput-bound.
- You have a CUDA GPU — vLLM is GPU-oriented.

For circuit *discovery* and the 6-pillar *faithfulness* evaluation, CircuitKit uses
TransformerLens / HuggingFace directly; vLLM is not involved there. vLLM only enters at
the **downstream benchmark evaluation** stage, after you have a concrete checkpoint on
disk.

## Installation

vLLM is not part of any CircuitKit extra. Install it yourself, alongside `lm-eval`:

```bash
pip install vllm
pip install "circuitkit[benchmarks]"   # pulls in lm-evaluation-harness
```

vLLM has its own CUDA and PyTorch requirements — see the
[vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).
If `vllm` is not importable, the `lm-eval` vLLM backend will raise an `ImportError`;
fall back to the default HuggingFace backend (the `HFLM` model class, or `--model hf`).

## Workflow: evaluate a pruned checkpoint with vLLM

The pattern has three steps. The first two are pure CircuitKit; the third is `lm-eval`.

### 1. Discover a circuit and select components to prune

Run any discovery algorithm to obtain component scores, then choose the components to
remove (for example, the lowest-scoring fraction):

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "meta-llama/Llama-3.2-1B-Instruct"},
    "discovery": {"algorithm": "eap-ig", "task": "boolq", "level": "node"},
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

### 2. Save a pruned HuggingFace checkpoint

`circuitkit.evaluation.hf_checkpoint.save_pruned_checkpoint` writes a standard
HuggingFace checkpoint with the selected components zeroed out. A standard checkpoint
is exactly what vLLM needs — there is no CircuitKit-specific format involved:

```python
from circuitkit.evaluation.hf_checkpoint import save_pruned_checkpoint

# `model` is a TransformerLens HookedTransformer; `pruned_nodes` is the list of
# node names to ablate (e.g. the bottom 30% by circuit score).
save_pruned_checkpoint(model, pruned_nodes, "./checkpoints/pruned", overwrite=True)
```

### 3. Evaluate the checkpoint with lm-eval's vLLM backend

Point `lm-eval` at the saved checkpoint directory and request the vLLM backend. This is
ordinary `lm-eval` usage — the only CircuitKit-specific input is the checkpoint path:

```python
from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM   # native lm-eval vLLM backend

lm = VLLM(
    pretrained="./checkpoints/pruned",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
)
results = evaluator.simple_evaluate(model=lm, tasks=["boolq", "winogrande"])
print(results["results"])
```

The equivalent command-line form (lm-eval's own CLI):

```bash
lm_eval --model vllm \
    --model_args pretrained=./checkpoints/pruned,tokenizer=meta-llama/Llama-3.2-1B-Instruct \
    --tasks boolq,winogrande
```

To run the same benchmarks **without** vLLM, swap `VLLM` for `HFLM`
(`from lm_eval.models.huggingface import HFLM`), or use `--model hf` on the CLI; the
default HuggingFace backend requires no extra install.

## Worked example

The pattern below (loading a model, pruning to a checkpoint, then evaluating with vLLM) is the complete worked example. Combine it with [`ck.prune`](../api-reference/flat-api.md#prune) and [`ck.export_checkpoint`](../api-reference/flat-api.md#export_checkpoint) from the Flat API to materialise the checkpoint first.

The core of the evaluation step is:

```python
from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM

lm = VLLM(pretrained=checkpoint_dir, tokenizer=tokenizer)
results = evaluator.simple_evaluate(model=lm, tasks=[lm_task])
```

To run the identical pipeline against the default HuggingFace backend instead (useful if vLLM is not available in your environment), swap `VLLM` for `HFLM` (or pass `--model hf` on the CLI), as shown earlier in this guide.

## Notes and caveats

- **vLLM is user-installed and optional.** `pip install vllm` is required to use
  the vLLM backend (`--model vllm`); CircuitKit will not install it.
- **CircuitKit does not reimplement vLLM.** The acceleration is entirely `lm-eval` +
  vLLM upstream. Any vLLM-specific tuning (tensor parallelism, dtype, GPU memory
  fraction) is passed through `lm-eval`'s `model_args`, not through CircuitKit.
- **Use vLLM only at the benchmark stage.** Discovery and 6-pillar faithfulness
  evaluation run on TransformerLens / HuggingFace and are unaffected by vLLM.
- vLLM and TransformerLens can contend for GPU memory if held simultaneously. The
  workflow above separates selection from evaluation by the on-disk checkpoint, so the
  two never need to be resident at once.
