# vLLM Evaluation

CircuitKit supports fast downstream benchmark evaluation via **vLLM**, accessed through lm-evaluation-harness's native vLLM backend. CircuitKit does not reimplement or wrap vLLM — it writes a standard HuggingFace checkpoint and hands it to `lm-eval`, which handles the vLLM serving.

---

## When to Use vLLM

Use vLLM at the **benchmark evaluation stage** (after pruning/quantization), not during circuit discovery or faithfulness evaluation.

| Stage | Tool | vLLM? |
|-------|------|:-----:|
| Circuit discovery | TransformerLens | No |
| 6-pillar faithfulness | TransformerLens | No |
| Downstream benchmarks (`boolq`, etc.) | lm-eval | Optional |

vLLM provides a speedup for throughput-bound benchmark evaluation. For GPT-2 scale, it is usually not worth the setup overhead. For Llama-3B / Gemma-4B at large sample counts, it can be 3–5× faster.

---

## Installation

vLLM is not part of any CircuitKit extra:

```bash
pip install vllm
pip install -e ".[benchmarks]"   # installs lm-eval from GitHub
```

See the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) for CUDA requirements.

---

## Workflow

### Step 1: Discover a circuit and save a pruned checkpoint

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "meta-llama/Llama-3.2-1B-Instruct"},
    "discovery": {"algorithm": "eap-ig", "task": "boolq", "level": "node",
                  "data_params": {"num_examples": 128}},
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

### Step 2: Export a pruned HuggingFace checkpoint

```python
import circuitkit as ck

model = ck.load_model("meta-llama/Llama-3.2-1B-Instruct", dtype="bfloat16")
circuit = ck.load_scores("./circuit.pt")
pruned = ck.prune(model, circuit, sparsity=0.3, scope="heads")
ck.export_checkpoint(pruned, circuit, "./checkpoints/pruned")
```

Or, using `save_pruned_checkpoint` directly:

```python
from circuitkit.evaluation.hf_checkpoint import save_pruned_checkpoint

save_pruned_checkpoint(model, circuit.nodes, "./checkpoints/pruned", overwrite=True)
```

### Step 3: Evaluate with lm-eval + vLLM backend

```python
from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM

lm = VLLM(
    pretrained="./checkpoints/pruned",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
)
results = evaluator.simple_evaluate(model=lm, tasks=["boolq", "winogrande"])
print(results["results"])
```

The vLLM engine is a distinct model class (`--model vllm` on the CLI), not an `HFLM` option — `HFLM`'s `backend` argument selects causal vs. seq2seq, not the inference engine.

**CLI equivalent (lm-eval's own CLI):**

```bash
lm_eval --model vllm \
    --model_args pretrained=./checkpoints/pruned,tokenizer=meta-llama/Llama-3.2-1B-Instruct \
    --tasks boolq,winogrande \
    --num_fewshot 0
```

---

## Without vLLM (Default HuggingFace Backend)

To run the same benchmark without vLLM — useful when vLLM is not available:

```python
from lm_eval.models.huggingface import HFLM

hflm = HFLM(
    pretrained="./checkpoints/pruned",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
)
results = evaluator.simple_evaluate(model=hflm, tasks=["boolq"])
```

Or via CircuitKit's own benchmark wrapper:

```python
import circuitkit as ck
scores = ck.benchmark("./checkpoints/pruned", tasks=["boolq", "winogrande"], limit=100)
```

---

## Memory Considerations

vLLM and TransformerLens can contend for GPU memory if held simultaneously. The recommended pattern (and what the example scripts do) is:

1. Run discovery + export checkpoint → release TransformerLens model
2. Start vLLM evaluation using the on-disk checkpoint
3. vLLM loads the checkpoint fresh — the two never co-occupy GPU memory

```python
# Safe pattern
model = ck.load_model(...)
circuit = ck.discover(model, ...)
ck.export_checkpoint(pruned, circuit, "./checkpoint")

del model
torch.cuda.empty_cache()

# Now run lm-eval vLLM benchmark — no contention
```

---

## Reference Scripts

The worked examples above (and the `ck.benchmark()` / `circuitkit benchmark` calls referenced throughout this page) are the canonical reference; see [Flat API: benchmark](../api-reference/flat-api.md#benchmark) and [Evaluation: Benchmarking](../evaluation/benchmarking.md).

---

## Next Steps

- [Evaluation: Downstream Benchmarking](../evaluation/benchmarking.md) — `ck.benchmark` wrapper
- [Applications](../user-guide/applications.md) — pruning and checkpoint export
- [CLI Reference: Applications](../cli/applications.md) — `benchmark` CLI command
