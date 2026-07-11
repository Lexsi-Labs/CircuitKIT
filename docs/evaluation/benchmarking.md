# Downstream Benchmarking

After applying circuit-guided interventions (pruning, quantization), CircuitKit integrates with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to measure downstream task performance.

**Requires:** `pip install -e ".[benchmarks]"` — installs `lm-eval` from GitHub and `datasets>=2.20`.

---

## Quick Start

```python
import circuitkit as ck

# After prune + export
scores = ck.benchmark(
    "./output/pruned_checkpoint",      # path to HF checkpoint
    tasks=["boolq", "winogrande"],     # lm-eval task names
    limit=100,                         # cap examples per task
)

for task, metrics in scores.items():
    print(f"{task}: {metrics['acc,none']:.3f}")
```

**Via Pipeline:**

```python
pipe.discover(algorithm="eap-ig", n_examples=128, sparsity=0.3)
pipe.prune(sparsity=0.3)
pipe.export("./output/checkpoint")
pipe.benchmark(tasks=["boolq", "winogrande", "hellaswag"], limit=100)
```

**Via CLI:**

There is no `--checkpoint` or `--limit` flag on `circuitkit benchmark` — that command runs a discovery + intervention + baseline comparison sweep, not an lm-eval-harness run on a checkpoint (see [CLI Reference](../cli/applications.md#circuitkit-benchmark)). To benchmark an exported checkpoint with lm-eval-harness, use the Python API shown above (`ck.benchmark()`).

---

## Available Benchmark Tasks

Any `lm-eval` task name is valid. Common choices:

| Task | Description | Metric |
|------|-------------|--------|
| `boolq` | Boolean QA | Accuracy |
| `winogrande` | Coreference resolution | Accuracy |
| `hellaswag` | Commonsense NLI | Accuracy (normalized) |
| `mmlu` | Massive multitask language understanding | Accuracy |
| `arc_easy`, `arc_challenge` | ARC Science QA | Accuracy |
| `truthfulqa_mc` | TruthfulQA | MC accuracy |
| `gsm8k` | Math word problems | Exact match |
| `wmdp_bio`, `wmdp_chem`, `wmdp_cyber` | Hazardous knowledge (unlearning benchmarks) | Accuracy |

---

## Interpreting Results

Benchmark results measure **extrinsic performance** — whether the pruned/quantized model still performs well on tasks beyond the one the circuit was discovered for.

**What to look for:**

1. **Baseline vs. pruned comparison:** Run `ck.benchmark` on the original model and the pruned model. The gap is the cost of circuit-guided compression.

2. **Target task vs. transfer tasks:** If you discovered a circuit for IOI and pruned at 30% sparsity, benchmark on both IOI (within-task) and BoolQ, Winogrande (out-of-task). A good circuit-guided compression should degrade transfer tasks less than random pruning.

3. **Circuit vs. random pruning:** Compare `ck.benchmark` after circuit-guided pruning vs. random pruning at the same sparsity. The CircuitKit paper shows circuit-guided pruning retains ~5-15% more accuracy on unrelated tasks.

---

## Full Benchmarking Workflow

```python
import circuitkit as ck

model = ck.load_model("gpt2", dtype="float32")
circuit = ck.load_scores("./circuit.pt")

# 1. Baseline (unpruned)
ck.export_checkpoint(model, None, "./output/baseline")
baseline = ck.benchmark("./output/baseline", tasks=["boolq", "winogrande"], limit=200)

# 2. Circuit-guided pruning at 30%
pruned = ck.prune(model, circuit, sparsity=0.3, scope="heads")
ck.export_checkpoint(pruned, circuit, "./output/pruned_0.3")
pruned_30 = ck.benchmark("./output/pruned_0.3", tasks=["boolq", "winogrande"], limit=200)

# 3. Print comparison
for task in ["boolq", "winogrande"]:
    b = baseline[task]["acc,none"]
    p = pruned_30[task]["acc,none"]
    print(f"{task}: baseline={b:.3f}, pruned_30={p:.3f}, drop={b-p:.3f}")
```

---

## PEFT Benchmarking

For circuit-restricted fine-tuning, CircuitKit includes a PEFT benchmark utility:

```python
from circuitkit.applications.finetuning.benchmark_peft import PEFTBenchmark

bench = PEFTBenchmark(model, method="lora", rank=8, device="cuda")
metrics = bench.run(num_batches=50, batch_size=8)

print(f"Parameter efficiency: {metrics.param_efficiency:.2%}")
print(f"Memory footprint: {metrics.peak_memory_mb:.0f} MB")
print(f"Throughput: {metrics.batches_per_second:.2f} batches/sec")
```

---

## Next Steps

- [Applications](../user-guide/applications.md) — pruning and quantization workflows
- [CLI Reference: Applications](../cli/applications.md) — `benchmark` CLI command
- [Framework Overview](framework.md) — faithfulness evaluation before benchmarking
