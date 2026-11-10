# Quick Start

Get your first circuit in 5 minutes. This walkthrough uses **GPT-2** because it runs on CPU with no setup — but every step takes a `model` field, and CircuitKIT is tested across GPT-2, Pythia, Qwen 2.5 / Qwen 3, Llama 3, and Gemma 2 / Gemma 3 (see [Use a different model](#use-a-different-model) below).

## Step 1: Install

`pip install circuitkit` — or from source, see [Installation](installation.md). GPT-2 runs on CPU, so no GPU is needed for this walkthrough; the larger models below want a GPU or Apple-Silicon MPS.

## Step 2: Discover a circuit

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {
        "algorithm": "eap-ig",          # stable default
        "task": "ioi",                  # Indirect Object Identification
        "level": "node",
        "data_params": {"num_examples": 32},
    },
    "pruning": {"target_sparsity": 0.3, "scope": "both"},
    "output_path": "./circuit.pt",
})

print(circuit)   # e.g. ['A0.1', 'A2.3', 'MLP 5', ...]
```

This takes ~1–3 minutes on CPU for GPT-2 with 32 examples.

### Use a different model

`model.name` accepts any TransformerLens-supported HuggingFace model. Only that one field changes — algorithm, task, and evaluation stay identical:

```python
# small + open, runs on CPU/MPS with no gating:
"model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"}
"model": {"name": "EleutherAI/pythia-160m"}

# larger, GPU/MPS recommended (Llama & Gemma are gated — accept the license on HF first):
"model": {"name": "meta-llama/Llama-3.2-1B-Instruct"}
"model": {"name": "google/gemma-2-2b-it"}      # or google/gemma-3-1b-it
"model": {"name": "Qwen/Qwen3-4B"}
```

GPT-2 is only the fast default here — for a task like `ioi` an instruct model gives a cleaner circuit, and safety/steering work (see the [examples](../examples/overview.md)) needs an instruct-tuned model since GPT-2 has no refusal behavior.

## Step 3: Evaluate faithfulness

```python
from circuitkit.api import evaluate_circuit

results = evaluate_circuit({
    "model": {"name": "gpt2"},
    "discovery": {"algorithm": "eap-ig", "task": "ioi", "level": "node"},
    "pruning": {"target_sparsity": 0.3, "scope": "both"},
    "output_path": "./circuit.pt",
})
print(results.patching_score)      # Pillar 1 (causal patching): 1.0, clipped
print(results.patching_raw_ratio)  # 1.035: just above the full model
print(results.ablation_score)      # Pillar 2 (zero ablation): 0.0, clipped
print(results.ablation_raw_ratio)  # -0.21: below the corrupt baseline, see the note below
# results is a FaithfulnessReport; the full-faithfulness path also populates
# .stability / .robustness / .baseline_comparison / .generalization /
# .intervention_reliability. See results.summary() for a formatted view.
```

These values were measured on 2026-10-11 by running Steps 2 and 3 exactly as printed, on one Tesla T4:
`gpt2`, `eap-ig`, `ioi`, 32 discovery examples, 30% pruned (46 of 156 components removed). Step 3
states no seed or example count, so the defaults apply: seed 42 and 256 evaluation examples. The
same steps on a CPU gave 1.029 and -0.20. An earlier CPU figure printed here (0.94 and -0.07) was
not reproduced.

Both scores are clipped to the range 0 to 1. On GPT-2 `ioi`, zero ablation takes the model below
the corrupt baseline, so `ablation_score` prints as 0.0 even for a good circuit; judge this circuit
by `patching_score` and read [the raw ratio](../evaluation/causal-patching.md#the-raw-ratio-and-scores-outside-0-to-1) before drawing a conclusion from a
score of exactly 0.0 or 1.0.

## Step 4: Prune and export

```python
import circuitkit as ck

circuit = ck.load_scores("./circuit.pt")
model = ck.load_model("gpt2", dtype="float32")
pruned = ck.prune(model, circuit, sparsity=0.3, scope="both")
ck.export_checkpoint(pruned, circuit, "./output/ioi_pruned")
```

This writes a reloadable HuggingFace checkpoint to `./output/ioi_pruned`.

## Step 5: Benchmark

```python
scores = ck.benchmark("./output/ioi_pruned", tasks=["boolq", "winogrande"], limit=100)
for task, metrics in scores.items():
    print(f"{task}: {metrics}")
```

!!! note "benchmarks extra required"
    `pip install -e ".[benchmarks]"`

## Alternative: Pipeline (stateful)

```python
from circuitkit import Pipeline

pipe = Pipeline("gpt2", task="ioi", output_dir="./results")
pipe.discover(algorithm="eap-ig", level="node", n_examples=128, sparsity=0.3)
pipe.evaluate(pillars=["patching", "ablation", "baselines"])
pipe.prune(sparsity=0.3)
pipe.export("./results/checkpoint")
pipe.summary()  # prints a Rich table; summary() returns None
```

Best for notebooks and multi-step experiments. See [Pipeline Overview](../user-guide/pipeline-overview.md).

## Alternative: CLI

```bash
circuitkit discover --model gpt2 --algorithm eap-ig --task ioi \
    --sparsity 0.3 --level node --output ./circuit.pt
circuitkit evaluate --model gpt2 --artifact ./circuit.pt
circuitkit prune --model gpt2 --artifact ./circuit.pt --sparsity 0.3 --output ./pruned
```

Benchmarking a checkpoint is Python-only — use `ck.benchmark("./pruned", tasks=["boolq", "winogrande"])` from Step 5.

## What's next

- [Core Concepts](core-concepts.md) — circuits, tasks, faithfulness, interventions
- [Configuration](configuration.md) — full comparison of all three interfaces
- [Algorithms](../algorithms/overview.md) — how to choose the right discovery algorithm
- [Pipeline Overview](../user-guide/pipeline-overview.md) — stateful Pipeline deep dive
