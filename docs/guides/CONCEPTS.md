# Core Concepts in CircuitKit

Understanding these fundamental concepts will help you use CircuitKit effectively.

## Table of Contents

1. [Circuits](#circuits)
2. [Circuit Discovery](#circuit-discovery)
3. [Tasks and Datasets](#tasks-and-datasets)
4. [Faithfulness Evaluation](#faithfulness-evaluation)
5. [Corruption Strategies](#corruption-strategies)
6. [Intervention Methods](#intervention-methods)
7. [Auto-Schema Detection](#auto-schema-detection)

## Circuits

### What is a Circuit?

A **circuit** is a minimal subgraph of a neural network that performs a specific computation. In the context of transformers, circuits identify which attention heads, MLP layers, and neurons are crucial for particular behaviors.

**Example**: For the IOI (Indirect Object Identification) task, a circuit might include:
- Attention heads that identify subject, object, and indirect object positions
- MLP layers that integrate information
- Output layer that predicts the correct indirect object

### Circuit Representations

CircuitKit represents circuits as directed graphs with:
- **Nodes**: Attention heads, MLP layers, or individual neurons
- **Edges**: Information flow between components
- **Attributes**: Weight, type (attention/mlp), layer information

### Node vs Neuron Level

- **Node Level**: Identifies entire attention heads or MLP layers as units
- **Neuron Level**: Identifies specific neurons within layers as units

Neuron-level circuits are more granular and sparse but computationally more expensive.

## Circuit Discovery

### Overview

Circuit discovery is the process of identifying which parts of a neural network are responsible for a particular behavior or computation.

### Discovery Algorithms

CircuitKit ships **13 discovery algorithms** across 4 backends (EAP, ACDC, IBCircuit,
CD-T). They are organized into explicit **stability tiers** — see the README's
[stability-tier table](../../README.md#discovery-algorithms-and-stability-tiers) and the
`STABILITY` map in `circuitkit.backends`. `discover_circuit` emits a `UserWarning` when
an experimental- or research-tier algorithm is used.

To pick an algorithm for a given model size, consult the README's
[capability matrix](../../README.md#capability-matrix-algorithm--model-scale), which
crosses each algorithm with model scale (GPT-2 → 7B+) and flags expected outcome
(Stable / Experimental / OOM-risk) with rough time and memory notes.

The most commonly used algorithms:

#### EAP-IG (EAP with Integrated Gradients) — **Stable, default**
- Combines EAP with integrated gradients for better gradient-based attribution
- The recommended default (`default_algorithm()` returns `"eap-ig"`)
- **Best for**: general use, including small Llama/Gemma models

#### EAP (Edge Attribution Patching) — **Stable**
- Direct attribution-based edge importance
- **Best for**: a fast, validated baseline

#### ACDC (Automatic Circuit DisCovery) — **Experimental**
- Greedy edge-pruning to minimize circuit size
- Validated on GPT-2 IOI; may fail or OOM on larger models
- **Best for**: GPT-2-scale exploratory work

#### IBCircuit (Information Bottleneck) — **Experimental**
- Information-theoretic approach to component importance
- Trains on a single batch; can OOM above ~3B parameters
- **Best for**: GPT-2-scale information-flow studies

The research-tier algorithms (`eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`,
`eap-ifr`, `cdt`) are implemented but validated only on GPT-2 IOI — use them for
research and exploration, not production.

### Discovery Hyperparameters

```python
{
    "algorithm": "acdc",           # Which algorithm to use
    "threshold": 0.01,             # Importance threshold (0-1)
    "max_samples": None,           # Limit discovery samples
    "granularity": "node",         # "node" or "neuron"
    "max_iterations": 100,         # Max discovery iterations
    "patience": 10,                # Early stopping patience
}
```

## Tasks and Datasets

### What is a Task?

A **task** defines:
- The dataset and how to load it
- Which features are inputs and outputs
- How to generate predictions
- How to evaluate correctness

### Built-in Tasks

CircuitKit registers **16 built-in tasks** (`ioi`, `sva`, `gender_bias`,
`capital_country`, `hypernymy`, `greater_than`, `double_io`, `boolq`, `glue`,
`mmlu`, `winogrande`, `winogrande_mc`, `truthfulqa`, `ifeval`, `wmdp`, `gsm8k`).
A few examples:
- **IOI (Indirect Object Identification)**: Identify the indirect object in "A and B went to C's house..."
- **SVA (Subject-Verb Agreement)**: Predict correct verb form in number agreement
- **Gender Bias**: Identify gender biases in pronouns
- **Capital-Country**: Predict capital city from country name
- **MMLU**: Multiple-choice question answering
- **GLUE**: Text classification benchmarks
- **WinoGrande**: Binary commonsense coreference — cloze task scored with a
  **suffix log-likelihood** metric (`metric="suffix_loglik"`), not the
  single-token logit-difference used by IOI / BoolQ
- **WinoGrande-MC**: A multiple-choice reformulation of WinoGrande (explicit
  question, single-token logit-difference metric via an option-swap corruption);
  unlike the cloze `winogrande`, it can be wrapped in a chat template for
  instruction-tuned models
- **GSM8K**: Grade-school math word problems — open-ended generation discovery
  scored with a differentiable NLL on the answer span

> **Discovery metrics differ per task.** Most tasks score a single answer token
> at the last query position with a logit-difference metric. `winogrande`
> (suffix log-likelihood) and `gsm8k` (answer-span NLL) are deliberate
> exceptions — the metric is chosen per task, not uniformly.

### Custom Tasks

Tasks are represented by `TaskSpec` (and its `GenericTaskSpec` subclass). The simplest
way to add a dataset is the auto-schema factory below, or build a `NormalizedTaskSpec`
from a dataset adapter and `register_task` it (see the README's "Tasks and datasets"
section).

```python
from circuitkit.tasks import TaskSpec, GenericTaskSpec, register_task
```

### Auto-Schema Detection

Automatically detect task schema for any Hugging Face dataset:

```python
from circuitkit.tasks import auto_task_from_hf

task = auto_task_from_hf(
    dataset_name="glue",
    subset="sst2",            # dataset config name
    split="validation",
)
# Automatically detects:
# - Which columns are inputs
# - Which column is the label
# - How to evaluate predictions
```

## Faithfulness Evaluation

### The 6-Pillar Framework

CircuitKit evaluates circuit faithfulness across six pillars, implemented as classes
under `circuitkit.evaluation.pillars` and orchestrated by `run_full_faithfulness`. Pass a
subset of pillar keys to `run_full_faithfulness(..., pillars=[...])` to skip expensive
ones.

| # | Pillar (class / key) | Measures |
|---|----------------------|----------|
| 1 | `Pillar1_CausalPatching` (`"patching"`) | Does the circuit explain model behavior under causal patching? |
| 2 | `Pillar2_Ablation` (`"ablation"`) | Does ablating out-of-circuit components degrade behavior? |
| 3 | `Pillar3_Stability` (`"stability"`) | Is the discovered circuit stable across re-discovery seeds? |
| 4 | `Pillar4_Robustness` (`"robustness"`) | Does the circuit hold up under input corruptions? |
| 5 | `Pillar5_Baselines` (`"baselines"`) | How does the circuit compare to random/magnitude baselines? |
| 6 | `Pillar6_Generalization` (`"generalization"`) | Does the circuit transfer to related tasks? |

> Pillar 6 (generalization) is implemented but has not been validated at scale; treat
> its scores as preliminary. `run_full_faithfulness` also accepts an optional auxiliary
> `"intervention_reliability"` pillar that measures cross-seed reproducibility.

```python
from circuitkit.evaluation.full import run_full_faithfulness

report = run_full_faithfulness(
    model, graph, task_spec, discovery_cfg,
    pillars=["patching", "ablation"],   # fast subset
)
```

### Metrics

CircuitKit provides domain-specific metrics:

- **Accuracy Metrics**: Accuracy, F1, Precision, Recall
- **Ranking Metrics**: MRR (Mean Reciprocal Rank), NDCG
- **QA Metrics**: Exact Match (EM), Span F1
- **Generation Metrics**: BLEU, ROUGE, BERTScore
- **Semantic Metrics**: Perplexity, Embedding similarity
- **WikiText-2 perplexity**: A general-language-modelling evaluation metric
  (token-weighted perplexity on the `wikitext-2-raw-v1` test split). It is the
  canonical Wanda / SparseGPT / GPTQ evaluation metric and is used to measure
  the language-modelling cost of compression interventions (pruning /
  quantization) on held-out general text, alongside task accuracy.

Discovery metrics, by contrast, are chosen **per task**: most tasks use a
single-token logit-difference, while `winogrande` uses a suffix
log-likelihood and `gsm8k` uses an answer-span NLL.

## Corruption Strategies

### What is Corruption?

**Corruption** tests circuit robustness by modifying inputs while preserving semantic meaning.

### Built-in Corruption Strategies

#### Entity Swap
Replace named entities with other entities:
- Original: "John gave the book to Mary"
- Corrupted: "Alice gave the book to Bob"

#### Paraphrase
Rephrase while preserving semantics:
- Original: "The capital of France"
- Corrupted: "What is the main city of France"

#### Distractor Injection
Add misleading but grammatical text:
- Original: "Subject verb object"
- Corrupted: "Subject [distractor clause] verb object"

#### Role Swap
Swap semantic roles:
- Original: "A gave B to C"
- Corrupted: "C gave B to A" (invalid but syntactic)

#### Token Swap
Swap random tokens:
- Original: "The quick brown fox"
- Corrupted: "The brown quick fox" (or random)

### Using Corruption

Build a `CorruptionPipeline` from one or more `CorruptionStrategy` instances and apply
it to a dataset:

```python
from circuitkit.corruption import (
    CorruptionPipeline, EntitySwapCorruption, ParaphraseCorruption,
)

pipeline = CorruptionPipeline(
    strategies=[EntitySwapCorruption(), ParaphraseCorruption()],
    n_variants=5,
)
corrupted = pipeline.corrupt_dataset(task_samples)
```

The robustness pillar (`Pillar4_Robustness`) consumes corrupted variants; see
`run_full_faithfulness`'s `corruption_variants` / `corruption_dataloaders` parameters.

## Application Methods

All application modules live under `circuitkit.applications`.

### Pruning

**Remove** components identified by the circuit, with real parameter reduction.

```python
from circuitkit.applications.pruning import StructuralPruner

pruner = StructuralPruner()
result = pruner.prune(model, circuit_scores, sparsity=0.3, dry_run=False)
```

### Circuit-restricted fine-tuning

**Restore** or adapt the model with LoRA restricted to circuit-identified layers.

```python
from circuitkit.applications.finetuning import CircuitTuner

tuner = CircuitTuner(model, node_scores=circuit_scores)
result = tuner.fit(prompts=[...], targets=[...])
```

### Steering

**Modify** activations to change model behavior.

```python
from circuitkit.applications.steering import ActivationSteering

steering = ActivationSteering(model, circuit_scores, score_threshold=0.5)
steering.compute_steering_vector(source_examples, target_examples)
out = steering.steer(prompt, coefficient=1.0)
```

### Knowledge Editing

**Surgically edit** specific knowledge at circuit-identified MLP layers.

```python
from circuitkit.applications.editing import CircuitKnowledgeEditor

CircuitKnowledgeEditor(model).edit_via_circuit(
    prompt="The capital of France is", subject="France",
    target="Lyon", circuit=circuit, method="rome",
)
```

## Auto-Schema Detection

### The Problem

Creating task definitions is tedious and error-prone. Each new dataset requires custom code.

### The Solution

Auto-schema detection automatically infers:
- Which columns contain inputs
- Which column contains labels
- How to process them
- How to evaluate predictions

### Usage

```python
from circuitkit.tasks import auto_task_from_hf

# Zero configuration!
task = auto_task_from_hf(
    dataset_name="glue",
    subset="sst2",
)

# Supports classification, QA, generation, and more
task = auto_task_from_hf("squad")                    # QA dataset
task = auto_task_from_hf("wmt14", subset="de-en")    # Translation
```

### How It Works

1. **Schema Inference**: Analyzes dataset structure
2. **Feature Detection**: Identifies input and output features
3. **Task Type Detection**: Determines if classification, QA, generation, etc.
4. **Validation**: Tests on a small sample
5. **Optimization**: Applies task-specific optimizations

---

## Key Takeaways

1. **Circuits** are minimal subgraphs responsible for specific behaviors
2. **Discovery** algorithms find these circuits automatically
3. **Tasks** define datasets and evaluation criteria
4. **Faithfulness** evaluation uses 6 pillars to assess circuit quality
5. **Corruption** tests robustness to input variations
6. **Interventions** allow you to modify or manipulate circuits
7. **Auto-schema** eliminates manual task definition work

## Further Reading

- [Quick Start](../getting-started/quickstart.md) — get started in 10 minutes
- [API Reference](../API_REFERENCE.md) — complete public API reference
- GitHub Issues: ask questions and report issues

---

**Next**: Try the [Quick Start](../getting-started/quickstart.md) for hands-on examples!
