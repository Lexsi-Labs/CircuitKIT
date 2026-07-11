# Taxonomy

> **This file is the single source of truth for the CircuitKit taxonomy.**
> README, this document, and the package docstring all reference this document;
> they do not redefine it.

CircuitKit is a **discover → evaluate → intervene** toolkit for mechanistic interpretability. Each step has multiple independent alternatives — you pick the combination that matches your use case.

```text
CircuitKit — discover · evaluate · intervene
│
├── 1. DISCOVER — find the circuit for a behaviour
│   │   input: model + task (built-in or custom) + algorithm config
│   │   output: circuit artifact (.pt + scores)
│   │
│   ├── Algorithms (pick one):
│   │   ├── EAP family (Stable): eap-ig, eap
│   │   ├── EAP research (Research): eap-ig-activations, eap-clean-corrupted, eap-exact, atp-gd, eap-gp, relp, peap, eap-ifr
│   │   ├── ACDC / IBCircuit (Experimental)
│   │   └── CD-T (Research)
│   │
│   └── Granularity:
│       ├── Node-level → list of component names (fast, default)
│       └── Neuron-level → dict of indices (granular, expensive)
│
├── 2. EVALUATE — score how faithful the circuit is
│   │   input: model + circuit artifact + task
│   │   output: FaithfulnessReport (6 pillars)
│   │
│   └── Pillars (pick any subset):
│       ├── 1 Causal Patching (fast)
│       ├── 2 Ablation (fast)
│       ├── 3 Stability (expensive — re-runs discovery N×)
│       ├── 4 Robustness (moderate — needs corruption extra)
│       ├── 5 Baselines (moderate — random + magnitude)
│       └── 6 Generalization (expensive — needs target task)
│
└── 3. INTERVENE — act on the circuit
    │   input: model + circuit artifact
    │   output: compressed model / edited model / HF checkpoint
    │
    └── Applications (pick one or more):
        ├── Prune → structural removal of low-scoring components
        ├── Quantize → circuit-guided mixed-precision (optimum-quanto)
        ├── Edit → ROME/MEMIT knowledge editing at circuit nodes
        ├── Steer → activation steering / contrastive weight steering
        └── Fine-tune → circuit-restricted LoRA
```

## Step 1: Discover

Every experiment starts with discovery. You give CircuitKit a model and a task; it scores every component by importance and returns the top-K as the circuit.

| Decision | Options | Default |
|---|---|---|
| Algorithm | `eap-ig`, `eap`, `acdc`, `ibcircuit`, `cdt`, ... | `eap-ig` (Stable) |
| Granularity | `node` or `neuron` | `node` |
| Sparsity | fraction of components to remove | 0.25 (keep top 75%) |
| Data | built-in task or custom CSV/JSONL/HF dataset | — |

## Step 2: Evaluate

After discovery, evaluate how faithfully the circuit explains the model's behaviour. Run a fast subset (Pillars 1+2) for iteration; run all 6 for publication-quality audits.

| Pillar | Question | Cost | Requires |
|---|---|---|---|
| 1 Patching | Does patching back circuit nodes recover behaviour? | Fast | — |
| 2 Ablation | Does removing circuit nodes degrade behaviour? | Fast | — |
| 3 Stability | Is the circuit consistent across re-discovery seeds? | Expensive | — |
| 4 Robustness | Does the circuit hold under input corruptions? | Moderate | `en_core_web_sm` model |
| 5 Baselines | Better than random/magnitude selection? | Moderate | — |
| 6 Generalization | Does it transfer to a related task? | Expensive | `target_task` |

## Step 3: Intervene

The step that makes CircuitKit different: once you have a circuit, act on it. Each application produces a real HuggingFace checkpoint you can reload and benchmark.

| Application | What it does | Output |
|---|---|---|
| Prune | Remove lowest-scoring components structurally | Masked model + HF checkpoint |
| Quantize | Apply mixed-precision guided by circuit importance | Quantized model |
| Edit | Rewrite facts at circuit-identified MLP layers | Edited model |
| Steer | Modify activations at circuit nodes | Hooked model (reversible) |
| Fine-tune | Restrict LoRA adapters to circuit components | Fine-tuned LoRA |

## Five interfaces, same pipeline

| Interface | Entry point | Best for |
|---|---|---|
| Dict-config API | `discover_circuit({...})` | Full control, YAML-driven, custom corruption |
| Flat typed API | `ck.discover(model, ...)` | Clean keyword-arg calls |
| Stateful Pipeline | `Pipeline(model, task)` | Multi-step chained experiments |
| CLI | `circuitkit discover ...` | Shell scripts, CI |
| YAML | `circuitkit run pipeline.yaml` | Reproducible experiments |

## How to choose

New to CircuitKit? Start here:

1. **Discover** with `eap-ig` (default) at node-level, 30% sparsity, on a built-in task (IOI or SVA).
2. **Evaluate** with Pillars 1+2+5 (fast subset) to confirm basic faithfulness.
3. **Prune** the model down to the circuit and export a HuggingFace checkpoint.
4. **Benchmark** the checkpoint with `ck.benchmark()`.

This is the path the audit paper follows. Once you have a working pipeline, explore other algorithms, applications, and custom data.
