# CircuitKit Features Matrix

## Version 1.0.0 - Current Feature Set

> First stable release, following a correctness-hardening cycle in which an audit found
> and fixed 10+ serious bugs in nominally-complete code. The Stable-tier discovery path
> is validated; experimental- and research-tier backends remain explicitly marked. See
> `CHANGELOG.md` and the README's *Validated configurations* section.

### Circuit Discovery Algorithms

CircuitKit ships **13 discovery algorithms** across 4 backends with explicit stability
tiers. Status below reflects validation maturity, not whether the code runs.

| Algorithm | Backend | Tier | Notes |
|-----------|---------|------|-------|
| `eap-ig` (default) | EAP |  Stable | EAP + integrated gradients; node/neuron level |
| `eap` | EAP |  Stable | Edge attribution patching at module level |
| `acdc` | ACDC |  Experimental | Validated on GPT-2 IOI; may fail on larger models |
| `ibcircuit` | IBCircuit |  Experimental | Single-batch training; OOM above ~3B params |
| `eap-ig-activations` | EAP |  Research | IG over node activations; validated only on GPT-2 IOI |
| `eap-clean-corrupted` | EAP |  Research | EAP with clean/corrupted activations; validated only on GPT-2 IOI |
| `eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`, `eap-ifr` | EAP |  Research | Validated only on GPT-2 IOI |
| `cdt` | CD-T |  Research | Frozen-RoPE / 50-50 gated-MLP approximations |

`discover_circuit` emits a `UserWarning` for experimental and research algorithms. The
stability map is defined in `circuitkit.backends`.

| Other | Status | Notes |
|-------|--------|-------|
| Circuit Scores Artifact | ✅ Full | Unified JSON schema for cross-backend compatibility |

### Supported Tasks

16 built-in tasks are registered on first use: `ioi`, `sva`, `gender_bias`,
`capital_country`, `hypernymy`, `greater_than`, `double_io`, `boolq`, `glue`, `mmlu`,
`winogrande`, `winogrande_mc`, `truthfulqa`, `ifeval`, `wmdp`, `gsm8k`.

| Task | Data | Notes |
|------|------|-------|
| IOI (Indirect Object Identification) | Synthetic | Core benchmark task |
| SVA (Subject-Verb Agreement) | Synthetic | Grammar evaluation |
| Gender Bias, Capital-Country, Hypernymy, Greater Than, Double-IO | Synthetic | Knowledge / reasoning circuits |
| BoolQ, GLUE, MMLU, TruthfulQA, IFEval, WMDP | Real-world | HuggingFace-backed benchmarks (single-token logit-diff metric) |
| WinoGrande | Real-world | Cloze task; suffix log-likelihood metric (`metric="suffix_loglik"`) — scores the multi-token text after the blank, not a single answer token |
| WinoGrande-MC | Real-world | Multiple-choice reformulation of WinoGrande; single-token logit-diff metric on the answer-letter tokens (option-swap corruption). Chat-templatable, unlike the cloze `winogrande` |
| GSM8K | Real-world | Open-ended generation discovery; differentiable NLL on the answer span |
| Custom Tasks | User-provided | Generic task adapter + `auto_task_from_hf` |

> **Discovery metrics are not uniform across tasks.** Most tasks score a single
> answer token at the last query position (logit difference). `winogrande` and
> `gsm8k` are exceptions — see the notes column above.

#### Chat-template policy

Each task declares a `chat_template_mode` — `"auto"`, `"on"`, or `"off"` —
controlling whether discovery prompts are wrapped in the model's chat template.
`"auto"` wraps iff the model is a chat model (its tokenizer ships a
`chat_template`); `"on"` always wraps; `"off"` always uses raw text. `"auto"` is
the default for downstream-behavior tasks (`boolq`, `glue`, `mmlu`,
`truthfulqa`, `ifeval`, `wmdp`, `gsm8k`, `winogrande_mc`) and for custom tasks;
`"off"` is the default for diagnostic minimal-pair tasks (the IOI family, `sva`,
`greater_than`, `gender_bias`, `capital_country`, `hypernymy`, `double_io`) and
for the cloze `winogrande` task. Discovery and downstream lm-eval benchmarking
use the same setting — the mode is resolved fresh on every call and is *not*
persisted into the artifact, so you must pass the same mode (and model type) at
each stage to keep a discovered circuit from being misattributed to a prompt
distribution the model is never run on. Override it with the `--chat-template-mode` CLI flag on
`discover`, or a `chat_template_mode:` field in task YAML. Helper module:
`circuitkit.tasks._chat`.

### Faithfulness Evaluation (6-Pillar Framework)

| Pillar | Class | Status | Details |
|--------|-------|--------|---------|
| 1 | **Causal Patching** | ✅ Implemented | Does the circuit explain model behavior? |
| 2 | **Ablation** | ✅ Implemented | Does ablating the circuit degrade behavior? |
| 3 | **Stability** | ✅ Implemented | Is the circuit stable across re-discovery seeds? |
| 4 | **Robustness** | ✅ Implemented | Performance under corruption variants |
| 5 | **Baselines** | ✅ Implemented | Comparison to random/magnitude baselines |
| 6 | **Generalization** | ⚠️ Implemented, not yet validated at scale | Transfer across tasks/datasets |

### Corruption Strategies

| Corruption Type | Status | Use Case |
|-----------------|--------|----------|
| Paraphrase | ✅ Full | Semantic equivalence testing |
| Entity Swap | ✅ Full | Role-sensitive evaluation |
| Distractor Insertion | ✅ Full | Attention robustness |
| Role Swap | ✅ Full | Position-sensitive testing |
| Token Swap | ✅ Full | Local perturbation |
| Custom Corruptions | ✅ Full | User-defined strategies |

### Intervention Methods

| Method | Status | Implementation | Use Case |
|--------|--------|-----------------|----------|
| Pruning | ✅ Full | Structural (parameter removal) | Model compression |
| Patching | ✅ Full | Activation intervention | Causal analysis |
| Ablation (Zero) | ✅ Full | Set activations to zero | Necessity testing |
| Mean Imputation | ✅ Full | Replace with dataset mean | Baseline comparison |
| Mean-Positional | ✅ Full | Position-aware mean | Contextualized baseline |
| Soft Healing | ✅ Full | LoRA-based restoration | Learned recovery |
| Steering | ✅ Full | Vector-based control | Behavior manipulation |

### Model Support

| Model Family | Status | Examples |
|--------------|--------|----------|
| GPT-2 | ✅ Full | gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| LLaMA | ✅ Full | meta-llama/Meta-Llama-3-8B, 70B |
| Mistral | ✅ Full | mistralai/Mistral-7B-v0.1 |
| OLMo | ✅ Full | allenai/OLMo-7B |
| Phi | ✅ Full | microsoft/phi-2 |
| Gemma | ✅ Full | google/gemma-7b |
| Other HF Models | ✅ Full | Any transformer-based model |

### Benchmarking & Evaluation

| Component | Status | Coverage |
|-----------|--------|----------|
| LM Evaluation Harness | ✅ Full | 100+ benchmark tasks |
| GSM8K (Math) | ✅ Full | Grade school math |
| MMLU (Knowledge) | ✅ Full | 57 subjects |
| TruthfulQA | ✅ Full | Truthfulness evaluation |
| HumanEval | ✅ Full | Code generation |
| HellaSwag | ✅ Full | Commonsense reasoning |
| Custom Benchmarks | ✅ Full | User-defined evaluation |

### Data & I/O

| Feature | Status | Format |
|---------|--------|--------|
| Circuit Scores Serialization | ✅ Full | JSON with versioning |
| Model Checkpoint Saving | ✅ Full | PyTorch .pt format |
| Artifact Management | ✅ Full | Organized storage and retrieval |
| Experiment Logging | ✅ Full | Structured output |
| Visualization | ✅ Full | Plotly and Matplotlib |

### CLI & API

| Interface | Status | Features |
|-----------|--------|----------|
| Command-Line Interface | ✅ Full | Rich terminal UI with progress bars |
| Discover API | ✅ Full | Main circuit discovery entry point |
| Evaluate API | ✅ Full | Faithfulness and benchmark evaluation |
| Analysis API | ✅ Full | Circuit properties and metrics |
| Memory Monitoring | ✅ Full | GPU/CPU usage tracking |
| Smart Model Selection | ✅ Full | Automatic device/precision selection |

### Development & Extensibility

| Feature | Status | Details |
|---------|--------|---------|
| Task Registration | ✅ Full | Plugin-style task system |
| Custom Algorithms | ✅ Full | Backend adapter framework |
| Type Hints | ✅ Full | Full Python 3.10+ type coverage |
| Unit Tests | ✅ Full | 100+ test cases |
| Documentation | ✅ Full | API docs, tutorials, examples |
| Examples | ✅ Full | Runnable demonstrations |

### Version & Maintenance

| Item | Status | Details |
|------|--------|---------|
| Semantic Versioning | ✅ Full | v1.0.0 (first stable release) |
| Changelog | ✅ Full | See CHANGELOG.md |
| License | ✅ Full | LSAL v1.1 (source-available) |
| Contributing Guide | ✅ Full | Development guidelines |
| Citation Info | ✅ Full | CITATION.cff |
| Python Support | ✅ Full | Python 3.10+ |

## Feature Adoption Timeline

### Version 0.1.0 (historical)
- Basic ACDC and EAP-IG algorithms
- IOI task support
- CLI interface
- Basic evaluation

### Version 1.0.0 (Current)
- Added corruption pipeline and robustness evaluation
- Generic task adapter for custom tasks
- Complete 6-pillar faithfulness framework
- Real pruning with structural parameter removal
- Soft healing capability
- Steering vectors
- Unified CircuitScores artifact
- Comprehensive benchmarking suite
- Cross-dataset generalization metrics

## Next Priorities (Future Releases)

- Distributed discovery across multiple GPUs
- DistilledCircuit compression format
- Advanced visualization dashboards
- Automated circuit caching
- Performance optimizations for large models
- Additional intervention methods
