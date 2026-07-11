# CircuitKit API Reference v1.0.0

Reference for CircuitKit's public API. For conceptual background see
[`docs/guides/CONCEPTS.md`](../guides/CONCEPTS.md); for runnable examples see
[`docs/tutorials/`](docs/tutorials/).

> **External libraries.** Application scripts that wrap standard ML methods delegate to
> upstream libraries: ROME / MEMIT knowledge editing can run through
> [EasyEdit](https://github.com/zjunlp/EasyEdit), and LoRA-based healing uses
> [PEFT](https://github.com/huggingface/peft). CircuitKit's own code handles circuit
> discovery, faithfulness evaluation, and circuit-guided target selection.

---

## Table of Contents

1. [Top-level API](#top-level-api)
2. [Discovery backends and stability tiers](#discovery-backends-and-stability-tiers)
3. [Faithfulness evaluation](#faithfulness-evaluation)
4. [Tasks and datasets](#tasks-and-datasets)
5. [Corruption strategies](#corruption-strategies)
6. [Token alignment audit](#token-alignment-audit)
7. [CD-T backend](#cd-t-backend)
8. [Architecture registry](#architecture-registry)
9. [PEFT benchmarking](#peft-benchmarking)
10. [Applications](#applications)
11. [CLI](#cli)

---

## Top-level API

**Module**: `circuitkit.api` (re-exported from `circuitkit`)

### `discover_circuit(config) -> Union[List[str], Dict]`

Run a discovery algorithm and return a pruning artifact.

- `config` (`Union[str, Dict]`): Path to a YAML config file or a config dict. Required
  top-level keys: `model`, `discovery`, `pruning`. Key fields:
  - `model.name` (str): HuggingFace / TransformerLens model identifier.
  - `model.precision` (str): Torch dtype string, e.g. `"float32"`, `"bfloat16"`.
  - `discovery.algorithm` (str): One of the 13 discovery algorithms (see
    [stability tiers](#discovery-backends-and-stability-tiers)).
  - `discovery.task` (str): Registered task name, e.g. `"ioi"`, `"mmlu"`.
  - `discovery.level` (str): `"node"` or `"neuron"`.
  - `discovery.chat_template_mode` (str, optional): `"auto"`, `"on"`, or `"off"` —
    whether discovery prompts are wrapped in the model's chat template. When
    omitted, the task's own default applies (see [Tasks and datasets](#tasks-and-datasets)).
  - `pruning.target_sparsity` (float): Fraction of components to remove.
  - `pruning.scope` (str): `"heads"`, `"mlp"`, or `"both"`.
  - `output_path` (str, optional): Where to save the pruning artifact.

**Returns**: For node-level discovery, a list of node-name strings to prune
(e.g. `['A0.1', 'MLP 3']`). For neuron-level discovery, a dict with `mlp`, `heads`, and
`_meta` keys. A `_scores.pt` side-car is saved next to the artifact for use by
`evaluate_circuit`.

**Warnings**: Emits a `UserWarning` when an experimental- or research-tier algorithm is
requested (see [stability tiers](#discovery-backends-and-stability-tiers)).

### `evaluate_circuit(config, pruned_artifact_path=None, scores_path=None) -> FaithfulnessReport`

Evaluate circuit faithfulness with the 6-pillar framework. Thin wrapper around
`run_full_faithfulness()`; reconstructs the circuit graph from the saved scores file.

- `config` (`Union[str, Dict]`): Same shape as `discover_circuit` (`model`, `discovery`,
  `pruning` keys). Used to reconstruct the graph and load the model.
- `pruned_artifact_path` (str, optional): Path to the `.pt` artifact. Defaults to
  `config['output_path']`.
- `scores_path` (str, optional): Path to the `_scores.pt` file. Auto-derived if omitted.

**Returns**: A `FaithfulnessReport` with `.patching_score` (Pillar 1, causal patching)
and `.ablation_score` (Pillar 2, ablation) — normalized faithfulness ratios in [0, 1] —
plus optional pillar fields (`.stability`, `.robustness`, `.baseline_comparison`,
`.generalization`, `.intervention_reliability`) and a `.metadata` dict. When a
random-circuit baseline is requested, its value is in `.metadata["random_avg"]`.

### `load_circuit(circuit_path) -> Union[List[str], Dict]`

Load a previously saved circuit artifact from disk.

### Package-root re-exports

`circuitkit.discover_circuit`, `circuitkit.evaluate_circuit`, `circuitkit.load_circuit`,
`circuitkit.get_task`, `circuitkit.list_tasks`, `circuitkit.register_task`,
`circuitkit.__version__`.

### Flat front-door API

**Module**: `circuitkit.quick` (re-exported from `circuitkit`, lazily imported)

For object-oriented use without config dicts, CircuitKit exposes a flat typed API at the
package root. These functions take and return Python objects directly (a loaded model, a
`Circuit`) instead of a config dict + artifact path:

- `circuitkit.load_model(name, *, dtype="bfloat16", device=None, algorithm=None)` — load a
  TransformerLens `HookedTransformer` with the hook flags discovery needs already set.
- `circuitkit.discover(model, task, *, algorithm="eap-ig", level="node", n_examples=..., scope=..., ...)` — run discovery, return a `Circuit`.
- `circuitkit.faithfulness(model, circuit, ...)` — score a `Circuit` with the 6-pillar framework.
- `circuitkit.prune(model, circuit, *, sparsity=..., scope=..., protect_layers=None, ...)` — structural pruning down to the circuit.
- `circuitkit.quantize(model, circuit, *, protect_layers=None, ...)` — circuit-aware quantization.
- `circuitkit.export_checkpoint(...)` — write a reloadable HuggingFace checkpoint.
- `circuitkit.benchmark(...)` — lm-eval benchmarking of a checkpoint.
- `circuitkit.Circuit` — the circuit object returned by `discover`.

The flat API and the dict-config `discover_circuit`/`evaluate_circuit` API are two views
of the same pipeline; pick whichever fits your workflow.

---

## Discovery backends and stability tiers

**Module**: `circuitkit.backends`

CircuitKit ships **13 discovery algorithms** across 4 backends (EAP, ACDC, IBCircuit,
CD-T). Each algorithm has an explicit stability tier. The stability map is the single
source of truth in `src/circuitkit/backends/__init__.py`.

```python
from circuitkit.backends import (
    STABILITY, STABLE_ALGORITHMS, EXPERIMENTAL_ALGORITHMS, RESEARCH_ALGORITHMS,
    is_stable, is_experimental, is_research, default_algorithm,
)

default_algorithm()       # "eap-ig"
is_stable("eap-ig")       # True
STABILITY["cdt"]          # "research"
```

| Tier | Algorithms | Notes |
|------|-----------|-------|
| **Stable** | `eap`, `eap-ig` | Validated across GPT-2 and small Llama/Gemma models. |
| **Experimental** | `acdc`, `ibcircuit` | Works on GPT-2 IOI; may fail or OOM on larger models. |
| **Research** | `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`, `eap-ifr`, `cdt` | Only validated on GPT-2 IOI. Unvalidated elsewhere. |

`discover_circuit` emits a `UserWarning` whenever an experimental- or research-tier
algorithm is requested.

> Note: `STABILITY` also includes selector/baseline keys (`random`, `magnitude`,
> `taylor`, `wanda`, `gptq`, `awq`, `tacq`, `multi_granular`) used by the
> compression selectors — these are *not* discovery algorithms. The 13 discovery
> algorithms are the ones listed in the table above (the `DISCOVERY_ALGORITHMS`
> frozenset exported from `circuitkit.backends`).

---

## Faithfulness evaluation

### 6-pillar framework

**Module**: `circuitkit.evaluation.pillars`

The faithfulness framework has **6 pillars**, one class each:

| # | Class | Question |
|---|-------|----------|
| 1 | `Pillar1_CausalPatching` | Does the circuit explain model behavior? |
| 2 | `Pillar2_Ablation` | Does ablating the circuit degrade behavior? |
| 3 | `Pillar3_Stability` | Is the circuit stable across re-discovery seeds? |
| 4 | `Pillar4_Robustness` | Does the circuit withstand input corruptions? |
| 5 | `Pillar5_Baselines` | How does the circuit compare to baselines? |
| 6 | `Pillar6_Generalization` | Does the circuit transfer to related tasks? |

Pillar 6 (generalization) is **implemented but not yet validated at scale** — it has
not been run on safety datasets or in a production sweep. Treat its scores as
preliminary.

### `run_full_faithfulness(model, graph, task_spec, discovery_cfg, ...) -> FaithfulnessReport`

**Module**: `circuitkit.evaluation.full`

Orchestrates the pillars end-to-end, ordered by cost (fast pillars first).

Key parameters:

- `pillars` (List[str], optional): Subset to run. Default: all of
  `["patching", "ablation", "baselines", "robustness", "stability", "generalization", "intervention_reliability"]`.
- `n_stability_runs` (int): Discovery runs for Pillar 3. Default 5.
- `n_reliability_seeds` (int): Seeds for the optional intervention-reliability pillar.
  Default 3.
- `target_task_spec` / `target_dataloader`: Required for Pillar 6; if omitted, Pillar 6
  is skipped.
- `pruning_cfg` (Dict, optional): Passed through to the re-discovery pillars.

**Valid pillar keys**: `patching`, `ablation`, `baselines`, `robustness`, `stability`,
`generalization`, `intervention_reliability`.

> `intervention_reliability` is an *optional auxiliary* pillar measuring cross-seed
> reproducibility (see below). The canonical numbered framework remains the 6 pillars
> above.

### `FaithfulnessReport`

**Module**: `circuitkit.evaluation.report`

Dataclass fields (all optional, `None` if the pillar was not run): `patching_score`,
`ablation_score`, `stability`, `robustness`, `baseline_comparison`, `generalization`,
`intervention_reliability`, `metadata`.

### `evaluate_graph(model, graph, dataloader, metrics, ...) -> Tensor | List[Tensor]`

**Module**: `circuitkit.evaluation.evaluate`

Score a circuit's faithfulness by running the model with out-of-circuit edges ablated.

- `metrics`: A metric callable, or a list of them, with signature
  `(logits, clean_logits, input_lengths, labels) -> Tensor`.
- `intervention` (str): `"patching"` (default), `"zero"`, `"mean"`, or
  `"mean-positional"`. `"mean"`/`"mean-positional"` require `intervention_dataloader`.
- `quiet` (bool): Suppress the progress bar. Default `False`.
- `skip_clean` (bool): Default `True`.

Call `graph.apply_topn()` or `graph.apply_threshold()` before this to define the
circuit. `evaluate_baseline` is also exported from the same module.

### Optional pillar: intervention reliability

**Module**: `circuitkit.evaluation.pillars.intervention_reliability`

`run_intervention_reliability(model, graph, task_spec, discovery_cfg, pruning_cfg, device, metric_fn, dataloader, n_seeds=3, seeds=None) -> Dict`

Measures cross-seed circuit reproducibility. Returns a dict with `r1_seed_consistency`
(mean Spearman rho across seed pairs), `r2_effect_magnitude`, `r3_effect_variance`,
`reliability_index` (harmonic mean, [0, 1]), `n_seeds`, and `per_seed`.

---

## Tasks and datasets

**Module**: `circuitkit.tasks`

16 built-in tasks are registered on first use: `ioi`, `sva`, `gender_bias`,
`capital_country`, `hypernymy`, `greater_than`, `double_io`, `boolq`, `glue`, `mmlu`,
`winogrande`, `winogrande_mc`, `truthfulqa`, `ifeval`, `wmdp`, `gsm8k`.

> **Per-task discovery metrics differ.** Most classification / MCQ tasks
> (`ioi`, `boolq`, `sva`, `winogrande_mc`, ...) score a single answer token at
> the last query position with a logit-difference metric. Two tasks do *not*:
> `winogrande` uses a **suffix log-likelihood** metric (`metric="suffix_loglik"`
> — it scores the multi-token text *after* the blank, because WinoGrande's
> disambiguating cue lies there), and `gsm8k` is an open-ended-generation task
> scored with a differentiable **negative-log-likelihood** on the answer span.
> Both remain differentiable and EAP-compatible, but the metric is not uniform
> across tasks. `winogrande_mc` is a separate multiple-choice reformulation of
> WinoGrande (explicit question, single-token logit-diff via an option-swap
> corruption) — unlike the cloze `winogrande`, it can be wrapped in a chat
> template for instruction-tuned models.

```python
import circuitkit
from circuitkit.tasks import get_task, list_tasks, register_task

list_tasks()           # registered task names
spec = get_task("ioi")
```

### Chat-template policy (`chat_template_mode`)

Every task spec carries a `chat_template_mode` attribute — `"auto"`, `"on"`, or
`"off"` — controlling whether discovery prompts are wrapped in the model's chat
template. `"auto"` wraps iff the model is a chat model (its tokenizer ships a
`chat_template`); `"on"` always wraps; `"off"` always uses raw text. `"auto"` is
the default for downstream-behavior tasks (`boolq`, `glue`, `mmlu`,
`truthfulqa`, `ifeval`, `wmdp`, `gsm8k`, `winogrande_mc`) and for custom
`GenericTaskSpec` / `NormalizedTaskSpec` tasks; `"off"` is the default for
diagnostic minimal-pair tasks (the IOI family, `sva`, `greater_than`,
`gender_bias`, `capital_country`, `hypernymy`, `double_io`) and for the cloze
`winogrande` task. The mode is overridable per run via the
`discovery.chat_template_mode` config key, the `--chat-template-mode` CLI flag,
or a `chat_template_mode:` field in task YAML; the resolved boolean is frozen
into the discovery artifact metadata so downstream stages read back an identical
setting.

The `circuitkit.tasks._chat` helper module implements the policy:

- `resolve_chat_template(mode, model) -> bool` — collapse a declared mode
  against a concrete model into a single boolean (raises `ValueError` on an
  unrecognized mode).
- `model_is_chat(model) -> bool` — `True` iff the model's tokenizer ships a
  `chat_template`.
- `wrap_prompt(model, user_text, assistant_prefix="", *, apply)` — format one
  task prompt, wrapping it in the chat template when `apply` is `True`.
- `to_tokens(model, text, *, templated)` — tokenize with BOS handled correctly
  (templated text already carries its own BOS, so `prepend_bos=False`).

### Custom HuggingFace datasets

Adapt a HF dataset into a `NormalizedTaskSpec` and register it:

```python
from datasets import load_dataset
from circuitkit.data.adapters.mcq import MCQAdapter
from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.tasks.registry import register_task

raw = list(load_dataset("cais/mmlu", "high_school_world_history",
                        split="test", streaming=True).take(24))
ds = MCQAdapter().adapt(raw, name="mmlu_hist", max_records=20)
# MCQAdapter yields unpaired records; apply a corruption strategy to build
# the (clean, corrupt) pairs that NormalizedTaskSpec requires.
ds.records = [MCQChoiceSwap().apply(r) for r in ds.records]
ds.records = [r for r in ds.records if r.is_paired]
register_task(NormalizedTaskSpec(ds, name="mmlu_hist"))
```

### WMDP task factory

`circuitkit.tasks.builtins.wmdp.build_wmdp_spec(subset, split="test", name=..., max_records=...)`
returns a named, subset-pinned WMDP task spec. `subset` is a HF config name such as
`"wmdp-bio"`, `"wmdp-chem"`, `"wmdp-cyber"`.

---

## Corruption strategies

**Module**: `circuitkit.corruption`

Available strategies: `NegationCorruption`, `DistractorInjectionCorruption`,
`EntitySwapCorruption`, `ParaphraseCorruption`, `RoleSwapCorruption`,
`TokenSwapCorruption`, `PositionShiftCorruption`, plus `VoiceSwapCorruption` and
`DistractorVariationCorruption`.

### `PositionShiftCorruption`

**Module**: `circuitkit.corruption.position_shift`

Shuffles or rotates sentence-level segments in the prompt.

- `strategy` (str): `"shuffle"` (random reorder) or `"rotate"` (cyclic shift by 1).
  Default `"shuffle"`.
- `seed` (int, optional): Random seed.

Splits on sentence-boundary punctuation. Returns the record unchanged (with `validate()`
reporting invalid) when fewer than 2 segments are found.

`run_full_faithfulness` auto-builds rule-based corruption dataloaders for the
`logical_negation`, `format_distractor`, and `position_shift` variants when they appear
in `corruption_variants` but no dataloader was supplied. The auto-build path requires the
task spec to expose `.ds.records`.

---

## Token alignment audit

**Module**: `circuitkit.data.normalized_task`

`validate_token_alignment(task_spec, model=None) -> dict`

Audit a `NormalizedTaskSpec` for records that would be silently dropped or produce
degenerate EAP gradients. Returns a dict with `total`, `same_prompt`,
`same_prompt_frac`, `empty_prompt`, `empty_prompt_frac`, `multi_token_answer`,
`multi_token_answer_frac`, `dropped_frac`, and `records_ok`. When `model` is provided,
multi-token detection uses real tokenization; otherwise it falls back to whitespace
splitting.

---

## CD-T backend

**Module**: `circuitkit.backends.cdt`

> CD-T is a **research-tier** algorithm — validated only on GPT-2 IOI. It uses a
> frozen-RoPE attention approximation (Q/K are not decomposed) and a 50/50 gated-MLP
> cross-term split. Do not rely on its scores for non-GPT-2 models.

`run_cdt_discovery(...)` lives in `circuitkit.backends.cdt.adapter`. CD-T is normally
invoked through `discover_circuit({"discovery": {"algorithm": "cdt", ...}})` rather than
called directly. The package re-exports the lower-level `wrappers`, `core`, and `basic`
modules from `circuitkit.backends.cdt`.

---

## Architecture registry

**Module**: `circuitkit.applications` (and `circuitkit.applications.arch_registry`)

Cross-architecture support for pruning/quantization applications.

```python
from circuitkit.applications import (
    MODEL_ARCH_REGISTRY, SUPPORTED_FAMILIES, PRODUCTION_FAMILIES, READY_FAMILIES,
    get_model_family, detect_model_architecture, get_arch_config,
    get_layers, get_attn_proj, get_mlp_proj, get_head_dim,
)

family = get_model_family("llama")   # -> "llama"  (takes an HF model_type, not a repo id)
config = get_arch_config("llama")
```

- `MODEL_ARCH_REGISTRY` (Dict): Per-family architecture metadata.
- `SUPPORTED_FAMILIES`, `PRODUCTION_FAMILIES`, `READY_FAMILIES` (List[str]): Family keys
  filtered by maturity status.
- `detect_model_architecture(model)`: Detect the architecture family of a loaded model.
- `get_layers`, `get_attn_proj`, `get_mlp_proj`, `get_head_dim`: Unified component
  accessors.

Errors: `UnsupportedArchitectureError`, `ArchitectureValidationError`.

---

## PEFT benchmarking

**Module**: `circuitkit.applications.finetuning.benchmark_peft`

- `BenchmarkMetrics` — dataclass holding benchmark results (parameter efficiency,
  memory, throughput, latency, accuracy).
- `PEFTBenchmark(model, method="lora", rank=8, device="cuda")` — benchmark a single PEFT
  method; `.run(num_batches, batch_size)` returns a `BenchmarkMetrics`.
- `CrossArchitectureBenchmark(models, device="cuda")` — benchmark multiple methods
  across multiple models; `.run_all(...)` and `.generate_report()`.

---

## Applications

**Module**: `circuitkit.applications`

Sub-packages, all importable from `circuitkit.applications`:

### `applications.pruning`

- `StructuralPruner` — structural parameter removal with real model-size reduction.
- `NodePruner`, `get_nodes_to_prune` — node-level pruning helpers.
- `zero_attention_head_weights`, `get_attention_architecture_info`.

### `applications.quantization`

Circuit-aware mixed-precision quantization utilities and selectors (`awq`, `tacq`).

### `applications.editing`

- `CircuitKnowledgeEditor`, `EditResult`, `UnlearningReport`.
- `BatchKnowledgeEditor`, `UnlearningVerifier`.
- `RomeHandler` / `RomeWrapper`, `MemitHandler` — ROME / MEMIT wrappers.
- `CircuitGuidedEditor`, `MCircKEEditor` (multi-hop), `CaKEEditor` (circuit-aware).

### `applications.steering`

Three distinct methods — not interchangeable:

- `ActivationSteering` — per-head, per-position **activation steering**: runtime hooks, weights untouched, reversible. The standard literature baseline.
- `CircuitWeightSteering` — **C-ΔΘ contrastive weight steering**: permanently edits per-head W_Q/K/V/O slices via θ_pos − θ_neg. The paper-faithful method.
- `SteeringComposer`, `SafetyDatasetSynthesis` — compose multiple steerings / synthesize safety datasets (built on activation steering).

### `applications.finetuning`

- `CircuitTuner` — LoRA fine-tuning restricted to circuit-identified MLP layers.
- `CircuitLoRA`, `LoRALayer`, `CircuitPEFT`, `PEFTComposer`.
- `HealingMetrics`, `HealingEvaluator`, `compute_recovery_metrics`.

> **Maturity.** The application modules build on the discovery/evaluation core but vary
> in validation coverage. See `internal/PENDING.md` for the current known-issues
> list (e.g. IBCircuit OOM above ~3B parameters, ROME MLP-only target selection).

---

## CLI

```bash
circuitkit --help
circuitkit discover --model gpt2 --algorithm eap-ig --task ioi --sparsity 0.3
circuitkit discover-yaml --model gpt2 --task-yaml task.yaml --algorithm eap-ig
circuitkit discover-smart --model gpt2 --algorithm eap-ig --task ioi --check-memory
circuitkit evaluate --model gpt2 --artifact results.pt
```

The `--algorithm` option of the discovery commands is restricted to the 13 discovery
algorithms only (it does not accept pruning/quantization method names).

---

## Version

- **Version**: 1.0.0
- **Python**: >= 3.10
- **PyTorch**: >= 2.0

## API stability

The top-level API (`discover_circuit`, `evaluate_circuit`, `load_circuit`), the
`circuitkit.backends` stability map, and the `circuitkit.evaluation` 6-pillar surface are
considered the stable public API. Backends in the experimental and research tiers may
change without notice. Breaking changes are documented in
[`CHANGELOG.md`](../CHANGELOG.md).

## Support

- **GitHub Issues**: bug reports and feature requests.
- **Documentation**: see [`docs/`](docs/) and [`docs/tutorials/`](docs/tutorials/).
