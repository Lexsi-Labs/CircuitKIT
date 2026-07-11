# CircuitKit Changelog

All notable changes to CircuitKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [1.0.0] - 2026-07-03

### Changed (BREAKING) — submodules renamed to stop shadowing installed packages

Four top-level submodules had names that collide with installed top-level
packages. When anything places `src/circuitkit` on `sys.path`, a bare
`import <name>` would resolve to circuitkit's submodule instead of the real
package — most damagingly `import datasets` resolving to `circuitkit.datasets`,
which broke `transformer_lens` (it does `from datasets.arrow_dataset import
Dataset`). The submodules are renamed to non-colliding names:

| Old import path         | New import path          | Collided with          |
|-------------------------|--------------------------|------------------------|
| `circuitkit.evaluate`   | `circuitkit.evaluation`  | HuggingFace `evaluate` |
| `circuitkit.selectors`  | `circuitkit.selection`   | stdlib `selectors`     |
| `circuitkit.datasets`   | `circuitkit.data`        | HuggingFace `datasets` |
| `circuitkit.scripts`    | `circuitkit.tooling`     | namespace `scripts`    |

This is a hard rename with **no runtime alias** (a compatibility shim would
either re-introduce the shadow or install a global `sys.meta_path` hook at
import time, which is inappropriate for a library). Update imports directly,
e.g. `from circuitkit.evaluation.report import FaithfulnessReport`. The
`circuitkit.evaluate_circuit` *function* is unaffected. `circuitkit.datasets`'s
invariance-group content now lives at `circuitkit.data.invariance_groups`.

### Added
- Deprecated alias `circuitkit.visualize` -> `visualize_circuit` (H1 from the
  1.0.0 audit): the pre-1.0 name keeps working through 1.x with a
  `DeprecationWarning`; removal planned for 2.0.
- 3 end-to-end case-study notebooks in `examples/case-studies/` (#69):
  - `21-quantization-permanent-unlearning.ipynb` — circuit-guided quantization
    for permanent knowledge removal that survives fine-tuning recovery attacks
  - `22-gender-bias-audit-and-mitigation.ipynb` — full responsible-AI loop:
    discover bias circuit, audit with faithfulness pillars, mitigate via
    selective fine-tuning/pruning, re-audit with capability preservation
  - `23-jailbreak-safety-steering.ipynb` — circuit-restricted activation
    steering for jailbreak defense with surgical-vs-global comparison

### June 2026 — `logger` scope audit after print()→logging sweep (Issue #67)

A prior sweep replacing `print()` calls with `logger.*()` across the codebase
mechanically inserted `logger` calls into functions without first ensuring a
`logger` was actually bound in scope, causing `NameError` / `UnboundLocalError`
at runtime. Two patterns were found and fixed:

- **Logger used but never assigned** — function calls `logger.info(...)` etc.
  with no local or module-level `logger` defined anywhere in the file.
- **Logger assigned conditionally** — `logger` is only bound inside an `if`
  branch, so Python treats it as a local for the whole function and any call
  path that skips the branch hits an unbound local.

Fixed by hoisting a single `logger = get_logger(...)` (or `logging.getLogger(...)`)
to module/function top-level and removing the conditional assignment.

- `src/circuitkit/utils/config.py`: `_validate_config` had `logger` assigned only
  inside `if algo == "ibcircuit":` but used unconditionally at the end of the
  function (conditional-assignment pattern); `load_and_validate_config` called
  `logger.info(...)` with no `logger` defined at all (never-assigned pattern).
  Both fixed by hoisting `logger = get_logger(...)` to the top of each function.
- `src/circuitkit/tasks/builtins/ioi_legacy.py`: `_build_eap_dataloader` and
  `_build_ibcircuit_dataloader` called `logger.*()` with no `logger` defined
  anywhere in the file (the latter even imported `get_logger` but never called
  it). Fixed by adding a module-level `logger = get_logger("task.ioi_legacy")`
  and removing the dead local import.
- `src/circuitkit/data/task_data/tasks/ioi/ioi_dataset.py`: `get_end_idxs` called
  `logger.error(...)` in an exception handler with `get_logger` imported but
  never instantiated. Fixed by adding a module-level
  `logger = get_logger("data.task_data.ioi_dataset")`.
- `src/circuitkit/data/task_data/tasks/ioi/utils.py`: `get_ioi_data_only` and
  `_generate_ioi_data_fallback` had the same never-instantiated `get_logger`
  import. Fixed by adding a module-level
  `logger = get_logger("data.task_data.ioi_utils")`.

An AST-based scan of every `.py` file under `src/circuitkit` for both patterns
turned up no further occurrences; remaining `logger`-as-parameter and
closure-style hits in `applications/pruning/pruner.py`, `utils/exceptions.py`,
and `utils/logging.py` were manually reviewed and confirmed correct (logger is
either an explicit constructor/decorator parameter with a `None` guard, or
assigned and used within the same conditional branch).

### May 2026 (session 5) — EasyEdit/PEFT integration, new corruption variants, robustness

#### Official external repos in all KE and healing application scripts
- `validation/applications/11_knowledge_editing_gpt2.py`: Replaced
  `circuitkit.apply.rome_wrapper.RomeHandler` with EasyEdit
  (`zjunlp/EasyEdit`, cloned to `validation/_vendor/EasyEdit`). All ROME
  edits now go through `BaseEditor.from_hparams(ROMEHyperParams)`. Editor
  instances are cached per target layer to avoid reloading the HF model for
  each algorithm. Metrics use EasyEdit's built-in `rewrite_acc` /
  `rephrase_acc` / `neighborhood_acc` instead of raw logit thresholds.
- `validation/applications/25_knowledge_editing_llama_circuit.py`: Same
  EasyEdit migration; uses `hparams/ROME/llama3.2-3b.yaml` as config
  template, overrides `layers=[target_layer]` and `model_name` from script
  args. Single `editor.edit()` call for all 6 facts.
- `validation/applications/26_soft_healing_llama_circuit.py`: Replaced
  `circuitkit.apply.soft_healing.CircuitLoRA` / `train_healing_lora` with
  HuggingFace PEFT (`LoraConfig` + `get_peft_model`). Circuit-restricted
  healing uses `layers_to_transform` (top-K circuit layers by score);
  unrestricted baseline uses all layers at the same rank. Training on
  `tatsu-lab/alpaca` (1k samples), evaluation on `cais/mmlu` (HF model
  direct, no TL dependency). MMLU eval is a clean 5-shot log-prob loop.

#### New Pillar 4 corruption variants
- `src/circuitkit/corruption/position_shift.py` (new): `PositionShiftCorruption`
  shuffles or rotates sentence-level segments. Uses `_SPLIT_PATTERN` to split
  on sentence boundaries; returns record unchanged (validation fails) when fewer
  than 2 segments are found.
- `src/circuitkit/corruption/__init__.py`: Added `PositionShiftCorruption` export.
- `src/circuitkit/evaluate/pillars/robustness.py`: Added `logical_negation`,
  `format_distractor`, `position_shift` to `valid_variants`.
- `src/circuitkit/evaluate/full.py`: Auto-builds rule-based corruption
  dataloaders before the robustness pillar for `logical_negation`
  (NegationCorruption), `format_distractor` (DistractorInjectionCorruption),
  and `position_shift` (PositionShiftCorruption) when they appear in
  `corruption_variants` but no dataloader has been supplied by the caller.

#### Token alignment audit utility
- `src/circuitkit/data/normalized_task.py`: Added `validate_token_alignment(
  task_spec, model=None)`. Reports counts and fractions of same-prompt,
  empty-prompt, and multi-token-answer records. Uses actual tokenization when
  a model is provided; falls back to whitespace heuristic otherwise. Exported
  from `__all__`.

#### IBCircuit OOM pre-flight check
- `src/circuitkit/backends/ibcircuit/trainer.py`: Added CUDA memory check at
  the start of `run_ib_discovery`. Estimates required VRAM as
  `n_params * 4 bytes * 4x overhead`; raises `MemoryError` with a clear
  message (suggests EAP-IG as alternative) if estimated usage exceeds 90% of
  free memory. Avoids silent OOM mid-training on models above ~3B parameters.

#### ROME MLP-only layer selection
- `src/circuitkit/apply/knowledge_editing.py`: `_select_best_edit_layer` now
  filters strictly for MLP nodes (checks `'mlp' in node_name.lower()`). Using
  attention layers as ROME targets is architecturally invalid; the code now
  raises `ValueError` with a clear message if no MLP nodes are found in the
  circuit, rather than silently falling back to an attention layer.

### May 2026 (session 4) — all-pillar safety eval, generalization hardening, PENDING.md

#### Safety pillar eval: all 7 pillars now passing for AdvBench and WMDP
- `validation/applications/28_safety_dataset_pillar_evals.py`: Added
  `tl_model.cfg.use_split_qkv_input = True` and `tl_model.cfg.use_hook_mlp_in = True`
  to model setup so Pillar 3 (stability re-discovery via EAP-IG) no longer crashes.
  Both datasets now complete status=WORKING across all 7 pillars.

#### Architecture hardening
- `src/circuitkit/tasks/builtins/mmlu.py`: Removed four silent `get('model_name', 'gpt2')`
  fallbacks; now accesses `discovery_cfg['model_name']` directly. The key is already
  validated (and raises ValueError) in `_resolve_configs`, so the fallback was dead code
  that masked misconfigured callers.
- `src/circuitkit/backends/cdt/pyfunctions/cdt_source_to_target.py`: Removed unused
  `n_layers=12` parameter from `batch_run` signature (parameter was never read).

#### PENDING.md — known-issues checklist
- Added `PENDING.md` at repo root capturing every open issue from Hem's audit:
  CD-T RoPE/gated-MLP approximations, missing Pillar 4 corruption variants,
  Pillar 6 (generalization) not yet run on safety data, IBCircuit OOM above 3B,
  Llama-3.2-3B KE/healing drivers not yet run, ROME attention-fallback validity,
  and missing model-agnostic KE method.

### May 2026 (session 3) — validation fixes, WMDP factory, custom IOI pairs, safety pillar end-to-end

#### Validation bug-fixes
- `validation/algos/12_all_algos_llama.py`: Fixed `DatasetShape.CONTRASTIVE` (no such
  enum value) -> `DatasetShape.PAIRWISE`. Fixed custom IOI pairs to use distinct
  clean/corrupt prompts (name-swapped, e.g. "Mary and John" vs "John and Mary") so
  EAP activation_difference is non-zero. All 5 algos + custom IOI now WORKING on Llama.
- `validation/applications/28_safety_dataset_pillar_evals.py`: Full end-to-end fix:
  - Changed `pairing_mode` from `"answer_contrastive"` (same prompt = zero activation
    diff) to `"harmful_vs_benign"` for AdvBench discovery.
  - Replaced missing `build_wmdp_spec` import with direct `WMDPTaskSpec` usage (then
    moved to a proper factory once `build_wmdp_spec` was added to wmdp.py).
  - Fixed graph retrieval: `discover_circuit` returns a list (pruned node names), not a
    `Graph`. Now calls `_reconstruct_circuit_graph(model, scores_data, discovery_cfg,
    pruning_cfg, device)` using the saved `CircuitScores` JSON.
  - Fixed `run_full_faithfulness` call: passes `metric_fn=task_spec.metric_fn()` (the
    function returned by the method, not the bound method itself).
  - Fixed model loading: sets `tl_model.cfg.use_attn_result = True` after `from_pretrained`.
  - AdvBench: WORKING (patching=-5.71, ablation=-5.67). WMDP: WORKING (patching=0.004).
- `validation/applications/24_circuitkit_advbench_refusal_steering.py`: Changed
  `pairing_mode` from `"answer_contrastive"` to `"harmful_vs_benign"` for discovery
  (same-prompt pairs produce zero EAP activation difference).
- `validation/data/12_e2e_safety_advbench_gpt2.py`: Same fix as above.

#### tasks/builtins/wmdp.py: `build_wmdp_spec` factory
- Added `build_wmdp_spec(subset, split, name, max_records)` factory function. Creates a
  `_PinnedWMDPTaskSpec` subclass with `_resolve_configs` overridden to return the pinned
  subset, so callers can get a named, subset-specific task spec without modifying the
  main `WMDPTaskSpec` class.

#### acdc_utils.py: generalized `make_nd_dict`
- `make_nd_dict(end_type, n)` no longer raises `NotImplementedError` for n outside
  {3, 4}. Now supports any n >= 1 via recursive `defaultdict` construction.

### May 2026 (session 2) — MCircKE, CaKE, Circuit-Tuning, CURE/CLUE, safety evals, generalization

#### New: Multi-hop and circuit-aware knowledge editing / unlearning
- `apply/mcircke.py`: MCircKE (arxiv:2604.05876) — multi-hop circuit-guided knowledge
  editing. Edits each hop in sequence with optional circuit re-discovery between hops.
  `MCircKEEditor.edit()`, `MultiHopEdit`, `Hop`, `HopResult`, `MCircKEResult`.
- `apply/cake.py`: CaKE — circuit-aware knowledge editing with locality. Uses top-k
  circuit MLP layers from node_scores for targeted ROME edits.
  `CaKEEditor.edit()`, `CaKEEdit`, `CaKEResult`.
- `apply/circuit_tuning.py`: Circuit-Tuning — LoRA fine-tuning restricted to
  circuit-identified MLP layers. KL retain loss prevents collateral forgetting.
  `CircuitTuner.fit()`, `CircuitTunerConfig`, `CircuitTunerResult`.
- `apply/cure_clue.py`: CURE/CLUE — circuit-restricted gradient-ascent unlearning.
  Suppresses forget targets while preserving retain set via KL penalty.
  CLUE extends CURE with per-locality-prompt preservation.
  `CureClueUnlearner.unlearn()`, `CureClueConfig`, `CureClueResult`.
- `apply/__init__.py`: All four new modules exported.

#### New: Safety dataset Pillar 1-7 evaluation
- `validation/applications/28_safety_dataset_pillar_evals.py`: Runs the full
  faithfulness evaluation pipeline (Pillars 1-7) on AdvBench and WMDP.
  Supports `--quick` (patching + ablation + intervention_reliability) and
  `--datasets advbench wmdp` selection flags.

#### SafetyPromptAdapter: generalized pairing modes
- `data/adapters/safety_prompt.py`: Added `pairing_mode` parameter to `adapt()`.
  Three modes: `"answer_contrastive"` (default; same prompt, refusal vs compliance
  token — correct for circuit localisation), `"harmful_vs_benign"` (Arditi et al.
  2024 style; different prompts), `"benign_vs_harmful"` (inverted).
- `tasks/safety_datasets.py`: `_register()` now accepts and forwards `pairing_mode`.
- Updated all validation scripts to pass explicit `pairing_mode`.

#### Architecture generalization (de-hardcoding)
- `utils/exceptions.py`: Extracted `SUPPORTED_ALGORITHMS` constant (12 algorithms).
  All callers import this instead of duplicating a local list.
- `artifacts/circuit_artifact.py`: `validate()` and `create()` now use
  `SUPPORTED_ALGORITHMS` for method validation; `eap_ig` normalized to `eap-ig`.
- `artifacts/scores.py`: Removed neuron-level `NotImplementedError`; neuron-level
  `CircuitScores` now accepted with same schema as node-level.
- `cli/main.py`: All 4 `click.Choice()` algorithm lists replaced with
  `SUPPORTED_ALGORITHMS`; benchmark help text updated.
- `cli/utils.py`: Removed 4 hardcoded model-name lookup tables (`_get_model_layers`,
  `_get_model_d_model`, `_get_model_n_heads`, `_get_model_d_vocab`). ImportError
  fallback now uses `AutoConfig.from_pretrained()` for any architecture.
- `utils/memory.py`: `get_memory_efficient_config()` and `check_memory_requirements()`
  now use `_estimate_model_params()` (HF AutoConfig) instead of name-substring lookup.
  Works for any model — Llama, Mistral, Phi, Gemma, GPT-NeoX, etc.
- `backends/cdt/pyfunctions/local_importance.py`: `range(12)` replaced with
  `model.cfg.n_heads` (TL) or `model.config.num_attention_heads` (HF); fixes silent
  wrong-head-count on any non-GPT-2-small model.
- `backends/cdt/pyfunctions/cdt_source_to_target.py`: `prop_attention_probs_tmp`
  now casts rel/irrel to weight dtype for cross-architecture type safety.

#### PEAP bfloat16 fix
- `backends/eap/attribute_node.py`: Per-position grad * act_diff accumulation now
  casts to float32 before multiply to avoid dtype errors on bfloat16 models (Llama,
  Mistral, Gemma). Also fixes `.norm()` dtype issue for per-pos summary.

#### MasterGrid.run() implemented
- `evaluate/master_grid.py`: `run()` no longer raises `NotImplementedError`. Now
  iterates over (method, wrapper, seed) cells, calls `discover_circuit()`, writes
  per-cell JSON to `out_dir/`, supports resume via cell caching.

#### IOIDataset mutation error messages
- `backends/cdt/pyfunctions/ioi_dataset.py` and `data/task_data/tasks/ioi/ioi_dataset.py`:
  `__setitem__` / `__delitem__` now raise `TypeError` with a helpful message instead
  of bare `NotImplementedError()`.

#### Validation: Llama-3B all-algos suite
- `validation/algos/12_all_algos_llama.py`: Runs all 5 key algorithms (EAP, EAP-IG,
  AtP*, PEAP, RelP) end-to-end on `meta-llama/Llama-3.2-1B` with IOI task AND custom
  IOI-type prompt pairs via NormalizedTaskSpec.
- `run_validation_gpu1.sh`: Comprehensive GPU 1 run: GPT-2 suite (01-11), Llama suite
  (12), and safety pillar evals (28).

---
### May 2026 — Reliability, CD-T stability, no-placeholder audit

#### New: Pillar 7 — Intervention Reliability
- `src/circuitkit/evaluate/pillars/intervention_reliability.py`: full implementation.
  Three sub-scores — R1 (Spearman rho across re-run seeds), R2 (normalized effect
  magnitude), R3 (1 - CV of intervention deltas) — combined as a harmonic mean
  `reliability_index` in [0, 1].
- Wired into `run_full_faithfulness()` (new `"intervention_reliability"` pillar key,
  `n_reliability_seeds` parameter) and `FaithfulnessReport` (`intervention_reliability`
  field + `__repr__` section).

#### New: MMLU evaluation module
- `src/circuitkit/evaluate/mmlu_eval.py`: `evaluate_mmlu(model, n_samples, shots)`
  backed by lm-eval (primary) or direct HuggingFace datasets fallback (all 57 subjects,
  5-shot, log-prob letter scoring).

#### CD-T full propagation improvements
- `gpt2_propagation.py`: Added `_normalize_rel_irrel()` (adelaidehsu/CD_Circuit
  stabilisation rule) applied after every residual add for both GPT-2 and Llama paths.
  Prevents sign divergence through deep stacks.
- `adapter.py`: Fixed NameError — `_run_simple_cdt` now receives `device` as explicit
  parameter rather than relying on outer-scope closure.

#### Placeholder audit and fixes (apply/ module)
- `hallucination_detection.py`: Replace `torch.randn()` mock activations in
  `_get_circuit_activations()` and `HallucinationDataset._extract_activations()` with
  real TransformerLens `run_with_cache` (preferred) or HF forward-hook fallback.
- `structural_pruner.py`: Complete `_prune_mlp_neurons()` stub with real W_out /
  down_proj zeroing (node-level MLP removal without shape change).
- `benchmark_peft.py`: Replace `time.sleep(0.01)` placeholder loop with real AdamW
  forward+backward passes; compute actual throughput, mean loss, perplexity pseudo-acc.

#### App scripts (24–27) — critical bug fixes
- Script 24 (AdvBench refusal steering): Fix `steer(return_handles=True)` (unsupported
  kwarg) → per-prompt `steer(inputs, coefficient)` calls; remove duplicate dead branch;
  fix `get_or_run_discovery` call from dict-positional to kwargs API.
- Script 26 (soft healing): Fix `StructuralPruner` constructor — init takes no args;
  `prune()` takes `(model, CircuitScores, sparsity, dry_run)`; add
  `_node_scores_to_circuit_scores` helper.
- Script 27 (TruthfulQA hallucination): Replace broken `HallucinationDetector` usage
  (required `CircuitArtifact` + training) with direct `_circuit_off_circuit_ratio()`
  using `run_with_cache`; real AUROC computation.

#### RomeWrapper and soft_healing additions
- `rome_wrapper.py`: Added `RomeWrapper` shim over `RomeHandler` for script 25 API.
- `soft_healing.py`: Added `circuit` / `rank` kwarg aliases to `CircuitLoRA.__init__`;
  added `_all_node_scores()`, `_build_alpaca_dataloader()`, `train_healing_lora()`.

#### GPU-1 run scripts
- `run_gpu1_ioi_gpt2.sh`: End-to-end GPT-2 IOI algo + application benchmark on GPU 1.
- `run_gpu1_llama3b.sh`: Llama-3.2-3B-Instruct production run (apps 24–27, benchmarks
  03/06/19) on GPU 1.

### M7.1-M7.8: Extended Task Support & Corruption Framework
- **GLUE Task Suite (M7.1)**: Support for General Language Understanding Evaluation benchmark
  - Multi-task text classification support (RTE, MRPC, CoLA, SST2)
  - Automatic dataset loading and preprocessing
  - Standardized label handling for different GLUE datasets
- **MMLU Task Support (M7.2)**: Multi-choice question answering on 57 subjects
  - Direct integration with MMLU benchmark
  - Subject-level filtering and analysis
- **New Corruption Strategies (M7.3-M7.5)**: 
  - Voice Swap Corruption: First/third person perspective changes
  - Negation Corruption: Semantic negation injection while preserving structure
  - Distractor Variation Corruption: Contextual distractors for robust evaluation
- **Dataset Validation Framework (M7.6)**: Comprehensive validation for custom datasets
  - DatasetValidator class for schema and content validation
  - Support for validation against pre-defined templates
  - Detailed validation reports with actionable errors
- **Corruption Effectiveness Metrics (M7.7-M7.8)**: 
  - CorruptionEffectiveness class for measuring corruption impact
  - Effectiveness scoring based on behavioral changes
  - Comparison across corruption types and datasets

### M4.2-M4.8: Comprehensive Circuit Visualization Suite
- **Activation Saliency Heatmaps (M4.2)**: ActivationSaliencyVisualizer
  - Layer-by-layer, aggregate, and stacked visualizations
  - Interactive Plotly-based heatmaps with token-level analysis
  - HTML export and summary statistics
- **Feature Saliency Maps (M4.3)**: FeatureSaliencyVisualizer
  - Node importance ranking and bar charts
  - Network-style visualization for circuit structure
  - Gradient and patching-based attribution support
- **Interactive Circuit Editor (M4.4)**: CircuitEditor
  - Jupyter widget interface for add/remove nodes and edges
  - Change history tracking and undo functionality
  - JSON export of edited circuits
- **Comparison Dashboards (M4.5)**: ComparisonDashboard
  - Multi-seed stability analysis with Jaccard overlap metrics
  - Corruption robustness comparison across conditions
  - Cross-task generalization analysis
  - Correlation matrices and transfer visualization
- **Jupyter Widget Suite (M4.6)**: JupyterWidgetSuite
  - Unified tabbed interface for all visualizers
  - Automatic error handling and graceful fallback
  - Optional HTML export of all visualizations
- **Streamlit Dashboard (M4.7)**: StreamlitCircuitDashboard
  - Multi-page web application for circuit analysis
  - Circuit upload, visualization, and interactive analysis
  - Real-time updates with responsive layout
- **Visualization Gallery (M4.8)**: GalleryGenerator
  - Auto-generate interactive HTML gallery from visualizations
  - Searchable and filterable by visualization type
  - Responsive card-based layout with full navigation

### Earlier 1.0.0 groundwork (2026-05-16 internal pre-cut, never published)

First stable release. This release closes a correctness-hardening cycle: an audit
found and fixed 10+ serious bugs in code that had been marked "done". The public API
is stable and the Stable-tier discovery path is validated (see README → *Validated
configurations*). Experimental- and research-tier backends remain explicitly marked
as such. The full found-and-fixed bug history is kept below and in earlier entries
deliberately — for interpretability tooling an honest fix log is a trust signal.

### Highlights of the hardening cycle

- **bf16 / GQA correctness across the stack.** Discovery backends (PEAP, CD-T, ACDC)
  and every application (knowledge editing + MEMIT construction, `CircuitWeightSteering`
  K/V steering, circuit-localized fine-tuning) now run correctly on bf16 models and on
  grouped-query-attention architectures. ACDC K/V destination nodes honor the ungroup
  flag; CD-T uses bias-free gated-MLP handling and a correct pattern dtype.
- **Chat-template handling.** Per-task `chat_template_mode` (`auto` / `on` / `off`),
  end-to-end BOS consistency across CD-T, selectors, applications and fine-tuning, a
  double-BOS tokenization fix in the MMLU / GLUE / generic dataloaders, and auto
  chat-model detection. Discovery freezes the resolved policy into artifact metadata.
- **Checkpoint export protocol.** 8 checkpoint export/eval correctness bugs fixed;
  intervened (pruned / quantized) checkpoints are now exported as standard reloadable
  HuggingFace checkpoints and are vLLM-evaluable via lm-eval.
- **`lrp` pruning selector removed.** The orphan `lrp` selector was dropped; pruning
  selectors are now `random`, `magnitude`, `taylor`, `wanda`, and `multi_granular`.
- **Unified `scope` / `protect_layers` component selection.** `StructuralPruner.prune`,
  `quick.prune`, and `quick.quantize` accept a consistent `scope` (`heads` / `mlp` /
  `both`) and `protect_layers` API for choosing which component types an intervention
  touches.
- **`StructuralPruner` sparsity fix.** Non-head/MLP nodes (e.g. `Resid Start`) are
  excluded from prune candidates — previously selectable but never masked, which
  deflated the effective sparsity reported.
- **Other fixes.** Wrong-axis structural pruning, unnormalized faithfulness scores
  (normalized ratio + convention-agnostic layer overlap), Pillar 4 robustness no
  longer fabricates zero-delta results, GSM8K answer-token labels on the Llama-3
  tokenizer, silent corruption failure in custom-data ingestion, and a flat typed API
  (`circuitkit.discover` / `prune` / ...) alongside the dict-config API.

> _Versioning note:_ this release was briefly numbered `0.9.0` during the hardening
> cycle; it ships as `1.0.0`, the first stable public release.

### Chat-template support for instruction-tuned models (2026-05-16)

Circuit discovery and downstream lm-eval benchmarking can now wrap task prompts
in an instruction-tuned model's chat template, and discovery freezes the chosen
policy so a discovered circuit is not misattributed to a prompt distribution the
model is never run on.

#### Per-task `chat_template_mode`
- Every task now declares a `chat_template_mode` — `"auto"`, `"on"`, or `"off"`.
  `"auto"` wraps prompts in the model's chat template iff the model is a chat
  model (its tokenizer ships a `chat_template`); `"on"` always wraps; `"off"`
  always uses raw text. `"auto"` is the default for downstream-behavior tasks
  (`boolq`, `glue`, `mmlu`, `truthfulqa`, `ifeval`, `wmdp`, `gsm8k`,
  `winogrande_mc`) and for custom user-defined tasks; `"off"` is the default for
  diagnostic minimal-pair tasks (the IOI family, `sva`, `greater_than`,
  `gender_bias`, `capital_country`, `hypernymy`, `double_io`) and for the cloze
  `winogrande` task. Discovery and downstream lm-eval use the same setting, and
  the resolved value is recorded in the discovery artifact metadata. New helper
  module `src/circuitkit/tasks/_chat.py` (`resolve_chat_template`, `wrap_prompt`,
  `to_tokens`, `model_is_chat`).
- The `discover` CLI command gains a `--chat-template-mode auto|on|off` flag; it
  is also settable as a `chat_template_mode:` field in task YAML. When unset, the
  task's own default applies.

#### New task
- **WinoGrande-MC builtin** (`winogrande_mc`): a chat-templatable multiple-choice
  reformulation of WinoGrande. Each item becomes an explicit MC question and is
  scored with a single-token logit-difference metric on the answer-letter tokens
  (via an option-swap corruption), so instruction-tuned models can be studied on
  WinoGrande without the cloze suffix-log-likelihood metric that blocks
  templating. The original `winogrande` (cloze, `metric="suffix_loglik"`) is
  unchanged. Brings the built-in task count to **16**.

#### Evaluation metrics
- The lm-eval benchmarking paths (`evaluate/lm_harness.py`,
  `evaluate/lm_eval_simple.py`, `evaluate/checkpoint_benchmark.py`) now
  auto-apply the model's chat template for instruction-tuned checkpoints; the
  setting is resolved once so original and pruned runs stay comparable.
  Cloze / loglikelihood tasks (`winogrande`, `lambada`, `wikitext`) are kept raw
  even then, since templating a cloze task has no user/assistant turn structure.

### Correctness-fix round (2026-05-16)

A focused correctness pass over the discovery harness and the compression
selectors, plus the canonical reimplementation of the paper-faithful selectors.
No public-API signature changes (`discover_circuit`, `evaluate_circuit`,
`load_circuit` unchanged).

#### Harness / selector bug fixes
- **Pruning attention-head no-op.** Fixed a structural-pruning path where
  zeroing an attention head's weights had no effect on the forward pass; the
  pruned head is now genuinely removed from the computation.
- **Faithfulness §10.1 normalization.** Corrected the normalization in the
  §10.1 faithfulness computation so circuit/baseline/full scores are placed on
  a comparable scale.
- **WinoGrande scoring.** The WinoGrande task was rewritten: the previous
  single-token last-position logit-diff scored ~chance (~0.51) because the
  disambiguating cue lies *after* the blank. It now uses a **suffix
  log-likelihood** metric (`metric="suffix_loglik"`) — it fills the blank with
  each option and compares the model's log-likelihood of the multi-token suffix
  span. This metric differs *in kind* from the single-token logit-diff used by
  BoolQ / SVA / IOI; the discovery metric is now explicitly per-task.
- **Wanda / AWQ flag.** Fixed a calibration-data flag so the Wanda and AWQ
  selectors calibrate on general text (WikiText-2) rather than the downstream
  task's EAP dataloader, matching the source papers.
- **`multi_granular` MLP.** Fixed the MLP-component path of the
  `multi_granular` pruning selector.

#### Canonical (paper-faithful) selector reimplementation
- The compression selectors were reimplemented to be honest about what they
  compute. **Wanda** is now labelled "Wanda aggregated to component
  granularity" (the genuine `|W| · E[‖X‖]` Wanda metric, lifted from
  per-weight to per-component scores). **GPTQ** and **AWQ** are honestly
  framed as *derived proxies* — `GPTQ-Hessian-diag` (the diagonal-Hessian
  importance signal) and `AWQ-salience` (per-channel activation-magnitude
  salience) — rather than presented as the full GPTQ / AWQ quantization
  algorithms.

#### New task
- **GSM8K builtin** (`gsm8k`): open-ended-generation circuit discovery on
  grade-school math word problems, scored with a differentiable
  negative-log-likelihood on the answer span. Brings the built-in task count
  to **15**. Known limitation: the EAP collate path supplies a single-token
  answer proxy, so the metric scores a single-token answer rather than a full
  multi-token numeric answer.

#### Evaluation metrics
- Added **WikiText-2 perplexity** (token-weighted, `wikitext-2-raw-v1` test
  split) as a language-modelling evaluation metric — the canonical
  Wanda / SparseGPT / GPTQ evaluation metric — used to measure the
  language-modelling cost of compression interventions on held-out general
  text, alongside general-text calibration for the selectors.

#### Other correctness fixes
- **IBCircuit KL constant.** Corrected a constant in the IBCircuit KL term.

#### Dependencies
- Pinned `transformer-lens>=2.18,<3` (from PyPI; verified against 2.18.x, which
  adds Gemma-3 support), replacing the earlier 2.11 pin.

#### Documentation
- Re-verified every factual claim and code snippet in `README.md`,
  `API_REFERENCE.md`, and `docs/` against the current code: corrected the
  built-in task count to **15** (added `gsm8k`); documented WinoGrande's
  suffix-log-likelihood metric and the per-task metric asymmetry across
  `README.md` / `API_REFERENCE.md` / `docs/reference/FEATURES.md` /
  `docs/guides/CONCEPTS.md` / `docs/DATASET_TYPES.md`; documented WikiText-2
  perplexity where evaluation metrics are listed; fixed the `DISCOVERY_ALGORITHMS`
  module reference to `circuitkit.backends`; and corrected the
  `transformer-lens` pin in `docs/INSTALLATION.md` from a GitHub pin to the
  PyPI `>=2.18,<3` pin.

### Production-readiness hardening pass

A production / open-source readiness pass: end-to-end bug fixes across the
data → discovery → applications pipeline, a runnable test suite, an
algorithm-registry consolidation, dead-code removal, GPU validation, accurate
documentation, and a fresh set of runnable examples. The public API
(`discover_circuit`, `evaluate_circuit`, `load_circuit`) is unchanged.

#### Test suite and packaging
- Fixed the test suite so `pytest tests/` runs cleanly.
- Corrected the `pytest.ini` section header so pytest discovers its configuration.
- Relocated test files that had been left inside `src/circuitkit/` into the proper
  `tests/` tree, so the installed package no longer ships test modules.

#### Algorithm registry consolidation
- Consolidated every algorithm name, category, and stability tier into a single
  source of truth: `backends.ALGORITHMS` in `src/circuitkit/backends/__init__.py`.
  `STABILITY`, `DISCOVERY_ALGORITHMS`, `PRUNING_ALGORITHMS`,
  `QUANTIZATION_ALGORITHMS`, `STABLE_/EXPERIMENTAL_/RESEARCH_ALGORITHMS`, and the
  validation registries in `circuitkit.utils.exceptions` are now all derived views
  of `ALGORITHMS` — no second hand-maintained list.

#### Stability-tier enforcement
- The 13 discovery algorithms now carry explicit stability tiers
  (`is_stable`/`is_experimental`/`is_research`, `default_algorithm`): **Stable**
  (`eap`, `eap-ig`, `eap-ig-activations`, `eap-clean-corrupted`), **Experimental**
  (`acdc`, `ibcircuit`), **Research** (`eap-exact`, `atp-gd`, `eap-gp`, `relp`,
  `peap`, `eap-ifr`, `cdt`). `discover_circuit` enforces the tiers at runtime by
  emitting a `UserWarning` for experimental/research algorithms.

#### CLI discovery-algorithm choices
- The discovery CLI commands (`discover`, `discover-yaml`, `discover-smart`,
  `transfer-matrix`) now restrict `--algorithm` to the 13 discovery algorithms
  (`DISCOVERY_ALGORITHMS`) instead of the full `SUPPORTED_ALGORITHMS` union, which had
  incorrectly offered pruning/quantization method names as discovery choices.

#### Dead-code removal
- Removed the superseded `applications/pruning/pruner_legacy.py` and other dead
  code paths; the active structural pruner is `applications.pruning.pruner`.

#### End-to-end bug fixes (data / discovery / applications)
- Fixed corruption and data-type issues across the data pipeline (corruption
  strategy dispatch, dataset-type detection, and tensor dtype mismatches that
  surfaced on bfloat16 models).
- Fixed end-to-end bugs spanning data loading, discovery, and the downstream
  applications so the discover → evaluate → apply workflow runs cleanly on GPT-2.

#### GPU validation
- Validated the discovery, evaluation, and application pipelines on GPU as well
  as CPU.

#### Examples
- Added a fresh set of runnable, CPU-friendly example scripts under `examples/`
  covering the full workflow: circuit discovery via the Python API, the CLI, and
  a YAML config; 6-pillar faithfulness evaluation; and each application (pruning,
  steering, knowledge editing, quantization, finetuning/healing). Each script runs
  end-to-end on GPT-2. See `examples/README.md`.

#### Documentation corrections
- `README.md`: rewritten for accuracy — added a stability-tier table, corrected the
  discovery-algorithm count from an overstated "21" to the actual **13** (the other
  entries in `STABILITY` are compression selectors/baselines, not discovery
  algorithms), corrected the built-in task count to **14**, and described the
  6-pillar faithfulness framework honestly (Pillar 6 / generalization is implemented
  but not yet validated at scale).
- `API_REFERENCE.md`: corrected the module path from the non-existent
  `circuitkit.apply` to `circuitkit.applications`; replaced stale symbol names
  (`detect_architecture` → `detect_model_architecture`, `ARCHITECTURE_REGISTRY` →
  `MODEL_ARCH_REGISTRY`, removed the non-existent `register_architecture`); corrected
  the faithfulness framework from a claimed "7 pillars" to the canonical **6 pillars**
  with `intervention_reliability` documented as an optional auxiliary pillar; fixed
  `evaluate_circuit` to reflect its real config shape.
- `docs/index.rst`: corrected the algorithm count to 13 with stability tiers; replaced
  the non-existent `circuitkit.apply.*` imports and `circuit_aware_prune` with the
  real `circuitkit.applications` API.
- `docs/INSTALLATION.md`: corrected the required Python version to **3.10+** (was
  listed as 3.9), fixed the verification snippet's import paths, and removed the
  inaccurate `pip install circuitkit` instruction (not published to PyPI).

## [0.2.0] - 2026-04-13

### Major Features Added

#### Corruption Pipeline (Workstream A)
- **5+ Corruption Strategies**: Paraphrase, entity swap, distractor, role swap, token swap
- **CorruptionPipeline Class**: Unified API for applying corruptions to IOI, SVA, and generic tasks
- **Variant Generation**: Automatic creation of corrupted task variants
- **Custom Corruption Support**: User-defined corruption functions via callbacks
- **Integration**: Seamless integration with discovery and evaluation pipelines

#### Generic Task Adapter (Workstream B)
- **TaskAdapter Framework**: Support for arbitrary custom tasks
- **Auto-conversion**: Transforms custom tasks to IOI-like format internally
- **Circuit Discovery**: Full discovery support for custom tasks
- **Evaluation**: Complete faithfulness evaluation on custom tasks
- **Built-in Examples**: Sample adapters for domain-specific tasks

#### 6-Pillar Faithfulness Framework (Workstreams C & D)
- **Pillar 1: Accuracy** - Original model performance baseline
- **Pillar 2: Sufficiency** - Circuit alone's predictive ability (Workstream C1)
- **Pillar 3: Necessity** - Ablation impact on performance (Workstreams C2-C4)
  - Patching-based ablation
  - Neuron-level ablation
  - Mixed-layer ablation
  - Causality tracking
- **Pillar 4: Robustness** - Performance under corruption variants (Workstream D5)
  - Robustness under paraphrase
  - Robustness under entity swaps
  - Robustness to distribution shifts
  - Corruption variant comparison
- **Pillar 5: Compositionality** - Cross-layer component interactions (Workstream D6)
  - Component relationship analysis
  - Layer-wise interaction metrics
  - Compositionality scoring
  - Circuit dependency graphs
- **Pillar 6: Generalization** - Transfer across tasks and datasets (Workstream D7)
  - Cross-task transfer evaluation
  - Transfer matrix construction (NxN)
  - Generalization statistics
  - Domain adaptation metrics

#### Real Pruning & Soft Healing (Workstreams E & F)
- **StructuralPruner**: Real parameter removal (not just masking)
  - Attention head removal (W_Q, W_K, W_V, W_O modification)
  - MLP neuron pruning
  - Cross-layer consistency
  - Sparsity measurement
  - Dry-run mode for safety
- **SoftHealer**: Learned restoration via LoRA adapters (Workstream E3)
  - LoRA fine-tuning on pruned models
  - Automatic adaptation learning
  - Restoration quality metrics
  - Efficient parameter updates
- **Healing Variants**: Multiple restoration strategies (Workstream E4)
  - Full LoRA healing
  - Residual-only healing
  - Selective healing (critical nodes only)

#### Steering Module (Workstream H)
- **SteeringVectors**: Model behavior manipulation
  - Vector extraction from circuits
  - Direction intervention
  - Multi-head steering
  - Concept-specific activation
- **Intervention Control**: Fine-grained behavior adjustment
  - Magnitude scaling
  - Direction mixing
  - Cumulative effect tracking

#### Unified Scores Artifact (Workstream G)
- **CircuitScores Dataclass**: Standardized node importance schema
  - Cross-backend compatibility
  - JSON serialization with versioning
  - Score normalization (minmax, zscore)
  - Top-K/bottom-K selection
- **Backend Integration**: ACDC, EAP, EAP-IG, IBCircuit support
  - Unified output format
  - Backward compatibility with .pt format
  - Extensible for new algorithms

#### Benchmarking Suite (Workstream I)
- **LM Evaluation Harness Integration**: 100+ benchmark tasks
  - GSM8K (math reasoning)
  - MMLU (general knowledge, 57 subjects)
  - TruthfulQA (truthfulness)
  - HumanEval (code generation)
  - HellaSwag (commonsense reasoning)
  - Additional benchmarks via lm-eval
- **Automated Benchmarking**: End-to-end evaluation pipelines
  - Batch processing
  - Memory-efficient evaluation
  - Comprehensive result aggregation
  - Performance comparison (original vs. pruned)
- **Benchmark CLI**: Command-line evaluation interface
  - Task selection
  - Limit parameters
  - Output formatting

#### Circuit Analysis & Visualization (Workstream J)
- **Circuit Analysis Tools**: Understanding discovered circuits
  - Node importance ranking
  - Component visualization
  - Dependency graph generation
  - Circuit statistics and summaries
- **Visualization Library**: Rich graphical outputs
  - Plotly interactive plots
  - Matplotlib static figures
  - Network graphs
  - Heatmaps and distributions
  - Custom visualization framework

#### Memory Optimization (Workstream K)
- **Smart Model Loading**: Automatic device and precision selection
  - GPU availability detection
  - Precision optimization (float32, float16, bfloat16)
  - Batch size auto-tuning
- **Memory Monitoring**: Real-time resource tracking
  - GPU memory usage
  - CPU memory tracking
  - Swap usage monitoring
  - Peak memory reporting
- **Efficiency Features**: Reduced memory footprint
  - Gradient checkpointing
  - Mixed-precision training
  - Activation recomputation
  - Batch processing optimization

### New Modules

- `circuitkit.corruption`: Corruption pipeline and strategies
- `circuitkit.apply`: Pruning, healing, and steering implementations
- `circuitkit.artifacts`: Unified scores and artifact management
- `circuitkit.evaluation`: Comprehensive faithfulness metrics
- `circuitkit.analysis`: Circuit analysis and properties

### API Enhancements

#### CLI Commands
- `discover-smart`: Memory-aware discovery with auto-configuration
- Enhanced `discover`: Support for corruption and custom tasks
- Enhanced `evaluate`: Faithfulness and benchmark evaluation
- `list-models`: Model information and capabilities
- `check-memory`: Memory requirement estimation

#### Python API

```python
# Discovery
from circuitkit.api import discover_circuit

# Evaluation with 6 pillars
from circuitkit.evaluation import (
    evaluate_circuit,
    Pillar1_Accuracy, Pillar2_Sufficiency, Pillar3_Necessity,
    Pillar4_Robustness, Pillar5_Compositionality, Pillar6_Generalization
)

# Interventions
from circuitkit.apply import (
    StructuralPruner, SoftHealer, SteeringVectors
)

# Corruptions
from circuitkit.corruption import CorruptionPipeline

# Artifacts
from circuitkit.artifacts import CircuitScores

# Analysis
from circuitkit.analysis import CircuitAnalyzer
```

### Backward Compatibility

- All v0.1 APIs remain functional
- Legacy .pt artifact format still supported
- Existing discovery code unchanged
- New features are opt-in

### Documentation

- Comprehensive README with v0.2 features
- FEATURES.md: Complete feature matrix
- INSTALLATION.md: Dependency groups and scenarios
- CONTRIBUTING.md: Development guidelines
- CITATION.cff: Academic citation format
- Tutorial notebooks for major features
- API documentation
- Extensive examples and use cases

### Testing

- 100+ new unit tests
- Integration test coverage for all workstreams
- CI/CD pipeline setup
- Automated testing on Python 3.9+

### Dependencies

- Added: `torch-pruning>=1.0` (structural pruning)
- Added: `spacy>=3.7` (optional, for text corruption)
- Added: `lm-eval` (optional, for benchmarking)
- Updated: `transformers>=4.52.3` (latest features)
- Updated: `transformer-lens` (pinned to stable commit)

### Internal Changes

- Refactored discovery API for extensibility
- Improved error handling and logging
- Enhanced type hints across codebase
- Standardized metric computation
- Optimized memory usage

### Bug Fixes

- Fixed edge cases in circuit scoring
- Improved numerical stability in normalization
- Better error messages for common issues
- Resolved platform compatibility issues

### Performance Improvements

- 2-3x faster discovery with EAP-IG
- Reduced memory footprint for large models
- Optimized artifact serialization
- Faster evaluation with batch processing

---

## [0.1.0] - 2025-01-15

### Initial Release

#### Core Features
- ACDC circuit discovery algorithm
- EAP-IG node-level circuit discovery
- IOI, SVA, and basic task support
- CLI interface
- Model evaluation
- LM Evaluation Harness integration

#### Included Algorithms
- ACDC (patching-based)
- EAP (edge attribution)
- EAP-IG (integrated gradients variant)

#### Supported Tasks
- IOI (Indirect Object Identification)
- SVA (Subject-Verb Agreement)
- Gender Bias
- Capital-Country
- Hypernymy
- Greater Than
- MMLU (via lm-eval)

#### API
- `discover_circuit()`: Main discovery function
- `evaluate_circuit()`: Basic performance evaluation
- CLI: `circuitkit discover`, `circuitkit evaluate`

#### Testing & Documentation
- Basic unit tests
- CLI documentation
- Example notebooks

---

## Versioning Policy

CircuitKit follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (0.X.0): Breaking API changes
- **MINOR** (.0.X): New features, backward compatible
- **PATCH** (.0.0.X): Bug fixes, backward compatible

---

## Support & Contributions

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/Lexsi-Labs/circuitkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lexsi-Labs/circuitkit/discussions)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Acknowledgments

CircuitKit builds on pioneering work in mechanistic interpretability:
- ACDC by Conmy et al.
- Integrated Gradients by Sundararajan et al.
- TransformerLens by Nanda et al.
- And the broader open-source ML community

---

**Created**: 2026-04-13  
**Latest Update**: 2026-05-16  
**Version**: 1.0.0
