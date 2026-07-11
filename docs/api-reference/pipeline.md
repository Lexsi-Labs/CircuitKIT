# Pipeline Class

**Module**: `circuitkit.pipeline` (re-exported from `circuitkit`)

`Pipeline` is the high-level orchestrator for the Discover → Evaluate → Intervene workflow. It supports method chaining, lazy model loading, and execution history inspection.

---

## Constructors

### `Pipeline.__init__`

```python
Pipeline(
    model_name: str,
    *,
    task: Optional[str] = None,
    precision: str = "bfloat16",
    device: Optional[str] = None,
    output_dir: str = "./pipeline_output",
)
```

Create a Pipeline bound to a model name. The model is loaded lazily on first use. `task` is required before calling `.discover()` (pass it here, or use `from_custom_data()`).

```python
from circuitkit import Pipeline

pipe = Pipeline("gpt2", task="ioi", output_dir="./my_results")
```

### `Pipeline.from_artifact`

```python
Pipeline.from_artifact(
    artifact_path: Union[str, Path],
    model_name: str,
    *,
    task: Optional[str] = None,
    precision: str = "bfloat16",
    device: Optional[str] = None,
    output_dir: str = "./pipeline_output",
) -> Pipeline
```

Create a Pipeline from a previously saved circuit artifact. `._circuit` is populated immediately, so the Pipeline can go directly to `.evaluate()` or `.prune()`.

```python
pipe = Pipeline.from_artifact("./circuit.pt", "gpt2", task="ioi")
pipe.prune(sparsity=0.3)
```

### `Pipeline.from_scores`

```python
Pipeline.from_scores(
    scores_path: Union[str, Path],
    model_name: str,
    *,
    task: Optional[str] = None,
    precision: str = "bfloat16",
    device: Optional[str] = None,
    output_dir: str = "./pipeline_output",
) -> Pipeline
```

Create a Pipeline by loading pre-saved circuit scores (a `_scores.pt`/`_scores.json` side-car).

```python
pipe = Pipeline.from_scores("./circuit_scores.pt", model_name="gpt2")
pipe.selective_finetune(top_fraction=0.2)
```

### `Pipeline.from_custom_data`

```python
Pipeline.from_custom_data(
    model_name: str,
    data_path: Union[str, Path],
    *,
    clean_prompt: str,
    clean_answer: str,
    corrupt_prompt: Optional[str] = None,
    corrupt_answer: Optional[str] = None,
    task_name: Optional[str] = None,
    precision: str = "bfloat16",
    device: Optional[str] = None,
    output_dir: str = "./pipeline_output",
) -> Pipeline
```

Create a Pipeline backed by a custom CSV file, registered as a task at construction time. `data_path` points to the CSV; `clean_prompt`/`clean_answer` are required template strings. Omit `corrupt_prompt`/`corrupt_answer` for unpaired algorithms (IBCircuit, CD-T clean-only). `task_name` is auto-derived from the CSV filename if omitted.

```python
pipe = Pipeline.from_custom_data(
    "gpt2", "./my_task.csv",
    clean_prompt="{question}", clean_answer="{answer}",
)
pipe.discover(algorithm="eap-ig")
```

---

## Methods

### `discover`

```python
discover(
    *,
    algorithm: str = "eap-ig",
    level: str = "node",
    sparsity: float = 0.3,
    n_examples: int = 128,
    batch_size: int = 4,
    scope: str = "both",
    **kw,
) -> Pipeline
```

Run circuit discovery. Takes no `task` argument — `task` must be set on the constructor (or via `from_custom_data()`), not passed here. Raises `ValueError` if `task` was never set. Returns `self` for chaining.

!!! note "`scope` in `discover()` populates `pruning.scope` only"
    The `scope` parameter in `Pipeline.discover()` sets the **pruning scope** for the artifact metadata — it does NOT set `discovery.scope`. Discovery always runs across all node types.

```python
pipe = Pipeline("gpt2", task="ioi")
pipe.discover(algorithm="eap-ig", n_examples=128, sparsity=0.3, scope="both")
```

### `evaluate`

```python
evaluate(
    *,
    pillars: Optional[List[str]] = None,
    n_examples: int = 256,
    n_stability_runs: int = 5,
    target_task: Optional[str] = None,
    **kw,
) -> Pipeline
```

Run the 6-pillar faithfulness evaluation on the discovered circuit. `pillars` must be string names (e.g. `"patching"`) — see [Evaluation Framework](../evaluation/framework.md) for the full list. `n_stability_runs` controls the stability-pillar rediscovery count; `target_task` overrides the task used by the cross-task generalization pillar. Returns `self`.

```python
pipe.evaluate(pillars=["patching", "ablation"])
```

### `prune`

```python
prune(
    sparsity: float = 0.3,
    scope: str = "both",
    protect_layers: Optional[List[int]] = None,
    release_original: bool = False,
    **kw,
) -> Pipeline
```

Structurally mask the model to the circuit, storing the result as `._pruned_model` for `.export()`. `release_original=True` drops the unpruned `._model` reference after masking (frees GPU RAM, but it will be lazily reloaded if a later step needs it).

```python
pipe.prune(sparsity=0.3, scope="both")
```

### `quantize`

```python
quantize(
    bits: int = 4,
    high_fraction: float = 0.3,
    backend: str = "quanto",
    **kw,
) -> Pipeline
```

Circuit-aware mixed-precision quantization. There is no `protect_layers` parameter on this method. `bits` (3, 4, or 8) applies to the `"llmcompressor"` backend; the `"quanto"` backend assigns fixed integer qtype tiers and ignores `bits`.

### `selective_finetune`

```python
selective_finetune(
    top_fraction: float = 0.2,
    scope: str = "both",
    **kw,
) -> SelectionResult
```

Select components for circuit-guided selective fine-tuning. Requires a circuit (`.discover()` first, or construct via `from_artifact()` / `from_scores()`). Unlike the other intervention methods, this does **not** return `self` — it returns a `SelectionResult` with `.attn` and `.mlp` dicts, terminating the chain.

```python
result = pipe.selective_finetune(top_fraction=0.2)
print(result.attn.keys())
```

### `export`

```python
export(path: str, intervention: Optional[str] = None) -> str  # None = auto-detect
```

Write the intervened model as a HuggingFace checkpoint. Requires `.prune()` or `.quantize()` to have been called first (raises `RuntimeError` otherwise). Returns the checkpoint path — not `self` — terminating the chain.

```python
checkpoint_path = pipe.export("./results/checkpoint")
```

### `benchmark`

```python
benchmark(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    **kw,
) -> None
```

Run lm-evaluation-harness benchmarks on the discovered artifact. `tasks` defaults to the API's default task set if omitted. There is no `num_fewshot` parameter. Returns `None` — terminating the chain.

### `visualize`

```python
visualize(mode: str = "graph", output: Optional[str] = None, **kw) -> Any
```

Visualize the discovered circuit. `mode`: `"graph"`, `"comparison"`, `"dashboard"`. `output` is a path to save an HTML export; omit it to get an inline widget. Pass `second_circuit=` (via `**kw`) for `mode="comparison"`. Returns whatever the underlying visualizer returns — not `self` — terminating the chain.

### `evaluate_advanced`

```python
evaluate_advanced(mode: str, **kw) -> Any
```

Run advanced evaluation beyond the 6-pillar framework. `mode` is required: `"transfer"` (needs `tasks=`), `"master_grid"` (needs `methods=`, `wrappers=`, `seeds=`), or `"intervention_faithfulness"` (needs `cells=`). Raises `ValueError` for an unknown mode. Returns a mode-specific result — not `self` — terminating the chain.

### `summary`

```python
summary() -> None
```

Print a Rich table summarizing the Pipeline's model, task, steps run, circuit, and (if available) Pillar 1/2 scores to the console. Returns `None` — this is the terminal method.

```python
pipe.summary()  # prints a table; nothing to capture
```

---

## State Inspection

There is no `pipe.state` property. Inspect progress via these attributes instead:

```python
pipe.circuit      # Circuit object (after discover/from_artifact/from_scores)
pipe.model        # loaded HookedTransformer, or None until first use
pipe.pruned_model # masked/quantized model (after prune/quantize)
pipe.report  # FaithfulnessReport (after evaluate)
pipe.history      # List[str] of step names run so far, e.g. ["discover", "prune"]
```

---

## Chaining Example

```python
from circuitkit import Pipeline

pipe = (
    Pipeline("meta-llama/Llama-3.2-1B-Instruct", task="mmlu", output_dir="./llama_results")
    .discover(algorithm="eap-ig", n_examples=128, sparsity=0.3)
    .evaluate(pillars=["patching", "ablation", "baselines"])
    .prune(sparsity=0.3, scope="both")
)
checkpoint_path = pipe.export("./llama_results/pruned")
pipe.benchmark(["boolq", "winogrande"], limit=200)
pipe.summary()  # prints a Rich table
```

---

## Error States

There is no `PipelineStateError` in CircuitKit. Pipeline raises standard `RuntimeError` and `ValueError`:

| Situation | Error |
|-----------|-------|
| Call `.discover()` without `task` set | `ValueError` |
| Call `.evaluate()` / `.prune()` / `.quantize()` / `.selective_finetune()` / `.visualize()` before `.discover()` | `RuntimeError` |
| Call `.evaluate()` before the discovery artifact is saved to disk | `RuntimeError` |
| Call `.export()` before `.prune()` / `.quantize()` | `RuntimeError` |
| Call `.benchmark()` before `.discover()` | `RuntimeError` |
| `.export()` without a `path` argument | `TypeError` (missing required argument) |
| `.evaluate_advanced()` with an unknown `mode` | `ValueError` |

---

## Next Steps

- [Pipeline Overview](../user-guide/pipeline-overview.md) — constructors, lifecycle, chaining
- [Flat Typed API](flat-api.md) — function-based alternative
- [Circuit Artifacts](../advanced/circuit-artifacts.md) — loading existing artifacts
