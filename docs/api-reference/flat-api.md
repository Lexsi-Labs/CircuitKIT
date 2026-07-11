# Flat Typed API

**Module**: `circuitkit.quick` (re-exported from `circuitkit`)

The flat API provides typed keyword-argument functions instead of nested config dicts. It is a strict superset shortcut: it covers the common discovery → evaluation → intervention path. For anything it can't express, use the [Dict-Config API](dict-config.md) directly.

---

## `load_model`

```python
load_model(
    name: str,
    *,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> HookedTransformer
```

Load a TransformerLens model with the hook flags circuit discovery needs already set. Use this instead of `HookedTransformer.from_pretrained` — plain TL loading does not enable the attention-result and split-QKV hook points EAP-family algorithms require.

**Sets**: `use_attn_result=True`, `use_split_qkv_input=True`, `use_hook_mlp_in=True`, and `ungroup_grouped_query_attention=True` (when available).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | — | HF / TL model id, e.g. `"gpt2"`, `"meta-llama/Llama-3.2-1B-Instruct"` |
| `dtype` | `"bfloat16"` | Torch dtype string |
| `device` | auto | `"cuda"` if available, else `"cpu"` |
| `algorithm` | `None` | If given, only the flags that algorithm needs are set |

```python
import circuitkit as ck
model = ck.load_model("gpt2", dtype="float32")
```

---

## `discover`

```python
discover(
    model: HookedTransformer,
    task: str,
    *,
    algorithm: str = "eap-ig",
    level: str = "node",
    n_examples: int = 128,
    batch_size: int = 4,
    sparsity: float = 0.3,
    scope: str = "both",
    output_path: str = "./circuit_discovery_results.pt",
    **kw,
) -> Circuit
```

Run circuit discovery and return a `Circuit` object.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Loaded `HookedTransformer` from `load_model` |
| `task` | — | Registered task name, e.g. `"ioi"`, `"mmlu"` |
| `algorithm` | `"eap-ig"` | Discovery algorithm |
| `level` | `"node"` | `"node"` (heads/MLPs) or `"neuron"` |
| `n_examples` | 128 | Number of examples to attribute over |
| `batch_size` | 4 | Discovery batch size |
| `sparsity` | 0.3 | Target pruning sparsity |
| `scope` | `"both"` | Pruning scope: `"heads"`, `"mlp"`, or `"both"` |
| `output_path` | `"./circuit_discovery_results.pt"` | Where to write artifacts |
| `**kw` | — | Extra keys forwarded into the discovery block (e.g. `ig_steps=3`) |

```python
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=64, ig_steps=3)
print(circuit.top_nodes(5))
```

---

## `faithfulness`

```python
faithfulness(
    model: HookedTransformer,
    circuit: Circuit,
    task: str,
    *,
    pillars: Optional[List[str]] = None,
    n_examples: int = 256,
    batch_size: int = 16,
    device: Optional[str] = None,
    **kw,
) -> FaithfulnessReport
```

Score a discovered circuit with the 6-pillar faithfulness framework.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pillars` | all | Subset to run, e.g. `["patching", "ablation"]` |
| `n_examples` | 256 | Evaluation examples |
| `**kw` | — | Forwarded to `run_full_faithfulness` (e.g. `n_stability_runs`) |

```python
report = ck.faithfulness(model, circuit, "ioi", pillars=["patching", "ablation"])
print(report.patching_score)
```

---

## `prune`

```python
prune(
    model: HookedTransformer,
    circuit: Circuit,
    *,
    sparsity: float = 0.3,
    scope: str = "both",
    protect_layers: Optional[List[int]] = None,
    inplace: bool = False,
    dry_run: Optional[bool] = None,
    **kw,
) -> HookedTransformer
```

Structurally mask a model to the circuit by zeroing out-of-circuit attention heads and/or MLP layers. By default this operates on a deep copy and leaves `model` untouched; pass `inplace=True` to mask `model` directly.

| Parameter | Default | Description |
|-----------|---------|--------------|
| `sparsity` | `0.3` | Target fraction of nodes to mask |
| `scope` | `"both"` | `"heads"`, `"mlp"`, or `"both"` |
| `protect_layers` | `None` | Layer indices to never mask |
| `inplace` | `False` | Mask `model` in place instead of a copy |
| `dry_run` | `None` | Deprecated alias for `not inplace`; `inplace` wins if both are set |

```python
pruned = ck.prune(model, circuit, sparsity=0.3, scope="both")
```

---

## `quantize`

```python
quantize(
    model: Any,
    circuit: Circuit,
    *,
    n_layers: Optional[int] = None,
    high_fraction: float = 0.3,
    protect_layers: Optional[List[int]] = None,
    backend: str = "quanto",
    bits: int = 3,
    tokenizer: Any = None,
    **kw,
) -> Dict[str, Any]
```

Circuit-aware mixed-precision quantization. `model` is a HuggingFace `AutoModelForCausalLM` (not a `HookedTransformer`) and is modified in place. Layers important to the circuit are kept at high precision; the rest are quantized more aggressively.

| Parameter | Default | Description |
|-----------|---------|--------------|
| `n_layers` | auto | Inferred from `model.config` if omitted |
| `high_fraction` | `0.3` | Fraction of layers kept at high precision |
| `protect_layers` | `None` | Specific layer indices to leave at native precision |
| `backend` | `"quanto"` | `"quanto"` (integer 2/4/8-bit tiers) or `"llmcompressor"` (GPTQ, vLLM-compatible) |
| `bits` | `3` | Bit-width for the `"llmcompressor"` backend |
| `tokenizer` | `None` | HF tokenizer for GPTQ calibration (`"llmcompressor"` only) |

Returns a tier-assignment plan (`"quanto"`) or a summary dict (`"llmcompressor"`).

```python
plan = ck.quantize(model, circuit)
```

---

## `export_checkpoint`

```python
export_checkpoint(
    model: Any,
    artifact: Union[Circuit, List[str], Dict[str, Any], None],
    path: str,
    *,
    intervention: str = "pruning",
    overwrite: bool = True,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    hub_private: bool = True,
    **kw,
) -> str
```

Write an intervened model (pruned or quantized) as a reloadable HuggingFace checkpoint. The checkpoint can be passed directly to `lm-eval` for benchmarking. Returns the checkpoint directory path.

| Parameter | Default | Description |
|-----------|---------|--------------|
| `model` | — | A `HookedTransformer` for `intervention="pruning"`, or an already-quantized HF `AutoModelForCausalLM` for `intervention="quantization"` |
| `artifact` | — | Pruning artifact (`Circuit`, node-name list, or neuron dict). Ignored for quantization — pass `None` |
| `path` | — | Destination directory for the HF checkpoint |
| `intervention` | `"pruning"` | `"pruning"` or `"quantization"` |
| `overwrite` | `True` | Overwrite `path` if it already exists |
| `push_to_hub` | `False` | Upload the checkpoint to the HuggingFace Hub after writing it |
| `hub_repo` | `None` | Target Hub repo id (`"org/name"`); required when `push_to_hub=True` |
| `hub_private` | `True` | Create the Hub repo as private |

```python
ck.export_checkpoint(pruned, circuit, "./checkpoints/pruned")
```

---

## `benchmark`

```python
benchmark(
    checkpoint_path: str,
    tasks: Union[str, List[str]],
    *,
    backend: str = "hf",
    limit: Optional[int] = None,
    fewshot: int = 0,
    device: Optional[str] = None,
    dtype: str = "float32",
    **kw,
) -> Dict[str, Dict[str, float]]
```

Run lm-evaluation-harness on a checkpoint. Returns `{task: {metric: value}}`.

| Parameter | Default | Description |
|-----------|---------|--------------|
| `backend` | `"hf"` | `"hf"` (HFLM) or `"vllm"` if installed |
| `limit` | `None` | Cap examples per task — useful for smoke tests |
| `fewshot` | `0` | Few-shot example count |
| `device` | auto | Torch device for the `"hf"` backend |
| `dtype` | `"float32"` | Model dtype string for the `"hf"` backend |

```python
scores = ck.benchmark("./checkpoints/pruned", tasks=["boolq", "winogrande"], limit=100)
print(scores["boolq"])
```

---

## `load_scores`

```python
load_scores(
    path: str,
    *,
    scores_path: Optional[str] = None,
) -> Circuit
```

Load a previously saved circuit artifact from disk. `scores_path` optionally points to an explicit scores side-car; it's auto-derived from `path` when omitted. Returns a `Circuit` object with `.nodes`, `.scores`, `.top_nodes()`, etc.

```python
circuit = ck.load_scores("./circuit.pt")
print(circuit.top_nodes(10))
```

---

## `selective_finetune`

```python
selective_finetune(
    circuit: Circuit,
    *,
    model_name: Optional[str] = None,
    top_fraction: float = 0.2,
    scope: str = "both",
    exclude_first_n: int = 0,
    exclude_last_n: int = 0,
    n_layers: Optional[int] = None,
    n_q_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> SelectionResult
```

Selects components (attention heads and/or MLP layers) for circuit-guided selective fine-tuning, based on the circuit's top `top_fraction` by score. This does **not** take a `model` argument — only `circuit`. Architecture parameters (`n_layers`, `n_q_heads`, `n_kv_heads`, `head_dim`) are auto-loaded from the HuggingFace config when `model_name` is given and any are omitted.

| Parameter | Default | Description |
|-----------|---------|--------------|
| `model_name` | `None` | HF model name for auto-loading architecture params; defaults to `circuit.model_name` |
| `top_fraction` | `0.2` | Fraction of components to select (0.0–1.0) |
| `scope` | `"both"` | `"attn"`, `"mlp"`, or `"both"` |
| `exclude_first_n` / `exclude_last_n` | `0` | Exclude the first/last N layers from selection |
| `n_layers`, `n_q_heads`, `n_kv_heads`, `head_dim` | auto | Architecture params; auto-loaded via `model_name` when omitted |

Returns a `SelectionResult` with `.attn` and `.mlp` dicts mapping component keys to index lists.

!!! note
    `selective_finetune` is NOT in `applications.__all__`. Access it only via `ck.selective_finetune()` or `Pipeline.selective_finetune()`.

```python
result = ck.selective_finetune(circuit, top_fraction=0.2)
print(result.attn.keys())
```

---

## `visualize_circuit`

```python
visualize_circuit(
    circuit: Circuit,
    *,
    mode: str = "graph",
    output: Optional[str] = None,
    second_circuit: Optional[Circuit] = None,
    **kw,
) -> Any
```

Visualize a circuit. `mode` options: `"graph"`, `"comparison"`, `"dashboard"`. Pass `second_circuit` when `mode="comparison"` to compare two circuits.

```python
ck.visualize_circuit(circuit, mode="graph", output="./circuit.html")
```

---

## `Circuit` Object

`Circuit` is returned by `ck.discover` and `ck.load_scores`.

```python
circuit.nodes          # List[str] — node names
circuit.scores         # Dict[str, float] — {node: importance_score}
circuit.level          # "node" or "neuron"
circuit.task           # task name
circuit.algorithm      # algorithm used
circuit.model_name     # model name
circuit.artifact_path  # path to .pt file

circuit.top_nodes(n)       # top N nodes by score
len(circuit)               # number of nodes
```

---

## Next Steps

- [Pipeline Class](pipeline.md) — object-oriented method chaining
- [Dict-Config API](dict-config.md) — for parameters not in the flat API
- [Circuit Artifacts](../advanced/circuit-artifacts.md) — file format and loading
