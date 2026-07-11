# Python Scripts

All scripts in `examples/` run on CPU with GPT-2 and are verified end-to-end. They are the recommended starting point for integration, automation, and CI workflows.

---

## `01-quickstart.py` — 5-Minute Onramp

The fastest path to a discovered circuit via the flat `ck.*` API.

```bash
python examples/01-quickstart.py
```

```python
import circuitkit as ck

model   = ck.load_model("gpt2", dtype="float32")
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=16)
print(circuit.top_nodes(5))
```

---

## `02-discover-python-api.py` — Python API Discovery

Full discovery workflow using `discover_circuit` (dict-config API) with all key parameters explained.

```bash
python examples/02-discover-python-api.py
```

Covers: model loading, algorithm selection, output paths, artifact inspection.

---

## `03-discover-cli.sh` — CLI Discovery

Runs the `discover` CLI command end to end (and echoes the related `discover-smart` / `evaluate` commands as next steps).

```bash
bash examples/03-discover-cli.sh
```

Covers: `--model`, `--algorithm`, `--task`, `--level`, `--scope`, `--sparsity`, `--batch-size`, `--num-examples`, `--ig-steps`, `--output`.

---

## `04-discover-yaml.py` — YAML Task Config

Discovery from a custom YAML task config file.

```bash
python examples/04-discover-yaml.py
```

Covers: YAML task schema (source, schema, corruption, metric), `discover-yaml` CLI and Python equivalent.

---

## `05-evaluate-faithfulness.py` — 6-Pillar Evaluation

Faithfulness evaluation for a discovered circuit.

```bash
python examples/05-evaluate-faithfulness.py
```

Covers: `evaluate_circuit`, pillar subset selection, `FaithfulnessReport` interpretation.

---

## `06-applications.py` — Pruning, Quantization, Finetuning

All three application operations on the discovered circuit.

```bash
python examples/06-applications.py
```

Covers:
- `ck.prune(model, circuit, sparsity=0.3, scope="heads")` — structural pruning
- `ck.quantize(model, circuit)` — circuit-aware quantization
- `ck.selective_finetune(circuit, top_fraction=0.2)` — select the top attention heads and MLP layers for finetuning
- `ck.export_checkpoint(...)` — writing a HuggingFace checkpoint

---

## `07-pipeline.py` — Pipeline Class

Stateful multi-step workflow using `Pipeline`.

```bash
python examples/07-pipeline.py
```

Covers: `Pipeline(model_name)`, method chaining, `from_artifact`, state inspection, `summary()`.

---

## `08-custom-data.py` — Bringing Your Own Dataset

Custom data via all three paths: CSV, HuggingFace dataset, full custom TaskSpec.

```bash
python examples/08-custom-data.py
```

Covers:
- CSV with YAML task config
- `MCQAdapter` + `MCQChoiceSwap` + `NormalizedTaskSpec`
- `validate_token_alignment` audit
- Registering and using a custom task

---

## `09-load-and-reuse.py` — Load and Reuse Artifacts

Reload a saved circuit artifact and continue the workflow without re-running discovery.

```bash
python examples/09-load-and-reuse.py
```

Covers:
- `ck.load_scores("./circuit.pt")` — load Circuit object
- `Pipeline.from_artifact(...)` — continue from artifact
- `Pipeline.from_scores(...)` — required for `selective_finetune`
- Normalizing scores for cross-method comparison

---

## Further Scripts

Beyond the 9 above, `examples/` also has:

- `10-steering.py` — activation steering (see [Steering Guide](../guides/steering.md))
- `11-knowledge-editing.py` — ROME/MEMIT knowledge editing
- `12-custom-corruption.py` — writing a custom `CorruptionStrategy`
- `13-transfer-matrix.py` — cross-task transfer matrix via `Pipeline.evaluate_advanced(mode="transfer")`

## Sub-directory Examples

`examples/discovery/`, `examples/visualization/`, `examples/pruning/`, `examples/quantization/`, and `examples/finetuning/` hold feature-specific and architecture-specific examples for advanced use cases; see `examples/README.md` in the repository for the full list. The domain-framed end-to-end walkthroughs in `examples/case-studies/` have [their own page](case-studies.md).

---

## Next Steps

- [Notebooks](notebooks.md) — interactive Colab versions of the same workflows
- [Case Studies](case-studies.md) — domain-framed end-to-end walkthroughs (14–24)
- [Getting Started: Quick Start](../getting-started/quickstart.md) — guided first circuit
