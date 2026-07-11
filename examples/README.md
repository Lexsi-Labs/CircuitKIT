# CircuitKit Examples

Runnable, self-contained examples for the full CircuitKit workflow:
**discover → evaluate → intervene**. Every numbered script runs end-to-end on
**GPT-2 on CPU** in a few minutes — no GPU and no external data required.

> **GPT-2 is the fast default, not the only option.** The numbered tutorials use
> GPT-2 so they run anywhere with zero setup, but every script takes a model id you
> can swap. For real models see [`pruning/prune-llama.py`](pruning/prune-llama.py)
> and [`pruning/prune-qwen.py`](pruning/prune-qwen.py), the
> [case studies](case-studies/) (Qwen 2.5 / 3, Llama 3, Gemma 2 / 3, Pythia), and
> the [notebooks](notebooks/) (Gemma 2, Qwen 2.5, Llama 3.2). Instruct-tuned models
> are required for the safety/steering work — GPT-2 has no refusal behavior.

## Quick start

```bash
pip install -e ".[benchmarks]"   # from repo root
python examples/01-quickstart.py
```

## Tutorial scripts

| Script | What it shows |
|--------|---------------|
| `01-quickstart.py` | The 5-minute onramp: Discover → Evaluate → Prune → Export |
| `02-discover-python-api.py` | Circuit discovery via `discover_circuit({...})` |
| `03-discover-cli.sh` | Same discovery through the `circuitkit` CLI |
| `04-discover-yaml.py` | Discovery driven by a YAML task config |
| `05-evaluate-faithfulness.py` | 6-pillar faithfulness evaluation |
| `06-applications.py` | Pruning, quantization, selective finetuning |
| `07-pipeline.py` | Stateful `Pipeline` class with chaining |
| `08-custom-data.py` | Bring your own dataset (three patterns) |
| `09-load-and-reuse.py` | Load saved circuits, re-apply with different settings |
| `10-steering.py` | Compute steering vectors and steer at inference |
| `11-knowledge-editing.py` | Rewrite facts using ROME/MEMIT at circuit layers |
| `12-custom-corruption.py` | Implement and register a custom corruption strategy |
| `13-transfer-matrix.py` | Cross-task circuit transfer evaluation |

## Case studies (industry use cases)

The `case-studies/` directory contains domain-specific examples targeting
real-world deployment scenarios:

| Script | Domain | What it demonstrates |
|--------|--------|---------------------|
| `case-studies/14-faithfulness-audit-for-compliance.py` | Regulated AI | 6-pillar faithfulness audit for financial compliance |
| `case-studies/15-compression-for-deployment.py` | Enterprise MLOps | Ship compressed models to production with trust |
| `case-studies/16-tabular-model-audit.py` | Orion-MSP | Auditing tabular foundation model decisions |
| `case-studies/17-quantization-unlearning.py` | AI Safety | Permanent knowledge removal via quantization |
| `case-studies/18-banking-safety-steering.py` | Banking | Inference-time chatbot safety (no retraining) |
| `case-studies/19-trade-finance-document-classification.py` | Fintra | CPU-only on-prem document classification |
| `case-studies/20-transit-edge-deployment.py` | Smart Mobility | Edge deployment for AFC/gate hardware |
| `case-studies/21-quantization-permanent-unlearning.ipynb` | AI Safety | Permanent knowledge removal via circuit-guided quantization |
| `case-studies/22-gender-bias-audit-and-mitigation.ipynb` | Responsible AI | Localize, fix, and re-audit gender bias end-to-end |
| `case-studies/23-jailbreak-safety-steering.ipynb` | LLM Safety | Circuit-restricted activation steering for jailbreak defense |

## Application examples

| Script | What it shows |
|--------|---------------|
| `pruning/basic-prune.py` | General structural pruning |
| `pruning/prune-llama.py` | Pruning a Llama model |
| `pruning/prune-qwen.py` | Pruning a Qwen model |
| `quantization/quantize-llama.py` | Circuit-aware quantization on Llama |
| `quantization/quantize-qwen.py` | Circuit-aware quantization on Qwen |
| `finetuning/finetune-llama.py` | Selective finetuning on Llama |
| `finetuning/finetune-qwen.py` | Selective finetuning on Qwen |

## Advanced reference

| Directory | What it contains |
|-----------|-----------------|
| `discovery/` | Multi-algorithm benchmarks and algorithm comparisons |
| `visualization/` | Circuit graph visualization with threshold filtering |

## Data files

- `simple_csv_task.yaml` — example YAML task config for a custom CSV dataset
- `simple_task_sample.csv` — the CSV dataset referenced by YAML configs
