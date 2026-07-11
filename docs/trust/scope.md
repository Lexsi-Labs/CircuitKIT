# Scope & Limitations

CircuitKit is a **unified discover → evaluate → intervene toolkit** for mechanistic interpretability. This page documents what the library does, what it does not do, and where the algorithms are known to have limitations.

## Stability tiers — the single source of truth

Every algorithm in CircuitKit is labelled with a stability tier. The definitive list lives in `circuitkit/backends/__init__.py`; this page mirrors it.

| Algorithm | Backend | Tier | Scope | GQA? | Chat template? | >3B? |
|---|---|---|---|---|---|---|
| `eap-ig` | EAP |  Stable | All models | ✅ | ✅ | ✅ |
| `eap` | EAP |  Stable | All models | ✅ | ✅ | ✅ |
| `acdc` | ACDC |  Experimental | GPT-2 scale | ⚠️ | ⚠️ | ❌ |
| `ibcircuit` | IBCircuit |  Experimental | ≤3B | ❌ | ❌ | ❌ |
| `eap-ig-activations` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `eap-clean-corrupted` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `eap-exact` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `atp-gd` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `eap-gp` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `relp` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `peap` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `eap-ifr` | EAP |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |
| `cdt` | CD-T |  Research | GPT-2 IOI | ❌ | ❌ | ❌ |

✅ Validated  ⚠️ Experimental / may fail  ❌ Not validated

### Tier definitions

| Tier | Meaning | You should |
|---|---|---|
|  **Stable** | Validated across model families (GPT-2, Llama-3, Gemma, Qwen) at multiple scales (124M–4B). Works with GQA, RoPE, SwiGLU, and chat templates. | Use in production experiments. Cite in papers. |
|  **Experimental** | Implemented and runs without errors on GPT-2 scale. May produce wrong results on larger or instruction-tuned models. GQA/SwiGLU handling is not validated. | Use for exploratory work. Do not cite as primary evidence. |
|  **Research** | Implemented and matches the paper description. Only validated on GPT-2 IOI (the paper's evaluation setting). No GQA, chat-template, or scaling validation. | Use only for algorithm comparison or paper replication. |

## Algorithm selection

Start with `eap-ig` for any new experiment. It is the default, the most validated, and the algorithm used in the CircuitKit audit paper. Use `eap` if discovery speed is the bottleneck.

## Known limitations

### Models

- **GPT-2 (124M–1.5B):** fully validated, CPU-friendly. All 13 algorithms run on GPT-2.
- **Llama 3.x (1B–3B):** Stable-tier EAP validated. Experimental and Research algorithms are not validated on Llama-3.
- **Gemma 2/3 (2B–4B):** Stable-tier EAP validated on Gemma-2-2B. GQA is detected at runtime (when `n_kv != n_heads`), not separately validated per model.
- **Qwen 2.5 (0.5B–7B):** Stable-tier EAP validated. Chat-template auto-detection works.
- **Larger models (>7B):** Not systematically validated. Stable EAP should work but may require GPU with ≥24 GB VRAM.

### Tasks

- **16 built-in tasks** are registered and tested. The loader-supported metrics (`logit_diff`, `kl`, `accuracy`) cover classification, cloze, and generative evaluations.
- **Custom tasks** via the YAML adapter support CSV, JSONL, and HuggingFace datasets. The auto-schema detection is heuristic — validate your schema with `circuitkit validate-config`.

### Interventions

- **Structural pruning** writes standard HuggingFace checkpoints. Confirmed reloadable with `transformers.AutoModelForCausalLM.from_pretrained`.
- **Quantization** has two backends: `optimum-quanto` (fixed `qint2`/`qint4`/`qint8` qtype tiers, weights-only linear quantization) and `llmcompressor` (true low-bit GPTQ with `num_bits` in {3, 4, 8}, including a genuine 3-bit type unavailable in optimum-quanto).
- **Knowledge editing** (ROME/MEMIT) updates MLP weights at circuit-identified layers. For ROME, the context token is auto-inferred from the subject string — verify the inferred token matches the target.
- **Activation steering** is hook-based and reversible. Not serialisable — the hooks must be re-installed on model reload.

### Framework

- **Pillar 6 (Generalization)** is implemented but has not been validated at scale. Treat its scores as preliminary. If `target_task` is not supplied, Pillar 6 is skipped automatically.

## What CircuitKit does not do

- **Training from scratch** — CircuitKit discovers circuits in pretrained models. It does not train new models.
- **Prompt engineering** — CircuitKit expects a task with a defined metric. It does not optimise prompts.
- **Safety evaluation** — CircuitKit measures faithfulness and downstream utility. It does not measure model safety or bias. For safety tools, see [SafeTune](https://github.com/Lexsi-Labs/SafeTune).
- **Distributed training** — CircuitKit runs on single-GPU or CPU. Multi-GPU discovery is not supported.
- **Auto-selection of algorithm** — The user must choose an algorithm. CircuitKit does not grid-search algorithms.

## Reproducibility

The audit paper's results were produced with a fixed environment (NVIDIA NeMo 25.09, CUDA 13, torch 2.9). The library itself targets torch ≥2.0, CUDA 12.6, and CPU. Numbers may differ slightly across environments. See `ENVIRONMENT.md` in the repository root for the exact pin.
