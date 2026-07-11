# Case Studies

End-to-end, domain-framed walkthroughs in
[`examples/case-studies/`](https://github.com/Lexsi-Labs/circuitkit/tree/main/examples/case-studies)
— each takes a realistic scenario (compliance, safety, deployment) through the
full **discover → evaluate → intervene** workflow. Unlike the numbered
tutorials (which default to GPT-2 for zero-setup CPU runs), the case studies
default to the model that fits their domain: an instruct model where refusal
behavior matters, a tiny model where the target is edge hardware.

Every script exposes the model as a `MODEL_NAME` constant you can swap; each
notes `"gpt2"` as a fast CPU smoke-test of the pipeline (which validates the
plumbing, not the domain claim).

## Index

| # | Case study | Domain | Default model | Format |
|---|-----------|--------|---------------|--------|
| 14 | [Faithfulness audit for compliance](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/14-faithfulness-audit-for-compliance.py) | Regulated AI / audit reports | `EleutherAI/pythia-410m` | script |
| 15 | [Compression for deployment](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/15-compression-for-deployment.py) | Enterprise MLOps | `Qwen/Qwen2.5-0.5B-Instruct` | script |
| 16 | [Tabular model audit](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/16-tabular-model-audit.py) | Tabular foundation models (Orion-MSP) | `gpt2` (stand-in) | script |
| 17 | [Quantization unlearning](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/17-quantization-unlearning.py) | Permanent knowledge removal | `Qwen/Qwen2.5-1.5B-Instruct` | script |
| 18 | [Banking safety steering](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/18-banking-safety-steering.py) | Chatbot safety at inference time | `meta-llama/Llama-3.2-1B-Instruct` | script |
| 19 | [Trade finance document classification](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/19-trade-finance-document-classification.py) | CPU-only on-prem deployment (Fintra) | `google/gemma-3-1b-it` | script |
| 20 | [Transit edge deployment](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/20-transit-edge-deployment.py) | ARM edge hardware (AFC gates) | `Qwen/Qwen2.5-0.5B-Instruct` | script |
| 21 | [Quantization-permanent unlearning](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/21-quantization-permanent-unlearning.ipynb) | Unlearning that survives fine-tuning | `Qwen/Qwen2.5-1.5B-Instruct` | notebook |
| 22 | [Gender bias audit & mitigation](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/22-gender-bias-audit-and-mitigation.ipynb) | Responsible AI loop | `Qwen/Qwen2.5-1.5B-Instruct` | notebook |
| 23 | [Jailbreak safety steering](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/23-jailbreak-safety-steering.ipynb) | Jailbreak defense | `Qwen/Qwen2.5-1.5B-Instruct` | notebook |
| 24 | [Safety refusal on custom data](https://github.com/Lexsi-Labs/circuitkit/blob/main/examples/case-studies/24-safety-refusal-custom-data.py) | Custom contrastive pairs (safety) | `gpt2` | script |

## Which one should I read?

- **"Prove the model is trustworthy"** → 14 (compliance audit) or 22 (bias audit & mitigation)
- **"Make it smaller / cheaper"** → 15 (compression), 19 (CPU-only), 20 (edge)
- **"Make it forget something — permanently"** → 17 / 21 (quantization-permanent unlearning)
- **"Make it safer at inference time, no retraining"** → 18 (banking) or 23 (jailbreak steering)
- **"Use my own dataset"** → 24 (explicit contrastive pairs), plus [Bring Your Own Data](../user-guide/custom-data.md)

## Hardware & access notes

- Notebooks 21–23 and the instruct-model scripts want a GPU or Apple-Silicon MPS.
- Llama and Gemma models are **gated** on Hugging Face — accept the license
  first, or swap `MODEL_NAME` for an open model (Qwen 2.5, Pythia, gpt2).
- Studies that prune/quantize/export require a **registered architecture**
  (gpt2, Llama, Qwen, Gemma, Mistral, Phi); Pythia is discovery/eval-only —
  see the [Architecture Registry](../advanced/architecture-registry.md).

## Next steps

- [Python Scripts](scripts.md) — the numbered tutorial scripts (01–13)
- [Notebooks](notebooks.md) — the Colab tutorial notebooks (00–08)
- [Applications guide](../user-guide/applications.md) — the intervention APIs the case studies use
