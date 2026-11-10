# Capability Matrix

Algorithm × model scale × expected outcome with rough runtime and memory estimates on a single A100 40GB.

## Algorithm × model scale

| Algorithm | GPT-2 (124M) | Pythia-1B | Llama-1B | Llama-3B | Gemma-4B | 7B+ |
|---|---|---|---|---|---|---|
| `eap-ig`  | ✅ ~2min | ✅ ~5min | ✅ ~8min | ✅ ~15min | ✅ ~20min | ⚠️ may OOM |
| `eap`  | ✅ ~1.5min | ✅ ~3min | ✅ ~5min | ✅ ~10min | ✅ ~14min | ⚠️ may OOM |
| `eap-ig-activations` (Research)  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `eap-clean-corrupted` (Research)  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `acdc`  | ✅ ~15-30min | ⚠️ slow | ⚠️ slow | ❌ likely OOM | ❌ | ❌ |
| `ibcircuit`  | ✅ ~20min | ⚠️ | ⚠️ | ❌ OOM | ❌ OOM | ❌ OOM |
| `cdt`  | ✅ ~5min | ❌ | ❌ | ❌ | ❌ | ❌ |
| Research  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

✅ Validated  ⚠️ May fail or be slow  ❌ Known failure

All runtimes: `n_examples=128`, `batch_size=4`, `level="node"`, A100 40GB.

## Memory requirements

| Model | Base VRAM | EAP-IG peak | ACDC peak | IBCircuit peak |
|---|---|---|---|---|
| GPT-2 (124M) | ~1 GB | ~2 GB | ~3 GB | ~2 GB |
| Llama-1B | ~2.5 GB | ~4 GB | — | ~5 GB |
| Llama-3B | ~7 GB | ~12 GB | — | ❌ OOM |
| Gemma-4B | ~10 GB | ~16 GB | — | ❌ OOM |
| Llama-7B | ~18 GB | ~30 GB | — | ❌ OOM |

Measured at `bfloat16`, `ig_steps=5` (the previous default), `batch_size=4`. Reduce `batch_size` if you hit OOM; `ig_steps` now defaults to 3.

## Task compatibility

| Task type | EAP family | ACDC | IBCircuit | CD-T |
|---|---|---|---|---|
| Paired (clean + corrupted) | ✅ | ✅ | ✅ | ✅ |
| Clean only (no corruption) | ❌ | ❌ | ✅ | ✅ |
| Multiple-choice (MCQ) | ✅ | ✅ | ⚠️ | ✅ |
| Chat-templated | ✅ | ✅ | ✅ | ❌ |

## Validated combinations (from the audit paper)

The GPT-2 IOI `eap-ig` row was re-measured on 2026-10-11 (Tesla T4, seed 42, 256 examples, 30% pruned; command on the [Audit Results](../trust/results.md#gpt-2-ioi-re-measured) page) and reports the patching score, the [measure for comparing circuits](../evaluation/causal-patching.md#which-measure-to-compare-circuits-with). The Llama and Gemma IOI rows were measured before the `ioi` scored position was corrected and not re-measured (gated models). The `acdc` row is not affected.

| Model | Task | Algorithm | Result |
|---|---|---|---|
| GPT-2 | IOI | eap-ig | patching_score 0.945 (zero-ablation ratio -0.276; was ablation_score 0.91 before the correction) |
| Llama-1B | IOI | eap-ig | ablation_score 0.88 *(before the correction, not re-measured)* |
| Llama-3B | IOI | eap-ig | ablation_score 0.85 *(before the correction, not re-measured)* |
| Gemma-2B | IOI | eap-ig | ablation_score 0.87 *(before the correction, not re-measured)* |
| Llama-1B | SVA | eap-ig | ablation_score 0.87 |
| Llama-1B | Gender Bias | eap-ig | ablation_score 0.90 |
| GPT-2 | IOI | acdc | ablation_score 0.76 |

## Next steps

- [:octicons-arrow-right-24: Memory Optimization](../advanced/memory-optimization.md)
- [:octicons-arrow-right-24: Stability Tiers](stability-tiers.md)
