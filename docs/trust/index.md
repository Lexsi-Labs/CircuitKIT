# Trust & Audit

CircuitKit ships **13 discovery algorithms** across 4 backends with explicit stability tiers, and a 6-pillar faithfulness evaluation framework. Shipping is not validation: 6 (`eap`, `eap-ig`, `eap-gp`, `acdc`, `ibcircuit`, `cdt`) are stable tier and have been tested across the GPT-2, Llama, Gemma, and Qwen families, and the other 7 are research tier, implemented and validated on GPT-2/IOI but not yet exercised at scale or across architectures. This section documents what has been validated, which caveats apply, and how the algorithms were audited.

## Where to look

| | |
|---|---|
| **[Scope & Limitations](scope.md)** | Stability tier definitions, algorithm maturity, known limitations |
| **[Audit Results](results.md)** | What the algorithms actually did on real models — empirical results |

## How CircuitKit labels algorithms

Every algorithm has a stability tier, displayed next to its name throughout these docs:

| Tier | Badge | Meaning | Count |
|---|---|---|---|
| **Stable** |  | Tested across the GPT-2, Llama, Gemma, and Qwen families. `acdc`, `ibcircuit`, and `cdt` carry documented caveats (see [Scope & Limitations](scope.md)). | 6 |
| **Experimental** |  | Works on GPT-2 scale; may fail on larger models, GQA, or instruction-tuned architectures. None currently for discovery. | 0 |
| **Research** |  | Implemented and validated on GPT-2/IOI but not yet exercised at scale or across architectures. For algorithm comparison studies or paper replication. | 7 |

## The honest finding

The CircuitKit audit paper ("Faithfulness Is Not Actionability", EMNLP Findings 2026) evaluated the core discovery algorithms across 6 faithfulness pillars and found:

- **`eap` and `eap-ig`** produce circuits with high faithfulness (≥0.85 ablation_score for `eap-ig`) across GPT-2, Llama-3.2-3B, and Gemma-2-2B.
- **ACDC and IBCircuit** produce smaller circuits but with lower faithfulness. The audit scored them on GPT-2 IOI only. ACDC is slow, and IBCircuit has a memory ceiling on multi-billion-parameter models (it can OOM above ~3B parameters on a single GPU at aggressive settings).
- **Research algorithms** are unvalidated beyond GPT-2 IOI. Do not cite findings from them without independent verification.

See [Audit Results](results.md) for per-algorithm scores.
