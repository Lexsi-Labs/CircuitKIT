# Trust & Audit

CircuitKit ships **13 discovery algorithms** across 4 backends with explicit stability tiers, and a 6-pillar faithfulness evaluation framework. Shipping is not validation: only 2 (`eap`, `eap-ig`) are validated at production scale, `acdc` and `ibcircuit` are experimental (GPT-2 scale), and the other 9 are research (GPT-2 IOI only). This section documents what has been validated, what is experimental, and how the algorithms were audited.

## Where to look

| | |
|---|---|
| **[Scope & Limitations](scope.md)** | Stability tier definitions, algorithm maturity, known limitations |
| **[Audit Results](results.md)** | What the algorithms actually did on real models — empirical results |

## How CircuitKit labels algorithms

Every algorithm has a stability tier, displayed next to its name throughout these docs:

| Tier | Badge | Meaning | Count |
|---|---|---|---|
| **Stable** |  | Validated across model families (GPT-2 through 3B+). Used in the audit paper. | 2 |
| **Experimental** |  | Works on GPT-2 scale; may fail on larger models, GQA, or instruction-tuned architectures. | 2 |
| **Research** |  | GPT-2 IOI only. For algorithm comparison studies or paper replication. Not validated on modern architectures. | 9 |

## The honest finding

The CircuitKit audit paper ("Faithfulness Is Not Actionability", EMNLP Findings 2026) evaluated the core discovery algorithms across 6 faithfulness pillars and found:

- **Stable-tier algorithms** (EAP family) produce circuits with high faithfulness (≥0.85 ablation_score) across GPT-2, Llama-3.2-3B, and Gemma-2-2B.
- **Experimental algorithms** (ACDC, IBCircuit) produce smaller circuits but with lower faithfulness — they are valid for research but not yet production-ready. Both were only validated at GPT-2 scale, and IBCircuit OOMs above ~3B parameters on a single GPU.
- **Research algorithms** are unvalidated beyond GPT-2 IOI. Do not cite findings from them without independent verification.

See [Audit Results](results.md) for per-algorithm scores.
