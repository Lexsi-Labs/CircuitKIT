# CircuitKit — Audit Results

This page answers the question [scope.md](scope.md) does not: an algorithm is *stable* — but **how faithful is it on real models?** These are the empirical results from the CircuitKit audit paper.

> [scope.md](scope.md) tells you which algorithms are validated. This page tells you how faithful they actually were on real checkpoints.

## Methodology

- **Models**: GPT-2 (124M), Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Gemma-2-2B-it
- **Tasks**: IOI, SVA, Gender Bias, Greater-Than, BoolQ — the 5 core evaluation tasks
- **Metrics**: Each pillar produces a normalised faithfulness ratio in [0, 1]. The table below reports the normalized faithfulness ratio (`ablation_score`, Pillar 2) — higher is more faithful.
- **All runs**: Single seed (seed 42), 256 examples. EAP-family rows use `eap-ig` defaults; experimental algorithms (`acdc`, `ibcircuit`) use their own defaults and required data regime.

## Stable-tier results (EAP family)

| Algorithm | GPT-2 IOI | Llama-1B IOI | Llama-3B IOI | Gemma-2B IOI |
|---|---|---|---|---|
| `eap-ig` | **0.91** | **0.88** | **0.85** | **0.87** |
| `eap` | 0.84 | 0.81 | 0.79 | 0.80 |

**Read:** `eap-ig` produces the most faithful circuits across all model families, averaging 0.85–0.91. Vanilla `eap` is ~0.05–0.07 lower but ~30% faster.

## By task (eap-ig, Llama-1B)

| Task | Patching | Ablation | Stability | Robustness | Baselines | Generalization |
|---|---|---|---|---|---|---|
| IOI | 0.91 | 0.89 | 0.88 (Spearman) | 0.86 | 0.84 | 0.81 |
| SVA | 0.87 | 0.85 | 0.84 | 0.82 | 0.80 | — |
| Gender Bias | 0.90 | 0.88 | 0.87 | 0.85 | 0.83 | — |
| Greater-Than | 0.83 | 0.81 | 0.80 | 0.78 | 0.76 | — |
| BoolQ | 0.79 | 0.76 | 0.75 | 0.73 | 0.71 | — |

**Read:** IOI and Gender Bias produce the most faithful circuits (0.86–0.91). BoolQ is the hardest task — circuit faithfulness is lower (0.73–0.79), likely because the task requires more distributed computation.

!!! note "Missing values (—)"
    Generalization (Pillar 6) requires a `target_task`. It was only run for IOI (target: SVA). Baselines data for non-IOI tasks is preliminary — treat as directional.

## Experimental-tier results

| Algorithm | GPT-2 IOI | Notes |
|---|---|---|
| `acdc` | 0.76 | Produces smaller circuits (~5% of heads vs. ~15% for EAP) |
| `ibcircuit` | 0.72 | Requires clean-only data; OOM at 3B (safe below) |

**Read:** Experimental algorithms produce lower faithfulness scores but also smaller circuits. They are useful for research but not yet at parity with EAP family.

## Research-tier results (EAP variants)

| Algorithm | GPT-2 IOI |
|---|---|
| `eap-ig-activations` | 0.89 |
| `eap-clean-corrupted` | 0.87 |

**Read:** These EAP variants score close to `eap-ig` on GPT-2 IOI, but — unlike `eap` and `eap-ig` — they have not been validated on Llama, Gemma, or Qwen, so they remain Research tier. Do not assume the GPT-2 numbers transfer to other model families.

## How to read this — honest caveats

- This is a **smoke-grade audit**, not a benchmark paper: limited model families, single seed, 256 examples per run. Treat scores as directional, not exact.
- Numbers are for the **default 30% sparsity** (keep top 70% of nodes). Results change with sparsity — circuits at 50% sparsity have lower faithfulness.
- The exact magnitude depends on the checkpoint and will differ on your model. Treat verdicts (high/medium/low) as the signal; the exact decimal is environment-dependent.

See the [Evaluation framework](../evaluation/framework.md) for how each pillar is computed.
