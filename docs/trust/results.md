# CircuitKIT — Audit Results

This page answers the question [scope.md](scope.md) does not: an algorithm is *stable* — but **how faithful is it on real models?** These are the empirical results from the CircuitKIT audit paper.

> [scope.md](scope.md) tells you which algorithms are validated. This page tells you how faithful they actually were on real checkpoints.

!!! warning "Which numbers on this page are current"
    Two corrections changed every `ioi` number from the EAP family: the task now reads its metric
    at the token that predicts the answer, and padding is masked on the side that was padded
    (see [where the prompts end](../evaluation/causal-patching.md#where-the-ioi-and-double_io-prompts-end)).

    - **Re-measured on 2026-10-11:** the GPT-2 IOI rows for `eap-ig`, `eap`, `eap-ig-activations`
      and `eap-clean-corrupted`. They are in the tables marked *re-measured*, with the command.
    - **Measured before the corrections and not re-measured:** every Llama and Gemma IOI number and
      the BoolQ row. The models are gated and were not available to the re-measurement run. These
      cells are marked *(before)*. Do not compare them with current runs.
    - **Not affected:** the `acdc` and `ibcircuit` rows, which build their own batches and already
      read the predicting position.

## Methodology

- **Models**: GPT-2 (124M), Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Gemma-2-2B-it
- **Tasks**: IOI, SVA, Gender Bias, Greater-Than, BoolQ — the 5 core evaluation tasks
- **Metrics**: Each pillar produces a faithfulness ratio clipped to [0, 1]. The re-measured tables report the patching score (Pillar 1), which is [the measure for comparing circuits](../evaluation/causal-patching.md#which-measure-to-compare-circuits-with), with the zero-ablation ratio (Pillar 2) next to it. The tables marked *(before)* report `ablation_score` as the audit recorded it.
- **All runs**: Single seed (seed 42), 256 examples. EAP-family rows use `eap-ig` defaults; `acdc` and `ibcircuit` use their own defaults and required data regime.

## Stable-tier results (EAP family)

### GPT-2 IOI, re-measured

| Algorithm | Patching score | Zero-ablation ratio (clipped score) | `ablation_score` before the corrections |
|---|:---:|:---:|:---:|
| `eap-ig` | **0.945** | -0.276 (0.0) | 0.91 |
| `eap` | **0.971** | 0.113 (0.113) | 0.84 |

Measured 2026-10-11 on one Tesla T4 and reproduced on 2026-11-10 with the same command on CPU and on one RTX 6000 Ada (`eap-ig` 0.9446 and -0.2763, `eap` 0.9710 and 0.1132 on both). Model `gpt2`, task `ioi`, seed 42, 256 discovery examples and
256 evaluation examples, node level, 30% pruned (46 of 156 components removed), `float32`:

```python
model = ck.load_model("gpt2", dtype="float32")
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=256, sparsity=0.3, seed=42)
report = ck.faithfulness(model, circuit, "ioi", n_examples=256, seed=42)
report.patching_score, report.ablation_raw_ratio
```

**Read:** both algorithms recover about 95% of the clean-minus-corrupt logit difference under
patching (clean 4.29, corrupt -0.10). The old `ablation_score` values of 0.84 to 0.91 do not
survive: with zero ablation the same circuits score between -0.28 and 0.11. Run time was not
compared in this run.

### Llama and Gemma IOI (before, not re-measured)

| Algorithm | Llama-1B IOI *(before)* | Llama-3B IOI *(before)* | Gemma-2B IOI *(before)* |
|---|---|---|---|
| `eap-ig` | 0.88 | 0.85 | 0.87 |
| `eap` | 0.81 | 0.79 | 0.80 |

These are `ablation_score` values measured before the corrections. Given what happened to the GPT-2
column, expect them to change.

## By task (eap-ig, Llama-1B)

The IOI and BoolQ rows were measured before the corrections and not re-measured (`ioi`: scored
position; `boolq`: pads left, so it attended to padding on a right-padding tokenizer). The SVA,
Gender Bias and Greater-Than rows are not touched by the position correction; the padding
correction reaches them only on a tokenizer that pads left.

| Task | Patching | Ablation | Stability | Robustness | Baselines | Generalization |
|---|---|---|---|---|---|---|
| IOI *(before)* | 0.91 | 0.89 | 0.88 (Spearman) | 0.86 | 0.84 | 0.81 |
| SVA | 0.87 | 0.85 | 0.84 | 0.82 | 0.80 | — |
| Gender Bias | 0.90 | 0.88 | 0.87 | 0.85 | 0.83 | — |
| Greater-Than | 0.83 | 0.81 | 0.80 | 0.78 | 0.76 | — |
| BoolQ *(before)* | 0.79 | 0.76 | 0.75 | 0.73 | 0.71 | — |

**Read:** as recorded by the audit, IOI and Gender Bias produced the most faithful circuits (0.86–0.91) and BoolQ the least (0.73–0.79). The IOI and BoolQ ends of that comparison rest on numbers that have not been re-measured.

!!! note "Missing values (—)"
    Generalization (Pillar 6) requires a `target_task`. It was only run for IOI (target: SVA). Baselines data for non-IOI tasks is preliminary — treat as directional.

## Stable-tier results (ACDC, IBCircuit)

| Algorithm | GPT-2 IOI | Notes |
|---|---|---|
| `acdc` | 0.76 | Produces smaller circuits (~5% of heads vs. ~15% for EAP) |
| `ibcircuit` | 0.72 | Requires clean-only data; OOM at 3B (safe below) |

**Read:** ACDC and IBCircuit produce smaller circuits. These two numbers are `ablation_score` values and are not on the same measure as the re-measured EAP rows above, so the two tables cannot be ranked against each other. The audit scored them on GPT-2 IOI only.

## Research-tier results (EAP variants)

GPT-2 IOI, re-measured 2026-10-11 with the command above (`algorithm=` changed; seed 42, 256 examples, 30% pruned, Tesla T4):

| Algorithm | Patching score | Zero-ablation ratio (clipped score) | `ablation_score` before the corrections |
|---|:---:|:---:|:---:|
| `eap-ig-activations` | 0.971 | 0.006 (0.006) | 0.89 |
| `eap-clean-corrupted` | 0.977 | -0.029 (0.0) | 0.87 |

**Read:** These EAP variants score close to `eap-ig` on GPT-2 IOI under patching, but — unlike `eap` and `eap-ig` — they have not been validated on Llama, Gemma, or Qwen, so they remain Research tier. Do not assume the GPT-2 numbers transfer to other model families.

## How to read this — honest caveats

- This is a **smoke-grade audit**, not a benchmark paper: limited model families, single seed, 256 examples per run. Treat scores as directional, not exact.
- Numbers are for the **default 30% sparsity** (keep top 70% of nodes). Results change with sparsity — circuits at 50% sparsity have lower faithfulness.
- The exact magnitude depends on the checkpoint and will differ on your model. Treat verdicts (high/medium/low) as the signal; the exact decimal is environment-dependent.

See the [Evaluation framework](../evaluation/framework.md) for how each pillar is computed.
