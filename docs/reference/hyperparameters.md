# Hyperparameters

Every tunable parameter has two ranges:

| Range | Meaning | What CircuitKIT does outside it |
|---|---|---|
| **Valid** | Values that can work at all. | Raises `HyperparameterError` (a `ValueError`) naming the parameter and the allowed range. |
| **Sensible** | Values inside the valid range that earlier runs or the literature show to be risky. | Emits a `HyperparameterWarning`. |

Integer parameters (`ig_steps`, `n_examples`, ...) must be integers: `3.0` is rejected, as it
would otherwise fail later inside the backend. A few older argument checks run before these
(for example the flat API's `sparsity` check) and raise a plain `ValueError` with the same
meaning, so `except ValueError` handles every route.

A value outside the *sensible* range is not wrong. It is a prompt to check that you meant it:
for example `n_examples=8` runs, but attribution scores from eight examples are noisy.

## How the checks behave

- **Only what you set is warned about.** Defaults are always inside the sensible range, and a
  default never produces a warning.
- **Checked before any expensive work.** A bad value in a dict/YAML config fails in
  `discover_circuit` / `evaluate_circuit` / `circuitkit validate-config`, not after discovery has run.
  Parameters an algorithm does not read are not checked (`ig_steps` is ignored for `eap`).
  The entry points that do not take a config check their own arguments: `ck.prune`,
  `ck.quantize` and `ck.faithfulness`.
- **Warnings are standard Python warnings.** `HyperparameterWarning` derives from `Warning`, not
  `UserWarning`, because importing `circuitkit.api` ignores all `UserWarning`s. Silence or
  escalate it with the `warnings` module:

  ```python
  import warnings
  from circuitkit.utils.hparams import HyperparameterWarning

  warnings.filterwarnings("ignore", category=HyperparameterWarning)   # silence
  warnings.filterwarnings("error", category=HyperparameterWarning)    # escalate
  ```

- **Strict mode** turns every sensible-range warning into an error, which suits CI and
  reproduction runs. Enable it in a dict/YAML config or with an environment variable:

  ```yaml
  validation:
    strict: true
  ```

  ```bash
  export CIRCUITKIT_STRICT_HPARAMS=1
  ```

  The `validation.strict` key applies to dict/YAML configs only (`discover_circuit`,
  `evaluate_circuit`, `Pipeline`, `circuitkit validate-config`). `ck.prune`, `ck.quantize` and
  `ck.faithfulness` have no config, so for them only the environment variable turns strict
  mode on. A quoted `strict: "false"` is read as false; an unrecognised value is an error.

- **`eval.pillars`** takes a list of pillar names, or the string `"all"` for every pillar, in
  a dict/YAML config as well as in `Pipeline.evaluate` and `ck.faithfulness`.
- **`circuitkit validate-config --config my.yaml`** runs the same checks on a config file and
  prints each problem; add `--strict` to fail on warnings. **`circuitkit hparams`** prints the
  table below in the terminal.

## What is (and is not) listed

Each range comes from our own runs or from a cited paper, and the **Confidence** column says
which kind of evidence it rests on. Ranges that have not been validated yet (activation steering,
weight steering, knowledge editing, LoRA healing) are intentionally not listed and are not
checked. "Checked by" tells you what enforces a row:

- *valid-range error + sensible-range warning*: enforced by this module.
- *valid-range error*: only the valid range is enforced; the sensible range is not warned on.
- *existing check in code*: enforced by an older check elsewhere in the code base.
- *documented only*: reference information; nothing enforces it.

The tables below are generated from `circuitkit/utils/hparams.py`; do not edit them by hand
(run `python -m circuitkit.utils.hparams --write docs/reference/hyperparameters.md`).

<!-- hparams:start -->

### Discovery

| Parameter | Default | Valid | Sensible | Checked by | Confidence |
|---|---|---|---|---|---|
| `num_examples` | `128` | >= 1 | 32 to 1024 (log scale) | valid-range error + sensible-range warning | medium |
| `batch_size` | `4` | >= 1 | — | valid-range error | medium |
| `ig_steps` | `3` | >= 1 | — | valid-range error | high |
| `tao_bases` | [1, 5] (ACDC backend default) | non-empty list, each > 0 | — | valid-range error | medium |
| `tao_exps` | [-5, -4, -3, -2] (ACDC backend default) | non-empty list of integers | — | valid-range error | medium |
| `num_epochs` | `1000` | >= 1 | 500 to 1500 (not warned on) | valid-range error | medium |

**`num_examples`**: Examples used to estimate attribution scores. Set via dict `discovery.data_params.num_examples` · flat/Pipeline `n_examples` · CLI `--num-examples`. Scores from very few examples are noisy: in our IOI runs the top-30% node sets of three seeds overlapped only ~0.6-0.7 (Jaccard) at 128 examples, and the EAP-IG, ACDC and CD-T papers all work with ~100 examples. Cost grows linearly, so beyond ~1000 there is little to gain.

**`batch_size`**: Examples per attribution batch. Set via dict `discovery.batch_size` · flat/Pipeline `batch_size` · CLI `--batch-size`. Memory-scaled, so there is no fixed sensible range: our runs used 2 for 3-4B models and 16 for models up to 1.5B on 80-98 GB GPUs. `batch_size=1` is always safe, just slower.

**`ig_steps`**: Integration steps for the EAP-IG family (eap-ig, eap-ig-activations, eap-gp). Set via dict `discovery.ig_steps` · CLI `--ig-steps` · flat/Pipeline via `**kw`. Read by: eap-ig, eap-ig-activations, eap-gp. Ignored by every other algorithm. One step is plain EAP. The EAP-IG paper (Hanna et al., 2024) tested 2 to 50 steps: 2 is unfaithful on some tasks, every value above 2 is similarly faithful. Cost grows linearly with steps. EAP-GP reads the same key; its paper uses k=5.

**`tao_bases`**: ACDC threshold grid: tau = base x 10^exp for every (base, exp) pair. Set via dict `discovery.tao_bases`. Read by: acdc. ACDC sweeps a log-spaced grid of thresholds and keeps one circuit per tau. Published ACDC circuits use tau between about 4e-3 and 1e-1 (KL divergence).

**`tao_exps`**: Exponents of the ACDC threshold grid (see `tao_bases`). Set via dict `discovery.tao_exps`. Read by: acdc. A threshold of 1 or more keeps almost every edge, so such a grid point is flagged.

**`num_epochs`**: Training epochs for the IBCircuit mask. Set via dict `discovery.num_epochs`. Read by: ibcircuit. IBCircuit learns its mask by gradient descent, so it needs a real training budget. The existing IBCircuit check still warns below 100 epochs; the upstream IBCircuit repo trains 1300 epochs and CircuitKIT's runs use 1000.

### Pruning

| Parameter | Default | Valid | Sensible | Checked by | Confidence |
|---|---|---|---|---|---|
| `target_sparsity` | `0.3` | [0, 1] | <= 0.4 | valid-range error + sensible-range warning | medium |

**`target_sparsity`**: Fraction of heads/MLPs removed (the circuit keeps the rest). Set via dict `pruning.target_sparsity` · flat/Pipeline/CLI `sparsity`. At 30% sparsity accuracy retention ranged from 0.65 (random) to 0.99 (Taylor) on Llama-3.2-3B and Gemma-3-4B. At 50% every selector tried collapsed (perplexity 365 to 220,000) without recovery fine-tuning.

### Evaluation

| Parameter | Default | Valid | Sensible | Checked by | Confidence |
|---|---|---|---|---|---|
| `num_examples` | `256` | >= 1 | 100 to 1000 (log scale) | valid-range error + sensible-range warning | medium |
| `n_stability_runs` | `3` | >= 1 | 3 to 10 | valid-range error + sensible-range warning | medium |
| `pillars` | `patching` + `ablation` (Pipeline, flat API); `"all"` runs every pillar | `patching` \| `ablation` \| `baselines` \| `robustness` \| `stability` \| `generalization` \| `intervention_reliability` | — | valid-range error | high |

**`num_examples`**: Held-out examples used by the faithfulness pillars. Set via dict `eval.num_examples` · flat `n_examples` · Pipeline.evaluate `n_examples`. Faithfulness is a ratio of two noisy averages. The EAP-IG paper evaluates on 100 examples; our paper runs use 300 held-out examples.

**`n_stability_runs`**: Independent re-discoveries compared by the stability pillar. Set via dict `eval.n_stability_runs` · Pipeline.evaluate `n_stability_runs`. Stability is the overlap between re-discovered circuits. With 1 run there is nothing to compare, so the pillar reports a perfect overlap of 1.0; 2 runs give a single pairwise overlap. Seed-to-seed variance is large: in a replicate of our runs, patch faithfulness flipped sign in 3 of 5 cells. Each run repeats discovery, so cost grows linearly.

**`pillars`**: Faithfulness pillars to compute. Set via dict `eval.pillars` · flat `pillars` · Pipeline.evaluate `pillars`. Names are checked up front so a typo fails before any discovery or evaluation work starts. Stability and generalization re-run discovery and are the expensive ones.

### Benchmarking

| Parameter | Default | Valid | Sensible | Checked by | Confidence |
|---|---|---|---|---|---|
| `limit` | unset (full task) | > 0 | — | documented only | medium |

**`limit`**: Cap on examples per lm-eval task. Set via flat `benchmark(limit=...)` · Pipeline.benchmark `limit` · CLI `--limit`. lm-evaluation-harness documents `limit` as for testing (a value below 1 is a fraction of the task). Leave it unset for numbers you report.

### Quantization

| Parameter | Default | Valid | Sensible | Checked by | Confidence |
|---|---|---|---|---|---|
| `bits` | 4 (Pipeline, CLI) · 3 (flat `ck.quantize`) | `3` \| `4` \| `8` | — | existing check in code | medium |
| `high_fraction` | 0.3 (code) · 0.05 in our runs | [0, 1] | 0.05 to 0.3 (not warned on) | valid-range error | medium |

**`bits`**: Bit-width of the quantized tier (llmcompressor backend). Set via flat `quantize(bits=...)` · Pipeline.quantize `bits` · CLI `--bits`. The llmcompressor backend accepts 3, 4 or 8 and raises otherwise. The quanto backend ignores `bits` (it uses qint2/qint4/qint8 tiers).

**`high_fraction`**: Fraction of layers kept at high precision. Set via flat `quantize(high_fraction=...)` · Pipeline.quantize · CLI `--high-fraction`. With a 4-bit base, protecting 5% of layers kept 0.98 accuracy retention for every selector. At a 3-bit base the choice of selector only started to matter at 15% protected, so 3-bit with less than 15% is flagged.

<!-- hparams:end -->
