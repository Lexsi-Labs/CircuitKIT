# Pillar 4: Robustness

Robustness tests whether the discovered circuit continues to explain the model's behavior when inputs are perturbed in ways that preserve the task semantics. If the circuit is genuinely capturing the computational mechanism (not just the surface form), it should hold under input variations.

**Requires:** the spaCy `en_core_web_sm` model for the spaCy-backed corruption strategies (spaCy itself ships with the base install): `python -m spacy download en_core_web_sm`.

---

## The Measurement

CircuitKit runs the patching evaluation (Pillar 1) on **corrupted variants** of the original dataset:

1. Apply one or more corruption strategies to the clean examples
2. Re-run Pillar 1 (causal patching) with the corrupted inputs
3. Compare the patching score on corrupted inputs vs. original inputs

A robust circuit maintains a high patching score even when the surface form of the inputs changes.

$$\text{robustness score} = \frac{\text{patching score}(\text{corrupted inputs})}{\text{patching score}(\text{original inputs})}$$

A score near 1.0 means the circuit is equally faithful on corrupted and original inputs.

!!! warning "The ratio is only interpretable when the underlying patching score is bounded and non-negative"
    Patching scores use a signed metric (`logit_diff`), so a corruption that inverts
    the task can push the *variant* score below zero or above the original — the ratio
    then escapes `[0, 1]` (e.g. `-0.8` or `1.9`) and has no "robustness" meaning. When
    the original score is `≤ 0` or the variant score is `< 0`, Pillar 4 reports
    `status: "invalid"` instead of a misleading number. Treat robustness as meaningful
    only when corruption *gently degrades* the score; on syntactic tasks like IOI, many
    corruption strategies break the task rather than perturb it.

---

## Running Pillar 4

```python
# Via Pipeline
pipe.evaluate(pillars=["robustness"], n_examples=256)
print(pipe.report.robustness)
# report.robustness is keyed by corruption-variant name, and each value is the
# variant's own result dict (not a bare float). There is no top-level "overall" key.
# {
#   "logical_negation": {"robustness_ratio": 0.79, "original_score": ..., "variant_score": ..., ...},
#   "format_distractor": {"robustness_ratio": 0.71, ...},
#   "position_shift":    {"robustness_ratio": 0.83, ...},
# }

# To read one variant's ratio:
print(pipe.report.robustness["logical_negation"]["robustness_ratio"])
```

---

## Corruption Strategies

Pillar 4 automatically applies three rule-based corruptions (no external LLM needed):

| Strategy | What it does | Example |
|----------|-------------|---------|
| **Logical negation** | Negate key claims in the prompt | "X is true" → "X is not true" |
| **Format distractor** | Inject an irrelevant-but-plausible distractor sentence | Append an unrelated sentence (correct answer unchanged) |
| **Position shift** | Shuffle or rotate sentence segments | Reorder clauses while preserving meaning |

Additional corruption strategies are available from `circuitkit.corruption` and can be applied manually:

```python
import random
from circuitkit.corruption import EntitySwapCorruption, ParaphraseCorruption

entity_strategy = EntitySwapCorruption(entity_pool="auto")
paraphrase_strategy = ParaphraseCorruption()

# corrupt() takes a {"prompt": ...} example and a keyword-only rng
rng = random.Random(42)
corrupted = [
    entity_strategy.corrupt({"prompt": r.clean_prompt}, rng=rng)
    for r in task_spec.ds.records
]
```

---

## Available Corruption Classes

| Class | Module | What it does |
|-------|--------|-------------|
| `EntitySwapCorruption` | `circuitkit.corruption` | Replace names/entities |
| `ParaphraseCorruption` | `circuitkit.corruption` | Rephrase while preserving semantics |
| `DistractorInjectionCorruption` | `circuitkit.corruption` | Insert misleading clauses |
| `TokenSwapCorruption` | `circuitkit.corruption` | Swap random tokens |
| `NegationCorruption` | `circuitkit.corruption` | Negate key claims |
| `RoleSwapCorruption` | `circuitkit.corruption` | Swap subject/object roles |
| `PositionShiftCorruption` | `circuitkit.corruption` | Shuffle/rotate sentence segments |

**`PositionShiftCorruption` parameters:**

```python
from circuitkit.corruption import PositionShiftCorruption

strategy = PositionShiftCorruption(
    strategy="shuffle",   # "shuffle" (random reorder) or "rotate" (cyclic shift)
    seed=42,
)
```

---

## Interpreting Robustness Scores

| Robustness Score | Interpretation |
|:---:|----------------|
| ≥ 0.80 | **Robust** — circuit holds under input variations |
| 0.60 – 0.80 | **Moderate** — some degradation under corruption |
| < 0.60 | **Brittle** — circuit is surface-form dependent |

**A brittle circuit** (low robustness) may indicate the algorithm is responding to shallow surface features of the training examples rather than the underlying computation. Consider:

1. Increasing `n_examples` (more examples reduce surface-form bias)
2. Using more diverse training examples
3. Checking that your corruption strategies produce valid inputs (`validate_token_alignment`)

---

## Auto-Build of Corruption Dataloaders

`run_full_faithfulness` automatically builds rule-based corruption dataloaders for the `logical_negation`, `format_distractor`, and `position_shift` variants when they appear in `corruption_variants` but no dataloader is supplied. The auto-build path requires the task spec to expose `.ds.records`.

---

## Next Steps

- [Baselines (Pillar 5)](baselines.md) — comparing to random selection
- [Stability (Pillar 3)](stability.md) — cross-seed consistency
- [Framework Overview](framework.md) — all 6 pillars
