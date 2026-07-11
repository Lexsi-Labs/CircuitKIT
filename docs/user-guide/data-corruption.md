# Corruption strategies

EAP-family discovery needs a corrupt prompt for every clean one. If your data already has both halves, supply them in explicit columns and you're done — an explicit counterfactual always wins. When you only have clean prompts, a corruption strategy generates the corrupt half at load time.

## The built-in strategies

Set `corruption.strategy` in a task YAML to one of these five names:

| Strategy | What it does |
|---|---|
| `entity_swap` | Swaps named entities (people, places) in the prompt |
| `token_swap` | Swaps tokens at chosen positions |
| `paraphrase` | Rewrites the prompt while keeping meaning |
| `distractor` | Injects a distracting element into the prompt |
| `role_swap` | Swaps subject/object roles |

```yaml
corruption:
  strategy: entity_swap
  config: {}          # optional strategy-specific keyword arguments
```

Only these five names are valid in the loader. Passing anything else raises `Unknown corruption strategy`. (The corruption package contains more classes — negation, position-shift, and others — but they aren't wired into the task-YAML `strategy:` map.)

## Writing a custom strategy

A corruption strategy implements the `CorruptionStrategy` protocol: a `name`, a `mode`, and a `corrupt` method that takes a clean example dict and returns a corrupted one.

```python
import random
from typing import Any, Dict, Optional

class SwapFirstNoun:
    name = "swap_first_noun"
    mode = "meaning-altering"   # or "meaning-preserving" | "role-swap"

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a corrupted copy of the example. Use rng, not random.*,
        so runs are reproducible."""
        corrupted = dict(example)
        # ... your perturbation logic, writing back into corrupted["prompt"] ...
        return corrupted
```

Wrap it in a `CorruptionPipeline` and pass it to `GenericTaskSpec.from_csv(..., corruption_strategy=pipeline)`, or register it into your own loader map. The protocol also defines optional `batch_corrupt` and `validate` methods; the default single-example `corrupt` is enough to get started.

## When the pair carries no signal

Here's the honest caveat. The built-in strategies were designed for **syntactic, template-style prompts** — IOI-style sentences where swapping a name or a token genuinely flips the answer. On instruction-tuned or safety-style prompts, they often can't find anything to change and leave the prompt untouched. Some also need optional dependencies (for example, `entity_swap` relies on spaCy for entity detection); when the dependency is missing, the strategy quietly produces no change.

When the corrupt prompt ends up identical to the clean one for every example, the contrastive pair carries no signal, and any circuit you get out is meaningless. CircuitKit does not let this pass silently:

- On the task-YAML / `GenericTaskSpec` path, discovery logs a loud warning when a corruption strategy (or explicit corrupt column) produces no change across the whole dataset — the clean and corrupt prompts are identical, so there is nothing to attribute against.
- On the normalized / `data.type: template` path, discovery raises a `ValueError` outright when a paired algorithm is handed data that isn't fully paired, telling you to apply a corruption strategy or switch to a clean-only algorithm (IBCircuit / CD-T).

If you're working with instruction-tuned or safety prompts, don't rely on the syntactic strategies. Supply explicit contrastive columns instead — write the clean and corrupt prompts yourself (a benign request paired with a length-matched harmful one, a true statement paired with a false one), map them with the `corrupted_prompt` / `corrupted_answer` schema keys, and skip the `corruption` block entirely. You keep full control over what the contrast actually isolates.

## Next steps

- [Bring your own data](custom-data.md) — supplying explicit corrupt columns
- [Data model](data.md) — why the pair matters
- [YAML configuration](../cli/yaml-config.md) — full schema reference
