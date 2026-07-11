#!/usr/bin/env python3
"""Custom corruption strategy: implement and register a new corruption.

A corruption strategy generates the *corrupt* half of a contrastive pair.
To add your own:

1. Subclass ``CorruptionStrategy`` and implement
   ``corrupt(record) -> CorruptionResult``.
2. Declare its ``length_contract`` (EAP needs token-aligned pairs, so
   length-preserving corruptions are strongly preferred — see
   ``LengthContract``).
3. Register it with the ``@register_strategy("name")`` decorator. The name
   then works everywhere a built-in strategy name does (YAML configs, CLI,
   ``get_strategy``).
"""

import random

from circuitkit.data.corruption import (
    CorruptionResult,
    CorruptionStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from circuitkit.data.corruption.base import LengthContract
from circuitkit.data.normalized import ContrastiveRecord


@register_strategy("synonym_swap")
class SynonymSwap(CorruptionStrategy):
    """Replace a known named entity with an alternative (meaning-changing,
    length-preserving when the replacement tokenizes to the same length)."""

    description = "Swap a known named entity for a same-role alternative."
    length_contract = LengthContract.PRESERVE

    SYNONYMS = {
        "Alice": ["Anna", "Ada", "Amy"],
        "Bob": ["Ben", "Bill", "Brad"],
        "France": ["Spain", "Italy", "Germany"],
    }

    def corrupt(self, record: ContrastiveRecord, *, rng=None, **_unused) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        prompt = record.clean_prompt
        for word, replacements in self.SYNONYMS.items():
            if word in prompt:
                replacement = rng.choice(replacements)
                return CorruptionResult(
                    corrupt_prompt=prompt.replace(word, replacement, 1),
                    corrupt_answer=record.clean_answer,
                    notes=f"swapped {word!r} -> {replacement!r}",
                    succeeded=True,
                )
        return CorruptionResult(
            None,
            None,
            notes="no known entity found in prompt",
            succeeded=False,
        )


if __name__ == "__main__":
    # The decorator has already registered the strategy at import time:
    print(f"Registered strategies: {list_strategies()}")
    assert "synonym_swap" in list_strategies()

    # Look it up by name, exactly as YAML/CLI configs do:
    strategy = get_strategy("synonym_swap")()

    record = ContrastiveRecord(
        record_id="demo-0",
        clean_prompt="Alice went to the store",
        clean_answer=" store",
    )
    result = strategy.corrupt(record, rng=random.Random(42))
    print(f"  Original:  {record.clean_prompt}")
    print(f"  Corrupted: {result.corrupt_prompt}   ({result.notes})")

    # The corrupt_example() bridge lets the same strategy run inside
    # GenericTaskSpec's dict-based corruption pipeline:
    example = {"prompt": "Bob asked France for help", "answer": " help"}
    corrupted = strategy.corrupt_example(example, rng=random.Random(0))
    print(f"  Dict pipeline: {example['prompt']!r} -> {corrupted['prompt']!r}")
