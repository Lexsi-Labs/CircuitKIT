"""token_swap — replace one content token with another, length-preserving.

Generic Zhang & Nanda 2023 String Token Replacement (STR) strategy:
in-distribution counterfactual that preserves token count. Picks a
random content token (excluding stopwords) and swaps it with a random
other content token from the same prompt — guaranteed token-aligned.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional, Set

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "and",
    "or",
    "but",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "his",
    "her",
    "their",
}


@register_strategy("token_swap")
class TokenSwap(CorruptionStrategy):
    description = (
        "Length-preserving String Token Replacement (Zhang & Nanda 2023). "
        "Swap a random content token in the prompt for another randomly "
        "chosen content token from the same prompt."
    )
    length_contract = LengthContract.PRESERVE

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        words = re.findall(r"\b[\w']+\b", record.clean_prompt)
        candidates = [
            w for w in words if w.lower() not in _STOPWORDS and len(w) >= 3 and w.isalpha()
        ]
        if len(candidates) < 2:
            return CorruptionResult(
                None,
                None,
                notes="too few content tokens to swap",
                succeeded=False,
            )
        a, b = rng.sample(candidates, 2)
        # Replace the *first* occurrence of `a` with `b`.
        new_prompt = re.sub(rf"\b{re.escape(a)}\b", b, record.clean_prompt, count=1)
        new_answer = record.clean_answer
        if record.clean_answer.strip() == a:
            new_answer = (" " if record.clean_answer.startswith(" ") else "") + b
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=new_answer,
            notes=f"swapped {a!r} -> {b!r}",
            succeeded=True,
        )
