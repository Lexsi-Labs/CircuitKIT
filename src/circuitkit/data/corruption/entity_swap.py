"""entity_swap — swap a salient noun/entity in the prompt.

Length-preserving when the swap is between two same-token-count entities.
Uses spaCy if installed; falls back to a simple capitalised-word heuristic
otherwise so it works in any environment.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

# A small built-in pool of common-name swap candidates so the strategy
# works without external NER deps. Length-grouped by character count so
# we can pick a same-length replacement.
_NAMES_BY_LEN: Dict[int, List[str]] = {}
for _name in [
    "Mary",
    "John",
    "Lisa",
    "Tom",
    "Sara",
    "Alex",
    "Beth",
    "Carol",
    "David",
    "Emily",
    "Frank",
    "Greta",
    "Henry",
    "Ivy",
    "Jack",
    "Kate",
    "Liam",
    "Mia",
    "Noah",
    "Olivia",
    "Peter",
    "Quinn",
    "Rachel",
    "Sam",
]:
    _NAMES_BY_LEN.setdefault(len(_name), []).append(_name)


def _word_tokens(s: str) -> List[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z']*\b", s)


def _capitalised(words: List[str]) -> List[str]:
    return [w for w in words if w[:1].isupper() and w not in {"I", "A", "An", "The"}]


@register_strategy("entity_swap")
class EntitySwap(CorruptionStrategy):
    description = (
        "Swap a salient capitalised entity in the clean prompt for a "
        "same-length alternative. Length-preserving when same-length swap "
        "is possible; falls back to UNKNOWN-contract drop if not."
    )
    length_contract = LengthContract.PRESERVE

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        candidates: Optional[List[str]] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        prompt = record.clean_prompt
        words = _word_tokens(prompt)
        ents = _capitalised(words)
        if not ents:
            return CorruptionResult(
                None,
                None,
                notes="no capitalised entity found in prompt",
                succeeded=False,
            )
        target = rng.choice(ents)
        # Find a same-length replacement that's not the same word.
        pool: List[str] = []
        if candidates is not None:
            pool = [c for c in candidates if len(c) == len(target) and c != target]
        if not pool:
            pool = [c for c in _NAMES_BY_LEN.get(len(target), []) if c != target]
        if not pool:
            # No same-length swap — soft fail (caller can fall back to token_swap)
            return CorruptionResult(
                None,
                None,
                notes=f"no same-length swap candidate for {target!r}",
                succeeded=False,
            )
        replacement = rng.choice(pool)
        # Replace only the first occurrence (caller can re-run if more needed)
        new_prompt = re.sub(rf"\b{re.escape(target)}\b", replacement, prompt, count=1)
        # Answer: if the original answer mentions the swapped entity, swap there too.
        new_answer = record.clean_answer
        if record.clean_answer.strip() == target:
            new_answer = (" " if record.clean_answer.startswith(" ") else "") + replacement
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=new_answer,
            notes=f"swapped {target!r} -> {replacement!r}",
            succeeded=True,
        )
