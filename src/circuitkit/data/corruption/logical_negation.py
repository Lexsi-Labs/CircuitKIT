"""logical_negation — flip the logical sense of a prompt by inserting/removing 'not'.

If the clean prompt contains an auxiliary verb ("is", "are", "was", "were",
"can", "will", "should") we insert ``not`` after it. If it contains
``not``/``no``, we remove it. The corrupt counterpart should now elicit
the *opposite* answer.

This isolates the negation circuit. Length contract is UNKNOWN because
inserting/removing one token changes total length by 1.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_AUX_VERBS = (
    "is",
    "are",
    "was",
    "were",
    "can",
    "will",
    "should",
    "could",
    "would",
    "did",
    "does",
    "do",
    "has",
    "have",
    "had",
    "may",
    "might",
    "must",
)
_NOT_PATTERNS = (
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bn't\b", re.IGNORECASE),
)


@register_strategy("logical_negation")
class LogicalNegation(CorruptionStrategy):
    description = (
        "Flip the logical polarity of the prompt: insert 'not' after an "
        "auxiliary verb, or remove an existing 'not'. Surfaces the model's "
        "negation circuit."
    )
    length_contract = LengthContract.UNKNOWN

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        prompt = record.clean_prompt
        # If the prompt already contains "not"/"no"/"n't", remove the first.
        for pat in _NOT_PATTERNS:
            if pat.search(prompt):
                new_prompt = pat.sub("", prompt, count=1)
                # collapse any double spaces produced by removal
                new_prompt = re.sub(r"\s{2,}", " ", new_prompt).strip()
                return CorruptionResult(
                    corrupt_prompt=new_prompt,
                    corrupt_answer=record.clean_answer,
                    notes="removed negation",
                    succeeded=True,
                )
        # Otherwise: insert "not" after the first auxiliary verb.
        for aux in _AUX_VERBS:
            pat = re.compile(rf"\b{aux}\b", re.IGNORECASE)
            m = pat.search(prompt)
            if m:
                start, end = m.span()
                new_prompt = prompt[:end] + " not" + prompt[end:]
                return CorruptionResult(
                    corrupt_prompt=new_prompt,
                    corrupt_answer=record.clean_answer,
                    notes=f"inserted 'not' after {aux!r}",
                    succeeded=True,
                )
        return CorruptionResult(
            None,
            None,
            notes="no auxiliary verb found and no existing negation; cannot apply",
            succeeded=False,
        )
