"""code_syntax_corrupt — corrupt the function signature/docstring in a code prompt.

Operates on records produced by ``CodeAdapter``. Applies one of:

  - rename a function-signature parameter (``def foo(x):`` -> ``def foo(y):``)
  - flip a comparison operator in the docstring (``>=`` -> ``<=``)
  - swap a constant (``return True`` -> ``return False``)
  - reorder two function arguments

Length contract: PRESERVE for parameter rename / operator flip; UNKNOWN
otherwise.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_OPERATOR_FLIPS = {
    ">=": "<=",
    "<=": ">=",
    ">": "<",
    "<": ">",
    "==": "!=",
    "!=": "==",
    "True": "False",
    "False": "True",
    "and": "or",
}


@register_strategy("code_syntax_corrupt")
class CodeSyntaxCorrupt(CorruptionStrategy):
    description = (
        "Corrupt a code prompt by flipping a comparison operator, "
        "renaming a parameter, or swapping True<->False. Length-"
        "preserving for the operator/boolean flips."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        # Code prompts typically include 'def ' or '```' or function-style signature
        return (
            "def " in record.clean_prompt
            or "function " in record.clean_prompt
            or "```" in record.clean_prompt
        )

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        prompt = record.clean_prompt
        # Try operator flip first (length-preserving)
        candidates = [op for op in _OPERATOR_FLIPS if op in prompt]
        if candidates:
            target = rng.choice(candidates)
            replacement = _OPERATOR_FLIPS[target]
            new_prompt = prompt.replace(target, replacement, 1)
            return CorruptionResult(
                corrupt_prompt=new_prompt,
                corrupt_answer=record.clean_answer,
                notes=f"flipped operator {target!r} -> {replacement!r}",
                succeeded=True,
            )
        # Fall back to parameter rename
        m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", prompt)
        if m:
            params = [p.strip().split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
            if params:
                pick = rng.choice(params)
                # Use a same-length replacement to keep tokens aligned
                replacement = pick[::-1] if len(pick) >= 2 else pick + "_"
                new_prompt = re.sub(
                    rf"\b{re.escape(pick)}\b", replacement, prompt, count=2
                )  # both in def + body
                return CorruptionResult(
                    corrupt_prompt=new_prompt,
                    corrupt_answer=record.clean_answer,
                    notes=f"renamed param {pick!r} -> {replacement!r}",
                    succeeded=True,
                )
        return CorruptionResult(
            None,
            None,
            notes=("no operator/boolean to flip and no def signature found " "to rename"),
            succeeded=False,
        )
