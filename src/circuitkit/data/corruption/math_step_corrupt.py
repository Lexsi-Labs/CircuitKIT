"""math_step_corrupt — corrupt one intermediate step in a math chain-of-thought.

Operates on records produced by ``MathAdapter``. Replaces an intermediate
arithmetic step in the solution text (e.g. ``2+3=5`` -> ``2+3=6``) so
the corrupt prompt now contains a known wrong intermediate computation
yet the final answer still matches the original. This isolates the
circuits that *trust* intermediate computation.

Length contract: PRESERVE (single-digit substitution) when possible.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

# Equation-like patterns: "2 + 3 = 5", "7 - 4 = 3"
_EQ_PAT = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)")


@register_strategy("math_step_corrupt")
class MathStepCorrupt(CorruptionStrategy):
    description = (
        "Replace one intermediate equation in a math solution with a "
        "wrong-but-plausible result (e.g. 2+3=5 -> 2+3=6). Surfaces the "
        "model's CoT-trust circuit."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        # Need a solution_text with at least one equation
        solution = record.meta.get("solution_text", "")
        return bool(_EQ_PAT.search(solution or record.clean_prompt))

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        # Look for an equation in the solution text first, fall back to prompt
        target_text = record.meta.get("solution_text") or record.clean_prompt
        matches = list(_EQ_PAT.finditer(target_text))
        if not matches:
            return CorruptionResult(
                None,
                None,
                notes="no arithmetic equation found in solution_text or prompt",
                succeeded=False,
            )
        m = rng.choice(matches)
        a, op, b, result = m.group(1), m.group(2), m.group(3), m.group(4)
        # Bump the result by +/-1 keeping same digit count
        try:
            res_int = int(result)
        except ValueError:
            return CorruptionResult(None, None, notes="non-integer result", succeeded=False)
        offset = rng.choice([1, -1])
        new_res = res_int + offset
        # Preserve digit count if possible
        if len(str(new_res)) != len(result):
            new_res = res_int + (-offset)  # try the other direction
            if len(str(new_res)) != len(result):
                return CorruptionResult(
                    None,
                    None,
                    notes=f"can't keep digit count for {result!r}",
                    succeeded=False,
                )
        new_eq = f"{a} {op} {b} = {new_res}"
        if record.meta.get("solution_text"):
            new_prompt = record.clean_prompt  # prompt itself unchanged
        else:
            new_prompt = target_text[: m.start()] + new_eq + target_text[m.end() :]
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=record.clean_answer,
            notes=f"corrupted equation: {m.group(0)!r} -> {new_eq!r}",
            succeeded=True,
        )
