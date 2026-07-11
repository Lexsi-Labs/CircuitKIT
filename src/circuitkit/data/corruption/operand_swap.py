"""operand_swap — Stolfo et al. 2023 EMNLP arithmetic-circuit corruption.

Pair two prompts that share the same arithmetic operator template
(e.g. "A + B =") but differ in operand values. The clean prompt
yields one numeric answer; the corrupt prompt yields a different
one. EAP attribution then isolates the components that compute
the operation as a function of the operands.

Source: Stolfo et al. 2023 EMNLP arxiv:2305.15054 — "A Mechanistic
Interpretation of Arithmetic Reasoning in Language Models using
Causal Mediation Analysis". Their canonical intervention type:
"two prompts differ in the value of the operands (for example,
'2 + 3 =' and '4 + 5 =')".

This strategy works on records whose `clean_prompt` matches a
small-integer arithmetic pattern; for free-form GSM8K word problems
use the question-pair construction in
`benchmark/_custom_tasks._materialise_math` instead.

Length contract: PRESERVE (single-digit -> single-digit swap).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_ARITHMETIC_RE = re.compile(r"(?P<a>-?\d+)\s*(?P<op>[+\-*/])\s*(?P<b>-?\d+)\s*=", re.UNICODE)


def _eval_op(a: int, op: str, b: int) -> Optional[int]:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and b != 0 and a % b == 0:
        return a // b
    return None


@register_strategy("operand_swap")
class OperandSwap(CorruptionStrategy):
    description = (
        "Swap operands in an arithmetic prompt 'A op B =' to produce a "
        "different correct answer. Same operator, same template — "
        "isolates the operation circuit (Stolfo et al. 2023 EMNLP)."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        return _ARITHMETIC_RE.search(record.clean_prompt) is not None

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        max_attempts: int = 8,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        match = _ARITHMETIC_RE.search(record.clean_prompt)
        if not match:
            return CorruptionResult(None, None, notes="no arithmetic pattern", succeeded=False)
        a, op, b = int(match.group("a")), match.group("op"), int(match.group("b"))
        original_result = _eval_op(a, op, b)
        if original_result is None:
            return CorruptionResult(None, None, notes="undefined op result", succeeded=False)

        # Pick a different (a', b') with same op such that the result differs.
        for _ in range(max_attempts):
            new_a = a + rng.choice([-3, -2, -1, 1, 2, 3])
            new_b = b + rng.choice([-3, -2, -1, 1, 2, 3])
            new_result = _eval_op(new_a, op, new_b)
            if new_result is None or new_result == original_result:
                continue
            new_substr = f"{new_a} {op} {new_b} ="
            new_prompt = (
                record.clean_prompt[: match.start()]
                + new_substr
                + record.clean_prompt[match.end() :]
            )
            new_answer = " " + str(new_result)
            return CorruptionResult(
                corrupt_prompt=new_prompt,
                corrupt_answer=new_answer,
                notes=f"({a} {op} {b}) -> ({new_a} {op} {new_b})",
                succeeded=True,
            )
        return CorruptionResult(
            None,
            None,
            notes=f"could not find distinct operand pair after " f"{max_attempts} attempts",
            succeeded=False,
        )
