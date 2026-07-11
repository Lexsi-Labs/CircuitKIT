"""final_answer_swap — produce a meaningful contrastive pair for math reasoning.

Two modes:

1. **Math-reasoning mode (preferred for GSM8K / CoT word problems).**
   When the record carries a chain-of-thought solution in
   ``meta['solution_text']`` containing GSM8K-style calculator
   annotations ``<<a op b=R>>``, this strategy perturbs the operand
   of the *final* calculation so the computed result — and therefore
   the final numeric answer — changes. The same numeric edit is applied
   both to the prompt (which conditions the model) and to the target
   answer. Clean and corrupt prompts genuinely differ, so this is a
   valid contrast for activation-patching / EAP circuit discovery, and
   the correct answer differs so the differentiable NLL metric has
   signal. This is an operand-swap in the spirit of Stolfo et al. 2023
   but applied to free-form word problems.

2. **Answer-only mode (fallback).** For records where ``clean_answer``
   is a bare numeric / boolean / single-letter answer and no math
   reasoning trace is available, perturb just the target token
   (``42`` -> ``43``, ``yes`` -> ``no``). The prompt is unchanged — this
   measures whether the circuit produces the correct token vs an
   adjacent one (useful for logit-difference metrics, not for
   activation patching).

Length contract: PRESERVE — numeric substitutions keep digit counts where
possible; the fallback mode never changes the prompt.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_BOOL_ANSWERS = {
    "yes": "no",
    "no": "yes",
    "true": "false",
    "false": "true",
    "True": "False",
    "False": "True",
    "Yes": "No",
    "No": "Yes",
}

# GSM8K calculator annotation: "<<48+24=72>>"
_CALC_STEP = re.compile(r"<<\s*([-\d.]+)\s*([+\-*/])\s*([-\d.]+)\s*=\s*([-\d.]+)\s*>>")
# Final-answer marker "#### 72"
_GSM_FINAL = re.compile(r"####\s*(-?\d[\d,]*)")


def _eval_op(a: float, op: str, b: float) -> Optional[float]:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/" and b != 0:
            return a / b
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _fmt_num(x: float) -> str:
    """Format a number without a trailing ``.0`` for integral values."""
    if x == int(x):
        return str(int(x))
    return "%g" % x


class _MathSwapResult:
    __slots__ = ("succeeded", "old_result", "new_result", "old_step", "new_step", "notes")


def _swap_final_calc(
    solution_text: str,
    rng: random.Random,
    max_attempts: int = 12,
) -> Optional[_MathSwapResult]:
    """Perturb the operand of the LAST calculator step so its result changes.

    Returns a description of the substitution, or None if no usable step
    was found.
    """
    matches = list(_CALC_STEP.finditer(solution_text))
    if not matches:
        return None
    m = matches[-1]  # the final calculation drives the final answer
    a_s, op, b_s, r_s = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        a, b = float(a_s), float(b_s)
        old_result = float(r_s)
    except ValueError:
        return None

    for _ in range(max_attempts):
        # Perturb one operand by a small non-zero delta.
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        if rng.random() < 0.5:
            new_a, new_b = a + delta, b
        else:
            new_a, new_b = a, b + delta
        new_result = _eval_op(new_a, op, new_b)
        if new_result is None:
            continue
        # Require a genuinely different — and integral, GSM8K-style — answer.
        if new_result == old_result:
            continue
        if new_result != int(new_result):
            continue
        res = _MathSwapResult()
        res.succeeded = True
        res.old_result = old_result
        res.new_result = new_result
        res.old_step = m.group(0)
        res.new_step = f"<<{_fmt_num(new_a)}{op}{_fmt_num(new_b)}={_fmt_num(new_result)}>>"
        res.notes = (
            f"final calc {a_s}{op}{b_s}={r_s} -> "
            f"{_fmt_num(new_a)}{op}{_fmt_num(new_b)}={_fmt_num(new_result)}"
        )
        return res
    return None


@register_strategy("final_answer_swap")
class FinalAnswerSwap(CorruptionStrategy):
    description = (
        "Math-reasoning corruption: perturbs the operand of the final "
        "calculator step in a chain-of-thought solution so the computed "
        "final answer changes. Both the prompt and the target answer are "
        "edited, giving a valid contrastive pair for circuit discovery. "
        "Falls back to answer-only perturbation (numbers +/-1, booleans "
        "inverted, MCQ letters shifted) when no reasoning trace is present."
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

        # --- Mode 1: math-reasoning prompt-level operand swap -----------
        solution = record.meta.get("solution_text", "") if record.meta else ""
        if solution:
            swap = _swap_final_calc(solution, rng)
            if swap is not None:
                old_ans = _fmt_num(swap.old_result)
                new_ans = _fmt_num(swap.new_result)
                # Apply the operand/result edit to the prompt if the final
                # calculator step (or the bare result) appears there.
                clean_prompt = record.clean_prompt
                corrupt_prompt = clean_prompt
                if swap.old_step in clean_prompt:
                    # GSM8K echoes the calculator result immediately after the
                    # "<<...=R>>" annotation as a bare number ("<<90+18=108>>108").
                    # Replace the step AND that trailing bare result together so
                    # the corrupt reasoning trace stays internally consistent;
                    # otherwise the prompt reads "<<93+18=111>>108".
                    old_step_with_result = swap.old_step + old_ans
                    new_step_with_result = swap.new_step + new_ans
                    if old_step_with_result in clean_prompt:
                        corrupt_prompt = clean_prompt.replace(
                            old_step_with_result, new_step_with_result, 1
                        )
                    else:
                        corrupt_prompt = clean_prompt.replace(swap.old_step, swap.new_step, 1)
                # Also fix up an explicit "#### N" marker if present.
                corrupt_prompt = _GSM_FINAL.sub(
                    lambda mm: mm.group(0).replace(mm.group(1), new_ans),
                    corrupt_prompt,
                )
                if corrupt_prompt == clean_prompt:
                    # The final step is not echoed in the prompt; fall back
                    # to swapping the bare result number near the end.
                    idx = clean_prompt.rfind(old_ans)
                    if idx != -1:
                        corrupt_prompt = (
                            clean_prompt[:idx] + new_ans + clean_prompt[idx + len(old_ans) :]
                        )
                prefix = " " if record.clean_answer.startswith(" ") else ""
                return CorruptionResult(
                    corrupt_prompt=corrupt_prompt,
                    corrupt_answer=prefix + new_ans,
                    notes=swap.notes,
                    succeeded=True,
                )

        # --- Mode 2: answer-only perturbation (prompt unchanged) --------
        ans = record.clean_answer.strip()
        prefix = " " if record.clean_answer.startswith(" ") else ""
        new = None
        m = re.match(r"^-?\d+$", ans)
        if m:
            n = int(ans)
            new = str(n + rng.choice([1, -1, 2, -2]))
        elif ans.lower() in _BOOL_ANSWERS:
            new = _BOOL_ANSWERS[ans] if ans in _BOOL_ANSWERS else _BOOL_ANSWERS[ans.lower()]
        elif len(ans) == 1 and ans.isalpha() and ans.isupper():
            offset = rng.choice([1, -1, 2, -2])
            new_code = ((ord(ans) - ord("A") + offset) % 26) + ord("A")
            new = chr(new_code)
            if new == ans:
                new = chr(((ord(ans) - ord("A") + 1) % 26) + ord("A"))
        if new is None:
            return CorruptionResult(
                None,
                None,
                notes=(
                    f"clean_answer {ans!r} is not numeric / boolean / "
                    f"single-letter and no solution_text was provided; "
                    f"final_answer_swap doesn't apply"
                ),
                succeeded=False,
            )
        return CorruptionResult(
            corrupt_prompt=record.clean_prompt,
            corrupt_answer=prefix + new,
            notes=f"answer {ans!r} -> {new!r}",
            succeeded=True,
        )
