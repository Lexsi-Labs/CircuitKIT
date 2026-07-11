"""mcq_choice_swap — for MCQ records, swap two choices' positions.

The clean prompt presents choices A/B/C/D in order; the corrupt prompt
swaps the correct choice's content with a distractor's content while
leaving the *letter labels* unchanged. So the correct letter (e.g. ``B``)
now points to a wrong content. The model's task: predict the letter that
maps to the correct content. After swap, the right letter is the
distractor's original letter.

Length contract: PRESERVE (lines just shuffle positions).

Requires the record's ``meta['choices']`` and ``meta['correct_idx']`` to
be populated by the MCQAdapter.
"""

from __future__ import annotations

import random
import string
from typing import Any, List, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_LETTERS = string.ascii_uppercase


@register_strategy("mcq_choice_swap")
class MCQChoiceSwap(CorruptionStrategy):
    description = (
        "Swap the correct choice with a randomly-picked distractor. The "
        "correct-letter target now points to wrong content; the new "
        "correct letter is the distractor's original letter."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        return (
            "choices" in record.meta
            and "correct_idx" in record.meta
            and len(record.meta.get("choices", [])) >= 2
        )

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        if not self.fits(record):
            return CorruptionResult(
                None,
                None,
                notes=("record needs meta['choices'] and meta['correct_idx'] " "(use MCQAdapter)"),
                succeeded=False,
            )
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        choices: List[str] = list(record.meta["choices"])
        correct_idx: int = int(record.meta["correct_idx"])
        if len(choices) < 2:
            return CorruptionResult(None, None, notes="<2 choices", succeeded=False)
        distractor_indices = [i for i in range(len(choices)) if i != correct_idx]
        swap_idx = rng.choice(distractor_indices)
        # Build the new choices list with correct ↔ distractor swap
        swapped = list(choices)
        swapped[correct_idx], swapped[swap_idx] = swapped[swap_idx], swapped[correct_idx]
        # Reconstruct prompt with the same letter labels
        try:
            choice_letter_format = "{letter}. {text}\n"
            choices_block = "".join(
                choice_letter_format.format(letter=_LETTERS[k], text=str(c).strip())
                for k, c in enumerate(swapped)
            )
            # Strip & rebuild from the original question (best-effort: pull the
            # bit before the first "A. ").
            split_at = record.clean_prompt.find(f"{_LETTERS[0]}. ")
            if split_at > 0:
                prefix = record.clean_prompt[:split_at]
                # Find where the choice block ends (line "Answer:" or end)
                ans_pos = record.clean_prompt.rfind("Answer:")
                suffix = record.clean_prompt[ans_pos:] if ans_pos > split_at else "Answer:"
                new_prompt = prefix + choices_block + suffix
            else:
                # Fallback: append corrupted choices block at end
                new_prompt = record.clean_prompt.replace(
                    "\nAnswer:", f"\n[CORRUPTED]\n{choices_block}Answer:"
                )
        except Exception as e:
            return CorruptionResult(
                None,
                None,
                notes=f"prompt reconstruction failed: {e}",
                succeeded=False,
            )
        # New correct letter is the *swap_idx* position because the original
        # correct content now lives at `swap_idx`.
        new_answer = " " + _LETTERS[swap_idx]
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=new_answer,
            notes=f"swapped choices {correct_idx}<->{swap_idx}; correct letter now {_LETTERS[swap_idx]}",
            succeeded=True,
        )
