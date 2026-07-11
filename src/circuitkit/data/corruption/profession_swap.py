"""profession_swap — Vig et al. 2020 NeurIPS bias-circuit corruption.

Pair sentences that share the same template but differ in a
gender-stereotyped profession. The clean prompt yields one pronoun;
the corrupt prompt yields the opposite-gendered pronoun.

Example:
  clean:   "The accountant said that"  -> " he"   (male-stereo profession)
  corrupt: "The nurse said that"       -> " she"  (female-stereo profession)

This is the canonical bias-circuit pairing from Vig et al. 2020
(arxiv:2004.12265). Different prompts AND different target tokens
make EAP attribution non-degenerate.

Length contract: PRESERVE.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

# Profession lexicon partitioned by historical gender stereotype
# (Bolukbasi et al. 2016 / Vig et al. 2020 word lists).
_MALE_STEREO = [
    "accountant",
    "architect",
    "carpenter",
    "doctor",
    "engineer",
    "lawyer",
    "mechanic",
    "physicist",
    "plumber",
    "scientist",
    "soldier",
    "surgeon",
]
_FEMALE_STEREO = [
    "nurse",
    "teacher",
    "secretary",
    "librarian",
    "receptionist",
    "stylist",
    "babysitter",
    "dietitian",
    "florist",
    "housekeeper",
    "midwife",
    "nanny",
]

_PROFESSION_RE = re.compile(
    r"\b(" + "|".join(_MALE_STEREO + _FEMALE_STEREO) + r")\b", re.IGNORECASE
)


def _stereotype_of(word: str) -> Optional[str]:
    w = word.lower()
    if w in _MALE_STEREO:
        return "male"
    if w in _FEMALE_STEREO:
        return "female"
    return None


@register_strategy("profession_swap")
class ProfessionSwap(CorruptionStrategy):
    description = (
        "Swap a stereotyped profession in the prompt to one of the opposite "
        "stereotype, AND swap the expected pronoun answer accordingly. "
        "Vig et al. 2020 bias-circuit pairing — different prompts and "
        "different target tokens make this a valid EAP setup."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        return _PROFESSION_RE.search(record.clean_prompt) is not None

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        m = _PROFESSION_RE.search(record.clean_prompt)
        if not m:
            return CorruptionResult(None, None, notes="no profession matched", succeeded=False)
        original = m.group(1)
        stereo = _stereotype_of(original)
        if stereo is None:
            return CorruptionResult(None, None, notes="profession not in lexicon", succeeded=False)
        # Pick a profession of the OPPOSITE stereotype.
        candidates = _FEMALE_STEREO if stereo == "male" else _MALE_STEREO
        new_word = rng.choice(candidates)
        # Preserve original capitalisation.
        if original[0].isupper():
            new_word = new_word.capitalize()
        new_prompt = record.clean_prompt[: m.start()] + new_word + record.clean_prompt[m.end() :]
        # Determine the corrupt answer: opposite-stereo pronoun.
        clean_ans = record.clean_answer.strip().lower().lstrip(".!?,;:'\"")
        if clean_ans in ("he", "him", "his"):
            corrupt_a = " she" if clean_ans == "he" else (" her" if clean_ans == "him" else " her")
        elif clean_ans in ("she", "her", "hers"):
            corrupt_a = " he" if clean_ans == "she" else (" him" if clean_ans == "her" else " his")
        else:
            # Default: predict the pronoun that matches the new profession's
            # stereotype.
            corrupt_a = " she" if stereo == "male" else " he"
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=corrupt_a,
            notes=f"profession {original!r} -> {new_word!r}; "
            f"pronoun {record.clean_answer!r} -> {corrupt_a!r}",
            succeeded=True,
        )
