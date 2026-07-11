"""
PositionShiftCorruption: Shift or shuffle clause/sentence positions in a prompt.

Reorders sentences or comma-separated clauses to test whether the circuit is
sensitive to the absolute or relative position of information rather than its
presence. The answer token is unchanged; only the prompt structure is shuffled.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .base import CorruptionValidation


class PositionShiftCorruption:
    """Shuffle or rotate the sentence/clause order of a prompt.

    Splits on sentence boundaries (". ", "? ", "! ") and clause boundaries
    ("; ", ": ") then reorders.  Prompts too short to split are returned
    unchanged (validated as invalid so the record is skipped).

    Attributes:
        name: Strategy identifier, "position_shift".
        mode: "structure-altering" (changes word order, not semantics).
    """

    name = "position_shift"
    mode = "structure-altering"

    _SPLIT_PATTERN = re.compile(r"(?<=[.?!;:])\s+")

    def __init__(self, strategy: str = "shuffle", seed: Optional[int] = None):
        """
        Args:
            strategy: "shuffle" (random permutation) or "rotate" (cyclic shift by 1).
            seed: Optional fixed seed for the internal RNG.
        """
        if strategy not in ("shuffle", "rotate"):
            raise ValueError(f"strategy must be 'shuffle' or 'rotate', got {strategy!r}")
        self.strategy = strategy
        self._seed = seed

    def _split_segments(self, text: str) -> List[str]:
        parts = self._SPLIT_PATTERN.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _rejoin(self, segments: List[str]) -> str:
        return " ".join(segments)

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        if not prompt:
            return example

        segments = self._split_segments(prompt)
        if len(segments) < 2:
            return example  # nothing to shift; validate() will mark as invalid

        if self.strategy == "shuffle":
            shuffled = segments[:]
            rng.shuffle(shuffled)
            # avoid identity permutations on short lists
            attempts = 0
            while shuffled == segments and attempts < 10:
                rng.shuffle(shuffled)
                attempts += 1
            new_prompt = self._rejoin(shuffled)
        else:  # rotate
            new_prompt = self._rejoin(segments[1:] + [segments[0]])

        result = example.copy()
        result["prompt"] = new_prompt
        return result

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        cp = clean.get("prompt", "")
        pp = corrupted.get("prompt", "")
        if cp == pp:
            return CorruptionValidation(
                is_valid=False,
                reason="position_shift did not change the prompt (too short to split)",
                severity=0.0,
            )
        return CorruptionValidation(is_valid=True, reason=None, severity=0.3)
