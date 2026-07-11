"""
ColorSwapCorruption Strategy for Colors Task (M7.0.4)

Implements a simple corruption strategy that swaps the correct answer color
with a plausible alternative color. This maintains sentence structure while
changing the semantic meaning (meaning-altering corruption).

Example:
    Clean:     "The sky is blue"
    Corrupted: "The sky is green"
    Answer:    blue → green
"""

import random
from typing import Any, Dict, List, Optional

from .base import CorruptionStrategy, CorruptionValidation


class ColorSwapCorruption(CorruptionStrategy):
    """
    Swaps the correct answer color with a plausible alternative.

    Attributes:
        name: "color_swap"
        mode: "meaning-altering" (changes semantic content)

    Color pool: {blue, green, red, white, black}

    For each example, randomly selects a different color from the pool.
    """

    name = "color_swap"
    mode = "meaning-altering"

    # Define the color pool
    COLOR_POOL = ["blue", "green", "red", "white", "black"]

    def __init__(self):
        """Initialize the corruption strategy."""
        super().__init__()

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Corrupt a single example by swapping the answer color.

        Args:
            example: Dictionary with 'prompt' and 'answer' keys.
                Example: {'prompt': 'The sky is', 'answer': 'blue'}
            rng: Random number generator for reproducibility
            metadata: Optional metadata (not used in this strategy)

        Returns:
            Corrupted example with answer swapped to a different color.
        """
        corrupted = example.copy()

        # Get the correct answer (original color)
        correct_answer = example.get("answer", "").lower().strip()

        # Validate that the answer is a color in our pool
        if correct_answer not in self.COLOR_POOL:
            # If answer not in pool, just return unchanged (shouldn't happen)
            # but keep validation consistent
            return corrupted

        # Find available colors (all except the correct answer)
        available_colors = [c for c in self.COLOR_POOL if c != correct_answer]

        # Pick a random incorrect color
        incorrect_color = rng.choice(available_colors)

        # Corrupt the answer
        corrupted["answer"] = incorrect_color

        return corrupted

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Corrupt a batch of examples.

        Default implementation calls corrupt() on each example.

        Args:
            examples: List of example dictionaries
            rng: Random number generator
            metadata: Optional metadata

        Returns:
            List of corrupted examples
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """
        Validate that the corruption is well-formed.

        Checks:
        1. Both examples have 'prompt' and 'answer' keys
        2. Answer is a valid color from the pool
        3. Answers are different (corruption changed something)
        4. Prompt is preserved

        Args:
            clean: Original example
            corrupted: Corrupted example

        Returns:
            CorruptionValidation with validity and severity
        """
        # Check required fields exist
        if "prompt" not in clean or "answer" not in clean:
            return CorruptionValidation(
                is_valid=False, reason="Clean example missing 'prompt' or 'answer' field"
            )

        if "prompt" not in corrupted or "answer" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' or 'answer' field"
            )

        clean_answer = clean["answer"].lower().strip()
        corrupted_answer = corrupted["answer"].lower().strip()

        # Check that answers are valid colors
        if clean_answer not in self.COLOR_POOL:
            return CorruptionValidation(
                is_valid=False,
                reason=f"Clean answer '{clean_answer}' not in color pool: {self.COLOR_POOL}",
            )

        if corrupted_answer not in self.COLOR_POOL:
            return CorruptionValidation(
                is_valid=False,
                reason=f"Corrupted answer '{corrupted_answer}' not in color pool: {self.COLOR_POOL}",
            )

        # Check that corruption actually changed the answer
        if clean_answer == corrupted_answer:
            return CorruptionValidation(
                is_valid=False, reason="Corruption did not change the answer"
            )

        # Check that prompt is preserved
        if clean["prompt"] != corrupted["prompt"]:
            return CorruptionValidation(
                is_valid=False, reason="Prompt was modified during corruption"
            )

        # Compute severity as a simple metric: always high since answer is completely different
        # In practice, could use semantic similarity or edit distance
        severity = 1.0  # Complete color swap = maximum severity

        return CorruptionValidation(is_valid=True, severity=severity)
