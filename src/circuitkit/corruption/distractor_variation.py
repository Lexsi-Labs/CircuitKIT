"""
DistractorVariationCorruption: Vary distractor quality in multiple-choice questions.

Implements a meaning-altering corruption strategy for MCQ tasks that modifies
distractor (wrong answer) options. Can make distractors easier or harder while
keeping the correct answer unchanged.

Supported variations:
- Easy: Replace distractors with obviously wrong answers
- Hard: Replace distractors with plausible but incorrect answers
- Random: Randomly vary distractor difficulty
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class DistractorVariationCorruption:
    """Vary distractor quality in MCQ questions.

    Attributes:
        name: Strategy identifier, "distractor_variation".
        mode: "meaning-altering" (changes MCQ difficulty).
    """

    name = "distractor_variation"
    mode: Literal["meaning-altering"] = "meaning-altering"

    def __init__(
        self,
        variation_type: str = "random",
        distractor_pool: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize DistractorVariationCorruption.

        Args:
            variation_type: Type of variation ("easy", "hard", "random").
                           "easy": Replace with obviously wrong answers
                           "hard": Replace with plausible answers
                           "random": Randomly toggle between easy/hard
            distractor_pool: Optional dict mapping question contexts to distractor lists.
                            If None, uses hardcoded generic distractors.
        """
        self.variation_type = variation_type
        self.distractor_pool = distractor_pool or self._get_default_distractors()

    def _get_default_distractors(self) -> Dict[str, List[str]]:
        """Get default distractor pools for common question types.

        Returns:
            Dict mapping context keys to lists of distractor options
        """
        return {
            "factual": {
                "easy": ["unknown", "impossible", "not applicable", "not available"],
                "hard": ["similar option A", "similar option B", "related answer"],
            },
            "math": {
                "easy": ["0", "-1", "infinity", "undefined"],
                "hard": ["close correct answer", "off by one", "related value"],
            },
            "multiple_choice": {
                "easy": ["None", "All", "Invalid", "Unknown"],
                "hard": ["Plausible wrong option 1", "Plausible wrong option 2"],
            },
        }

    def _detect_mcq_structure(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect if example is an MCQ and extract structure.

        Args:
            example: Example dict potentially containing MCQ structure

        Returns:
            Dict with MCQ structure or None if not an MCQ
        """
        # Look for choices field
        if "choices" not in example:
            return None

        choices = example.get("choices", [])
        if not isinstance(choices, list) or len(choices) < 2:
            return None

        correct_choice_idx = example.get("correct_choice_idx")
        if correct_choice_idx is None:
            return None

        if correct_choice_idx is None or not (0 <= correct_choice_idx < len(choices)):
            return None

        return {
            "choices": choices,
            "correct_idx": correct_choice_idx,
            "correct_choice": choices[correct_choice_idx],
            "distractor_indices": [i for i in range(len(choices)) if i != correct_choice_idx],
        }

    def _get_easy_distractors(self, num_needed: int, context: str = "multiple_choice", rng: Optional[random.Random] = None) -> List[str]:
        """Get obviously wrong distractor options.

        Args:
            num_needed: Number of distractor options needed
            context: Question context type (factual, math, multiple_choice)
            rng: Random number generator for reproducibility

        Returns:
            List of easy (obviously wrong) distractor options
        """
        easy_options = self.distractor_pool.get(context, {}).get(
            "easy", ["False", "Unknown", "None", "Invalid", "Error"]
        )

        # Return random selection
        choice = rng.choice if rng is not None else random.choice
        result = []
        for _ in range(num_needed):
            if easy_options:
                result.append(choice(easy_options))

        return result if result else ["Wrong answer"] * num_needed

    def _get_hard_distractors(self, num_needed: int, context: str = "multiple_choice", rng: Optional[random.Random] = None) -> List[str]:
        """Get plausible but incorrect distractor options.

        Args:
            num_needed: Number of distractor options needed
            context: Question context type
            rng: Random number generator for reproducibility

        Returns:
            List of hard (plausible) distractor options
        """
        hard_options = self.distractor_pool.get(context, {}).get(
            "hard",
            [
                "Partially correct",
                "Related but wrong",
                "Common mistake",
                "Opposite of correct",
                "Similar concept",
            ],
        )

        choice = rng.choice if rng is not None else random.choice
        result = []
        for _ in range(num_needed):
            if hard_options:
                result.append(choice(hard_options))

        return result if result else ["Plausible wrong answer"] * num_needed

    def _create_varied_choices(
        self,
        mcq_structure: Dict[str, Any],
        variation: str,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Create varied choices for MCQ.

        Args:
            mcq_structure: MCQ structure dict
            variation: Variation type ("easy" or "hard")
            rng: Random number generator for reproducibility

        Returns:
            New choice list with varied distractors
        """
        choices = mcq_structure["choices"]
        correct_idx = mcq_structure["correct_idx"]
        correct_choice = mcq_structure["correct_choice"]
        distractor_indices = mcq_structure["distractor_indices"]

        new_choices = [None] * len(choices)
        new_choices[correct_idx] = correct_choice

        # Generate replacement distractors
        num_distractors = len(distractor_indices)

        if variation == "easy":
            distractors = self._get_easy_distractors(num_distractors, rng=rng)
        else:  # hard
            distractors = self._get_hard_distractors(num_distractors, rng=rng)

        # Place distractors in non-correct positions
        for i, distractor_idx in enumerate(distractor_indices):
            if i < len(distractors):
                new_choices[distractor_idx] = distractors[i]

        # Fill any remaining None with generic wrong answers
        new_choices = [choice if choice is not None else "Wrong answer" for choice in new_choices]

        return new_choices

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt example by varying MCQ distractors.

        Args:
            example: Dict potentially containing MCQ structure
            rng: Random number generator for reproducibility.
            metadata: Optional task-specific metadata.

        Returns:
            Corrupted example with varied distractors, or original if not MCQ
        """
        # Check if this is an MCQ
        mcq_structure = self._detect_mcq_structure(example)
        if mcq_structure is None:
            return example

        # Determine variation
        variation = self.variation_type
        if variation == "random":
            variation = rng.choice(["easy", "hard"])

        # Create varied choices
        new_choices = self._create_varied_choices(mcq_structure, variation, rng=rng)

        # Build corrupted example
        result = example.copy()
        result["choices"] = new_choices
        result["_distractor_variation"] = variation

        return result

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples.

        Args:
            examples: List of example dictionaries.
            rng: Random number generator for reproducibility.
            metadata: Task-specific metadata.

        Returns:
            List of corrupted examples.
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate distractor variation corruption.

        Args:
            clean: Original example.
            corrupted: Corrupted example.

        Returns:
            CorruptionValidation result.
        """
        # Check MCQ structure exists
        if "choices" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Missing 'choices' field (not an MCQ)", severity=0.0
            )

        clean_choices = clean.get("choices", [])
        corrupted_choices = corrupted.get("choices", [])

        # Ensure choice lists have same length
        if len(clean_choices) != len(corrupted_choices):
            return CorruptionValidation(
                is_valid=False, reason="Choice list length mismatch after corruption", severity=1.0
            )

        # Ensure correct choice index is still valid
        correct_idx = corrupted.get("correct_choice_idx")
        if correct_idx is None or correct_idx >= len(corrupted_choices):
            return CorruptionValidation(
                is_valid=False, reason="Correct choice index invalid after corruption", severity=1.0
            )

        # Ensure correct choice is unchanged
        clean_correct = clean_choices[clean.get("correct_choice_idx", 0)]
        corrupted_correct = corrupted_choices[correct_idx]

        if clean_correct != corrupted_correct:
            return CorruptionValidation(
                is_valid=False, reason="Correct choice was modified during corruption", severity=1.0
            )

        # Check that at least one distractor changed
        changed_distractors = 0
        for i in range(len(clean_choices)):
            if i != correct_idx and clean_choices[i] != corrupted_choices[i]:
                changed_distractors += 1

        if changed_distractors == 0:
            return CorruptionValidation(
                is_valid=False, reason="No distractors were modified", severity=0.0
            )

        # Calculate severity based on fraction of distractors changed
        num_distractors = len(clean_choices) - 1
        severity = min(1.0, changed_distractors / num_distractors if num_distractors > 0 else 0.5)

        return CorruptionValidation(is_valid=True, reason=None, severity=severity)
