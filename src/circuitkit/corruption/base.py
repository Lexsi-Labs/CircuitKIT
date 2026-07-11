"""
CorruptionStrategy Protocol Definition

Defines the interface that all corruption strategies must implement.
Corruption strategies systematically modify clean examples to create corrupted
versions for behavioral analysis, circuit discovery, and robustness evaluation.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol


@dataclass
class CorruptionValidation:
    """Result of validating a corruption.

    Attributes:
        is_valid: Whether the corruption is well-formed and usable.
        reason: Optional explanation if invalid (e.g., "prompt too short").
        severity: Severity score in [0.0, 1.0] where higher = more severe change.
                 Can be used for weighted analysis or filtering.
    """

    is_valid: bool
    reason: Optional[str] = None
    severity: float = 0.0


class CorruptionStrategy(Protocol):
    """Protocol that all corruption strategies must implement.

    Corruption strategies transform clean examples into corrupted versions
    while maintaining or altering semantic meaning depending on mode.

    Implementations must support three modes:
    - "meaning-preserving": Corruption changes surface form, not semantic content.
    - "meaning-altering": Corruption changes semantic content (e.g., entity swap).
    - "role-swap": Special case where roles/relationships are swapped.

    Attributes:
        name: Canonical strategy name, e.g., "entity_swap", "paraphrase".
        mode: One of "meaning-preserving", "meaning-altering", "role-swap".
    """

    name: str
    mode: Literal["meaning-preserving", "meaning-altering", "role-swap"]

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt a single example.

        Args:
            example: Clean example dictionary. Expected to contain keys like
                    'prompt' and 'answer', but structure is task-specific.
            rng: Random number generator for reproducibility. Use this instead
                of calling random.* directly to ensure deterministic behavior
                across runs.
            metadata: Optional task-specific metadata such as entity pools,
                     POS taggers, token vocabularies, etc. Format is defined
                     by the task consuming this corruption.

        Returns:
            Corrupted example dictionary with the same structure and keys as
            the input example, with selected fields modified.

        Example (EntitySwap strategy that swaps two random entities):
            Accepts metadata["entities"] as entity pool, samples two random
            entities, swaps them in prompt using rng for reproducibility.
        """
        ...

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples (optional optimization).

        Default implementation calls corrupt() on each example in sequence.
        Strategies may override for vectorized or parallel processing.

        Args:
            examples: List of clean example dictionaries.
            rng: Random number generator for reproducibility.
            metadata: Task-specific metadata (same as corrupt()).

        Returns:
            List of corrupted examples in the same order as input.

        Example (Paraphrase strategy with batch optimization):
            Default calls corrupt() sequentially. Implementations may override
            for vectorized inference (e.g., batch LLM paraphrase) by extracting
            prompts, processing in parallel, and reassembling examples.
        """
        ...

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate that corruption is well-formed.

        This method checks that the corrupted example:
        1. Contains all required fields from the clean example.
        2. Does not introduce invalid data types or NaN values.
        3. Respects strategy-specific constraints (e.g., minimum length).

        Args:
            clean: Original clean example.
            corrupted: Result of calling corrupt() on the clean example.

        Returns:
            CorruptionValidation with:
            - is_valid: True if corruption is usable.
            - reason: Explanation of any validation failure.
            - severity: Severity of the corruption on [0.0, 1.0].

        Example:
            Checks that required fields exist, enforces minimum constraints
            (e.g., prompt length), computes severity as character-level Levenshtein
            distance or other similarity metric between clean and corrupted.
        """
        ...
