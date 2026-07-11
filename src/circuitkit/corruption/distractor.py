"""
DistractorInjectionCorruption Strategy

Injects irrelevant-but-plausible distractor sentences into prompts for QA/MCQ/long-context
tasks to evaluate model robustness. The distractor is semantically plausible but unrelated
to the main context, increasing task difficulty without changing the correct answer.

Corruption Mode: meaning-altering (adds semantic noise/diversion)
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class DistractorInjectionCorruption:
    """Injects irrelevant-but-plausible distractors into task prompts.

    This strategy is designed for QA, MCQ, and long-context tasks where:
    - The task has a main context/question and answer options
    - We want to inject semantically plausible but unrelated sentences
    - The correct answer should remain unchanged despite the distraction

    The distractor is inserted before or after the main context, making the task
    harder but not changing ground truth.
    """

    name = "distractor_injection"
    mode = "meaning-altering"

    def __init__(
        self,
        position: Literal["before", "after"] = "after",
        distractor_source: Literal["task", "corpus"] = "task",
        corpus: Optional[List[str]] = None,
        connection_phrase: str = "Additionally, ",
    ):
        """Initialize DistractorInjectionCorruption.

        Args:
            position: Where to inject distractor relative to main context.
                     "before" = prepend to prompt, "after" = append to prompt.
            distractor_source: Where distractors come from.
                             "task" = sample from example dict keys/metadata
                             "corpus" = use provided corpus list
            corpus: Optional list of distractor sentences for corpus-based injection.
                   If distractor_source="corpus" and corpus is None, will raise error
                   at corruption time.
            connection_phrase: Phrase to connect distractor to context (e.g., "Additionally, ")
        """
        self.position = position
        self.distractor_source = distractor_source
        self.corpus = corpus or []
        self.connection_phrase = connection_phrase

        # If None is passed, use the default corpus.
        # If [] is passed, preserve it to trigger the empty corpus error.
        if corpus is None:
            self.corpus = [
                "The Great Wall of China is over 13,000 miles long.",
                "Mount Everest is the tallest mountain in the world.",
                "The Amazon rainforest produces about 20% of the world's oxygen.",
                "The human brain contains approximately 86 billion neurons.",
                "The speed of light is approximately 299,792 kilometers per second.",
                "Paris is the capital of France.",
                "The Titanic sank in 1912.",
                "The human heart beats about 100,000 times per day.",
                "The Pacific Ocean is the largest ocean on Earth.",
                "Leonardo da Vinci painted the Mona Lisa.",
            ]
        else:
            self.corpus = corpus

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt an example by injecting a distractor sentence.

        Args:
            example: Clean example dictionary with 'prompt' key.
            rng: Random number generator for reproducibility.
            metadata: Optional metadata containing task-specific info.
                     May contain 'distractors' key with list of distractor sentences.

        Returns:
            Corrupted example with distractor injected into prompt.
            Includes metadata about injection: 'distractor_text', 'distractor_position'

        Raises:
            ValueError: If prompt is missing, corpus is empty, or distractor can't be generated.
        """
        if "prompt" not in example:
            raise ValueError("Example must contain 'prompt' key")

        prompt = example["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        # Select distractor based on source
        distractor = self._select_distractor(example, metadata, rng)

        if not distractor or not isinstance(distractor, str):
            raise ValueError("Could not generate valid distractor sentence")

        # Inject distractor at specified position
        if self.position == "before":
            corrupted_prompt = f"{self.connection_phrase}{distractor} {prompt}"
        else:  # after
            corrupted_prompt = f"{prompt} {self.connection_phrase}{distractor}"

        # Create corrupted example with metadata
        corrupted_example = example.copy()
        corrupted_example["prompt"] = corrupted_prompt

        # Add distractor metadata
        if "distractor_metadata" not in corrupted_example:
            corrupted_example["distractor_metadata"] = {}

        corrupted_example["distractor_metadata"]["distractor_text"] = distractor
        corrupted_example["distractor_metadata"]["distractor_position"] = self.position
        corrupted_example["distractor_metadata"]["connection_phrase"] = self.connection_phrase

        return corrupted_example

    def _select_distractor(
        self,
        example: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        rng: random.Random,
    ) -> str:
        """Select a distractor sentence from the appropriate source.

        Args:
            example: The example being corrupted.
            metadata: Optional task metadata.
            rng: Random number generator.

        Returns:
            Selected distractor sentence.

        Raises:
            ValueError: If no distractor can be selected.
        """
        if self.distractor_source == "corpus":
            if not self.corpus:
                raise ValueError(
                    "distractor_source='corpus' but corpus is empty. "
                    "Provide corpus in constructor or set distractor_source='task'."
                )
            return rng.choice(self.corpus)

        elif self.distractor_source == "task":
            # Try to get distractors from metadata or example dict
            distractors = None

            # Check metadata first
            if metadata and "distractors" in metadata:
                distractors = metadata["distractors"]

            # Fallback: check example dict for candidate distractors
            if not distractors:
                distractors = self._extract_distractors_from_example(example)

            # If still no distractors, use fallback corpus
            if not distractors:
                distractors = self.corpus

            if not distractors:
                raise ValueError(
                    "Cannot select distractor: no distractors in metadata, "
                    "example, or fallback corpus"
                )

            return rng.choice(distractors)

        else:
            raise ValueError(f"Unknown distractor_source: {self.distractor_source}")

    def _extract_distractors_from_example(self, example: Dict[str, Any]) -> List[str]:
        """Extract potential distractors from example dict.

        For MMLU-style tasks with options A/B/C/D, we can use option texts.
        For other tasks, we look for common keys like 'context', 'background', etc.

        Args:
            example: The example dictionary.

        Returns:
            List of potential distractor sentences.
        """
        distractors = []

        # For MCQ tasks with options
        for key in [
            "optionA",
            "optionB",
            "optionC",
            "optionD",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
        ]:
            if key in example and isinstance(example[key], str):
                # Take first sentence or phrase from option
                text = example[key].strip()
                if text:
                    # Extract first sentence (up to period)
                    sentence = text.split(".")[0].strip()
                    if sentence:
                        distractors.append(sentence)

        # For tasks with additional context/background
        for key in ["context", "background", "passage", "document"]:
            if key in example and isinstance(example[key], str):
                text = example[key].strip()
                if text:
                    # Extract first sentence
                    sentences = text.split(".")
                    for sent in sentences:
                        sent = sent.strip()
                        if sent and len(sent) > 10:  # Non-trivial sentence
                            distractors.append(sent)
                            break

        return distractors

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples.

        Default implementation calls corrupt() on each example.

        Args:
            examples: List of clean examples.
            rng: Random number generator.
            metadata: Optional metadata.

        Returns:
            List of corrupted examples in same order as input.
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate corruption is well-formed.

        Checks:
        1. Both examples have 'prompt' key.
        2. Prompts are non-empty strings.
        3. Corrupted prompt is longer (contains additional distractor).
        4. Corrupted example has distractor metadata.
        5. Severity based on character length change.

        Args:
            clean: Original clean example.
            corrupted: Corrupted example from corrupt().

        Returns:
            CorruptionValidation with validity and severity score.
        """
        # Check required fields
        if "prompt" not in clean or "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Missing 'prompt' key in clean or corrupted example"
            )

        clean_prompt = clean.get("prompt", "")
        corrupted_prompt = corrupted.get("prompt", "")

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        if not clean_prompt.strip() or not corrupted_prompt.strip():
            return CorruptionValidation(is_valid=False, reason="Prompts must be non-empty")

        # Check that corrupted is actually longer (contains distractor)
        if len(corrupted_prompt) <= len(clean_prompt):
            return CorruptionValidation(
                is_valid=False, reason="Corrupted prompt is not longer than clean prompt"
            )

        # Check for distractor metadata
        has_metadata = "distractor_metadata" in corrupted and isinstance(
            corrupted.get("distractor_metadata"), dict
        )

        if not has_metadata:
            return CorruptionValidation(
                is_valid=False, reason="Missing distractor_metadata in corrupted example"
            )

        # Verify distractor was inserted
        distractor_text = corrupted["distractor_metadata"].get("distractor_text", "")
        if not distractor_text or distractor_text not in corrupted_prompt:
            return CorruptionValidation(
                is_valid=False, reason="Distractor text not found in corrupted prompt"
            )

        # Compute severity as proportion of added characters
        clean_len = len(clean_prompt)
        corrupted_len = len(corrupted_prompt)
        added_len = corrupted_len - clean_len
        severity = added_len / max(corrupted_len, 1)

        return CorruptionValidation(is_valid=True, reason=None, severity=severity)
