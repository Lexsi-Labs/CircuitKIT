"""
Tests for DistractorInjectionCorruption strategy.

Tests verify:
1. Distractor injection at various positions
2. Corpus-based and task-based distractor selection
3. Validation of corrupted examples
4. MMLU-style MCQ distraction
5. Severity scoring
"""

import random
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from circuitkit.corruption.distractor import (  # noqa: E402 - import after intentional pre-import setup
    DistractorInjectionCorruption,
)


class TestDistractorInjectionBasic:
    """Test basic distractor injection functionality."""

    def test_initialization_with_corpus(self):
        """Test initialization with provided corpus."""
        corpus = [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius.",
            "The Earth orbits the Sun.",
        ]
        strategy = DistractorInjectionCorruption(
            position="after", distractor_source="corpus", corpus=corpus
        )

        assert strategy.name == "distractor_injection"
        assert strategy.mode == "meaning-altering"
        assert strategy.corpus == corpus
        assert strategy.position == "after"

    def test_initialization_default_corpus(self):
        """Test that default corpus is populated for corpus-based injection."""
        strategy = DistractorInjectionCorruption(distractor_source="corpus")
        assert len(strategy.corpus) > 0

    def test_corrupt_with_corpus_after(self):
        """Test distractor injection after prompt using corpus."""
        corpus = ["The Great Wall of China is over 13,000 miles long."]
        strategy = DistractorInjectionCorruption(
            position="after", distractor_source="corpus", corpus=corpus
        )

        example = {"prompt": "What is the capital of France?"}
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng)

        # Check structure
        assert "prompt" in corrupted
        assert "distractor_metadata" in corrupted

        # Check that prompt is longer
        assert len(corrupted["prompt"]) > len(example["prompt"])

        # Check that distractor is in the prompt
        assert "Great Wall" in corrupted["prompt"]

        # Check metadata
        assert corrupted["distractor_metadata"]["distractor_text"] == corpus[0]
        assert corrupted["distractor_metadata"]["distractor_position"] == "after"

    def test_corrupt_with_corpus_before(self):
        """Test distractor injection before prompt using corpus."""
        corpus = ["Mount Everest is the tallest mountain."]
        strategy = DistractorInjectionCorruption(
            position="before", distractor_source="corpus", corpus=corpus
        )

        example = {"prompt": "What is the capital of France?"}
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng)

        # Check structure
        assert "prompt" in corrupted
        assert corrupted["distractor_metadata"]["distractor_position"] == "before"

        # Check that distractor comes first
        assert "Mount Everest" in corrupted["prompt"]
        assert corrupted["prompt"].startswith("Additionally")

    def test_corrupt_missing_prompt(self):
        """Test that missing prompt raises error."""
        strategy = DistractorInjectionCorruption()
        example = {"question": "What is 2+2?"}  # No 'prompt' key
        rng = random.Random(42)

        with pytest.raises(ValueError, match="must contain 'prompt'"):
            strategy.corrupt(example, rng=rng)

    def test_corrupt_empty_prompt(self):
        """Test that empty prompt raises error."""
        strategy = DistractorInjectionCorruption()
        example = {"prompt": ""}
        rng = random.Random(42)

        with pytest.raises(ValueError, match="non-empty"):
            strategy.corrupt(example, rng=rng)


class TestDistractorSelection:
    """Test distractor selection from various sources."""

    def test_select_distractor_from_corpus(self):
        """Test that distractors are selected from corpus."""
        corpus = ["Distractor A", "Distractor B", "Distractor C"]
        strategy = DistractorInjectionCorruption(distractor_source="corpus", corpus=corpus)

        example = {"prompt": "Main question"}
        rng = random.Random(42)

        # Generate multiple corruptions and verify all come from corpus
        for _ in range(5):
            corrupted = strategy.corrupt(example, rng=rng)
            distractor = corrupted["distractor_metadata"]["distractor_text"]
            assert distractor in corpus

    def test_select_distractor_from_task_metadata(self):
        """Test distractor selection from task metadata."""
        strategy = DistractorInjectionCorruption(distractor_source="task")

        example = {"prompt": "What is 2+2?"}
        metadata = {"distractors": ["5", "6", "3"]}
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Verify distractor comes from metadata
        distractor = corrupted["distractor_metadata"]["distractor_text"]
        assert distractor in metadata["distractors"]

    def test_select_distractor_from_example_mcq(self):
        """Test distractor extraction from MCQ-style example."""
        strategy = DistractorInjectionCorruption(distractor_source="task")

        # MMLU-style example with options
        example = {
            "prompt": "Which is the capital of France?",
            "optionA": "Paris. It is in northern France.",
            "optionB": "Lyon",
            "optionC": "Marseille",
            "optionD": "Nice",
        }
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng)

        # Verify distractor was extracted from one of the options
        distractor = corrupted["distractor_metadata"]["distractor_text"]
        assert any(
            distractor.lower() in option.lower()
            for option in [
                example["optionA"],
                example["optionB"],
                example["optionC"],
                example["optionD"],
            ]
        )

    def test_fallback_to_default_corpus_when_empty(self):
        """Test that fallback corpus is used when task has no distractors."""
        strategy = DistractorInjectionCorruption(distractor_source="task")

        example = {"prompt": "Simple question"}  # No distractors in example
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng)

        # Should succeed because fallback corpus is available
        assert "distractor_metadata" in corrupted
        assert corrupted["distractor_metadata"]["distractor_text"] is not None


class TestDistractorValidation:
    """Test corruption validation."""

    def test_validate_valid_corruption(self):
        """Test validation of a valid corruption."""
        corpus = ["Test distractor."]
        strategy = DistractorInjectionCorruption(distractor_source="corpus", corpus=corpus)

        example = {"prompt": "Original prompt"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        validation = strategy.validate(example, corrupted)

        assert validation.is_valid
        assert validation.reason is None
        assert validation.severity > 0

    def test_validate_missing_prompt_in_clean(self):
        """Test validation fails if clean example lacks prompt."""
        strategy = DistractorInjectionCorruption()

        clean = {"question": "test"}  # No prompt
        corrupted = {"prompt": "test"}

        validation = strategy.validate(clean, corrupted)

        assert not validation.is_valid
        assert "prompt" in validation.reason.lower()

    def test_validate_missing_prompt_in_corrupted(self):
        """Test validation fails if corrupted example lacks prompt."""
        strategy = DistractorInjectionCorruption()

        clean = {"prompt": "test"}
        corrupted = {"question": "test"}  # No prompt

        validation = strategy.validate(clean, corrupted)

        assert not validation.is_valid

    def test_validate_missing_metadata(self):
        """Test validation fails if corrupted lacks distractor metadata."""
        strategy = DistractorInjectionCorruption()

        clean = {"prompt": "test"}
        corrupted = {"prompt": "test with stuff"}  # No distractor_metadata

        validation = strategy.validate(clean, corrupted)

        assert not validation.is_valid
        assert "metadata" in validation.reason.lower()

    def test_validate_corrupted_not_longer(self):
        """Test validation fails if corrupted is not longer than clean."""
        strategy = DistractorInjectionCorruption()

        clean = {"prompt": "original prompt"}
        corrupted = {
            "prompt": "short",
            "distractor_metadata": {"distractor_text": "test"},
        }

        validation = strategy.validate(clean, corrupted)

        assert not validation.is_valid
        assert "longer" in validation.reason.lower()

    def test_severity_scoring(self):
        """Test that severity scores increase with distractor length."""
        corpus_short = ["Hi"]
        corpus_long = ["This is a very long distractor sentence with many words"]

        strategy_short = DistractorInjectionCorruption(
            distractor_source="corpus", corpus=corpus_short
        )
        strategy_long = DistractorInjectionCorruption(
            distractor_source="corpus", corpus=corpus_long
        )

        example = {"prompt": "Test prompt"}
        rng = random.Random(42)

        corrupted_short = strategy_short.corrupt(example, rng=rng)
        corrupted_long = strategy_long.corrupt(example, rng=rng)

        val_short = strategy_short.validate(example, corrupted_short)
        val_long = strategy_long.validate(example, corrupted_long)

        # Longer distractor should have higher severity
        assert val_long.severity > val_short.severity


class TestDistractorBatchProcessing:
    """Test batch corruption processing."""

    def test_batch_corrupt(self):
        """Test batch corruption of multiple examples."""
        corpus = ["Distractor 1", "Distractor 2"]
        strategy = DistractorInjectionCorruption(distractor_source="corpus", corpus=corpus)

        examples = [
            {"prompt": "Question 1?"},
            {"prompt": "Question 2?"},
            {"prompt": "Question 3?"},
        ]
        rng = random.Random(42)

        corrupted_batch = strategy.batch_corrupt(examples, rng=rng)

        # Check results
        assert len(corrupted_batch) == len(examples)
        for orig, corr in zip(examples, corrupted_batch):
            assert len(corr["prompt"]) > len(orig["prompt"])
            assert "distractor_metadata" in corr

    def test_batch_corrupt_empty_list(self):
        """Test batch corruption with empty list."""
        strategy = DistractorInjectionCorruption()
        examples = []
        rng = random.Random(42)

        corrupted_batch = strategy.batch_corrupt(examples, rng=rng)

        assert corrupted_batch == []


class TestMMluDistractorScenario:
    """Test on MMLU-style MCQ scenarios."""

    def test_mmlu_distractor_variant(self):
        """Test distractor injection on MMLU-style example."""
        # MMLU-style example
        mmlu_example = {
            "prompt": "What is the capital of France?",
            "optionA": "Paris",
            "optionB": "Lyon",
            "optionC": "Marseille",
            "optionD": "Nice",
            "answer": "A",
        }

        strategy = DistractorInjectionCorruption(position="after", distractor_source="task")
        rng = random.Random(42)

        corrupted = strategy.corrupt(mmlu_example, rng=rng)

        # Verify corruption structure
        assert "prompt" in corrupted
        assert corrupted["answer"] == mmlu_example["answer"]  # Answer unchanged
        assert "distractor_metadata" in corrupted

        # The prompt should be longer and contain both original + distractor
        assert len(corrupted["prompt"]) > len(mmlu_example["prompt"])

        # Options should be preserved
        for option in ["optionA", "optionB", "optionC", "optionD"]:
            assert corrupted[option] == mmlu_example[option]

    def test_mmlu_distractor_does_not_change_answer(self):
        """Verify distractor injection doesn't change ground truth answer."""
        mmlu_example = {
            "prompt": "What is 2 + 2?",
            "optionA": "4",
            "optionB": "5",
            "optionC": "3",
            "optionD": "6",
            "answer": "A",
            "correct_idx": 0,
        }

        strategy = DistractorInjectionCorruption(
            distractor_source="corpus",
            corpus=["The speed of light is 299,792 km/s"],
        )
        rng = random.Random(42)

        corrupted = strategy.corrupt(mmlu_example, rng=rng)

        # Answer should remain the same
        assert corrupted["answer"] == "A"
        assert corrupted["correct_idx"] == 0

        # Options should be unchanged
        assert corrupted["optionA"] == "4"

    def test_multiple_mmlu_examples_with_different_distractors(self):
        """Test multiple MMLU examples get different random distractors."""
        corpus = [
            "Einstein developed the theory of relativity.",
            "The Earth rotates around the Sun.",
            "Photosynthesis is how plants make food.",
        ]

        strategy = DistractorInjectionCorruption(distractor_source="corpus", corpus=corpus)

        examples = [
            {"prompt": "What is the capital of France?"},
            {"prompt": "What is the largest planet?"},
            {"prompt": "Who painted the Mona Lisa?"},
        ]
        rng = random.Random(42)

        corrupted_examples = [strategy.corrupt(ex, rng=rng) for ex in examples]

        # Each should have a distractor
        distractors = [ex["distractor_metadata"]["distractor_text"] for ex in corrupted_examples]

        # All should be from corpus
        for dist in distractors:
            assert dist in corpus


class TestDistractorConnectionPhrase:
    """Test custom connection phrases."""

    def test_custom_connection_phrase(self):
        """Test that custom connection phrase is used."""
        strategy = DistractorInjectionCorruption(
            position="after",
            distractor_source="corpus",
            corpus=["Test distractor"],
            connection_phrase="Note that ",
        )

        example = {"prompt": "Main question"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        # Check that custom phrase appears in output
        assert "Note that" in corrupted["prompt"]
        # Default phrase should not appear
        assert "Additionally" not in corrupted["prompt"]

    def test_default_connection_phrase(self):
        """Test that default connection phrase is used when not specified."""
        strategy = DistractorInjectionCorruption()

        example = {"prompt": "Main question"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        # Default phrase should appear
        assert "Additionally" in corrupted["prompt"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_prompt_with_special_characters(self):
        """Test corruption of prompts with special characters."""
        strategy = DistractorInjectionCorruption(
            distractor_source="corpus",
            corpus=["Test distractor"],
        )

        example = {"prompt": "What is 2+2? (Math question)"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        assert "prompt" in corrupted
        assert len(corrupted["prompt"]) > len(example["prompt"])

    def test_very_long_prompt(self):
        """Test corruption of very long prompts."""
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 100

        strategy = DistractorInjectionCorruption(
            distractor_source="corpus",
            corpus=["Short distractor"],
        )

        example = {"prompt": long_prompt}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        assert len(corrupted["prompt"]) > len(example["prompt"])

    def test_very_short_prompt(self):
        """Test corruption of very short prompts."""
        strategy = DistractorInjectionCorruption(
            distractor_source="corpus",
            corpus=["Distractor"],
        )

        example = {"prompt": "Q?"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        assert len(corrupted["prompt"]) > len(example["prompt"])

    def test_corpus_empty_error(self):
        """Test that using corpus-based with empty corpus raises error."""
        strategy = DistractorInjectionCorruption(distractor_source="corpus", corpus=[])

        example = {"prompt": "Question"}
        rng = random.Random(42)

        with pytest.raises(ValueError, match="corpus is empty"):
            strategy.corrupt(example, rng=rng)

    def test_unicode_in_prompt_and_distractor(self):
        """Test handling of unicode characters."""
        strategy = DistractorInjectionCorruption(
            distractor_source="corpus",
            corpus=["L'électricité est une forme d'énergie. 你好世界"],
        )

        example = {"prompt": "Question avec des caractères spéciaux"}
        corrupted = strategy.corrupt(example, rng=random.Random(42))

        assert "prompt" in corrupted
        assert len(corrupted["prompt"]) > len(example["prompt"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
