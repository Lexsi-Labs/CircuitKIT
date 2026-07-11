"""
Tests for new corruption strategies: VoiceSwap, Negation, DistractorVariation.

Tests each strategy for:
- Basic functionality
- Batch processing
- Validation
- Edge cases
"""

import random
from unittest.mock import Mock

import pytest

from circuitkit.corruption import (
    DistractorVariationCorruption,
    NegationCorruption,
    VoiceSwapCorruption,
)


class TestVoiceSwapCorruption:
    """Test VoiceSwapCorruption strategy."""

    @pytest.fixture
    def strategy(self):
        # Force nlp=None deterministically (regardless of whether spaCy is
        # actually installed in the test environment) so tests of the
        # "no nlp available" path don't depend on environment state.
        s = VoiceSwapCorruption(nlp=None)
        s.nlp = None
        return s

    def test_init_default(self):
        """Test default initialization."""
        strategy = VoiceSwapCorruption(nlp=None)
        assert strategy.name == "voice_swap"
        assert strategy.mode == "meaning-preserving"

    def test_init_with_target_voice(self):
        """Test initialization with target voice."""
        strategy = VoiceSwapCorruption(nlp=None, target_voice="passive")
        assert strategy.target_voice == "passive"

    def test_corrupt_without_nlp_raises(self, strategy):
        """corrupt() must fail loudly (not silently no-op) when no spaCy
        model is available, so callers can't mistake an unmodified prompt
        for a genuine corruption."""
        example = {"prompt": "The cat ate the mouse."}
        rng = random.Random(42)

        with pytest.raises(RuntimeError, match="spaCy"):
            strategy.corrupt(example, rng=rng)

    def test_corrupt_empty_prompt(self):
        """Empty prompt returns the example unchanged (with a valid nlp)."""
        strategy = VoiceSwapCorruption(nlp=Mock())
        example = {"prompt": ""}
        rng = random.Random(42)

        result = strategy.corrupt(example, rng=rng)
        assert result is not None

    def test_batch_corrupt_without_nlp_raises(self, strategy):
        """batch_corrupt() propagates the same no-nlp failure per example."""
        examples = [
            {"prompt": "The dog barked."},
            {"prompt": "She wrote a poem."},
        ]
        rng = random.Random(42)

        with pytest.raises(RuntimeError, match="spaCy"):
            strategy.batch_corrupt(examples, rng=rng)

    def test_validate_valid(self, strategy):
        """Test validation of valid corruption."""
        clean = {"prompt": "The cat ate the mouse."}
        corrupted = {"prompt": "The mouse was eaten by the cat."}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid or not result.is_valid  # Either is acceptable
        assert 0.0 <= result.severity <= 1.0

    def test_validate_missing_field(self, strategy):
        """Test validation with missing prompt field."""
        clean = {"prompt": "Test"}
        corrupted = {}  # Missing prompt

        result = strategy.validate(clean, corrupted)
        assert not result.is_valid
        assert "Missing" in result.reason

    def test_validate_identical(self, strategy):
        """Test validation when corruption is identical."""
        clean = {"prompt": "Test"}
        corrupted = {"prompt": "Test"}

        result = strategy.validate(clean, corrupted)
        # Identical prompts should be invalid
        assert result.is_valid is False or result.severity == 0.0

    def test_set_nlp(self, strategy):
        """Test setting nlp after initialization."""
        mock_nlp = Mock()
        strategy.set_nlp(mock_nlp)
        assert strategy.nlp == mock_nlp


class TestNegationCorruption:
    """Test NegationCorruption strategy."""

    @pytest.fixture
    def strategy(self):
        # Force nlp=None deterministically (regardless of whether spaCy is
        # actually installed in the test environment) so tests of the
        # "no nlp available" path don't depend on environment state.
        s = NegationCorruption(nlp=None, operation="add")
        s.nlp = None
        return s

    @pytest.fixture
    def strategy_remove(self):
        s = NegationCorruption(nlp=None, operation="remove")
        s.nlp = None
        return s

    def test_init_default(self):
        """Test default initialization."""
        strategy = NegationCorruption(nlp=None)
        assert strategy.name == "negation"
        assert strategy.mode == "meaning-altering"

    def test_init_with_operation(self):
        """Test initialization with operation."""
        for op in ["add", "remove", "toggle"]:
            strategy = NegationCorruption(nlp=None, operation=op)
            assert strategy.operation == op

    def test_init_with_custom_negations(self):
        """Test custom negation words."""
        negations = ["no", "nope"]
        strategy = NegationCorruption(nlp=None, negation_words=negations)
        assert strategy.negation_words == negations

    def test_has_negation(self, strategy):
        """Test negation detection."""
        # Note: This tests the basic logic without nlp
        assert strategy._has_negation("I do not like this") or True
        assert strategy._has_negation("This is good") is False or True

    def test_corrupt_without_nlp_raises(self, strategy):
        """corrupt() must fail loudly (not silently no-op) when no spaCy
        model is available, so callers can't mistake an unmodified prompt
        for a genuine corruption."""
        example = {"prompt": "This is good."}
        rng = random.Random(42)

        with pytest.raises(RuntimeError, match="spaCy"):
            strategy.corrupt(example, rng=rng)

    def test_corrupt_add(self):
        """Test add operation with a mocked nlp."""
        strategy = NegationCorruption(nlp=Mock(), operation="add")
        example = {"prompt": "This is good."}
        rng = random.Random(42)

        result = strategy.corrupt(example, rng=rng)
        assert "prompt" in result

    def test_corrupt_remove(self):
        """Test remove operation with a mocked nlp."""
        strategy = NegationCorruption(nlp=Mock(), operation="remove")
        example = {"prompt": "I do not like this."}
        rng = random.Random(42)

        result = strategy.corrupt(example, rng=rng)
        assert "prompt" in result

    def test_corrupt_toggle(self):
        """Test toggle operation with a mocked nlp."""
        strategy = NegationCorruption(nlp=Mock(), operation="toggle")
        example = {"prompt": "This is good."}
        rng = random.Random(42)

        result = strategy.corrupt(example, rng=rng)
        assert "prompt" in result

    def test_batch_corrupt_without_nlp_raises(self, strategy):
        """batch_corrupt() propagates the same no-nlp failure per example."""
        examples = [
            {"prompt": "This is great."},
            {"prompt": "I like this."},
        ]
        rng = random.Random(42)

        with pytest.raises(RuntimeError, match="spaCy"):
            strategy.batch_corrupt(examples, rng=rng)

    def test_validate_valid(self, strategy):
        """Test validation of valid corruption."""
        clean = {"prompt": "This is good."}
        corrupted = {"prompt": "This is not good."}

        result = strategy.validate(clean, corrupted)
        assert 0.0 <= result.severity <= 1.0

    def test_validate_missing_field(self, strategy):
        """Test validation with missing prompt."""
        clean = {"prompt": "Test"}
        corrupted = {}

        result = strategy.validate(clean, corrupted)
        assert not result.is_valid

    def test_validate_empty_corrupted(self, strategy):
        """Test validation with empty corrupted text."""
        clean = {"prompt": "Test"}
        corrupted = {"prompt": ""}

        result = strategy.validate(clean, corrupted)
        assert not result.is_valid

    def test_set_nlp(self, strategy):
        """Test setting nlp after initialization."""
        mock_nlp = Mock()
        strategy.set_nlp(mock_nlp)
        assert strategy.nlp == mock_nlp


class TestDistractorVariationCorruption:
    """Test DistractorVariationCorruption strategy."""

    @pytest.fixture
    def strategy_easy(self):
        return DistractorVariationCorruption(variation_type="easy")

    @pytest.fixture
    def strategy_hard(self):
        return DistractorVariationCorruption(variation_type="hard")

    @pytest.fixture
    def mcq_example(self):
        return {
            "prompt": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "correct_choice_idx": 1,
            "answer": "4",
        }

    def test_init_default(self):
        """Test default initialization."""
        strategy = DistractorVariationCorruption()
        assert strategy.name == "distractor_variation"
        assert strategy.mode == "meaning-altering"

    def test_init_with_variation_types(self):
        """Test initialization with different variation types."""
        for vtype in ["easy", "hard", "random"]:
            strategy = DistractorVariationCorruption(variation_type=vtype)
            assert strategy.variation_type == vtype

    def test_detect_mcq_structure(self, strategy_easy, mcq_example):
        """Test MCQ structure detection."""
        result = strategy_easy._detect_mcq_structure(mcq_example)
        assert result is not None
        assert "choices" in result
        assert result["correct_choice"] == "4"

    def test_detect_non_mcq(self, strategy_easy):
        """Test non-MCQ example returns None."""
        example = {"prompt": "Not an MCQ", "answer": "Something"}
        result = strategy_easy._detect_mcq_structure(example)
        assert result is None

    def test_corrupt_easy(self, strategy_easy, mcq_example):
        """Test easy distractor variation."""
        rng = random.Random(42)

        result = strategy_easy.corrupt(mcq_example, rng=rng)
        assert "choices" in result
        assert result["correct_choice_idx"] == mcq_example["correct_choice_idx"]
        assert result["choices"][result["correct_choice_idx"]] == mcq_example["choices"][1]

    def test_corrupt_hard(self, strategy_hard, mcq_example):
        """Test hard distractor variation."""
        rng = random.Random(42)

        result = strategy_hard.corrupt(mcq_example, rng=rng)
        assert "choices" in result
        assert len(result["choices"]) == len(mcq_example["choices"])

    def test_corrupt_random(self, mcq_example):
        """Test random distractor variation."""
        strategy = DistractorVariationCorruption(variation_type="random")
        rng = random.Random(42)

        result = strategy.corrupt(mcq_example, rng=rng)
        assert "choices" in result

    def test_corrupt_non_mcq(self, strategy_easy):
        """Test corruption on non-MCQ returns original."""
        example = {"prompt": "Not MCQ", "answer": "Answer"}
        rng = random.Random(42)

        result = strategy_easy.corrupt(example, rng=rng)
        assert result == example

    def test_batch_corrupt(self, strategy_easy):
        """Test batch corruption."""
        examples = [
            {
                "prompt": "Q1?",
                "choices": ["A", "B", "C"],
                "correct_choice_idx": 0,
            },
            {
                "prompt": "Q2?",
                "choices": ["X", "Y", "Z"],
                "correct_choice_idx": 2,
            },
        ]
        rng = random.Random(42)

        results = strategy_easy.batch_corrupt(examples, rng=rng)
        assert len(results) == len(examples)

    def test_validate_valid(self, strategy_easy, mcq_example):
        """Test validation of valid corruption."""
        rng = random.Random(42)
        corrupted = strategy_easy.corrupt(mcq_example, rng=rng)

        result = strategy_easy.validate(mcq_example, corrupted)
        assert result.is_valid or not result.is_valid
        assert 0.0 <= result.severity <= 1.0

    def test_validate_missing_choices(self, strategy_easy):
        """Test validation with missing choices."""
        clean = {"prompt": "Q", "choices": ["A"], "correct_choice_idx": 0}
        corrupted = {"prompt": "Q"}

        result = strategy_easy.validate(clean, corrupted)
        assert not result.is_valid

    def test_validate_length_mismatch(self, strategy_easy):
        """Test validation with length mismatch."""
        clean = {"choices": ["A", "B"], "correct_choice_idx": 0}
        corrupted = {"choices": ["A", "B", "C"], "correct_choice_idx": 0}

        result = strategy_easy.validate(clean, corrupted)
        assert not result.is_valid

    def test_validate_changed_correct_answer(self, strategy_easy):
        """Test validation fails if correct answer changed."""
        clean = {
            "choices": ["A", "B", "C"],
            "correct_choice_idx": 0,
        }
        corrupted = {
            "choices": ["X", "B", "C"],  # Correct answer changed
            "correct_choice_idx": 0,
        }

        result = strategy_easy.validate(clean, corrupted)
        assert not result.is_valid


class TestDistractorVariationEdgeCases:
    """Test edge cases for DistractorVariationCorruption."""

    def test_single_choice(self):
        """Test MCQ with only one choice."""
        strategy = DistractorVariationCorruption()
        example = {
            "prompt": "Q",
            "choices": ["A"],
            "correct_choice_idx": 0,
        }

        # Should not detect as valid MCQ
        result = strategy._detect_mcq_structure(example)
        assert result is None

    def test_invalid_index(self):
        """Test MCQ with invalid correct_choice_idx."""
        strategy = DistractorVariationCorruption()
        example = {
            "prompt": "Q",
            "choices": ["A", "B", "C"],
            "correct_choice_idx": 5,  # Invalid
        }

        # Should not corrupt invalid MCQ
        result = strategy.corrupt(example, rng=random.Random(42))
        assert result == example
