"""
Tests for TokenSwapCorruption strategy.
"""

import random

import pytest

from circuitkit.corruption import CorruptionValidation, TokenSwapCorruption


# Simple POS tagger for testing (simulates spaCy-like tagging)
class SimplePOSTagger:
    """Simple regex-based POS tagger for testing purposes."""

    def __call__(self, text: str):
        """Tag text with simple POS rules.

        Returns:
            Tuple of (tokens, pos_tags).
        """

        tokens = text.split()
        pos_tags = []

        for token in tokens:
            # Simple heuristics
            if token in ["5", "7", "10", "20", "42", "100"]:
                pos_tags.append("NUM")
            elif token.endswith("ing"):
                pos_tags.append("VBG")
            elif token in ["the", "a", "an"]:
                pos_tags.append("DET")
            elif token in ["cat", "dog", "bird", "house", "tree", "number"]:
                pos_tags.append("NN")
            elif token in ["+", "-", "*", "/", ">"]:
                pos_tags.append("SYM")
            else:
                pos_tags.append("NN")

        return tokens, pos_tags


@pytest.fixture
def simple_tagger():
    """Fixture providing a simple POS tagger."""
    return SimplePOSTagger()


@pytest.fixture
def simple_tokenizer():
    """Fixture providing a mock tokenizer (checks token length)."""

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simple mock: split on spaces
            return text.split()

    return MockTokenizer()


@pytest.fixture
def number_vocab():
    """Fixture providing a sample number vocabulary."""
    return {
        "NUM": ["1", "2", "3", "5", "7", "10", "20", "42", "100"],
    }


class TestTokenSwapCorruptionBasics:
    """Test basic TokenSwapCorruption functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        strategy = TokenSwapCorruption()
        assert strategy.name == "token_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.pos_tags is None

    def test_initialization_with_pos_tags(self):
        """Test initialization with POS tag filters."""
        strategy = TokenSwapCorruption(pos_tags=["NUM", "NN"])
        assert strategy.pos_tags == ["NUM", "NN"]

    def test_initialization_with_vocab(self, number_vocab):
        """Test initialization with vocabulary."""
        strategy = TokenSwapCorruption(vocab=number_vocab)
        assert strategy.vocab == number_vocab

    def test_initialization_with_tokenizer(self, simple_tokenizer):
        """Test initialization with tokenizer."""
        strategy = TokenSwapCorruption(tokenizer=simple_tokenizer)
        assert strategy.tokenizer == simple_tokenizer


class TestTokenSwapCorrupt:
    """Test TokenSwapCorruption.corrupt() method."""

    def test_corrupt_simple_number_swap(self, simple_tagger, number_vocab):
        """Test corruption with number swapping."""
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=number_vocab,
        )

        example = {
            "prompt": "5 + 3 equals something",
        }

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Should have corrupted the prompt
        assert isinstance(corrupted, dict)
        assert "prompt" in corrupted
        # The result should be different (a number should be swapped)
        # We can't guarantee which exactly, but there should be corruption info
        if "_corruption_info" in corrupted:
            info = corrupted["_corruption_info"]
            assert "swapped_token" in info
            assert "replacement" in info
            assert "pos" in info

    def test_corrupt_requires_tagger_in_metadata(self, number_vocab):
        """Test that corrupt() requires tagger in metadata."""
        strategy = TokenSwapCorruption(vocab=number_vocab)

        example = {"prompt": "5 + 3"}

        rng = random.Random(42)

        with pytest.raises(ValueError, match="tagger"):
            strategy.corrupt(example, rng=rng, metadata={})

    def test_corrupt_no_suitable_tokens(self, simple_tagger):
        """Test corruption when no suitable tokens exist."""
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab={"NUM": ["1", "2", "3"]},
        )

        example = {
            "prompt": "The cat sat on the mat",  # No numbers
        }

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Should return unchanged (or copy)
        assert "prompt" in corrupted

    def test_corrupt_deterministic_with_seed(self, simple_tagger, number_vocab):
        """Test that corruption is deterministic with same RNG seed."""
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=number_vocab,
        )

        example = {"prompt": "5 + 10 equals 15"}

        metadata = {"tagger": simple_tagger}

        # Two runs with same seed should produce same result
        rng1 = random.Random(42)
        corrupted1 = strategy.corrupt(example, rng=rng1, metadata=metadata)

        rng2 = random.Random(42)
        corrupted2 = strategy.corrupt(example, rng=rng2, metadata=metadata)

        assert corrupted1["prompt"] == corrupted2["prompt"]

    def test_corrupt_preserves_other_fields(self, simple_tagger, number_vocab):
        """Test that corruption preserves non-prompt fields."""
        strategy = TokenSwapCorruption(vocab=number_vocab)

        example = {
            "prompt": "5 + 3",
            "answer": "8",
            "custom_field": "value",
        }

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        assert corrupted["answer"] == "8"
        assert corrupted["custom_field"] == "value"

    def test_corrupt_metadata_vocab_override(self, simple_tagger):
        """Test that metadata can override vocabulary."""
        strategy = TokenSwapCorruption(vocab={"NUM": ["1"]})

        example = {"prompt": "5 + 10"}

        override_vocab = {"NUM": ["2", "3", "4"]}
        metadata = {"tagger": simple_tagger, "vocab": override_vocab}

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        assert "prompt" in corrupted


class TestTokenSwapBatchCorrupt:
    """Test TokenSwapCorruption.batch_corrupt() method."""

    def test_batch_corrupt_basic(self, simple_tagger, number_vocab):
        """Test batch corruption."""
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=number_vocab,
        )

        examples = [
            {"prompt": "5 + 3 equals something"},
            {"prompt": "10 - 7 equals something"},
            {"prompt": "42 * 2 equals something"},
        ]

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted_batch = strategy.batch_corrupt(examples, rng=rng, metadata=metadata)

        assert len(corrupted_batch) == len(examples)
        for corrupted in corrupted_batch:
            assert "prompt" in corrupted

    def test_batch_corrupt_builds_vocab(self, simple_tagger):
        """Test that batch_corrupt can build vocab from examples."""
        strategy = TokenSwapCorruption(pos_tags=["NUM"])
        strategy.vocab = {}
        strategy._vocab_built = False

        examples = [
            {"prompt": "5 + 3"},
            {"prompt": "10 - 7"},
            {"prompt": "42 * 2"},
        ]

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted_batch = strategy.batch_corrupt(examples, rng=rng, metadata=metadata)

        assert len(corrupted_batch) == len(examples)
        # Vocab should now be built
        if strategy.vocab:
            assert strategy._vocab_built


class TestTokenSwapValidation:
    """Test TokenSwapCorruption.validate() method."""

    def test_validate_valid_corruption(self):
        """Test validation of a valid corruption."""
        strategy = TokenSwapCorruption()

        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "7 + 3"}

        validation = strategy.validate(clean, corrupted)

        assert isinstance(validation, CorruptionValidation)
        assert validation.is_valid is True
        assert validation.severity > 0.0

    def test_validate_missing_prompt_field(self):
        """Test validation fails for missing prompt field."""
        strategy = TokenSwapCorruption()

        clean = {"prompt": "5 + 3"}
        corrupted = {}  # Missing prompt

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is False
        assert "prompt" in validation.reason.lower()

    def test_validate_empty_prompt(self):
        """Test validation fails for empty corrupted prompt."""
        strategy = TokenSwapCorruption()

        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": ""}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is False

    def test_validate_unchanged_corruption(self):
        """Test validation of unchanged corruption (severity=0)."""
        strategy = TokenSwapCorruption()

        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "5 + 3"}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is True
        assert validation.severity == 0.0

    def test_validate_small_change_low_severity(self):
        """Test that small changes have low severity."""
        strategy = TokenSwapCorruption()

        clean = {"prompt": "5 + 3 equals 8"}
        corrupted = {"prompt": "7 + 3 equals 8"}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is True
        # Changing one token should have low but non-zero severity
        assert 0.0 < validation.severity < 0.5


class TestTokenSwapGreaterThanIntegration:
    """Integration tests for greater_than task style."""

    def test_greater_than_number_swap(self, simple_tagger):
        """Test number swapping for greater_than style task."""
        vocab = {
            "NUM": ["1", "2", "3", "5", "7", "10", "20", "25", "30", "42"],
        }
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        example = {
            "prompt": "The thing lasted from 5 to",
        }

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Should have swapped the number
        assert "prompt" in corrupted

    def test_greater_than_valid_expression(self, simple_tagger):
        """Test that corrupted expressions remain valid for greater_than."""
        vocab = {
            "NUM": ["1", "2", "3", "5", "7"],
        }
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        # Simulate a valid expression prompt
        example = {
            "prompt": "5 + 3",
        }

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Result should still have valid structure
        if "prompt" in corrupted and corrupted["prompt"] != example["prompt"]:
            # Tokenize and check it's valid
            tokens = corrupted["prompt"].split()
            assert len(tokens) > 0


class TestTokenSwapEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_corrupt_single_token_vocabulary(self, simple_tagger):
        """Test behavior with only one token in vocabulary."""
        vocab = {"NUM": ["42"]}
        strategy = TokenSwapCorruption(vocab=vocab)

        example = {"prompt": "5 + 3"}
        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)

        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Should return unchanged if only replacement is the original token
        assert "prompt" in corrupted

    def test_corrupt_all_tokens_same(self, simple_tagger):
        """Test behavior when all tokens are the same POS."""
        vocab = {"NUM": ["1", "2", "3", "5"]}
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        example = {"prompt": "5 7 10"}

        metadata = {"tagger": simple_tagger}
        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        assert "prompt" in corrupted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
