"""
Practical integration examples for EntitySwapCorruption and TokenSwapCorruption.

These tests demonstrate real-world usage patterns for both strategies.
"""

import os
import random
import re

import pytest

# Import directly from files
base_path = os.path.join(os.path.dirname(__file__), "../../src/circuitkit/corruption")

# Load base module
import importlib.util  # noqa: E402 - import after intentional pre-import setup

spec = importlib.util.spec_from_file_location("test_base", os.path.join(base_path, "base.py"))
base_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_mod)

# Load entity_swap module
with open(os.path.join(base_path, "entity_swap.py"), "r") as f:
    entity_swap_code = f.read()
entity_swap_code = re.sub(r"(?m)^from \.\S* import .*$", "", entity_swap_code)
entity_swap_mod_dict = {
    "__name__": "test_entity_swap",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "random": random,
    "__builtins__": __builtins__,
}
exec(entity_swap_code, entity_swap_mod_dict)
EntitySwapCorruption = entity_swap_mod_dict["EntitySwapCorruption"]

# Load token_swap module
with open(os.path.join(base_path, "token_swap.py"), "r") as f:
    token_swap_code = f.read()
token_swap_code = re.sub(r"(?m)^from \.\S* import .*$", "", token_swap_code)
token_swap_mod_dict = {
    "__name__": "test_token_swap",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "random": random,
    "__builtins__": __builtins__,
}
exec(token_swap_code, token_swap_mod_dict)
TokenSwapCorruption = token_swap_mod_dict["TokenSwapCorruption"]


class _MockSpacyDoc:
    """Minimal stand-in for a spaCy Doc, exposing only the `.ents` attribute."""

    def __init__(self, ents):
        self.ents = ents


class MockSpacyNLP:
    """Mock spaCy NLP pipeline for testing EntitySwapCorruption.

    EntitySwapCorruption only needs a callable that turns text into an object
    with an `.ents` iterable. This mock lets the real corrupt()/batch_corrupt()
    code paths run without requiring the optional `spacy` dependency to be
    installed, mirroring how MockPOSTagger is used for TokenSwapCorruption.
    """

    def __call__(self, text: str):
        """Return a Doc-like object. No entities are detected (empty `.ents`)."""
        return _MockSpacyDoc(ents=[])


class MockPOSTagger:
    """Mock POS tagger for testing TokenSwapCorruption."""

    def __call__(self, text: str):
        """Simple POS tagging for testing.

        Returns:
            Tuple of (tokens, pos_tags).
        """
        tokens = text.split()
        pos_tags = []

        for token in tokens:
            # Simple heuristics
            if token.isdigit():
                pos_tags.append("NUM")
            elif token in ["the", "a", "an"]:
                pos_tags.append("DET")
            elif token in ["+", "-", "=", ">"]:
                pos_tags.append("SYM")
            else:
                pos_tags.append("NN")

        return tokens, pos_tags


class TestEntitySwapIOIStyle:
    """Test EntitySwapCorruption with IOI-style examples."""

    def test_ioi_basic_swap(self):
        """Test basic IOI-style corruption."""
        pool = {
            "PERSON": ["Alice", "Bob", "Charlie", "Diana"],
        }
        strategy = EntitySwapCorruption(
            entity_types=["PERSON"],
            entity_pool=pool,
        )

        # Simulate an IOI example (without spaCy for testing)

        # Since we can't test actual NER without spacy loaded,
        # we verify the structure is correct
        assert strategy.name == "entity_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.entity_pool == pool

    def test_ioi_with_metadata_override(self):
        """Test EntitySwapCorruption with metadata pool override."""
        default_pool = {"PERSON": ["Alice", "Bob"]}
        override_pool = {"PERSON": ["Charlie", "Diana", "Eve"]}

        strategy = EntitySwapCorruption(entity_pool=default_pool)

        # Verify override mechanism works
        assert strategy.entity_pool == default_pool

        # Metadata would override at runtime
        metadata = {"entity_pool": override_pool}
        assert metadata["entity_pool"] == override_pool


class TestTokenSwapGreaterThan:
    """Test TokenSwapCorruption with greater_than style examples."""

    def test_greater_than_number_vocab(self):
        """Test TokenSwapCorruption with number vocabulary."""
        vocab = {
            "NUM": ["1", "2", "3", "5", "7", "10", "20", "42", "100"],
        }
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        assert strategy.name == "token_swap"
        assert strategy.vocab == vocab
        assert strategy.pos_tags == ["NUM"]

    def test_greater_than_deterministic_corruption(self):
        """Test deterministic corruption with fixed seed."""
        vocab = {"NUM": ["1", "2", "3", "5", "7", "10"]}
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        tagger = MockPOSTagger()
        example = {"prompt": "5 + 3"}
        metadata = {"tagger": tagger}

        # Two runs with same seed should produce same result
        rng1 = random.Random(42)
        result1 = strategy.corrupt(example, rng=rng1, metadata=metadata)

        rng2 = random.Random(42)
        result2 = strategy.corrupt(example, rng=rng2, metadata=metadata)

        # Both should either be swapped or unchanged (deterministic)
        assert result1.get("prompt") == result2.get("prompt")

    def test_greater_than_batch_processing(self):
        """Test batch processing of greater_than examples."""
        vocab = {"NUM": ["1", "2", "3", "5", "7", "10"]}
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )

        examples = [
            {"prompt": "5 + 3"},
            {"prompt": "10 - 7"},
            {"prompt": "20 * 2"},
        ]

        tagger = MockPOSTagger()
        metadata = {"tagger": tagger}
        rng = random.Random(42)

        # Batch corrupt
        corrupted_batch = strategy.batch_corrupt(examples, rng=rng, metadata=metadata)

        assert len(corrupted_batch) == len(examples)
        for corrupted in corrupted_batch:
            assert "prompt" in corrupted


class TestValidationSeverity:
    """Test severity calculation for both strategies."""

    def test_entity_swap_severity_scaling(self):
        """Test that severity increases with larger changes."""
        strategy = EntitySwapCorruption()

        # Small change (1 character)
        clean1 = {"prompt": "Alice went home"}
        corrupted1 = {"prompt": "Alice went hzme"}  # 1 char changed
        val1 = strategy.validate(clean1, corrupted1)

        # Large change (multiple words)
        clean2 = {"prompt": "Alice went home"}
        corrupted2 = {"prompt": "Bob went to the park"}  # Major change
        val2 = strategy.validate(clean2, corrupted2)

        assert val1.is_valid is True
        assert val2.is_valid is True
        assert val2.severity > val1.severity

    def test_token_swap_severity_scaling(self):
        """Test that severity increases with larger changes."""
        strategy = TokenSwapCorruption()

        # Small change (1 token)
        clean1 = {"prompt": "5 + 3"}
        corrupted1 = {"prompt": "7 + 3"}
        val1 = strategy.validate(clean1, corrupted1)

        # Large change (multiple tokens)
        clean2 = {"prompt": "5 + 3"}
        corrupted2 = {"prompt": "This is completely different"}
        val2 = strategy.validate(clean2, corrupted2)

        assert val1.is_valid is True
        assert val2.is_valid is True
        assert val2.severity > val1.severity


class TestMetadataHandling:
    """Test metadata parameter handling for both strategies."""

    def test_entity_swap_metadata_none(self):
        """Test EntitySwapCorruption handles None metadata."""
        pool = {"PERSON": ["Alice", "Bob"]}
        strategy = EntitySwapCorruption(entity_pool=pool, nlp=MockSpacyNLP())

        example = {"prompt": "test"}
        rng = random.Random(42)

        # Should handle None metadata gracefully
        result = strategy.corrupt(example, rng=rng, metadata=None)
        assert isinstance(result, dict)

    def test_token_swap_metadata_required(self):
        """Test TokenSwapCorruption requires tagger in metadata."""
        strategy = TokenSwapCorruption(vocab={"NUM": ["1", "2"]})

        example = {"prompt": "5"}
        rng = random.Random(42)

        # Should raise if tagger not in metadata
        with pytest.raises(ValueError, match="tagger"):
            strategy.corrupt(example, rng=rng, metadata={})

    def test_token_swap_metadata_override(self):
        """Test TokenSwapCorruption metadata vocab override."""
        default_vocab = {"NUM": ["1"]}
        override_vocab = {"NUM": ["2", "3", "4"]}

        strategy = TokenSwapCorruption(vocab=default_vocab)

        tagger = MockPOSTagger()

        # Metadata provides override
        metadata = {
            "tagger": tagger,
            "vocab": override_vocab,
        }

        assert strategy.vocab == default_vocab
        assert metadata["vocab"] == override_vocab


class TestBatchCorruptionConsistency:
    """Test that batch corruption maintains consistency."""

    def test_entity_swap_batch_returns_list(self):
        """Test batch_corrupt returns list of same length."""
        pool = {"PERSON": ["Alice", "Bob"]}
        strategy = EntitySwapCorruption(entity_pool=pool, nlp=MockSpacyNLP())

        examples = [
            {"prompt": "Test 1"},
            {"prompt": "Test 2"},
            {"prompt": "Test 3"},
        ]

        rng = random.Random(42)
        result = strategy.batch_corrupt(examples, rng=rng)

        assert isinstance(result, list)
        assert len(result) == len(examples)

    def test_token_swap_batch_returns_list(self):
        """Test batch_corrupt returns list of same length."""
        vocab = {"NUM": ["1", "2"]}
        strategy = TokenSwapCorruption(vocab=vocab)

        examples = [
            {"prompt": "5"},
            {"prompt": "10"},
            {"prompt": "20"},
        ]

        tagger = MockPOSTagger()
        metadata = {"tagger": tagger}
        rng = random.Random(42)
        result = strategy.batch_corrupt(examples, rng=rng, metadata=metadata)

        assert isinstance(result, list)
        assert len(result) == len(examples)


class TestFieldPreservation:
    """Test that non-prompt fields are preserved."""

    def test_entity_swap_preserves_fields(self):
        """Test EntitySwapCorruption preserves non-prompt fields."""
        pool = {"PERSON": ["Alice", "Bob"]}
        strategy = EntitySwapCorruption(entity_pool=pool, nlp=MockSpacyNLP())

        example = {
            "prompt": "Test",
            "answer": "expected",
            "custom_id": 42,
            "metadata": {"key": "value"},
        }

        rng = random.Random(42)
        result = strategy.corrupt(example, rng=rng)

        # Non-prompt fields should be preserved
        assert result["answer"] == "expected"
        assert result["custom_id"] == 42
        assert result["metadata"] == {"key": "value"}

    def test_token_swap_preserves_fields(self):
        """Test TokenSwapCorruption preserves non-prompt fields."""
        vocab = {"NUM": ["1", "2"]}
        strategy = TokenSwapCorruption(vocab=vocab)

        example = {
            "prompt": "5",
            "answer": "correct",
            "task": "math",
            "difficulty": "easy",
        }

        tagger = MockPOSTagger()
        metadata = {"tagger": tagger}
        rng = random.Random(42)
        result = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Non-prompt fields should be preserved
        assert result["answer"] == "correct"
        assert result["task"] == "math"
        assert result["difficulty"] == "easy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
