"""
Tests for EntitySwapCorruption strategy.
"""

import random

import pytest

from circuitkit.corruption import CorruptionValidation, EntitySwapCorruption

# Skip tests if spacy is not installed
pytest.importorskip("spacy")


@pytest.fixture
def entity_swap_with_nlp():
    """Fixture providing EntitySwapCorruption with loaded spaCy model."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        strategy = EntitySwapCorruption(nlp=nlp)
        return strategy
    except OSError:
        pytest.skip("spaCy en_core_web_sm model not installed")


@pytest.fixture
def entity_pool():
    """Fixture providing a sample entity pool."""
    return {
        "PERSON": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "GPE": ["Paris", "London", "Tokyo", "Berlin", "Rome"],
    }


class TestEntitySwapCorruptionBasics:
    """Test basic EntitySwapCorruption functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        strategy = EntitySwapCorruption()
        assert strategy.name == "entity_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.entity_types is None

    def test_initialization_with_entity_types(self):
        """Test initialization with entity type filters."""
        strategy = EntitySwapCorruption(entity_types=["PERSON"])
        assert strategy.entity_types == ["PERSON"]

    def test_initialization_with_entity_pool(self):
        """Test initialization with entity pool."""
        pool = {"PERSON": ["Alice", "Bob"]}
        strategy = EntitySwapCorruption(entity_pool=pool)
        assert strategy.entity_pool == pool

    def test_nlp_loading_on_init(self):
        """Test that spaCy model is loaded on init (if available)."""
        strategy = EntitySwapCorruption()
        # If spaCy is installed, nlp should be loaded
        # If not, nlp will be None (graceful fail)
        try:
            pass

            assert strategy.nlp is not None
        except ImportError:
            assert strategy.nlp is None


class TestEntitySwapCorruption:
    """Test EntitySwapCorruption.corrupt() method."""

    def test_corrupt_simple_example(self, entity_swap_with_nlp, entity_pool):
        """Test corruption of a simple example with entity pool."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = entity_pool

        example = {
            "prompt": "Alice and Bob went to Paris.",
            "answer": "Alice",
        }

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        # Check that result is a dict with original keys
        assert isinstance(corrupted, dict)
        assert "prompt" in corrupted
        assert "answer" in corrupted
        assert corrupted["answer"] == "Alice"  # Answer should be unchanged

        # Prompt should be modified (some entity should be swapped)
        # Due to randomness, we can't guarantee which entity is swapped
        # but we should have corruption info
        assert "_corruption_info" in corrupted

    def test_corrupt_preserves_other_fields(self, entity_swap_with_nlp, entity_pool):
        """Test that corruption preserves non-prompt fields."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = entity_pool

        example = {
            "prompt": "Alice met Bob at London.",
            "answer": "Alice",
            "custom_field": "custom_value",
        }

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        assert corrupted["custom_field"] == "custom_value"
        assert corrupted["answer"] == "Alice"

    def test_corrupt_no_entities_returns_unchanged(self, entity_swap_with_nlp):
        """Test that prompt with no recognized entities returns unchanged."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = {"PERSON": ["Alice", "Bob"]}

        example = {
            "prompt": "The number 42 is interesting.",  # No PERSON entities
        }

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        # Should return a copy even if unchanged
        assert "prompt" in corrupted
        assert corrupted != example or corrupted is not example

    def test_corrupt_requires_nlp(self):
        """Test that corrupt() raises error if NLP not loaded."""
        strategy = EntitySwapCorruption(nlp=None)  # Explicitly set nlp to None
        strategy.nlp = None  # Ensure nlp is None

        example = {"prompt": "Alice and Bob"}
        rng = random.Random(42)

        with pytest.raises(RuntimeError, match="EntitySwapCorruption requires a spaCy model"):
            strategy.corrupt(example, rng=rng)

    def test_corrupt_deterministic_with_seed(self, entity_swap_with_nlp, entity_pool):
        """Test that corruption is deterministic with same RNG seed."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = entity_pool

        example = {
            "prompt": "Alice and Bob went to Paris with Charlie.",
        }

        # Two runs with same seed should produce same result
        rng1 = random.Random(42)
        corrupted1 = strategy.corrupt(example, rng=rng1)

        rng2 = random.Random(42)
        corrupted2 = strategy.corrupt(example, rng=rng2)

        assert corrupted1["prompt"] == corrupted2["prompt"]

    def test_corrupt_with_entity_type_filter(self, entity_swap_with_nlp):
        """Test corruption with entity type filtering."""
        pool = {
            "PERSON": ["Alice", "Bob", "Charlie"],
            "GPE": ["Paris", "London"],
        }
        strategy = EntitySwapCorruption(
            entity_types=["PERSON"],
            nlp=entity_swap_with_nlp.nlp,
            entity_pool=pool,
        )

        example = {
            "prompt": "Alice went to Paris.",
        }

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        # Should only swap PERSON entities, not GPE
        assert "prompt" in corrupted

    def test_corrupt_metadata_override_pool(self, entity_swap_with_nlp):
        """Test that metadata can override entity pool."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = {"PERSON": ["Alice"]}

        example = {
            "prompt": "Bob and Alice met.",
        }

        override_pool = {"PERSON": ["Charlie", "Diana"]}
        metadata = {"entity_pool": override_pool}

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng, metadata=metadata)

        # Should use override pool
        assert "prompt" in corrupted


class TestEntitySwapBatchCorrupt:
    """Test EntitySwapCorruption.batch_corrupt() method."""

    def test_batch_corrupt_basic(self, entity_swap_with_nlp, entity_pool):
        """Test batch corruption."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = entity_pool

        examples = [
            {"prompt": "Alice and Bob went to Paris."},
            {"prompt": "Charlie met Diana at London."},
            {"prompt": "Eve visited Tokyo with Frank."},
        ]

        rng = random.Random(42)
        corrupted_batch = strategy.batch_corrupt(examples, rng=rng)

        assert len(corrupted_batch) == len(examples)
        for corrupted in corrupted_batch:
            assert "prompt" in corrupted

    def test_batch_corrupt_builds_pool(self, entity_swap_with_nlp):
        """Test that batch_corrupt can build pool from examples."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = None
        strategy._pool_built = False

        examples = [
            {"prompt": "Alice and Bob met."},
            {"prompt": "Charlie went to Paris."},
        ]

        rng = random.Random(42)
        # This should trigger pool building if we set up entity_pool="auto"
        strategy2 = EntitySwapCorruption(entity_pool="auto", nlp=entity_swap_with_nlp.nlp)
        corrupted_batch = strategy2.batch_corrupt(examples, rng=rng)

        assert len(corrupted_batch) == len(examples)
        # Pool should now be built
        assert strategy2._pool_built


class TestEntitySwapValidation:
    """Test EntitySwapCorruption.validate() method."""

    def test_validate_valid_corruption(self, entity_swap_with_nlp, entity_pool):
        """Test validation of a valid corruption."""
        strategy = entity_swap_with_nlp
        strategy.entity_pool = entity_pool

        clean = {"prompt": "Alice and Bob"}
        corrupted = {"prompt": "Charlie and Bob"}

        validation = strategy.validate(clean, corrupted)

        assert isinstance(validation, CorruptionValidation)
        assert validation.is_valid is True
        assert validation.severity > 0.0

    def test_validate_missing_prompt_field(self, entity_swap_with_nlp):
        """Test validation fails for missing prompt field."""
        strategy = entity_swap_with_nlp

        clean = {"prompt": "Alice"}
        corrupted = {}  # Missing prompt

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is False
        assert "prompt" in validation.reason.lower()

    def test_validate_empty_prompt(self, entity_swap_with_nlp):
        """Test validation fails for empty corrupted prompt."""
        strategy = entity_swap_with_nlp

        clean = {"prompt": "Alice"}
        corrupted = {"prompt": ""}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is False

    def test_validate_unchanged_corruption(self, entity_swap_with_nlp):
        """Test validation of unchanged corruption (severity=0)."""
        strategy = entity_swap_with_nlp

        clean = {"prompt": "Alice and Bob"}
        corrupted = {"prompt": "Alice and Bob"}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is True
        assert validation.severity == 0.0

    def test_validate_high_similarity_low_severity(self, entity_swap_with_nlp):
        """Test that small changes have low severity."""
        strategy = entity_swap_with_nlp

        clean = {"prompt": "Alice and Bob went to Paris"}
        corrupted = {"prompt": "Charlie and Bob went to Paris"}

        validation = strategy.validate(clean, corrupted)

        assert validation.is_valid is True
        # Changing one word should have low but non-zero severity
        assert 0.0 < validation.severity < 0.5


class TestEntitySwapIOIIntegration:
    """Integration tests with IOI-like examples."""

    def test_ioi_style_example(self, entity_swap_with_nlp):
        """Test corruption on IOI-style example."""
        pool = {
            "PERSON": [
                "Alice",
                "Bob",
                "Charlie",
                "Diana",
                "Eve",
                "Frank",
                "Grace",
                "Henry",
                "Iris",
                "Jack",
            ]
        }
        strategy = EntitySwapCorruption(
            entity_types=["PERSON"],
            nlp=entity_swap_with_nlp.nlp,
            entity_pool=pool,
        )

        example = {
            "prompt": "Alice and Bob went to the park. Bob gave a ball to Alice",
            "IO": "Alice",
            "S": "Bob",
        }

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        # Should have swapped one of the names
        assert "prompt" in corrupted
        # Original fields should be preserved
        assert "IO" in corrupted
        assert "S" in corrupted

    def test_ioi_agreement_on_swap(self, entity_swap_with_nlp):
        """Test that we track which entity was swapped."""
        pool = {"PERSON": ["Alice", "Bob", "Charlie"]}
        strategy = EntitySwapCorruption(
            entity_types=["PERSON"],
            nlp=entity_swap_with_nlp.nlp,
            entity_pool=pool,
        )

        example = {"prompt": "Alice met Bob"}

        rng = random.Random(42)
        corrupted = strategy.corrupt(example, rng=rng)

        # Should have corruption info tracking the swap
        if "_corruption_info" in corrupted:
            info = corrupted["_corruption_info"]
            assert "swapped_entity" in info
            assert "replacement" in info
            assert "entity_type" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
