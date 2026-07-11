"""
Integration tests for B2 (EntitySwapCorruption) and B3 (TokenSwapCorruption).

These tests verify the core functionality without relying on the full
circuitkit API, focusing on the corruption strategies themselves.
"""

import os
import sys

import pytest

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Standard import - this ensures class memory addresses match!
from circuitkit.corruption.base import (  # noqa: E402 - import after intentional pre-import setup
    CorruptionValidation,
)


class TestEntitySwapCorruptionDirect:
    """Direct tests of EntitySwapCorruption without full API."""

    def test_entity_swap_initialization(self):
        """Test EntitySwapCorruption can be imported and initialized."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption(entity_types=["PERSON"])
        assert strategy.name == "entity_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.entity_types == ["PERSON"]

    def test_entity_swap_with_pool(self):
        """Test EntitySwapCorruption with provided entity pool."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        pool = {"PERSON": ["Alice", "Bob", "Charlie"]}
        strategy = EntitySwapCorruption(
            entity_types=["PERSON"],
            entity_pool=pool,
        )
        assert strategy.entity_pool == pool

    def test_entity_swap_nlp_import(self):
        """Test that EntitySwapCorruption handles NLP loading gracefully."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        # Should not crash even if spacy not installed
        strategy = EntitySwapCorruption()
        # nlp may be None or loaded depending on spacy availability
        assert hasattr(strategy, "nlp")

    def test_entity_swap_has_corrupt_method(self):
        """Test that EntitySwapCorruption has required protocol methods."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        assert hasattr(strategy, "corrupt")
        assert callable(strategy.corrupt)
        assert hasattr(strategy, "validate")
        assert callable(strategy.validate)
        assert hasattr(strategy, "batch_corrupt")
        assert callable(strategy.batch_corrupt)


class TestTokenSwapCorruptionDirect:
    """Direct tests of TokenSwapCorruption without full API."""

    def test_token_swap_initialization(self):
        """Test TokenSwapCorruption can be imported and initialized."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption(pos_tags=["NUM", "NN"])
        assert strategy.name == "token_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.pos_tags == ["NUM", "NN"]

    def test_token_swap_with_vocab(self):
        """Test TokenSwapCorruption with provided vocabulary."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        vocab = {"NUM": ["1", "2", "3"]}
        strategy = TokenSwapCorruption(
            pos_tags=["NUM"],
            vocab=vocab,
        )
        assert strategy.vocab == vocab

    def test_token_swap_has_corrupt_method(self):
        """Test that TokenSwapCorruption has required protocol methods."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        assert hasattr(strategy, "corrupt")
        assert callable(strategy.corrupt)
        assert hasattr(strategy, "validate")
        assert callable(strategy.validate)
        assert hasattr(strategy, "batch_corrupt")
        assert callable(strategy.batch_corrupt)


class TestCorruptionValidation:
    """Test CorruptionValidation dataclass."""

    def test_validation_basic(self):
        """Test CorruptionValidation instantiation."""
        validation = CorruptionValidation(
            is_valid=True,
            reason=None,
            severity=0.5,
        )
        assert validation.is_valid is True
        assert validation.reason is None
        assert validation.severity == 0.5

    def test_validation_invalid(self):
        """Test CorruptionValidation for invalid cases."""
        validation = CorruptionValidation(
            is_valid=False,
            reason="Test failure",
            severity=1.0,
        )
        assert validation.is_valid is False
        assert validation.reason == "Test failure"
        assert validation.severity == 1.0


class TestEntitySwapBasicCorruption:
    """Basic functionality tests for EntitySwapCorruption."""

    def test_entity_swap_validate_returns_validation(self):
        """Test that validate() returns CorruptionValidation."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice met Bob"}
        corrupted = {"prompt": "Charlie met Bob"}

        result = strategy.validate(clean, corrupted)
        assert isinstance(result, CorruptionValidation)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "severity")

    def test_entity_swap_validate_missing_prompt(self):
        """Test validation fails for missing prompt."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice"}
        corrupted = {}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_entity_swap_validate_empty_prompt(self):
        """Test validation fails for empty prompt."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice"}
        corrupted = {"prompt": ""}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_entity_swap_validate_unchanged(self):
        """Test validation of unchanged corruption."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice and Bob"}
        corrupted = {"prompt": "Alice and Bob"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity == 0.0

    def test_entity_swap_validate_changed(self):
        """Test validation of changed corruption."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice and Bob met at school"}
        corrupted = {"prompt": "Charlie and Bob met at school"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity > 0.0
        assert result.severity < 1.0


class TestTokenSwapBasicCorruption:
    """Basic functionality tests for TokenSwapCorruption."""

    def test_token_swap_validate_returns_validation(self):
        """Test that validate() returns CorruptionValidation."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "7 + 3"}

        result = strategy.validate(clean, corrupted)
        assert isinstance(result, CorruptionValidation)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "severity")

    def test_token_swap_validate_missing_prompt(self):
        """Test validation fails for missing prompt."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_token_swap_validate_empty_prompt(self):
        """Test validation fails for empty prompt."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": ""}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_token_swap_validate_unchanged(self):
        """Test validation of unchanged corruption."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "5 + 3"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity == 0.0

    def test_token_swap_validate_changed(self):
        """Test validation of changed corruption."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3 equals 8"}
        corrupted = {"prompt": "7 + 3 equals 8"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity > 0.0
        assert result.severity < 1.0


class TestCorruptionProtocolCompliance:
    """Test that strategies comply with CorruptionStrategy protocol."""

    def test_entity_swap_protocol_attributes(self):
        """Test EntitySwapCorruption has protocol attributes."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "mode")
        assert isinstance(strategy.name, str)
        assert strategy.mode in ["meaning-preserving", "meaning-altering", "role-swap"]

    def test_token_swap_protocol_attributes(self):
        """Test TokenSwapCorruption has protocol attributes."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "mode")
        assert isinstance(strategy.name, str)
        assert strategy.mode in ["meaning-preserving", "meaning-altering", "role-swap"]

    def test_entity_swap_protocol_methods(self):
        """Test EntitySwapCorruption has all protocol methods."""
        from circuitkit.corruption.entity_swap import EntitySwapCorruption

        strategy = EntitySwapCorruption()
        # Check method signatures (can't fully test without proper setup)
        assert callable(getattr(strategy, "corrupt", None))
        assert callable(getattr(strategy, "batch_corrupt", None))
        assert callable(getattr(strategy, "validate", None))

    def test_token_swap_protocol_methods(self):
        """Test TokenSwapCorruption has all protocol methods."""
        from circuitkit.corruption.token_swap import TokenSwapCorruption

        strategy = TokenSwapCorruption()
        # Check method signatures (can't fully test without proper setup)
        assert callable(getattr(strategy, "corrupt", None))
        assert callable(getattr(strategy, "batch_corrupt", None))
        assert callable(getattr(strategy, "validate", None))


class TestCorruptionExports:
    """Test that strategies are properly exported from module."""

    def test_entity_swap_in_init(self):
        """Test EntitySwapCorruption is exported from __init__."""
        from circuitkit.corruption import EntitySwapCorruption

        assert EntitySwapCorruption is not None

    def test_token_swap_in_init(self):
        """Test TokenSwapCorruption is exported from __init__."""
        from circuitkit.corruption import TokenSwapCorruption

        assert TokenSwapCorruption is not None

    def test_validation_in_init(self):
        """Test CorruptionValidation is exported from __init__."""
        from circuitkit.corruption import CorruptionValidation

        assert CorruptionValidation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
