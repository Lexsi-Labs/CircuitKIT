"""
Standalone tests for B2 (EntitySwapCorruption) and B3 (TokenSwapCorruption).

These tests import the modules directly to avoid triggering the full
circuitkit API initialization.
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

# Strip relative imports (the needed names are injected into the exec dict below)
entity_swap_code = re.sub(r"(?m)^from \.\S* import .*$", "", entity_swap_code)

entity_swap_mod_dict = {
    "__name__": "test_entity_swap",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "random": random,
    "__builtins__": __builtins__,
}
exec(entity_swap_code, entity_swap_mod_dict)
EntitySwapCorruption_orig = entity_swap_mod_dict["EntitySwapCorruption"]

# Load token_swap module
with open(os.path.join(base_path, "token_swap.py"), "r") as f:
    token_swap_code = f.read()

# Strip relative imports (the needed names are injected into the exec dict below)
token_swap_code = re.sub(r"(?m)^from \.\S* import .*$", "", token_swap_code)

token_swap_mod_dict = {
    "__name__": "test_token_swap",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "random": random,
    "__builtins__": __builtins__,
}
exec(token_swap_code, token_swap_mod_dict)
TokenSwapCorruption_orig = token_swap_mod_dict["TokenSwapCorruption"]

# Extract classes
EntitySwapCorruption = EntitySwapCorruption_orig
TokenSwapCorruption = TokenSwapCorruption_orig
CorruptionValidation = base_mod.CorruptionValidation


class TestEntitySwapCorruptionInit:
    """Tests for EntitySwapCorruption initialization."""

    def test_default_init(self):
        """Test default initialization."""
        strategy = EntitySwapCorruption()
        assert strategy.name == "entity_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.entity_types is None

    def test_init_with_entity_types(self):
        """Test initialization with entity type filters."""
        strategy = EntitySwapCorruption(entity_types=["PERSON", "GPE"])
        assert strategy.entity_types == ["PERSON", "GPE"]

    def test_init_with_pool(self):
        """Test initialization with entity pool."""
        pool = {"PERSON": ["Alice", "Bob"], "GPE": ["Paris", "London"]}
        strategy = EntitySwapCorruption(entity_pool=pool)
        assert strategy.entity_pool == pool

    def test_init_with_auto_pool(self):
        """Test initialization with auto pool mode."""
        strategy = EntitySwapCorruption(entity_pool="auto")
        assert strategy.entity_pool is None
        assert strategy._pool_built is False


class TestEntitySwapValidation:
    """Tests for EntitySwapCorruption.validate() method."""

    def test_validate_returns_validation_object(self):
        """Test that validate returns CorruptionValidation."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice met Bob"}
        corrupted = {"prompt": "Charlie met Bob"}

        result = strategy.validate(clean, corrupted)
        assert isinstance(result, CorruptionValidation)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "severity")
        assert hasattr(result, "reason")

    def test_validate_missing_prompt_field(self):
        """Test that validation fails when prompt is missing."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice"}
        corrupted = {"other_field": "value"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False
        assert "prompt" in result.reason.lower()

    def test_validate_empty_prompt(self):
        """Test that validation fails for empty prompt."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice"}
        corrupted = {"prompt": ""}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_validate_unchanged_has_zero_severity(self):
        """Test that unchanged corruption has zero severity."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice and Bob"}
        corrupted = {"prompt": "Alice and Bob"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity == 0.0

    def test_validate_changed_has_nonzero_severity(self):
        """Test that changed corruption has non-zero severity."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice and Bob went to school"}
        corrupted = {"prompt": "Charlie and Bob went to school"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert 0.0 < result.severity < 1.0

    def test_validate_severity_increases_with_change(self):
        """Test that severity increases with larger changes."""
        strategy = EntitySwapCorruption()
        clean = {"prompt": "Alice"}

        # Small change
        corrupted_small = {"prompt": "Charlie"}
        result_small = strategy.validate(clean, corrupted_small)

        # Larger change
        corrupted_large = {"prompt": "This is a completely different sentence"}
        result_large = strategy.validate(clean, corrupted_large)

        # Larger change should have higher severity
        assert result_large.severity > result_small.severity


class TestTokenSwapCorruptionInit:
    """Tests for TokenSwapCorruption initialization."""

    def test_default_init(self):
        """Test default initialization."""
        strategy = TokenSwapCorruption()
        assert strategy.name == "token_swap"
        assert strategy.mode == "meaning-altering"
        assert strategy.pos_tags is None

    def test_init_with_pos_tags(self):
        """Test initialization with POS tag filters."""
        strategy = TokenSwapCorruption(pos_tags=["NUM", "NN"])
        assert strategy.pos_tags == ["NUM", "NN"]

    def test_init_with_vocab(self):
        """Test initialization with vocabulary."""
        vocab = {"NUM": ["1", "2", "3"]}
        strategy = TokenSwapCorruption(vocab=vocab)
        assert strategy.vocab == vocab

    def test_init_with_tokenizer(self):
        """Test initialization with tokenizer."""

        class MockTokenizer:
            pass

        tok = MockTokenizer()
        strategy = TokenSwapCorruption(tokenizer=tok)
        assert strategy.tokenizer is tok


class TestTokenSwapValidation:
    """Tests for TokenSwapCorruption.validate() method."""

    def test_validate_returns_validation_object(self):
        """Test that validate returns CorruptionValidation."""
        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "7 + 3"}

        result = strategy.validate(clean, corrupted)
        assert isinstance(result, CorruptionValidation)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "severity")
        assert hasattr(result, "reason")

    def test_validate_missing_prompt_field(self):
        """Test that validation fails when prompt is missing."""
        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"other_field": "value"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False
        assert "prompt" in result.reason.lower()

    def test_validate_empty_prompt(self):
        """Test that validation fails for empty prompt."""
        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": ""}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is False

    def test_validate_unchanged_has_zero_severity(self):
        """Test that unchanged corruption has zero severity."""
        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3"}
        corrupted = {"prompt": "5 + 3"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert result.severity == 0.0

    def test_validate_changed_has_nonzero_severity(self):
        """Test that changed corruption has non-zero severity."""
        strategy = TokenSwapCorruption()
        clean = {"prompt": "5 + 3 equals 8"}
        corrupted = {"prompt": "7 + 3 equals 8"}

        result = strategy.validate(clean, corrupted)
        assert result.is_valid is True
        assert 0.0 < result.severity < 1.0


class TestProtocolCompliance:
    """Test that both strategies comply with CorruptionStrategy protocol."""

    def test_entity_swap_has_required_attributes(self):
        """Test EntitySwapCorruption has protocol attributes."""
        strategy = EntitySwapCorruption()
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "mode")
        assert isinstance(strategy.name, str)
        assert strategy.mode in ["meaning-preserving", "meaning-altering", "role-swap"]

    def test_entity_swap_has_required_methods(self):
        """Test EntitySwapCorruption has protocol methods."""
        strategy = EntitySwapCorruption()
        assert callable(getattr(strategy, "corrupt", None))
        assert callable(getattr(strategy, "batch_corrupt", None))
        assert callable(getattr(strategy, "validate", None))

    def test_token_swap_has_required_attributes(self):
        """Test TokenSwapCorruption has protocol attributes."""
        strategy = TokenSwapCorruption()
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "mode")
        assert isinstance(strategy.name, str)
        assert strategy.mode in ["meaning-preserving", "meaning-altering", "role-swap"]

    def test_token_swap_has_required_methods(self):
        """Test TokenSwapCorruption has protocol methods."""
        strategy = TokenSwapCorruption()
        assert callable(getattr(strategy, "corrupt", None))
        assert callable(getattr(strategy, "batch_corrupt", None))
        assert callable(getattr(strategy, "validate", None))


class TestEntitySwapMethods:
    """Tests for EntitySwapCorruption method signatures."""

    def test_set_nlp_method(self):
        """Test that set_nlp method exists."""
        strategy = EntitySwapCorruption()
        assert hasattr(strategy, "set_nlp")
        assert callable(strategy.set_nlp)

        # Should not raise
        strategy.set_nlp(None)

    def test_build_pool_from_examples_method(self):
        """Test that _build_pool_from_examples method exists."""
        strategy = EntitySwapCorruption()
        assert hasattr(strategy, "_build_pool_from_examples")
        assert callable(strategy._build_pool_from_examples)


class TestTokenSwapMethods:
    """Tests for TokenSwapCorruption method signatures."""

    def test_set_tokenizer_method(self):
        """Test that set_tokenizer method exists."""
        strategy = TokenSwapCorruption()
        assert hasattr(strategy, "set_tokenizer")
        assert callable(strategy.set_tokenizer)

        # Should not raise
        strategy.set_tokenizer(None)

    def test_build_vocab_from_examples_method(self):
        """Test that _build_vocab_from_examples method exists."""
        strategy = TokenSwapCorruption()
        assert hasattr(strategy, "_build_vocab_from_examples")
        assert callable(strategy._build_vocab_from_examples)


class TestCorruptionValidationDataclass:
    """Tests for CorruptionValidation dataclass."""

    def test_create_valid_corruption(self):
        """Test creating a valid corruption result."""
        val = CorruptionValidation(is_valid=True, reason=None, severity=0.5)
        assert val.is_valid is True
        assert val.reason is None
        assert val.severity == 0.5

    def test_create_invalid_corruption(self):
        """Test creating an invalid corruption result."""
        val = CorruptionValidation(is_valid=False, reason="Prompt too short", severity=1.0)
        assert val.is_valid is False
        assert val.reason == "Prompt too short"
        assert val.severity == 1.0

    def test_default_reason(self):
        """Test that reason defaults to None."""
        val = CorruptionValidation(is_valid=True)
        assert val.reason is None

    def test_default_severity(self):
        """Test that severity defaults to 0.0."""
        val = CorruptionValidation(is_valid=True)
        assert val.severity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
