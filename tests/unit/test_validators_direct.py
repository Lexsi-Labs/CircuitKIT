"""
Direct test for validators (bypass package init).
Run with: python test_validators_direct.py
"""

import sys
from unittest.mock import Mock

from circuitkit.corruption.validators import (
    CompositeValidator,
    CorruptionValidationResult,
    LabelConsistencyValidator,
    LengthBudgetValidator,
    TokenizationValidator,
)


def test_length_budget_validator():
    """Test LengthBudgetValidator works correctly."""
    print("Testing LengthBudgetValidator...")
    validator = LengthBudgetValidator(tolerance=0.1)

    # Valid case: exact length
    clean = {"prompt": "The quick brown fox jumps"}
    corrupted = {"prompt": "The quick orange fox jumps"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is True
    print("  [OK] Valid corruption (exact length)")

    # Invalid case: too long
    clean = {"prompt": "Short"}
    corrupted = {"prompt": "This is a much much longer prompt that exceeds budget"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is False
    print("  [OK] Invalid corruption (too long)")

    # Invalid case: missing prompt
    clean = {}
    corrupted = {"prompt": "test"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is False
    print("  [OK] Invalid corruption (missing prompt)")


def test_label_consistency_validator():
    """Test LabelConsistencyValidator works correctly."""
    print("\nTesting LabelConsistencyValidator...")
    tokenizer = Mock()
    validator = LabelConsistencyValidator(tokenizer)

    # Valid case: answer present
    clean = {"answer": "Paris"}
    corrupted = {"prompt": "The capital of France is Paris."}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is True
    print("  [OK] Valid corruption (answer present)")

    # Invalid case: answer missing
    clean = {"answer": "Paris"}
    corrupted = {"prompt": "The capital of France is London."}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is False
    print("  [OK] Invalid corruption (answer missing)")

    # Valid case: case insensitive
    clean = {"answer": "PARIS"}
    corrupted = {"prompt": "The capital is paris."}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is True
    print("  [OK] Valid corruption (case insensitive)")


def test_tokenization_validator():
    """Test TokenizationValidator works correctly."""
    print("\nTesting TokenizationValidator...")
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x: x.split())

    validator = TokenizationValidator(tokenizer, max_length_diff=2)

    # Valid case: exact length
    clean = {"prompt": "hello world"}
    corrupted = {"prompt": "hello earth"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is True
    print("  [OK] Valid corruption (exact token length)")

    # Valid case: within tolerance
    clean = {"prompt": "hello world"}
    corrupted = {"prompt": "hello world foo"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is True
    print("  [OK] Valid corruption (within tolerance)")

    # Invalid case: exceeds tolerance
    validator = TokenizationValidator(tokenizer, max_length_diff=1)
    clean = {"prompt": "hello world"}
    corrupted = {"prompt": "hello world foo bar"}
    result = validator.validate(clean, corrupted)
    assert result.is_valid is False
    print("  [OK] Invalid corruption (exceeds tolerance)")


def test_composite_validator():
    """Test CompositeValidator works correctly."""
    print("\nTesting CompositeValidator...")
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x: x.split())

    validators = [
        LengthBudgetValidator(tolerance=0.2),
        LabelConsistencyValidator(tokenizer),
        TokenizationValidator(tokenizer, max_length_diff=2),
    ]

    composite = CompositeValidator(validators)

    # Valid case: all pass
    clean = {"prompt": "The answer is Paris", "answer": "Paris"}
    corrupted = {"prompt": "The capital is Paris", "answer": "Paris"}
    results = composite.validate(clean, corrupted)

    assert "LengthBudgetValidator" in results
    assert "LabelConsistencyValidator" in results
    assert "TokenizationValidator" in results
    print("  [OK] All validators executed")

    # Test is_all_valid
    is_valid = composite.is_all_valid(results)
    assert isinstance(is_valid, bool)
    print("  [OK] is_all_valid works")


def test_validation_result():
    """Test CorruptionValidationResult dataclass."""
    print("\nTesting CorruptionValidationResult...")

    result = CorruptionValidationResult(is_valid=True, reason="Test", severity=0.5)
    assert result.is_valid is True
    assert result.reason == "Test"
    assert result.severity == 0.5
    print("  [OK] Dataclass creation and access")

    result = CorruptionValidationResult(is_valid=False)
    assert result.reason is None
    assert result.severity == 0.0
    print("  [OK] Default values")


if __name__ == "__main__":
    try:
        test_length_budget_validator()
        test_label_consistency_validator()
        test_tokenization_validator()
        test_composite_validator()
        test_validation_result()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
