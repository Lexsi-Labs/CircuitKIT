"""
DEPRECATED: Use circuitkit.corruption.validators instead.

This module is maintained for backward compatibility only.
All functionality has been consolidated into the new validators module.

New code should import from:
    from circuitkit.corruption import (
        CorruptionValidationResult,
        ModelRequirementValidator,
        TokenConsistencyValidator,
        LengthBudgetValidator,
        LabelConsistencyValidator,
        TokenizationValidator,
        SemanticShiftValidator,
        CompositeValidator,
    )

Old code can continue importing from this module, but deprecation warnings
will be issued in future versions.
"""

import warnings
from typing import Any, Dict, List

# Re-export all validators from the new location
from circuitkit.corruption.validators import (
    CompositeValidator,
    CorruptionValidationResult,
    CorruptionValidator,
    LabelConsistencyValidator,
    LengthBudgetValidator,
    ModelRequirementValidator,
    SemanticShiftValidator,
    TokenConsistencyValidator,
    TokenizationValidator,
)

__all__ = [
    "CorruptionValidationResult",
    "CorruptionValidator",
    "LengthBudgetValidator",
    "LabelConsistencyValidator",
    "TokenizationValidator",
    "SemanticShiftValidator",
    "CompositeValidator",
    "ModelRequirementValidator",
    "TokenConsistencyValidator",
    # Legacy functions below
    "validate_model_for_corruption",
    "validate_corruption_output",
    "validate_corruption_config",
    "validate_corruption_batch_output",
    "filter_valid_corruptions",
    "check_circuit_discovery_safety",
    "compute_label_token_ids",
]


# Legacy convenience functions for backward compatibility
def validate_model_for_corruption(model) -> None:
    """
    DEPRECATED: Use ModelRequirementValidator.validate_model() instead.

    Validate model is suitable for corruption with helpful error messages.

    Args:
        model: Model instance to validate

    Raises:
        ValueError: If model is None or invalid
    """
    warnings.warn(
        "validate_model_for_corruption is deprecated. Use ModelRequirementValidator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    validator = ModelRequirementValidator()
    result = validator.validate_model(model)
    if not result.is_valid:
        raise ValueError(result.reason)


def validate_corruption_output(corruption_result: dict, model) -> bool:
    """
    DEPRECATED: Use CorruptionValidationResult and validators instead.

    Validate corruption output meets circuit discovery requirements.

    Args:
        corruption_result: Dictionary containing corruption result
        model: Model instance for validation

    Returns:
        True if corruption output is valid, False otherwise
    """
    warnings.warn(
        "validate_corruption_output is deprecated. Use specific validators instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(corruption_result, dict):
        return False

    # Check required fields
    required_fields = ["clean", "corrupted", "correct_idx", "incorrect_idx"]
    for field in required_fields:
        if field not in corruption_result:
            return False

    # Validate text fields
    clean = corruption_result["clean"]
    corrupted = corruption_result["corrupted"]

    if not isinstance(clean, str) or not isinstance(corrupted, str):
        return False

    if not clean.strip() or not corrupted.strip():
        return False

    # Validate token IDs
    correct_idx = corruption_result["correct_idx"]
    incorrect_idx = corruption_result["incorrect_idx"]

    if not isinstance(correct_idx, (int, type(__import__("torch").Tensor))):
        return False
    if not isinstance(incorrect_idx, (int, type(__import__("torch").Tensor))):
        return False

    # Convert to int if tensor
    if hasattr(correct_idx, "item"):
        correct_idx = correct_idx.item()
    if hasattr(incorrect_idx, "item"):
        incorrect_idx = incorrect_idx.item()

    # Check token IDs are valid for model vocabulary
    try:
        vocab_size = model.cfg.d_vocab
        if correct_idx < 0 or correct_idx >= vocab_size:
            return False
        if incorrect_idx < 0 or incorrect_idx >= vocab_size:
            return False
    except AttributeError:
        # If vocab size not available, just check they're non-negative
        if correct_idx < 0 or incorrect_idx < 0:
            return False

    # Check that clean and corrupted are different
    if clean == corrupted:
        return False

    return True


def validate_corruption_config(config: Dict[str, Any], strategy_name: str) -> None:
    """
    DEPRECATED: Use ModelRequirementValidator instead.

    Validate corruption configuration for a specific strategy.

    Args:
        config: Configuration dictionary
        strategy_name: Name of the corruption strategy

    Raises:
        ValueError: If configuration is invalid
    """
    warnings.warn(
        "validate_corruption_config is deprecated. Use ModelRequirementValidator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(config, dict):
        raise ValueError(f"{strategy_name}: config must be a dict")

    # Check for model parameter
    if "model" not in config:
        raise ValueError(f"{strategy_name}: 'model' is required. No default model.")

    if config["model"] is None:
        raise ValueError(
            f"{strategy_name}: 'model' cannot be None. Circuit discovery requires model-specific token IDs."
        )

    # Validate model
    validate_model_for_corruption(config["model"])


def validate_corruption_batch_output(outputs: List[Dict[str, Any]], model) -> List[bool]:
    """
    DEPRECATED: Use validators in batch mode instead.

    Validate a batch of corruption outputs.

    Args:
        outputs: List of corruption result dictionaries
        model: Model instance for validation

    Returns:
        List of boolean validation results for each output
    """
    warnings.warn(
        "validate_corruption_batch_output is deprecated. Use CompositeValidator with batch processing instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(outputs, list):
        return [False]

    validation_results = []
    for output in outputs:
        try:
            is_valid = validate_corruption_output(output, model)
            validation_results.append(is_valid)
        except Exception:
            validation_results.append(False)

    return validation_results


def filter_valid_corruptions(outputs: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use validators with filtering logic instead.

    Filter corruption outputs to keep only valid ones.

    Args:
        outputs: List of corruption result dictionaries
        model: Model instance for validation

    Returns:
        List of valid corruption results
    """
    warnings.warn(
        "filter_valid_corruptions is deprecated. Use validators for filtering instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    validation_results = validate_corruption_batch_output(outputs, model)
    return [output for output, is_valid in zip(outputs, validation_results) if is_valid]


def check_circuit_discovery_safety(clean: str, corrupted: str, model) -> Dict[str, Any]:
    """
    DEPRECATED: Use TokenConsistencyValidator instead.

    Check if a corruption meets circuit discovery safety requirements.

    Args:
        clean: Original text
        corrupted: Corrupted text
        model: Model instance for tokenization

    Returns:
        Dictionary with safety check results
    """
    warnings.warn(
        "check_circuit_discovery_safety is deprecated. Use TokenConsistencyValidator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        # Tokenize both texts
        clean_tokens = model.to_tokens(clean, prepend_bos=False)
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=False)

        # Check token count alignment
        clean_count = clean_tokens.shape[1]
        corrupted_count = corrupted_tokens.shape[1]
        token_count_aligned = clean_count == corrupted_count

        # Count differing tokens
        if token_count_aligned:
            differences = 0
            for i in range(clean_count):
                if clean_tokens[0, i] != corrupted_tokens[0, i]:
                    differences += 1
        else:
            differences = -1  # Cannot count if lengths differ

        # Check single token change constraint
        single_token_change = differences == 1

        # Check for reasonable length similarity
        clean_words = clean.split()
        corrupted_words = corrupted.split()
        length_ratio = len(corrupted_words) / len(clean_words) if clean_words else 0
        reasonable_length = 0.5 <= length_ratio <= 2.0

        return {
            "token_count_aligned": token_count_aligned,
            "differences": differences,
            "single_token_change": single_token_change,
            "reasonable_length": reasonable_length,
            "length_ratio": length_ratio,
            "safe_for_circuit_discovery": (
                token_count_aligned and single_token_change and reasonable_length
            ),
        }

    except Exception as e:
        return {
            "token_count_aligned": False,
            "differences": -1,
            "single_token_change": False,
            "reasonable_length": False,
            "length_ratio": 0,
            "safe_for_circuit_discovery": False,
            "error": str(e),
        }


def compute_label_token_ids(clean: str, corrupted: str, model) -> Dict[str, int]:
    """
    DEPRECATED: Use TokenConsistencyValidator instead.

    Compute correct/incorrect token IDs by finding the single differing token
    position between clean and corrupted texts using the provided model's tokenizer.

    Requirements:
    - Token sequences must be the same length
    - Exactly one token must differ

    Returns:
        A dict with keys 'correct_idx' and 'incorrect_idx' as Python ints

    Raises:
        ValueError if constraints are not met (length mismatch or != 1 diff)
    """
    warnings.warn(
        "compute_label_token_ids is deprecated. Use TokenConsistencyValidator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(clean, str) or not isinstance(corrupted, str):
        raise ValueError("compute_label_token_ids: clean and corrupted must be strings")

    # Tokenize without BOS to align with discovery datasets
    clean_tokens = model.to_tokens(clean, prepend_bos=False)
    corrupted_tokens = model.to_tokens(corrupted, prepend_bos=False)

    if clean_tokens.ndim != 2 or corrupted_tokens.ndim != 2:
        raise ValueError("compute_label_token_ids: unexpected token tensor shape")

    clean_len = int(clean_tokens.shape[1])
    corrupted_len = int(corrupted_tokens.shape[1])
    if clean_len != corrupted_len:
        raise ValueError("compute_label_token_ids: token length mismatch; require aligned lengths")

    diff_positions: List[int] = []
    for i in range(clean_len):
        if int(clean_tokens[0, i].item()) != int(corrupted_tokens[0, i].item()):
            diff_positions.append(i)

    if len(diff_positions) != 1:
        raise ValueError("compute_label_token_ids: require exactly one differing token position")

    pos = diff_positions[0]
    correct_idx = int(clean_tokens[0, pos].item())
    incorrect_idx = int(corrupted_tokens[0, pos].item())

    return {"correct_idx": correct_idx, "incorrect_idx": incorrect_idx}
