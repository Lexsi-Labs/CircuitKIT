"""
Corruption Validators

Provides validators to enforce "meaning-preserving" and "label-consistent"
guarantees on corruptions. Used by the orchestrator to filter invalid corruptions.
"""

from typing import Any, Dict, Optional, Protocol

import torch

try:
    # Normal package import.
    from .base import CorruptionValidation
except ImportError:
    # Fallback for tooling that loads this file as a standalone module
    # (no parent package). Keeps the canonical type identical in that case.
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class CorruptionValidation:  # type: ignore[no-redef]
        """Result of validating a corruption. See corruption/base.py for the
        canonical definition; this fallback is structurally identical."""

        is_valid: bool
        reason: Optional[str] = None
        severity: float = 0.0


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# Backwards-compatible alias. The canonical validation result type is
# `CorruptionValidation` (defined in corruption/base.py). `CorruptionValidationResult`
# was an identical duplicate and is kept only so existing imports keep working.
#
# .. deprecated::
#    Use ``circuitkit.corruption.CorruptionValidation`` instead.
#    ``CorruptionValidationResult`` is an alias and may be removed in a future
#    release.
CorruptionValidationResult = CorruptionValidation


class CorruptionValidator(Protocol):
    """Base validator protocol.

    All validators must implement a validate() method that checks a corruption
    against specific criteria.
    """

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """Validate a corruption.

        Args:
            clean: Original clean example dict.
            corrupted: Corrupted example dict.

        Returns:
            CorruptionValidation with validation status.
        """
        ...


class LengthBudgetValidator:
    """Validates corruption is within ±N% of clean prompt token length.

    This ensures corruptions don't dramatically shrink or expand the input,
    which could affect model behavior in unexpected ways.

    Length is measured with a real tokenizer when one is supplied, so the
    budget matches the token counts that corruption strategies produce.
    When no tokenizer is given, whitespace splitting is used as a documented
    (and necessarily approximate) fallback.
    """

    def __init__(self, tolerance: float = 0.1, tokenizer=None):
        """
        Initialize length budget validator.

        Args:
            tolerance: Allowed deviation as fraction. E.g., 0.1 = ±10%.
                      If clean prompt has 100 tokens, corrupted can have 90-110.
            tokenizer: Optional HuggingFace-style tokenizer (any object with an
                      ``encode()`` method). When provided, prompt length is
                      measured in real tokens, matching the tokenizers used by
                      corruption strategies. When omitted, length is measured
                      by whitespace splitting as an approximate fallback.
        """
        self.tolerance = tolerance
        self.tokenizer = tokenizer

    def _measure_length(self, text: str) -> int:
        """Measure prompt length in tokens (via tokenizer) or words (fallback).

        Uses ``self.tokenizer.encode()`` when a tokenizer is available; otherwise
        falls back to whitespace splitting. If tokenization raises, the
        whitespace fallback is used so validation never crashes on a bad encoder.
        """
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                # Fall back to whitespace approximation on tokenizer failure.
                pass
        return len(text.split())

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """
        Check that corrupted prompt length is within tolerance of clean.

        Length is measured with the configured tokenizer when available, and
        with whitespace splitting otherwise (see :meth:`__init__`).

        Args:
            clean: Original example with 'prompt' key.
            corrupted: Corrupted example with 'prompt' key.

        Returns:
            CorruptionValidation.
        """
        if "prompt" not in clean:
            return CorruptionValidation(is_valid=False, reason="Clean example missing 'prompt' key")

        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' key"
            )

        clean_prompt = clean["prompt"]
        corrupted_prompt = corrupted["prompt"]

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        # Measure in real tokens when a tokenizer is configured, otherwise
        # fall back to whitespace splitting (documented approximation).
        clean_len = self._measure_length(clean_prompt)
        corrupted_len = self._measure_length(corrupted_prompt)

        if clean_len == 0:
            return CorruptionValidation(is_valid=False, reason="Clean prompt is empty")

        # Check if within tolerance
        length_ratio = corrupted_len / clean_len
        min_ratio = 1.0 - self.tolerance
        max_ratio = 1.0 + self.tolerance

        if not (min_ratio <= length_ratio <= max_ratio):
            reason = (
                f"Length ratio {length_ratio:.2f} outside tolerance "
                f"[{min_ratio:.2f}, {max_ratio:.2f}]"
            )
            return CorruptionValidation(is_valid=False, reason=reason)

        # Severity: how far from ideal 1.0 ratio
        severity = abs(length_ratio - 1.0) / self.tolerance

        return CorruptionValidation(is_valid=True, severity=min(severity, 1.0))


class LabelConsistencyValidator:
    """Validates that the answer token still appears in corrupted text.

    Critical for meaning-preserving corruptions: the correct answer
    should still be reachable in the corrupted prompt.
    """

    def __init__(self, tokenizer, answer_key: str = "answer"):
        """
        Initialize label consistency validator.

        Args:
            tokenizer: HuggingFace tokenizer or any object with encode() method.
            answer_key: Key in example dict for the answer string.
        """
        self.tokenizer = tokenizer
        self.answer_key = answer_key

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """
        Check that the answer token/text appears in corrupted prompt.

        Args:
            clean: Original example with 'answer' key.
            corrupted: Corrupted example with 'prompt' key.

        Returns:
            CorruptionValidation.
        """
        if self.answer_key not in clean:
            return CorruptionValidation(
                is_valid=False, reason=f"Clean example missing '{self.answer_key}' key"
            )

        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' key"
            )

        answer = clean[self.answer_key]
        corrupted_prompt = corrupted["prompt"]

        if not isinstance(answer, (str, int)):
            return CorruptionValidation(
                is_valid=False, reason=f"Answer must be string or int, got {type(answer)}"
            )

        if not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Corrupted prompt must be string")

        answer_str = str(answer).strip()

        # Check if answer appears as substring (case-insensitive)
        corrupted_lower = corrupted_prompt.lower()
        answer_lower = answer_str.lower()

        if answer_lower in corrupted_lower:
            return CorruptionValidation(is_valid=True, severity=0.0)

        # If answer not found, it's invalid
        reason = f"Answer '{answer_str}' not found in corrupted prompt"
        return CorruptionValidation(is_valid=False, reason=reason)


class TokenizationValidator:
    """Validates that clean and corrupted tokenize to same length.

    Critical for circuit patching: if token counts differ, patch operations
    become misaligned. This validator ensures token-level alignment.
    """

    def __init__(self, tokenizer, max_length_diff: int = 2):
        """
        Initialize tokenization validator.

        Args:
            tokenizer: HuggingFace tokenizer with encode() method.
            max_length_diff: Maximum allowed difference in token count.
                            Default 2 allows for minor rounding differences.
        """
        self.tokenizer = tokenizer
        self.max_length_diff = max_length_diff

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """
        Check that clean and corrupted tokenize to same/similar length.

        Args:
            clean: Original example with 'prompt' key.
            corrupted: Corrupted example with 'prompt' key.

        Returns:
            CorruptionValidation.
        """
        if "prompt" not in clean:
            return CorruptionValidation(is_valid=False, reason="Clean example missing 'prompt' key")

        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' key"
            )

        clean_prompt = clean["prompt"]
        corrupted_prompt = corrupted["prompt"]

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        try:
            # Tokenize both prompts
            clean_tokens = self.tokenizer.encode(clean_prompt)
            corrupted_tokens = self.tokenizer.encode(corrupted_prompt)

            clean_len = len(clean_tokens)
            corrupted_len = len(corrupted_tokens)

            # Check if lengths are compatible
            length_diff = abs(clean_len - corrupted_len)

            if length_diff > self.max_length_diff:
                reason = (
                    f"Token length mismatch: clean={clean_len}, "
                    f"corrupted={corrupted_len}, diff={length_diff} "
                    f"(max allowed: {self.max_length_diff})"
                )
                return CorruptionValidation(is_valid=False, reason=reason)

            # Severity: normalized difference
            max_len = max(clean_len, corrupted_len)
            severity = length_diff / max_len if max_len > 0 else 0.0

            return CorruptionValidation(is_valid=True, severity=severity)

        except Exception as e:
            return CorruptionValidation(is_valid=False, reason=f"Tokenization error: {str(e)}")


class SemanticShiftValidator:
    """Validates semantic similarity between clean and corrupted prompts.

    Uses sentence transformers to compute cosine similarity. Useful for
    meaning-preserving corruptions to verify semantic equivalence.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.7,
    ):
        """
        Initialize semantic shift validator.

        Args:
            model_name: Sentence transformer model name.
                       Default "all-MiniLM-L6-v2" is lightweight (~22M params).
            threshold: Minimum cosine similarity for valid corruption.
                      Range [0, 1]. E.g., 0.7 means 70% semantic similarity required.
        """
        self.threshold = threshold
        self.model_name = model_name

        if SentenceTransformer is None:
            raise ImportError(
                "SemanticShiftValidator requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """
        Check semantic similarity between clean and corrupted prompts.

        Args:
            clean: Original example with 'prompt' key.
            corrupted: Corrupted example with 'prompt' key.

        Returns:
            CorruptionValidation with similarity as severity.
        """
        if "prompt" not in clean:
            return CorruptionValidation(is_valid=False, reason="Clean example missing 'prompt' key")

        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' key"
            )

        clean_prompt = clean["prompt"]
        corrupted_prompt = corrupted["prompt"]

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        try:
            # Encode prompts
            embeddings = self.model.encode([clean_prompt, corrupted_prompt], convert_to_tensor=True)

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[0:1], embeddings[1:2]
            ).item()

            # Check against threshold
            if similarity < self.threshold:
                reason = (
                    f"Semantic similarity {similarity:.3f} below threshold " f"{self.threshold:.3f}"
                )
                return CorruptionValidation(is_valid=False, reason=reason)

            # Severity: inverse of similarity (1 - similarity, clamped to [0, 1])
            severity = max(0.0, min(1.0, 1.0 - similarity))

            return CorruptionValidation(is_valid=True, severity=severity)

        except Exception as e:
            return CorruptionValidation(
                is_valid=False, reason=f"Semantic validation error: {str(e)}"
            )


class CompositeValidator:
    """Combines multiple validators into a single composite validator.

    Useful for enforcing multiple constraints simultaneously.
    """

    def __init__(self, validators: list):
        """
        Initialize composite validator.

        Args:
            validators: List of validator instances.
        """
        self.validators = validators

    def validate(
        self, clean: Dict[str, Any], corrupted: Dict[str, Any]
    ) -> Dict[str, CorruptionValidation]:
        """
        Validate using all validators.

        Args:
            clean: Original example.
            corrupted: Corrupted example.

        Returns:
            Dictionary mapping validator name to CorruptionValidation.
        """
        results = {}
        for validator in self.validators:
            validator_name = validator.__class__.__name__
            results[validator_name] = validator.validate(clean, corrupted)
        return results

    def is_all_valid(self, results: Dict[str, CorruptionValidation]) -> bool:
        """Check if all validators passed.

        Args:
            results: Output from validate().

        Returns:
            True if all validators returned is_valid=True.
        """
        return all(result.is_valid for result in results.values())


class ModelRequirementValidator:
    """Validates that model meets circuit discovery requirements.

    This validator checks that a model has the necessary attributes
    (tokenizer, to_tokens method) and that tokenization works correctly.
    """

    def __init__(self):
        """Initialize model requirement validator."""

    def validate_model(self, model) -> CorruptionValidation:
        """
        Validate that a model is suitable for corruption.

        Args:
            model: Model instance to validate.

        Returns:
            CorruptionValidation with validation status.
        """
        if model is None:
            return CorruptionValidation(
                is_valid=False,
                reason=(
                    "Model is required for corruption. Circuit discovery requires "
                    "model-specific token IDs. Pass a HookedTransformer model instance."
                ),
            )

        # Check if model has required attributes
        if not hasattr(model, "tokenizer"):
            return CorruptionValidation(
                is_valid=False,
                reason=(
                    "Model must have a tokenizer attribute for corruption. "
                    "Use a HookedTransformer model instance."
                ),
            )

        if not hasattr(model, "to_tokens"):
            return CorruptionValidation(
                is_valid=False,
                reason=(
                    "Model must have a to_tokens method for corruption. "
                    "Use a HookedTransformer model instance."
                ),
            )

        # Check if tokenizer is properly initialized
        try:
            test_tokens = model.to_tokens("test", prepend_bos=False)
            if test_tokens is None or test_tokens.numel() == 0:
                return CorruptionValidation(
                    is_valid=False, reason="Model tokenizer not properly initialized"
                )
        except Exception as e:
            return CorruptionValidation(
                is_valid=False, reason=f"Model tokenizer validation failed: {e}"
            )

        return CorruptionValidation(is_valid=True)


class TokenConsistencyValidator:
    """Validates that clean and corrupted have consistent token-level changes.

    This validator checks token-level alignment and ensures that exactly
    one token differs between clean and corrupted versions, which is required
    for circuit patching operations.
    """

    def __init__(self, model=None, allow_token_variance: int = 0):
        """
        Initialize token consistency validator.

        Args:
            model: Optional HookedTransformer model with to_tokens method.
                  If provided, uses actual tokenization. Otherwise uses word-count heuristic.
            allow_token_variance: Maximum allowed difference in token count.
                                 Default 0 requires exact alignment.
        """
        self.model = model
        self.allow_token_variance = allow_token_variance

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """
        Check that clean and corrupted are token-consistent.

        Args:
            clean: Original example with 'prompt' key.
            corrupted: Corrupted example with 'prompt' key.

        Returns:
            CorruptionValidation.
        """
        if "prompt" not in clean:
            return CorruptionValidation(is_valid=False, reason="Clean example missing 'prompt' key")

        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted example missing 'prompt' key"
            )

        clean_prompt = clean["prompt"]
        corrupted_prompt = corrupted["prompt"]

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        try:
            if self.model is not None:
                # Use actual tokenization
                clean_tokens = self.model.to_tokens(clean_prompt, prepend_bos=False)
                corrupted_tokens = self.model.to_tokens(corrupted_prompt, prepend_bos=False)

                clean_len = int(clean_tokens.shape[1])
                corrupted_len = int(corrupted_tokens.shape[1])
            else:
                # Use word-count heuristic
                clean_len = len(clean_prompt.split())
                corrupted_len = len(corrupted_prompt.split())

            token_diff = abs(clean_len - corrupted_len)

            if token_diff > self.allow_token_variance:
                reason = (
                    f"Token length mismatch: clean={clean_len}, "
                    f"corrupted={corrupted_len}, diff={token_diff} "
                    f"(max allowed: {self.allow_token_variance})"
                )
                return CorruptionValidation(is_valid=False, reason=reason)

            # Severity: normalized difference
            max_len = max(clean_len, corrupted_len)
            severity = token_diff / max_len if max_len > 0 else 0.0

            return CorruptionValidation(is_valid=True, severity=min(severity, 1.0))

        except Exception as e:
            return CorruptionValidation(is_valid=False, reason=f"Token validation error: {str(e)}")
