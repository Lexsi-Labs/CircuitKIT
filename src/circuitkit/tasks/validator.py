"""
DatasetValidator: Validate dataset schema completeness, field types, and format.

Provides comprehensive validation for datasets across different task types:
- Schema completeness (required fields present)
- Field type validation
- Task type auto-detection
- Answer format validation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a single validation error.

    Attributes:
        field: Field name where error occurred
        error_type: Type of error (missing, invalid_type, invalid_format, etc.)
        message: Human-readable error message
        examples: Optional example values that caused the error
    """

    field: str
    error_type: str
    message: str
    examples: Optional[List[Any]] = None


@dataclass
class ValidationResult:
    """Result of dataset validation.

    Attributes:
        is_valid: Whether dataset passed all validation checks
        detected_task_type: Auto-detected task type (classification, qa, mcq, ranking, open)
        total_examples: Number of examples in dataset
        valid_examples: Number of valid examples
        errors: List of validation errors
        warnings: List of non-blocking warnings
    """

    is_valid: bool
    detected_task_type: str
    total_examples: int
    valid_examples: int
    errors: List[ValidationError]
    warnings: List[str]

    def __str__(self) -> str:
        """Human-readable summary of validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        summary = f"Dataset Validation: {status}\n"
        summary += f"  Task Type: {self.detected_task_type}\n"
        summary += f"  Examples: {self.valid_examples}/{self.total_examples} valid\n"

        if self.errors:
            summary += f"  Errors ({len(self.errors)}):\n"
            for err in self.errors[:5]:  # Show first 5 errors
                summary += f"    - {err.field}: {err.message}\n"
            if len(self.errors) > 5:
                summary += f"    ... and {len(self.errors) - 5} more\n"

        if self.warnings:
            summary += f"  Warnings ({len(self.warnings)}):\n"
            for warn in self.warnings[:3]:
                summary += f"    - {warn}\n"
            if len(self.warnings) > 3:
                summary += f"    ... and {len(self.warnings) - 3} more\n"

        return summary


class DatasetValidator:
    """Validates datasets for use with CircuitKit tasks.

    Checks:
    1. Schema completeness: Required fields present
    2. Field types: Values match expected types
    3. Task type detection: Automatically infer task type from schema
    4. Answer format: Answers match expected format for task type
    5. Data quality: No null/NaN values in required fields
    """

    REQUIRED_FIELDS = {"prompt"}  # Minimum required field
    OPTIONAL_FIELDS = {
        "answer",
        "answers",
        "answer_tokens",  # Answer formats
        "context",
        "choices",
        "correct_choice_idx",  # Optional answer formats
        "id",
        "difficulty",
        "category",
        "metadata",  # Metadata
        "answer_start",
        "answer_end",  # Answer spans
    }

    TASK_TYPES = {"classification", "qa", "mcq", "ranking", "open"}

    def __init__(self):
        """Initialize DatasetValidator."""

    def validate(
        self,
        examples: List[Dict[str, Any]],
        schema: Optional[Dict[str, str]] = None,
        task_type: Optional[str] = None,
    ) -> ValidationResult:
        """Validate dataset examples.

        Args:
            examples: List of example dictionaries
            schema: Optional schema mapping (column_name -> field_name)
            task_type: Optional expected task type. If None, will auto-detect.

        Returns:
            ValidationResult with detailed validation information
        """
        if not examples:
            return ValidationResult(
                is_valid=False,
                detected_task_type="unknown",
                total_examples=0,
                valid_examples=0,
                errors=[
                    ValidationError(
                        field="dataset", error_type="empty", message="Dataset contains no examples"
                    )
                ],
                warnings=[],
            )

        errors = []
        warnings = []
        valid_count = 0

        # Auto-detect task type
        detected_type = task_type or self._detect_task_type(examples)

        # Validate each example
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                errors.append(
                    ValidationError(
                        field=f"example[{i}]",
                        error_type="invalid_type",
                        message=f"Example is {type(example).__name__}, expected dict",
                    )
                )
                continue

            # Check required fields
            if "prompt" not in example:
                errors.append(
                    ValidationError(
                        field=f"example[{i}].prompt",
                        error_type="missing",
                        message="Missing required 'prompt' field",
                    )
                )
                continue

            prompt = example.get("prompt")
            if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
                errors.append(
                    ValidationError(
                        field=f"example[{i}].prompt",
                        error_type="empty",
                        message="'prompt' field cannot be empty or null",
                    )
                )
                continue

            # Validate answer field(s) based on task type
            answer_valid = self._validate_answer_fields(example, detected_type, i, errors)

            if answer_valid:
                valid_count += 1
            else:
                errors.append(
                    ValidationError(
                        field=f"example[{i}]",
                        error_type="invalid_answer",
                        message=f"Invalid answer format for {detected_type} task",
                    )
                )

        # Check for null/NaN values
        null_warnings = self._check_null_values(examples, valid_count)
        warnings.extend(null_warnings)

        # Check for data quality issues
        quality_warnings = self._check_data_quality(examples)
        warnings.extend(quality_warnings)

        is_valid = len(errors) == 0 and valid_count > 0

        return ValidationResult(
            is_valid=is_valid,
            detected_task_type=detected_type,
            total_examples=len(examples),
            valid_examples=valid_count,
            errors=errors,
            warnings=warnings,
        )

    def _detect_task_type(self, examples: List[Dict[str, Any]]) -> str:
        """Auto-detect task type from example structure.

        Args:
            examples: List of examples

        Returns:
            Detected task type (classification, qa, mcq, ranking, open)
        """
        if not examples:
            return "open"

        example = examples[0]

        # Check for MCQ structure
        if "choices" in example:
            return "mcq"

        # Check for QA structure
        if "context" in example:
            return "qa"

        # Short wh-questions with a single answer are likely QA records.
        if "answer" in example and self._looks_like_qa_prompt(example.get("prompt")):
            return "qa"

        # Check for ranking/multi-answer
        if "answers" in example and isinstance(example.get("answers"), list):
            return "ranking"

        # Check for single answer (classification)
        if "answer" in example or "answer_tokens" in example:
            return "classification"

        # Default to open if only prompt
        return "open"

    def _looks_like_qa_prompt(self, prompt: Any) -> bool:
        """Heuristically detect question-answer prompts without context."""
        if not isinstance(prompt, str):
            return False

        stripped = prompt.strip().lower()
        question_starts = ("who", "what", "when", "where", "why", "how")
        return stripped.endswith("?") and stripped.startswith(question_starts)

    def _validate_answer_fields(
        self,
        example: Dict[str, Any],
        task_type: str,
        example_idx: int,
        errors: List[ValidationError],
    ) -> bool:
        """Validate answer fields based on task type.

        Args:
            example: Example dict
            task_type: Task type
            example_idx: Example index for error reporting
            errors: List to append errors to

        Returns:
            True if answer is valid, False otherwise
        """
        if task_type == "mcq":
            # Require choices and correct_choice_idx
            if "choices" not in example:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].choices",
                        error_type="missing",
                        message="MCQ task requires 'choices' field",
                    )
                )
                return False

            if "correct_choice_idx" not in example:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].correct_choice_idx",
                        error_type="missing",
                        message="MCQ task requires 'correct_choice_idx' field",
                    )
                )
                return False

            # Validate choices
            choices = example.get("choices", [])
            if not isinstance(choices, list) or len(choices) < 2:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].choices",
                        error_type="invalid_type",
                        message="'choices' must be list with at least 2 options",
                    )
                )
                return False

            # Validate index
            correct_idx = example.get("correct_choice_idx")
            if not isinstance(correct_idx, int) or correct_idx >= len(choices):
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].correct_choice_idx",
                        error_type="invalid_type",
                        message=f"'correct_choice_idx' must be valid index in [0, {len(choices)-1}]",
                    )
                )
                return False

            return True

        elif task_type == "qa":
            # Require context and answer(s)
            if "context" not in example:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].context",
                        error_type="missing",
                        message="QA task requires 'context' field",
                    )
                )
                return False

            has_answer = "answer" in example or "answers" in example
            if not has_answer:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].answer",
                        error_type="missing",
                        message="QA task requires 'answer' or 'answers' field",
                    )
                )
                return False

            return True

        elif task_type == "ranking":
            # Require multiple answers
            if "answers" not in example:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].answers",
                        error_type="missing",
                        message="Ranking task requires 'answers' field (list)",
                    )
                )
                return False

            answers = example.get("answers", [])
            if not isinstance(answers, list) or len(answers) < 2:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].answers",
                        error_type="invalid_type",
                        message="'answers' must be list with at least 2 items",
                    )
                )
                return False

            return True

        else:  # classification or open
            # Require at least one answer format
            has_answer = "answer" in example or "answers" in example or "answer_tokens" in example

            if not has_answer:
                errors.append(
                    ValidationError(
                        field=f"example[{example_idx}].answer",
                        error_type="missing",
                        message="Classification task requires 'answer', 'answers', or 'answer_tokens'",
                    )
                )
                return False

            return True

    def _check_null_values(
        self,
        examples: List[Dict[str, Any]],
        valid_count: int,
    ) -> List[str]:
        """Check for null/NaN values in required fields.

        Args:
            examples: List of examples
            valid_count: Number of valid examples

        Returns:
            List of warning messages
        """
        warnings = []

        # Sample first few examples
        for example in examples[: min(10, len(examples))]:
            if not isinstance(example, dict):
                continue

            prompt = example.get("prompt", "")
            if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
                warnings.append("Some examples have empty/null 'prompt' field")
                break

            answer = example.get("answer")
            if answer is None:
                warnings.append("Some examples have null 'answer' field")
                break

        return warnings

    def _check_data_quality(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[str]:
        """Check general data quality issues.

        Args:
            examples: List of examples

        Returns:
            List of warning messages
        """
        warnings = []

        if not examples:
            return warnings

        # Check for very short prompts
        short_prompts = sum(
            1 for ex in examples if isinstance(ex, dict) and len(str(ex.get("prompt", ""))) < 5
        )
        if short_prompts > len(examples) * 0.1:
            warnings.append(f"{short_prompts} examples have very short prompts (< 5 chars)")

        # Check for duplicate prompts
        prompts = [str(ex.get("prompt", "")) for ex in examples if isinstance(ex, dict)]
        duplicates = len(prompts) - len(set(prompts))
        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate prompts detected")

        # Check field consistency
        fields_per_example = [len(ex) for ex in examples if isinstance(ex, dict)]
        if fields_per_example and max(fields_per_example) != min(fields_per_example):
            warnings.append(
                f"Examples have inconsistent number of fields "
                f"({min(fields_per_example)} to {max(fields_per_example)})"
            )

        return warnings


def validate_dataset(
    examples: List[Dict[str, Any]],
    task_type: Optional[str] = None,
) -> ValidationResult:
    """Convenience function to validate a dataset.

    Args:
        examples: List of example dictionaries
        task_type: Optional expected task type

    Returns:
        ValidationResult with validation details
    """
    validator = DatasetValidator()
    return validator.validate(examples, task_type=task_type)
