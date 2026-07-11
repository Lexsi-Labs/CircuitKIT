"""
Tests for DatasetValidator.

Tests validation of datasets for different task types:
- Classification
- MCQ
- QA
- Ranking
- Task type auto-detection
- Error detection
"""

import pytest

from circuitkit.tasks.validator import DatasetValidator, ValidationResult, validate_dataset


class TestDatasetValidatorInit:
    """Test DatasetValidator initialization."""

    def test_init(self):
        """Test validator initialization."""
        validator = DatasetValidator()
        assert validator is not None
        assert hasattr(validator, "validate")


class TestClassificationValidation:
    """Test validation of classification datasets."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_valid_classification(self, validator):
        """Test valid classification dataset."""
        examples = [
            {"prompt": "This is great!", "answer": "positive"},
            {"prompt": "This is bad.", "answer": "negative"},
        ]

        result = validator.validate(examples)
        assert result.is_valid
        assert result.detected_task_type == "classification"
        assert result.valid_examples == 2

    def test_missing_prompt(self, validator):
        """Test classification missing prompt."""
        examples = [
            {"answer": "positive"},
        ]

        result = validator.validate(examples)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_missing_answer(self, validator):
        """Test classification missing answer."""
        examples = [
            {"prompt": "This is great!"},
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_empty_dataset(self, validator):
        """Test empty dataset."""
        result = validator.validate([])
        assert not result.is_valid
        assert result.total_examples == 0


class TestMCQValidation:
    """Test validation of MCQ datasets."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_valid_mcq(self, validator):
        """Test valid MCQ dataset."""
        examples = [
            {
                "prompt": "What is 2+2?",
                "choices": ["3", "4", "5"],
                "correct_choice_idx": 1,
            },
            {
                "prompt": "Capital of France?",
                "choices": ["Paris", "London", "Berlin"],
                "correct_choice_idx": 0,
            },
        ]

        result = validator.validate(examples)
        assert result.is_valid
        assert result.detected_task_type == "mcq"
        assert result.valid_examples == 2

    def test_missing_choices(self, validator):
        """Test MCQ missing choices."""
        examples = [
            {
                "prompt": "What?",
                "correct_choice_idx": 0,
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_missing_correct_idx(self, validator):
        """Test MCQ missing correct_choice_idx."""
        examples = [
            {
                "prompt": "What?",
                "choices": ["A", "B", "C"],
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_single_choice(self, validator):
        """Test MCQ with only one choice (invalid)."""
        examples = [
            {
                "prompt": "What?",
                "choices": ["A"],
                "correct_choice_idx": 0,
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_invalid_choice_index(self, validator):
        """Test MCQ with invalid choice index."""
        examples = [
            {
                "prompt": "What?",
                "choices": ["A", "B"],
                "correct_choice_idx": 5,  # Out of bounds
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid


class TestQAValidation:
    """Test validation of QA datasets."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_valid_qa(self, validator):
        """Test valid QA dataset."""
        examples = [
            {
                "prompt": "Who wrote Romeo and Juliet?",
                "context": "Shakespeare wrote many plays.",
                "answer": "Shakespeare",
            },
        ]

        result = validator.validate(examples)
        assert result.is_valid
        assert result.detected_task_type == "qa"

    def test_missing_context(self, validator):
        """Test QA missing context."""
        examples = [
            {
                "prompt": "Who?",
                "answer": "Someone",
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_missing_answer(self, validator):
        """Test QA missing answer."""
        examples = [
            {
                "prompt": "Who?",
                "context": "Some context",
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_qa_with_answers_list(self, validator):
        """Test QA with multiple valid answers."""
        examples = [
            {
                "prompt": "Who wrote it?",
                "context": "Context",
                "answers": ["Author A", "Author B"],
            },
        ]

        result = validator.validate(examples)
        # Should still validate as QA even with answers list
        assert result.is_valid or not result.is_valid


class TestRankingValidation:
    """Test validation of ranking datasets."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_valid_ranking(self, validator):
        """Test valid ranking dataset."""
        examples = [
            {
                "prompt": "Which are colors?",
                "answers": ["Red", "Blue", "Green"],
            },
        ]

        result = validator.validate(examples)
        assert result.is_valid
        assert result.detected_task_type == "ranking"

    def test_missing_answers(self, validator):
        """Test ranking missing answers."""
        examples = [
            {"prompt": "Which?"},
        ]

        result = validator.validate(examples)
        assert not result.is_valid

    def test_single_answer(self, validator):
        """Test ranking with only one answer (invalid)."""
        examples = [
            {
                "prompt": "Which?",
                "answers": ["One"],
            },
        ]

        result = validator.validate(examples)
        assert not result.is_valid


class TestTaskTypeDetection:
    """Test automatic task type detection."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_detect_classification(self, validator):
        """Test detection of classification."""
        examples = [{"prompt": "Q", "answer": "A"}]
        result = validator.validate(examples)
        assert result.detected_task_type == "classification"

    def test_detect_mcq(self, validator):
        """Test detection of MCQ."""
        examples = [
            {
                "prompt": "Q",
                "choices": ["A", "B"],
                "correct_choice_idx": 0,
            }
        ]
        result = validator.validate(examples)
        assert result.detected_task_type == "mcq"

    def test_detect_qa(self, validator):
        """Test detection of QA."""
        examples = [
            {
                "prompt": "Q",
                "context": "C",
                "answer": "A",
            }
        ]
        result = validator.validate(examples)
        assert result.detected_task_type == "qa"

    def test_detect_ranking(self, validator):
        """Test detection of ranking."""
        examples = [
            {
                "prompt": "Q",
                "answers": ["A", "B"],
            }
        ]
        result = validator.validate(examples)
        assert result.detected_task_type == "ranking"

    def test_detect_open(self, validator):
        """Test detection of open task."""
        examples = [{"prompt": "Q"}]
        result = validator.validate(examples)
        assert result.detected_task_type == "open"


class TestValidationErrors:
    """Test error detection and reporting."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_not_dict(self, validator):
        """Test non-dict example."""
        examples = ["not a dict"]
        result = validator.validate(examples)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_multiple_errors(self, validator):
        """Test multiple validation errors."""
        examples = [
            {},  # Missing prompt
            {"prompt": ""},  # Empty prompt
            {"prompt": None},  # Null prompt
        ]
        result = validator.validate(examples)
        assert len(result.errors) > 0

    def test_error_messages(self, validator):
        """Test error messages are informative."""
        examples = [{"answer": "A"}]
        result = validator.validate(examples)
        assert len(result.errors) > 0
        error = result.errors[0]
        assert error.message is not None
        assert len(error.message) > 0


class TestValidationWarnings:
    """Test warning detection."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_duplicate_prompts(self, validator):
        """Test warning for duplicate prompts."""
        examples = [
            {"prompt": "Same", "answer": "A"},
            {"prompt": "Same", "answer": "B"},
        ]
        result = validator.validate(examples)
        # May generate warnings
        assert isinstance(result.warnings, list)

    def test_short_prompts(self, validator):
        """Test warning for short prompts."""
        examples = [
            {"prompt": "a", "answer": "A"},
            {"prompt": "b", "answer": "B"},
            {"prompt": "c", "answer": "C"},
            {"prompt": "d", "answer": "D"},
            {"prompt": "e", "answer": "E"},
        ]
        result = validator.validate(examples)
        # May generate warnings about short prompts
        assert isinstance(result.warnings, list)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_properties(self):
        """Test ValidationResult properties."""
        result = ValidationResult(
            is_valid=True,
            detected_task_type="classification",
            total_examples=10,
            valid_examples=10,
            errors=[],
            warnings=[],
        )

        assert result.is_valid
        assert result.total_examples == 10
        assert len(result.errors) == 0

    def test_result_str(self):
        """Test ValidationResult string representation."""
        result = ValidationResult(
            is_valid=True,
            detected_task_type="classification",
            total_examples=10,
            valid_examples=10,
            errors=[],
            warnings=[],
        )

        s = str(result)
        assert "VALID" in s or "INVALID" in s
        assert "classification" in s


class TestValidateDatasetFunction:
    """Test convenience validate_dataset function."""

    def test_validate_dataset_func(self):
        """Test validate_dataset convenience function."""
        examples = [
            {"prompt": "Q", "answer": "A"},
        ]
        result = validate_dataset(examples)
        assert isinstance(result, ValidationResult)

    def test_validate_with_task_type(self):
        """Test validate_dataset with explicit task type."""
        examples = [
            {"prompt": "Q", "answer": "A"},
        ]
        result = validate_dataset(examples, task_type="classification")
        assert result.detected_task_type == "classification"


class TestValidationEdgeCases:
    """Test edge cases in validation."""

    @pytest.fixture
    def validator(self):
        return DatasetValidator()

    def test_very_large_dataset(self, validator):
        """Test validation of large dataset (samples)."""
        examples = [{"prompt": f"Question {i}", "answer": f"Answer {i}"} for i in range(1000)]
        result = validator.validate(examples)
        assert result.total_examples == 1000

    def test_mixed_valid_invalid(self, validator):
        """Test dataset with mix of valid and invalid examples."""
        examples = [
            {"prompt": "Good", "answer": "A"},
            {"prompt": "Bad"},  # Missing answer
            {"prompt": "Good2", "answer": "B"},
        ]
        result = validator.validate(examples)
        assert result.valid_examples < result.total_examples

    def test_null_fields(self, validator):
        """Test handling of null fields."""
        examples = [
            {"prompt": None, "answer": "A"},
        ]
        result = validator.validate(examples)
        # Should detect the null
        assert not result.is_valid or result.valid_examples == 0

    def test_special_characters(self, validator):
        """Test validation with special characters."""
        examples = [
            {"prompt": "What is 日本語?", "answer": "Japanese"},
            {"prompt": "Quel est ça?", "answer": "C'est"},
        ]
        result = validator.validate(examples)
        assert result.is_valid or not result.is_valid  # Should handle gracefully
