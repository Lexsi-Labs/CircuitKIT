"""
Test suite for Auto-Schema Detection feature.

Tests:
- SchemaAnalyzer column role inference
- Task type detection (QA, MCQ, Classification, Ranking, Paraphrase)
- auto_task_from_hf factory function
- Custom mapping overrides
- Preview and validation functions
- Edge cases and unknown schemas
"""

import pytest

from circuitkit.tasks import (
    SchemaAnalyzer,
    TaskType,
    auto_task_from_hf,
    list_compatible_datasets,
    preview_schema,
    validate_hf_dataset,
)

# ===== Fixtures =====


@pytest.fixture
def squad_like_dataset():
    """Simulate SQuAD dataset structure."""
    return [
        {
            "id": "56be4db0acb8001400a502ec",
            "title": "Super_Bowl_50",
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "context": "Super Bowl 50 was an American football game...",
            "answer": "Denver Broncos",
            "answer_start": 177,
        },
        {
            "id": "56be4db0acb8001400a502ed",
            "title": "Super_Bowl_50",
            "question": "What color was used to emphasize the 50th anniversary?",
            "context": "As an additional feature to the Super Bowl...",
            "answer": "gold",
            "answer_start": 155,
        },
    ]


@pytest.fixture
def mmlu_like_dataset():
    """Simulate MMLU MCQ dataset structure."""
    return [
        {
            "question": "Which planet is closest to the sun?",
            "choices": ["Mercury", "Venus", "Earth", "Mars"],
            "answer": 0,
        },
        {
            "question": "What is the chemical symbol for gold?",
            "choices": ["Go", "Gd", "Au", "Ag"],
            "answer": 2,
        },
    ]


@pytest.fixture
def glue_sst2_like_dataset():
    """Simulate GLUE SST-2 sentiment classification."""
    return [
        {
            "sentence": "This movie was amazing!",
            "label": 1,
        },
        {
            "sentence": "I didn't like it at all.",
            "label": 0,
        },
        {
            "sentence": "It was okay.",
            "label": 0,
        },
    ]


@pytest.fixture
def ranking_like_dataset():
    """Simulate ranking/retrieval dataset (BEIR-style)."""
    return [
        {
            "query": "What is machine learning?",
            "candidates": [
                "Machine learning is a subset of AI...",
                "ML is a programming paradigm...",
                "Learning by machines is artificial...",
            ],
            "relevance_scores": [0.9, 0.7, 0.3],
        },
        {
            "query": "How to train neural networks?",
            "candidates": [
                "Training involves backpropagation...",
                "Use gradient descent methods...",
            ],
            "relevance_scores": [0.85, 0.8],
        },
    ]


@pytest.fixture
def paraphrase_like_dataset():
    """Simulate paraphrase detection (MRPC-style)."""
    return [
        {
            "sentence1": "The cat is on the mat.",
            "sentence2": "A feline rests upon the mat.",
            "label": 1,
        },
        {
            "sentence1": "I like apples.",
            "sentence2": "Oranges are fruits.",
            "label": 0,
        },
    ]


@pytest.fixture
def unknown_dataset():
    """Dataset with unclear structure."""
    return [
        {
            "field_a": "some text",
            "field_b": "other text",
            "field_c": 42,
        },
        {
            "field_a": "another text",
            "field_b": "more text",
            "field_c": 99,
        },
    ]


# ===== Column Role Inference Tests =====


def test_infer_column_roles_squad(squad_like_dataset):
    """Test column role inference for SQuAD-like dataset."""
    columns = set(squad_like_dataset[0].keys())
    roles = SchemaAnalyzer.infer_column_roles(columns, squad_like_dataset)

    assert "question" in roles
    assert "context" in roles
    assert "answer" in roles
    assert roles["question"] == "question"
    assert roles["context"] == "context"
    assert roles["answer"] == "answer"


def test_infer_column_roles_mmlu(mmlu_like_dataset):
    """Test column role inference for MMLU-like dataset."""
    columns = set(mmlu_like_dataset[0].keys())
    roles = SchemaAnalyzer.infer_column_roles(columns, mmlu_like_dataset)

    assert "question" in roles
    assert "choices" in roles
    assert "correct_answer_idx" in roles


def test_infer_column_roles_glue_sst2(glue_sst2_like_dataset):
    """Test column role inference for GLUE SST-2."""
    columns = set(glue_sst2_like_dataset[0].keys())
    roles = SchemaAnalyzer.infer_column_roles(columns, glue_sst2_like_dataset)

    assert "question" in roles  # 'sentence' -> question
    assert "answer" in roles  # 'label' -> answer


def test_infer_column_roles_ranking(ranking_like_dataset):
    """Test column role inference for ranking dataset."""
    columns = set(ranking_like_dataset[0].keys())
    roles = SchemaAnalyzer.infer_column_roles(columns, ranking_like_dataset)

    assert "question" in roles or "query" in roles
    # Should infer candidates/answers from list structure
    assert "answers" in roles or "candidates" in roles


# ===== Task Type Detection Tests =====


def test_detect_qa_task(squad_like_dataset):
    """Test QA task detection."""
    columns = set(squad_like_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, squad_like_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, squad_like_dataset, inferred
    )

    assert task_type == TaskType.QA
    assert confidence > 0.7
    assert features["has_question"]
    assert features["has_context"]
    assert features["has_answer"]


def test_detect_mcq_task(mmlu_like_dataset):
    """Test MCQ task detection."""
    columns = set(mmlu_like_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, mmlu_like_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, mmlu_like_dataset, inferred
    )

    assert task_type == TaskType.MCQ
    assert confidence > 0.7
    assert features["has_choices"]
    assert features["has_correct_idx"]


def test_detect_classification_task(glue_sst2_like_dataset):
    """Test classification task detection."""
    columns = set(glue_sst2_like_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, glue_sst2_like_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, glue_sst2_like_dataset, inferred
    )

    assert task_type == TaskType.CLASSIFICATION
    assert confidence > 0.7


def test_detect_ranking_task(ranking_like_dataset):
    """Test ranking task detection."""
    columns = set(ranking_like_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, ranking_like_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, ranking_like_dataset, inferred
    )

    assert task_type == TaskType.RANKING
    assert confidence > 0.7


def test_detect_paraphrase_task(paraphrase_like_dataset):
    """Test paraphrase task detection."""
    columns = set(paraphrase_like_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, paraphrase_like_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, paraphrase_like_dataset, inferred
    )

    assert task_type == TaskType.PARAPHRASE
    assert confidence > 0.6


def test_detect_unknown_task(unknown_dataset):
    """Test detection of unknown task type."""
    columns = set(unknown_dataset[0].keys())
    inferred = SchemaAnalyzer.infer_column_roles(columns, unknown_dataset)

    task_type, confidence, features, reasoning = SchemaAnalyzer.detect_task_type(
        columns, unknown_dataset, inferred
    )

    assert task_type == TaskType.UNKNOWN
    assert confidence == 0.0


# ===== Schema Analysis Tests =====


def test_analyze_squad_dataset(squad_like_dataset):
    """Test full schema analysis for SQuAD."""
    detection = SchemaAnalyzer.analyze(squad_like_dataset)

    assert detection.task_type == TaskType.QA
    assert detection.confidence > 0.7
    assert "question" in detection.suggested_mapping
    assert "context" in detection.suggested_mapping
    assert "answer" in detection.suggested_mapping


def test_analyze_mmlu_dataset(mmlu_like_dataset):
    """Test full schema analysis for MMLU."""
    detection = SchemaAnalyzer.analyze(mmlu_like_dataset)

    assert detection.task_type == TaskType.MCQ
    assert detection.confidence > 0.7
    assert "choices" in detection.suggested_mapping


def test_analyze_empty_dataset():
    """Test analysis of empty dataset."""
    detection = SchemaAnalyzer.analyze([])

    assert detection.task_type == TaskType.UNKNOWN
    assert detection.confidence == 0.0


# ===== Mapping Suggestion Tests =====


def test_suggest_mapping_qa(squad_like_dataset):
    """Test mapping suggestion for QA task."""
    mapping = SchemaAnalyzer.suggest_mapping(squad_like_dataset, TaskType.QA)

    assert "prompt" in mapping
    assert "answer" in mapping
    assert "context" in mapping


def test_suggest_mapping_mcq(mmlu_like_dataset):
    """Test mapping suggestion for MCQ task."""
    mapping = SchemaAnalyzer.suggest_mapping(mmlu_like_dataset, TaskType.MCQ)

    assert "prompt" in mapping
    assert "choices" in mapping
    assert "correct_choice_idx" in mapping


def test_suggest_mapping_classification(glue_sst2_like_dataset):
    """Test mapping suggestion for classification."""
    mapping = SchemaAnalyzer.suggest_mapping(glue_sst2_like_dataset, TaskType.CLASSIFICATION)

    assert "prompt" in mapping
    assert "answer" in mapping


# ===== Factory Function Tests =====


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_auto_task_from_hf_squad():
    """Test auto_task_from_hf with SQuAD dataset."""
    task = auto_task_from_hf("squad", split="validation")

    assert task.name == "squad"
    assert task.task_type == "qa"
    assert "question" in task.schema
    assert "context" in task.schema


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_auto_task_from_hf_glue():
    """Test auto_task_from_hf with GLUE SST-2."""
    task = auto_task_from_hf("glue", subset="sst2", split="train")

    assert task.name == "glue_sst2"
    assert task.task_type == "classification"


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_auto_task_from_hf_custom_mapping():
    """Test auto_task_from_hf with custom mapping override."""
    custom_mapping = {"prompt": "question", "answer": "answer_text"}
    task = auto_task_from_hf("squad", split="validation", custom_mapping=custom_mapping)

    assert task.schema["prompt"] == "question"
    assert task.schema["answer"] == "answer_text"


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_auto_task_from_hf_force_task_type():
    """Test auto_task_from_hf with forced task type."""
    task = auto_task_from_hf("squad", split="validation", force_task_type="ranking")

    # Should still create valid task, even if type doesn't match
    assert task.name == "squad"


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_list_compatible_datasets():
    """Test listing compatible datasets."""
    datasets_list = list_compatible_datasets()

    assert len(datasets_list) > 0
    assert all("name" in d for d in datasets_list)
    assert all("task_type" in d for d in datasets_list)


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_list_compatible_datasets_filtered():
    """Test listing datasets filtered by task type."""
    qa_datasets = list_compatible_datasets(task_type="qa")

    assert len(qa_datasets) > 0
    assert all(d["task_type"] == "qa" for d in qa_datasets)


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_preview_schema():
    """Test schema preview function."""
    preview = preview_schema("squad", split="validation")

    assert preview.dataset_name == "squad"
    assert preview.detected_task_type == "qa"
    assert preview.num_examples > 0
    assert "question" in preview.suggested_mapping


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_validate_hf_dataset_valid():
    """Test dataset validation for valid dataset."""
    result = validate_hf_dataset("squad", split="validation")

    assert result["is_valid"]
    assert result["num_examples"] > 0
    assert "question" in result["columns"]


@pytest.mark.skip(reason="Requires datasets library and HF connection")
def test_validate_hf_dataset_invalid():
    """Test dataset validation for invalid dataset."""
    result = validate_hf_dataset("nonexistent_dataset_xyz")

    assert not result["is_valid"]
    assert result["error"] is not None


# ===== Mocked Integration Tests =====


def test_auto_task_from_hf_mocked(squad_like_dataset, monkeypatch):
    """Test auto_task_from_hf with mocked datasets library."""
    from unittest.mock import MagicMock

    # Mock datasets.load_dataset
    mock_dataset = MagicMock()
    mock_dataset.select.return_value = squad_like_dataset
    mock_dataset.__len__.return_value = len(squad_like_dataset)

    def mock_load_dataset(name, *args, **kwargs):
        return mock_dataset

    monkeypatch.setattr("datasets.load_dataset", mock_load_dataset)

    # Mock datasets import
    import sys

    mock_datasets = MagicMock()
    mock_datasets.load_dataset = mock_load_dataset
    sys.modules["datasets"] = mock_datasets

    try:
        task = auto_task_from_hf("squad", split="validation")

        assert task.name == "squad"
        assert task.task_type == "qa"
        assert "prompt" in task.schema
        assert "answer" in task.schema
    finally:
        # Cleanup
        if "datasets" in sys.modules and isinstance(sys.modules["datasets"], MagicMock):
            del sys.modules["datasets"]


def test_custom_mapping_override_mocked(squad_like_dataset, monkeypatch):
    """Test custom mapping override with mocked dataset."""
    from unittest.mock import MagicMock

    mock_dataset = MagicMock()
    mock_dataset.select.return_value = squad_like_dataset
    mock_dataset.__len__.return_value = len(squad_like_dataset)

    def mock_load_dataset(name, *args, **kwargs):
        return mock_dataset

    monkeypatch.setattr("datasets.load_dataset", mock_load_dataset)

    import sys

    mock_datasets = MagicMock()
    mock_datasets.load_dataset = mock_load_dataset
    sys.modules["datasets"] = mock_datasets

    try:
        custom_mapping = {"prompt": "question", "answer": "answer", "context": "context"}
        task = auto_task_from_hf("squad", split="validation", custom_mapping=custom_mapping)

        assert task.schema == custom_mapping
    finally:
        if "datasets" in sys.modules and isinstance(sys.modules["datasets"], MagicMock):
            del sys.modules["datasets"]


# ===== Edge Cases =====


def test_detect_helper_functions_squad(squad_like_dataset):
    """Test individual detection helper functions."""
    columns = set(squad_like_dataset[0].keys())
    samples = squad_like_dataset

    assert SchemaAnalyzer.detect_qa_task(columns, samples)
    assert not SchemaAnalyzer.detect_mcq_task(columns, samples)
    assert not SchemaAnalyzer.detect_classification_task(columns, samples)


def test_detect_helper_functions_mmlu(mmlu_like_dataset):
    """Test individual detection helper functions."""
    columns = set(mmlu_like_dataset[0].keys())
    samples = mmlu_like_dataset

    assert not SchemaAnalyzer.detect_qa_task(columns, samples)
    assert SchemaAnalyzer.detect_mcq_task(columns, samples)
    assert not SchemaAnalyzer.detect_classification_task(columns, samples)


def test_single_example_analysis(squad_like_dataset):
    """Test analysis with just one example."""
    single_sample = [squad_like_dataset[0]]
    detection = SchemaAnalyzer.analyze(single_sample)

    assert detection.task_type == TaskType.QA
    assert detection.confidence > 0.7


def test_max_samples_limit(squad_like_dataset):
    """Test that max_samples parameter is respected."""
    # Expand dataset to 1000 examples
    large_dataset = squad_like_dataset * 500

    detection = SchemaAnalyzer.analyze(large_dataset, max_samples=10)

    # Should complete quickly even with large dataset
    assert detection.task_type == TaskType.QA
