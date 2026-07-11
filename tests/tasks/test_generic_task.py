"""
Test suite for GenericTaskSpec.

Tests basic functionality:
- CSV loading with schema mapping
- Corruption strategy application
- EAP dataloader generation
- Finetuning dataset creation
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def toy_csv_file():
    """Create a simple 10-example CSV for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "question": [
                    "What is 2+2?",
                    "What is 3+3?",
                    "What is 4+4?",
                    "What is 5+5?",
                    "What is 6+6?",
                    "What is 7+7?",
                    "What is 8+8?",
                    "What is 9+9?",
                    "What is 10+10?",
                    "What is 11+11?",
                ],
                "answer": [
                    "4",
                    "6",
                    "8",
                    "10",
                    "12",
                    "14",
                    "16",
                    "18",
                    "20",
                    "22",
                ],
            }
        )
        df.to_csv(f, index=False)
        return Path(f.name)


@pytest.fixture
def toy_jsonl_file():
    """Create a simple 5-example JSONL for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            obj = {"prompt_text": f"Question {i}", "expected_answer": f"Answer {i}"}
            f.write(json.dumps(obj) + "\n")
        return Path(f.name)


def test_generic_task_from_csv_basic(toy_csv_file):
    """Test creating GenericTaskSpec from CSV."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
        metric_fn=dummy_metric,
    )

    assert task.name == "csv_" + toy_csv_file.stem
    assert task.source["type"] == "csv"
    assert task.source["path_or_id"] == str(toy_csv_file)
    assert task.schema["prompt"] == "question"
    assert task.schema["answer"] == "answer"


def test_generic_task_from_jsonl_basic(toy_jsonl_file):
    """Test creating GenericTaskSpec from JSONL."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec.from_jsonl(
        path=str(toy_jsonl_file),
        schema={"prompt": "prompt_text", "answer": "expected_answer"},
        metric_fn=dummy_metric,
    )

    assert task.name == "jsonl_" + toy_jsonl_file.stem
    assert task.source["type"] == "jsonl"
    assert task.schema["prompt"] == "prompt_text"


def test_generic_task_load_data_csv(toy_csv_file):
    """Test loading CSV data."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
    )

    data = task._load_data()
    assert len(data) == 10
    assert "question" in data[0]
    assert "answer" in data[0]


def test_generic_task_load_data_jsonl(toy_jsonl_file):
    """Test loading JSONL data."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec.from_jsonl(
        path=str(toy_jsonl_file),
        schema={"prompt": "prompt_text", "answer": "expected_answer"},
    )

    data = task._load_data()
    assert len(data) == 5
    assert "prompt_text" in data[0]
    assert "expected_answer" in data[0]


def test_generic_task_extract_fields(toy_csv_file):
    """Test field extraction with schema."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
    )

    raw_example = {"question": "What is 5?", "answer": "5", "extra": "ignored"}
    extracted = task._extract_fields(raw_example)

    assert extracted["prompt"] == "What is 5?"
    assert extracted["answer"] == "5"
    assert extracted["extra"] == "ignored"  # Original fields preserved


def test_generic_task_validate_discovery_config():
    """Test discovery configuration validation."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )

    # Valid config
    task.validate_discovery_config(
        {
            "algorithm": "eap",
            "batch_size": 32,
            "num_examples": 100,
        }
    )

    # acdc is now a supported discovery algorithm for GenericTaskSpec.
    task.validate_discovery_config(
        {
            "algorithm": "acdc",
        }
    )

    # Invalid algorithm
    with pytest.raises(ValueError, match="only supports discovery algorithms"):
        task.validate_discovery_config(
            {
                "algorithm": "not-a-real-algorithm",
            }
        )

    # Invalid batch_size
    with pytest.raises(ValueError, match="invalid 'batch_size'"):
        task.validate_discovery_config(
            {
                "algorithm": "eap",
                "batch_size": -1,
            }
        )


def test_generic_task_metric_fn():
    """Test metric_fn access."""
    from circuitkit.tasks import GenericTaskSpec

    def my_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=my_metric,
    )

    assert task.metric_fn() == my_metric


def test_generic_task_metric_fn_auto_generated():
    """Test that metric_fn is auto-generated from task_type when not provided.

    GenericTaskSpec auto-generates a metric from the detected task type when
    no explicit metric_fn is passed (documented behaviour), so metric_fn()
    returns a usable callable rather than raising.
    """
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
    )

    metric = task.metric_fn()
    assert callable(metric)


def test_generic_task_artifact_metadata(toy_csv_file):
    """Test artifact metadata generation."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
        metric_fn=dummy_metric,
    )

    discovery_cfg = {"algorithm": "eap", "batch_size": 16}
    metadata = task.artifact_metadata(discovery_cfg)

    assert metadata["task_name"] == task.name
    assert metadata["source_type"] == "csv"
    assert metadata["algorithm"] == "eap"
    assert metadata["corruption_strategy"] == "none"


def test_generic_task_build_finetuning_dataset(toy_csv_file):
    """Test finetuning dataset construction."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
    )

    # Use a real tokenizer (gpt2 is small and commonly available)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("Could not load gpt2 tokenizer")

    clean_texts, query_strings = task.build_finetuning_dataset(
        tokenizer=tokenizer,
        model_name="gpt2",
        n_examples=5,
        seed=42,
    )

    assert len(clean_texts) == 5
    assert len(query_strings) == 5
    # clean_texts should be prompt + answer
    for clean, query in zip(clean_texts, query_strings):
        assert clean.startswith(query)


def test_generic_task_no_corruption_by_default(toy_csv_file):
    """Test that examples are copied when no corruption strategy is provided."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec.from_csv(
        path=str(toy_csv_file),
        schema={"prompt": "question", "answer": "answer"},
        corruption_strategy=None,  # Explicitly no corruption
    )

    examples = [
        {"prompt": "Q1", "answer": "A1"},
        {"prompt": "Q2", "answer": "A2"},
    ]

    discovery_cfg = {"seed": 42}
    corrupted = task._apply_corruptions(examples, discovery_cfg)

    # Without corruption, prompts should be identical
    assert len(corrupted) == len(examples)
    for clean, corr in zip(examples, corrupted):
        assert corr["prompt"] == clean["prompt"]


# ========== Extended Schema Tests ==========


def test_extended_schema_validation():
    """Test that schema validation detects invalid configurations."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    # Invalid: no prompt
    with pytest.raises(ValueError, match="missing the required key 'prompt'"):
        GenericTaskSpec(
            name="test",
            source={"type": "csv", "path_or_id": "test.csv"},
            schema={"answer": "a"},  # Missing prompt
            metric_fn=dummy_metric,
        )

    # Invalid: no answer specification
    with pytest.raises(ValueError, match="at least one answer key"):
        GenericTaskSpec(
            name="test",
            source={"type": "csv", "path_or_id": "test.csv"},
            schema={"prompt": "q"},  # Missing answer
            metric_fn=dummy_metric,
        )

    # Valid: just prompt and answer
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "classification"


def test_task_type_detection():
    """Test auto-detection of task type from schema."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    # Classification: prompt + single answer
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "classification"

    # QA: has context
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "context": "c", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "qa"

    # MCQ: has choices
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "choices": "opts", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "mcq"

    # Ranking: multiple answers
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answers": "answers"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "ranking"

    # Open-ended: prompt only
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )
    # Will be classification if has answer
    assert task.task_type == "classification"


def test_context_extraction():
    """Test extraction of context field."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "context": "c", "answer": "a"},
    )

    example = {"q": "What is X?", "c": "Background text", "a": "X", "extra": "ignored"}

    extracted = task._extract_fields(example)
    assert extracted["prompt"] == "What is X?"
    assert extracted["context"] == "Background text"
    assert extracted["answer"] == "X"
    assert "context_boundary" in extracted


def test_multiple_answers_extraction():
    """Test extraction of multiple valid answers."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answers": "answers"},
    )

    example = {"q": "What is the capital?", "answers": ["Paris", "paris"]}

    extracted = task._extract_fields(example)
    assert extracted["prompt"] == "What is the capital?"
    assert extracted["answers"] == ["Paris", "paris"]


def test_choices_and_correct_idx_extraction():
    """Test extraction of MCQ choices and correct index."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "choices": "options", "correct_choice_idx": "idx", "answer": "a"},
    )

    example = {"q": "Pick the right one", "options": ["A", "B", "C"], "idx": 1, "a": "B"}

    extracted = task._extract_fields(example)
    assert extracted["choices"] == ["A", "B", "C"]
    assert extracted["correct_choice_idx"] == 1


def test_metadata_extraction():
    """Test extraction of metadata fields."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={
            "prompt": "q",
            "answer": "a",
            "id": "id_col",
            "difficulty": "diff",
            "category": "cat",
            "metadata": "meta",
        },
    )

    example = {
        "q": "Question",
        "a": "Answer",
        "id_col": "ex123",
        "diff": "hard",
        "cat": "history",
        "meta": '{"source": "wiki"}',
    }

    extracted = task._extract_fields(example)
    assert extracted["metadata"]["id"] == "ex123"
    assert extracted["metadata"]["difficulty"] == "hard"
    assert extracted["metadata"]["category"] == "history"
    assert extracted["metadata"]["source"] == "wiki"


def test_metadata_filter_matching():
    """Test metadata filter matching logic."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a", "difficulty": "diff"},
        metadata_filter={"difficulty": "hard"},
    )

    # Match
    example_hard = {"prompt": "Q1", "metadata": {"difficulty": "hard"}}
    assert task._matches_metadata_filter(example_hard)

    # No match
    example_easy = {"prompt": "Q2", "metadata": {"difficulty": "easy"}}
    assert not task._matches_metadata_filter(example_easy)

    # No metadata
    example_no_meta = {"prompt": "Q3"}
    assert not task._matches_metadata_filter(example_no_meta)


def test_answer_span_extraction():
    """Test extraction of answer spans."""
    from circuitkit.tasks import GenericTaskSpec

    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={
            "prompt": "q",
            "context": "c",
            "answer": "a",
            "answer_start": "start",
            "answer_end": "end",
        },
    )

    example = {"q": "What?", "c": "Long passage with answer", "a": "answer", "start": 20, "end": 26}

    extracted = task._extract_fields(example)
    assert extracted["answer_start"] == 20
    assert extracted["answer_end"] == 26


def test_artifact_metadata_extended():
    """Test extended artifact metadata generation."""
    from circuitkit.tasks import GenericTaskSpec

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec(
        name="squad_qa",
        source={"type": "csv", "path_or_id": "squad.csv"},
        schema={
            "prompt": "question",
            "context": "passage",
            "answer": "answer",
            "difficulty": "diff",
        },
        metric_fn=dummy_metric,
    )

    metadata = task.artifact_metadata({"algorithm": "eap", "batch_size": 32})

    assert metadata["task_name"] == "squad_qa"
    assert metadata["task_type"] == "qa"
    assert metadata["has_context"] is True
    assert metadata["has_choices"] is False
    assert metadata["has_multiple_answers"] is False
    assert metadata["source_type"] == "csv"


def test_csv_with_extended_schema(tmp_path):
    """Test loading CSV with extended schema."""
    from circuitkit.tasks import GenericTaskSpec

    # Create test CSV with extended fields
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame(
        {
            "question": ["Q1", "Q2"],
            "context": ["Context 1", "Context 2"],
            "answer": ["A1", "A2"],
            "difficulty": ["easy", "hard"],
            "category": ["math", "science"],
        }
    )
    df.to_csv(csv_file, index=False)

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec.from_csv(
        path=str(csv_file),
        schema={
            "prompt": "question",
            "context": "context",
            "answer": "answer",
            "difficulty": "difficulty",
            "category": "category",
        },
        metric_fn=dummy_metric,
    )

    assert task.task_type == "qa"

    data = task._load_data()
    assert len(data) == 2

    examples = [task._extract_fields(ex) for ex in data]
    assert len(examples) == 2
    assert "context" in examples[0]
    assert examples[0]["metadata"]["difficulty"] == "easy"


def test_metadata_filtering_in_dataloader(tmp_path):
    """Test metadata filtering during dataloader creation."""
    from circuitkit.tasks import GenericTaskSpec

    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame(
        {"q": ["Q1", "Q2", "Q3"], "a": ["A1", "A2", "A3"], "difficulty": ["easy", "hard", "hard"]}
    )
    df.to_csv(csv_file, index=False)

    def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
        return logits.mean() if mean else logits

    task = GenericTaskSpec.from_csv(
        path=str(csv_file),
        schema={"prompt": "q", "answer": "a", "difficulty": "difficulty"},
        metric_fn=dummy_metric,
        metadata_filter={"difficulty": "hard"},
    )

    # When building dataloader, only "hard" examples should be included
    # Note: This test verifies that metadata is extracted correctly;
    # actual dataloader building requires a model
    data = task._load_data()
    examples = [task._extract_fields(ex) for ex in data]

    # Filter manually to verify logic
    filtered = [ex for ex in examples if task._matches_metadata_filter(ex)]
    assert len(filtered) == 2  # Only Q2 and Q3 have difficulty="hard"


def test_finetuning_dataset_with_context(tmp_path):
    """Test finetuning dataset generation with context."""
    from circuitkit.tasks import GenericTaskSpec

    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"q": ["Q1", "Q2"], "c": ["Context 1", "Context 2"], "a": ["A1", "A2"]})
    df.to_csv(csv_file, index=False)

    task = GenericTaskSpec.from_csv(
        path=str(csv_file),
        schema={"prompt": "q", "context": "c", "answer": "a"},
    )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("Could not load gpt2 tokenizer")

    clean_texts, query_strings = task.build_finetuning_dataset(
        tokenizer=tokenizer,
        model_name="gpt2",
        n_examples=2,
        seed=42,
    )

    assert len(clean_texts) == 2
    assert len(query_strings) == 2

    # Queries should start with context
    for query in query_strings:
        assert "Context" in query


class TestCorruptionEffectivenessGuard:
    """_check_corruption_effectiveness fails loud on fully-degenerate corruption.

    Discovering on clean==corrupt pairs gives EAP no contrastive signal, so a
    corruption that leaves every prompt unchanged (e.g. entity_swap on prompts
    with no named entities) is an error rather than a silent meaningless run.
    """

    def test_all_identical_raises(self):
        import pytest
        from circuitkit.tasks import GenericTaskSpec
        with pytest.raises(ValueError, match="Every pair is identical"):
            GenericTaskSpec._check_corruption_effectiveness(4, 4, {}, context="x")

    def test_partial_identical_warns_not_raises(self, caplog):
        import logging
        from circuitkit.tasks import GenericTaskSpec
        with caplog.at_level(logging.WARNING):
            GenericTaskSpec._check_corruption_effectiveness(2, 4, {}, context="x")
        assert any("2/4" in r.message for r in caplog.records)

    def test_opt_out_allows_degenerate(self):
        from circuitkit.tasks import GenericTaskSpec
        # Must not raise when explicitly allowed.
        GenericTaskSpec._check_corruption_effectiveness(
            4, 4, {"allow_degenerate_corruption": True}, context="x"
        )

    def test_no_identical_is_noop(self):
        from circuitkit.tasks import GenericTaskSpec
        GenericTaskSpec._check_corruption_effectiveness(0, 4, {}, context="x")
