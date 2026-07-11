#!/usr/bin/env python3
"""Tests for extended task schema implementation."""

import json

import pytest

from circuitkit.tasks.generic import GenericDataLoader, GenericTaskSpec


def dummy_metric(logits, clean_logits, input_length, labels, mean=True):
    return logits.mean() if mean else logits


def test_schema_validation_missing_prompt():
    with pytest.raises(ValueError, match="prompt"):
        GenericTaskSpec(
            name="test",
            source={"type": "csv", "path_or_id": "test.csv"},
            schema={"answer": "a"},
            metric_fn=dummy_metric,
        )


def test_schema_validation_missing_answer():
    with pytest.raises(ValueError, match="at least one answer key"):
        GenericTaskSpec(
            name="test",
            source={"type": "csv", "path_or_id": "test.csv"},
            schema={"prompt": "q"},
            metric_fn=dummy_metric,
        )


def test_task_type_classification():
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "classification"


def test_task_type_qa():
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "context": "c", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "qa"


def test_task_type_mcq():
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "choices": "opts", "answer": "a"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "mcq"


def test_task_type_ranking():
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answers": "answers"},
        metric_fn=dummy_metric,
    )
    assert task.task_type == "ranking"


def test_field_extraction_with_context():
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
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answers": "answers"},
    )
    example = {"q": "What is the capital?", "answers": ["Paris", "paris"]}
    extracted = task._extract_fields(example)
    assert extracted["prompt"] == "What is the capital?"
    assert extracted["answers"] == ["Paris", "paris"]


def test_metadata_extraction():
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
        "meta": json.dumps({"source": "wiki"}),
    }
    extracted = task._extract_fields(example)
    assert extracted["metadata"]["id"] == "ex123"
    assert extracted["metadata"]["difficulty"] == "hard"
    assert extracted["metadata"]["category"] == "history"
    assert extracted["metadata"]["source"] == "wiki"


def test_metadata_filter_matching():
    task = GenericTaskSpec(
        name="test",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a", "difficulty": "diff"},
        metadata_filter={"difficulty": "hard"},
    )
    assert task._matches_metadata_filter({"prompt": "Q1", "metadata": {"difficulty": "hard"}})
    assert not task._matches_metadata_filter({"prompt": "Q2", "metadata": {"difficulty": "easy"}})


def test_answer_span_extraction():
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


def test_artifact_metadata():
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


def test_backward_compatibility():
    task = GenericTaskSpec(
        name="legacy_task",
        source={"type": "csv", "path_or_id": "test.csv"},
        schema={"prompt": "q", "answer": "a"},
        metric_fn=dummy_metric,
    )
    example = {"q": "Old style question", "a": "Old style answer"}
    extracted = task._extract_fields(example)
    assert "prompt" in extracted
    assert "answer" in extracted


def test_generic_dataloader_extended_fields():
    examples = [
        {
            "clean": "Q1",
            "corrupted": "Q1_corrupt",
            "correct_idx": 100,
            "incorrect_idx": 101,
            "context": "Context 1",
            "choices": ["A", "B", "C"],
            "metadata": {"difficulty": "easy"},
        },
        {"clean": "Q2", "corrupted": "Q2_corrupt", "correct_idx": 200, "incorrect_idx": 201},
    ]
    loader = GenericDataLoader(examples)
    assert len(loader) == 2
    assert loader.has_context is True
    assert loader.has_metadata is True
    item = loader[0]
    assert item["clean"] == "Q1"
    assert item["context"] == "Context 1"
    assert item["choices"] == ["A", "B", "C"]
    assert item["metadata"]["difficulty"] == "easy"
