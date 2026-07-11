"""
Test suite for multi-token answer support (M7.0.1).

Tests the complete pipeline:
1. GenericTaskSpec schema extension
2. DataLoader with answer spans
3. Multi-token metric functions
4. Backward compatibility
"""

import tempfile
from pathlib import Path

import pytest
import torch

from circuitkit.backends.eap.eap_utils import collate_EAP_with_spans
from circuitkit.backends.eap.metrics import accuracy_span, kl_div_span, logit_diff_span
from circuitkit.tasks.generic import GenericDataLoader, GenericTaskSpec


class TestGenericDataLoaderSpans:
    """Test GenericDataLoader with answer spans."""

    def test_dataloader_without_spans(self):
        """Test backward compatibility: examples without spans."""
        examples = [
            {
                "clean": "What is 2+2?",
                "corrupted": "What is 3+3?",
                "correct_idx": 100,
                "incorrect_idx": 101,
            }
        ]
        loader = GenericDataLoader(examples)
        assert len(loader) == 1
        assert not loader.has_answer_spans

        # GenericDataLoader.__getitem__ returns an extended example dict.
        item = loader[0]
        assert item["clean"] == "What is 2+2?"
        assert item["answer_span"] is None
        assert item["labels"] == [100, 101]

    def test_dataloader_with_spans(self):
        """Test new functionality: examples with answer spans."""
        examples = [
            {
                "clean": "Capital of France?",
                "corrupted": "Capital of Germany?",
                "correct_idx": 10,
                "incorrect_idx": 20,
                "answer_start": 5,
                "answer_end": 7,  # 2-token answer
            }
        ]
        loader = GenericDataLoader(examples)
        assert len(loader) == 1
        assert loader.has_answer_spans

        item = loader[0]
        assert item["answer_span"] == (5, 7)

    def test_mixed_examples(self):
        """Test mix of examples with and without spans."""
        examples = [
            {
                "clean": "Q1?",
                "corrupted": "Bad Q1?",
                "correct_idx": 100,
                "incorrect_idx": 101,
            },
            {
                "clean": "Q2?",
                "corrupted": "Bad Q2?",
                "correct_idx": 200,
                "incorrect_idx": 201,
                "answer_start": 3,
                "answer_end": 5,
            },
        ]
        loader = GenericDataLoader(examples)
        assert loader.has_answer_spans

        # First example (no span)
        assert loader[0]["answer_span"] is None

        # Second example (with span)
        assert loader[1]["answer_span"] == (3, 5)


class TestCollateWithSpans:
    """Test collate_EAP_with_spans function."""

    def test_collate_backward_compatible(self):
        """Test backward compatibility with 3-tuple format."""
        batch = [
            ("clean 1", "corr 1", [100, 101]),
            ("clean 2", "corr 2", [200, 201]),
        ]
        result = collate_EAP_with_spans(batch)

        # Should return 3-tuple (no spans)
        assert len(result) == 3
        clean_texts, corrupted_texts, labels = result
        assert len(clean_texts) == 2
        assert labels.shape[0] == 2

    def test_collate_with_spans(self):
        """Test collation with answer spans."""
        batch = [
            ("clean 1", "corr 1", [100, 101], (2, 4)),
            ("clean 2", "corr 2", [200, 201], (3, 5)),
            ("clean 3", "corr 3", [300, 301], None),
        ]
        result = collate_EAP_with_spans(batch)

        # Should return 4-tuple (with spans)
        assert len(result) == 4
        clean_texts, corrupted_texts, labels, answer_spans = result
        assert len(clean_texts) == 3
        assert len(answer_spans) == 3
        assert answer_spans[0] == (2, 4)
        assert answer_spans[1] == (3, 5)
        assert answer_spans[2] is None


class TestLogitDiffSpan:
    """Test logit_diff_span metric function."""

    def test_single_token_answer(self):
        """Test backward compatibility with single-token answers."""
        batch_size = 2
        seq_len = 10
        vocab_size = 50000

        # Create dummy logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[100, 101], [200, 201]])
        input_lengths = torch.tensor([10, 10])

        # Single-token answers: no spans provided
        result = logit_diff_span(logits, None, input_lengths, labels, answer_spans=None, mean=False)

        assert result.shape == (batch_size,)
        assert result.dtype == logits.dtype

    def test_multi_token_answer(self):
        """Test multi-token answer handling."""
        batch_size = 2
        seq_len = 15
        vocab_size = 50000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[100, 101], [200, 201]])
        input_lengths = torch.tensor([15, 15])

        # Multi-token answers: spans at (8, 11) and (6, 9)
        answer_spans = [(8, 11), (6, 9)]
        result = logit_diff_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        assert result.shape == (batch_size,)
        # With multiple tokens, averaging should reduce variance
        # Just check it's computed without error

    def test_mean_aggregation(self):
        """Test mean aggregation across batch."""
        batch_size = 4
        seq_len = 10
        vocab_size = 50000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[100, 101], [200, 201], [300, 301], [400, 401]])
        input_lengths = torch.full((batch_size,), 10)

        # Test mean=False (per-example)
        result_per_ex = logit_diff_span(logits, None, input_lengths, labels, mean=False)
        assert result_per_ex.shape == (batch_size,)

        # Test mean=True (aggregated)
        result_mean = logit_diff_span(logits, None, input_lengths, labels, mean=True)
        assert result_mean.shape == ()
        assert torch.allclose(result_mean, result_per_ex.mean())

    def test_span_clamping(self):
        """Test that spans are clamped to valid sequence length."""
        batch_size = 1
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[10, 11]])
        input_lengths = torch.tensor([10])

        # Span extending beyond sequence length
        answer_spans = [(8, 20)]  # 20 > seq_len
        result = logit_diff_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        # Should not crash; should clamp to [8, 10)
        assert result.shape == (batch_size,)


class TestAccuracySpan:
    """Test accuracy_span metric function."""

    def test_accuracy_single_token(self):
        """Test accuracy on single-token answers."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[42, 43], [87, 88]])
        input_lengths = torch.tensor([10, 10])

        # Set correct logits to be maximal for testing
        logits[0, 9, 42] = 100  # Correct answer for example 0
        logits[1, 9, 88] = 100  # Incorrect answer for example 1

        result = accuracy_span(logits, None, input_lengths, labels, answer_spans=None, mean=False)

        # Example 0: correct prediction (idx 42 > 43)
        # Example 1: incorrect prediction (idx 88 != predicted max)
        assert result.shape == (batch_size,)
        assert result[0].item() == 1.0  # Correct

    def test_accuracy_mean(self):
        """Test accuracy aggregation."""
        batch_size = 3
        seq_len = 8
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[10, 11], [20, 21], [30, 31]])
        input_lengths = torch.full((batch_size,), 8)

        # Make 2 correct, 1 incorrect
        logits[0, 7, 10] = 100  # Correct
        logits[1, 7, 20] = 100  # Correct
        logits[2, 7, 49] = 100  # Incorrect (wrong token, within vocab_size=50)

        result_mean = accuracy_span(logits, None, input_lengths, labels, mean=True)
        assert result_mean.item() == pytest.approx(2.0 / 3.0)


class TestKLDivSpan:
    """Test kl_div_span metric function."""

    def test_kl_div_single_token(self):
        """Test KL divergence on single-token answers."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        clean_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[42, 43], [87, 88]])
        input_lengths = torch.tensor([10, 10])

        result = kl_div_span(
            logits, clean_logits, input_lengths, labels, answer_spans=None, mean=False
        )

        assert result.shape == (batch_size,)
        # KL divergence should be non-negative
        assert (result >= -1e-6).all()  # Allow small numerical error

    def test_kl_div_multi_token(self):
        """Test KL divergence on multi-token answers."""
        batch_size = 1
        seq_len = 12
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        clean_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[42, 43]])
        input_lengths = torch.tensor([12])

        # 3-token answer
        answer_spans = [(5, 8)]
        result = kl_div_span(
            logits, clean_logits, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        assert result.shape == (batch_size,)
        assert (result >= -1e-6).all()


class TestGenericTaskSpecSchema:
    """Test GenericTaskSpec with multi-token answer schema."""

    def test_schema_documentation(self):
        """Verify schema is documented correctly."""
        task = GenericTaskSpec(
            name="test_task",
            source={"type": "csv", "path_or_id": "dummy.csv"},
            schema={
                "prompt": "question",
                "answer": "answer_text",  # Single or multi-token
                # Optional:
                # "answer_start": "answer_pos_start",
                # "answer_end": "answer_pos_end",
            },
        )
        assert task.schema["prompt"] == "question"
        assert task.schema["answer"] == "answer_text"

    def test_backward_compat_csv_loading(self):
        """Test backward compatibility with CSV loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("prompt,answer\nWhat is 2+2?,4\nWhat is 3+3?,6\n")

            task = GenericTaskSpec.from_csv(
                path=str(csv_path),
                schema={"prompt": "prompt", "answer": "answer"},
                name="test_csv",
            )
            assert task.name == "test_csv"
            assert task.schema["prompt"] == "prompt"


@pytest.mark.integration
class TestIntegrationMultiTokenWorkflow:
    """Integration tests for complete multi-token workflow."""

    def test_full_pipeline_single_token(self):
        """Test full pipeline with single-token answers (backward compat)."""
        # Create example data
        examples = [
            {
                "clean": "Q1?",
                "corrupted": "BadQ1?",
                "correct_idx": 100,
                "incorrect_idx": 101,
            },
            {
                "clean": "Q2?",
                "corrupted": "BadQ2?",
                "correct_idx": 200,
                "incorrect_idx": 201,
            },
        ]

        loader = GenericDataLoader(examples)
        assert not loader.has_answer_spans

        # GenericDataLoader yields dicts; eap_utils.collate_EAP_with_spans
        # consumes (clean, corrupted, labels) tuples, so adapt the items.
        batch = [
            (item["clean"], item["corrupted"], item["labels"]) for item in (loader[0], loader[1])
        ]
        clean_texts, corrupted_texts, labels = collate_EAP_with_spans(batch)

        # Should return 3-tuple
        assert clean_texts == ["Q1?", "Q2?"]
        assert len(labels) == 2

    def test_full_pipeline_multi_token(self):
        """Test full pipeline with multi-token answers."""
        examples = [
            {
                "clean": "Capital of France?",
                "corrupted": "Capital of Germany?",
                "correct_idx": 10,
                "incorrect_idx": 20,
                "answer_start": 3,
                "answer_end": 5,  # 2 tokens
            },
        ]

        loader = GenericDataLoader(examples)
        assert loader.has_answer_spans

        # Adapt dict items to the (clean, corrupted, labels, answer_span)
        # tuple format consumed by eap_utils.collate_EAP_with_spans.
        item = loader[0]
        batch = [(item["clean"], item["corrupted"], item["labels"], item["answer_span"])]
        result = collate_EAP_with_spans(batch)

        # Should return 4-tuple with answer_spans
        assert len(result) == 4
        clean_texts, corrupted_texts, labels, answer_spans = result
        assert answer_spans[0] == (3, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
