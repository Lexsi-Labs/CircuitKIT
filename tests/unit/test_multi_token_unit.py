"""
Unit tests for multi-token answer support (M7.0.1).

Tests without transformer_lens dependency to allow CI to run.
"""

import sys

# Test without circular imports
import unittest
from pathlib import Path

import pytest
import torch


class TestLogitDiffSpanMetric(unittest.TestCase):
    """Test logit_diff_span metric function."""

    def test_single_token_answer_computation(self):
        """Test logit difference computation for single-token answers."""
        # Import metrics directly
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import logit_diff_span

        batch_size = 2
        seq_len = 10
        vocab_size = 50000

        # Create dummy logits
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Set known values to test computation
        logits[0, 9, 100] = 5.0  # correct token
        logits[0, 9, 101] = 3.0  # incorrect token
        logits[1, 9, 200] = 10.0  # correct
        logits[1, 9, 201] = 8.0  # incorrect

        labels = torch.tensor([[100, 101], [200, 201]])
        input_lengths = torch.tensor([10, 10])

        # Single-token answers: no spans provided
        result = logit_diff_span(logits, None, input_lengths, labels, answer_spans=None, mean=False)

        assert result.shape == (batch_size,)
        # Check computed differences
        assert result[0].item() == pytest.approx(5.0 - 3.0)
        assert result[1].item() == pytest.approx(10.0 - 8.0)

    def test_multi_token_answer_averaging(self):
        """Test that multi-token answers are averaged before computing diff."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import logit_diff_span

        batch_size = 1
        seq_len = 15
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Set consistent values for span (3 tokens)
        logits[0, 5:8, 42] = 2.0  # correct token across span
        logits[0, 5:8, 43] = 1.0  # incorrect token across span

        labels = torch.tensor([[42, 43]])
        input_lengths = torch.tensor([15])

        # Multi-token answer spanning positions 5-8
        answer_spans = [(5, 8)]
        result = logit_diff_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        assert result.shape == (batch_size,)
        # Average of [2.0, 2.0, 2.0] - average of [1.0, 1.0, 1.0] = 2.0 - 1.0 = 1.0
        assert result[0].item() == pytest.approx(1.0)

    def test_mean_aggregation(self):
        """Test mean aggregation across batch."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import logit_diff_span

        batch_size = 4
        seq_len = 10
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        labels = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]])
        input_lengths = torch.full((batch_size,), 10)

        # Test mean=False (per-example)
        result_per_ex = logit_diff_span(logits, None, input_lengths, labels, mean=False)
        assert result_per_ex.shape == (batch_size,)

        # Test mean=True (aggregated)
        result_mean = logit_diff_span(logits, None, input_lengths, labels, mean=True)
        assert result_mean.shape == ()
        assert torch.allclose(result_mean, result_per_ex.mean())

    def test_span_clamping(self):
        """Test that invalid spans are clamped to valid range."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import logit_diff_span

        batch_size = 1
        seq_len = 10
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[0, 9, 10] = 5.0
        logits[0, 9, 11] = 3.0

        labels = torch.tensor([[10, 11]])
        input_lengths = torch.tensor([10])

        # Span extending beyond sequence (should be clamped to [8, 10))
        answer_spans = [(8, 20)]
        result = logit_diff_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        # Should not crash and should clamp appropriately
        assert result.shape == (batch_size,)
        assert not torch.isnan(result).any()


class TestAccuracySpanMetric(unittest.TestCase):
    """Test accuracy_span metric function."""

    def test_accuracy_computation(self):
        """Test accuracy computation for single-token answers."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import accuracy_span

        batch_size = 3
        seq_len = 10
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        labels = torch.tensor([[42, 43], [87, 88], [10, 11]])

        # Set correct answers to have highest logits
        logits[0, 9, 42] = 100  # Correct
        logits[1, 9, 99] = 100  # Incorrect (wrong token, label is 87)
        logits[2, 9, 10] = 100  # Correct

        input_lengths = torch.full((batch_size,), 10)

        result = accuracy_span(logits, None, input_lengths, labels, mean=False)

        assert result.shape == (batch_size,)
        assert result[0].item() == 1.0  # Correct
        assert result[1].item() == 0.0  # Incorrect
        assert result[2].item() == 1.0  # Correct

    def test_accuracy_mean(self):
        """Test accuracy mean aggregation."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import accuracy_span

        batch_size = 3
        seq_len = 8
        vocab_size = 50

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        labels = torch.tensor([[10, 11], [20, 21], [30, 31]])
        input_lengths = torch.full((batch_size,), 8)

        # Make 2 correct, 1 incorrect
        logits[0, 7, 10] = 100  # Correct
        logits[1, 7, 20] = 100  # Correct
        logits[2, 7, 49] = 100  # Incorrect (wrong token, label is 30)

        result_mean = accuracy_span(logits, None, input_lengths, labels, mean=True)

        assert result_mean.item() == pytest.approx(2.0 / 3.0, abs=1e-6)


class TestKLDivSpanMetric(unittest.TestCase):
    """Test kl_div_span metric function."""

    def test_kl_div_computation(self):
        """Test KL divergence computation."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import kl_div_span

        batch_size = 1
        seq_len = 10
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        clean_logits = torch.randn(batch_size, seq_len, vocab_size)

        labels = torch.tensor([[42, 43]])
        input_lengths = torch.tensor([10])

        result = kl_div_span(
            logits, clean_logits, input_lengths, labels, answer_spans=None, mean=False
        )

        assert result.shape == (batch_size,)
        # KL divergence should be non-negative
        assert (result >= -1e-6).all()

    def test_kl_div_multi_token(self):
        """Test KL divergence on multi-token answers."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.metrics import kl_div_span

        batch_size = 1
        seq_len = 12
        vocab_size = 100

        torch.manual_seed(42)
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


class TestCollateFunction(unittest.TestCase):
    """Test collate_EAP_with_spans function."""

    def test_backward_compatible_format(self):
        """Test backward compatibility with 3-tuple format."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.eap_utils import collate_EAP_with_spans

        batch = [
            ("clean 1", "corr 1", [100, 101]),
            ("clean 2", "corr 2", [200, 201]),
        ]
        result = collate_EAP_with_spans(batch)

        # Should return 3-tuple (no spans)
        assert len(result) == 3
        clean_texts, corrupted_texts, labels = result
        assert clean_texts == ["clean 1", "clean 2"]
        assert corrupted_texts == ["corr 1", "corr 2"]
        assert labels.tolist() == [[100, 101], [200, 201]]

    def test_multi_token_format(self):
        """Test collation with answer spans."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.backends.eap.eap_utils import collate_EAP_with_spans

        batch = [
            ("clean 1", "corr 1", [100, 101], (2, 4)),
            ("clean 2", "corr 2", [200, 201], (3, 5)),
            ("clean 3", "corr 3", [300, 301], None),
        ]
        result = collate_EAP_with_spans(batch)

        # Should return 4-tuple (with spans)
        assert len(result) == 4
        clean_texts, corrupted_texts, labels, answer_spans = result
        assert clean_texts == ["clean 1", "clean 2", "clean 3"]
        assert len(answer_spans) == 3
        assert answer_spans[0] == (2, 4)
        assert answer_spans[1] == (3, 5)
        assert answer_spans[2] is None


class TestGenericDataLoader(unittest.TestCase):
    """Test GenericDataLoader with answer spans."""

    def test_loader_without_spans(self):
        """Test backward compatibility: examples without spans."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.tasks.generic import GenericDataLoader

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

        item = loader[0]
        assert item["clean"] == "What is 2+2?"
        assert item["corrupted"] == "What is 3+3?"
        assert item["labels"] == [100, 101]
        assert item["answer_span"] is None

    def test_loader_with_spans(self):
        """Test new functionality: examples with answer spans."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.tasks.generic import GenericDataLoader

        examples = [
            {
                "clean": "Capital of France?",
                "corrupted": "Capital of Germany?",
                "correct_idx": 10,
                "incorrect_idx": 20,
                "answer_start": 5,
                "answer_end": 7,
            }
        ]
        loader = GenericDataLoader(examples)
        assert loader.has_answer_spans

        item = loader[0]
        assert item["answer_span"] == (5, 7)

    def test_mixed_examples(self):
        """Test mix of examples with and without spans."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from circuitkit.tasks.generic import GenericDataLoader

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


if __name__ == "__main__":
    unittest.main()
