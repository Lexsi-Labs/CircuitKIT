"""
Unit tests for the perplexity metric (simplified import).

Tests cover:
- Single-token perplexity (backward compatibility)
- Multi-token perplexity (span-based)
- Loss variant for optimization
- Correctness of perplexity computation
- Integration with evaluate_graph
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from functools import partial  # noqa: E402 - import after intentional pre-import setup

import numpy as np  # noqa: E402 - import after intentional pre-import setup
import pytest  # noqa: E402 - import after intentional pre-import setup
import torch  # noqa: E402 - import after intentional pre-import setup

# Import metrics directly
from circuitkit.backends.eap.metrics import (  # noqa: E402 - import after intentional pre-import setup
    perplexity_legacy,
    perplexity_loss_legacy,
    perplexity_loss_span,
    perplexity_span,
)


class TestPerplexitySingleToken:
    """Test perplexity metric with single-token answers (backward compatibility)."""

    def test_perplexity_basic_single_token(self):
        """Test basic perplexity computation for single token."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        # Create mock logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        # Compute perplexity
        ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)

        # Check shape and type
        assert ppl.shape == torch.Size([])
        assert ppl.dtype == logits.dtype
        assert ppl.item() > 0  # Perplexity is always positive

    def test_perplexity_per_example(self):
        """Test perplexity returns per-example scores when mean=False."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        # Compute perplexity without mean
        ppl = perplexity_span(logits, None, input_lengths, labels, mean=False)

        assert ppl.shape == torch.Size([batch_size])
        assert (ppl > 0).all()  # All values positive

    def test_perplexity_loss_variant(self):
        """Test that loss variant returns cross-entropy (negative log prob)."""
        batch_size = 2
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([5, 5])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        # Compute both versions
        ppl = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=False)
        loss = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=True)

        # loss is the cross-entropy (-mean_log_prob), so perplexity = exp(loss)
        expected_ppl = torch.exp(loss)
        assert torch.allclose(ppl, expected_ppl, rtol=1e-5)

    def test_perplexity_loss_span_wrapper(self):
        """Test the perplexity_loss_span wrapper function."""
        batch_size = 3
        seq_len = 8
        vocab_size = 64

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([8, 8, 8])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        loss = perplexity_loss_span(logits, None, input_lengths, labels, mean=True)

        assert loss.shape == torch.Size([])
        assert loss.dtype == logits.dtype

    def test_perplexity_perfect_confidence(self):
        """Test perplexity with very confident predictions."""
        batch_size = 1
        seq_len = 5
        vocab_size = 100

        # Create logits with very high score for correct token
        logits = torch.ones(batch_size, seq_len, vocab_size) * -100
        correct_token_idx = 42
        labels = torch.tensor([[correct_token_idx]])

        # Set correct token to have very high logit
        logits[0, -1, correct_token_idx] = 100

        ppl = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True)

        # When model is very confident (high logit for correct token),
        # perplexity should be low
        assert ppl < 2.0  # Should be close to 1

    def test_perplexity_random_baseline(self):
        """Test perplexity with random logits (baseline)."""
        batch_size = 1
        seq_len = 5
        vocab_size = 100

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[50]])

        ppl = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True)

        # For random logits the model is neither perfectly confident nor
        # certain, so perplexity is finite and strictly greater than 1.
        # It is NOT bounded by vocab_size: an arbitrary (possibly low-ranked)
        # target token can yield perplexity well above the vocabulary size.
        assert ppl > 1
        assert torch.isfinite(ppl).all()


class TestPerplexityMultiToken:
    """Test perplexity metric with multi-token answers."""

    def test_perplexity_two_token_span(self):
        """Test perplexity with 2-token answer span."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10])
        labels = torch.tensor([[42, 55], [10, 20]])
        answer_spans = [
            (8, 10),  # Last 2 tokens
            (7, 9),  # 2 tokens
        ]

        ppl = perplexity_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True
        )

        assert ppl.shape == torch.Size([])
        assert ppl > 0

    def test_perplexity_multi_token_per_example(self):
        """Test per-example scores with multi-token spans."""
        batch_size = 3
        seq_len = 10
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10, 10])
        labels = torch.randint(0, vocab_size, (batch_size, 3))
        answer_spans = [(7, 10), (5, 8), (6, 9)]

        ppl = perplexity_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        assert ppl.shape == torch.Size([batch_size])
        assert (ppl > 0).all()

    def test_perplexity_variable_span_lengths(self):
        """Test with different span lengths per example."""
        batch_size = 2
        seq_len = 15
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([15, 15])
        # Label tensor needs to accommodate longest span
        labels = torch.randint(0, vocab_size, (batch_size, 3))
        answer_spans = [
            (12, 15),  # 3 tokens
            (10, 13),  # 3 tokens
        ]

        ppl = perplexity_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True
        )

        assert ppl > 0


class TestPerplexitySpanHandling:
    """Test answer span parsing and boundary handling."""

    def test_perplexity_with_none_spans(self):
        """Test that None in answer_spans uses last token position."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 8])
        labels = torch.randint(0, vocab_size, (batch_size, 1))
        answer_spans = [None, None]

        ppl = perplexity_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=False
        )

        assert ppl.shape == torch.Size([batch_size])

    def test_perplexity_boundary_clamping(self):
        """Test that out-of-bounds spans are clamped correctly."""
        batch_size = 1
        seq_len = 10
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10])
        labels = torch.tensor([[42]])
        # Out-of-bounds span should be clamped
        answer_spans = [(20, 30)]

        # Should not raise an error
        ppl = perplexity_span(
            logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True
        )

        assert ppl > 0


class TestPerplexityLegacyCompatibility:
    """Test backward compatibility with legacy API."""

    def test_legacy_perplexity_function(self):
        """Test legacy wrapper for perplexity_span."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        ppl = perplexity_legacy(logits, None, input_lengths, labels, mean=True)

        assert ppl.shape == torch.Size([])
        assert ppl > 0

    def test_legacy_perplexity_loss_function(self):
        """Test legacy wrapper for perplexity_loss_span."""
        batch_size = 3
        seq_len = 10
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([10, 10, 10])
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        loss = perplexity_loss_legacy(logits, None, input_lengths, labels, mean=True)

        assert loss.shape == torch.Size([])


class TestPerplexityNumerical:
    """Test numerical correctness of perplexity computation."""

    def test_perplexity_computation_correctness(self):
        """Verify perplexity formula: exp(-mean_log_prob)."""
        batch_size = 1
        seq_len = 1
        vocab_size = 10

        # Create simple logits
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[0]])  # First token is correct

        ppl = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True)

        # With all-zero logits, softmax gives uniform distribution [1/10, 1/10, ...]
        # Correct token prob = 1/10, log_prob = -log(10)
        # Perplexity = exp(log(10)) = 10
        expected = torch.tensor(vocab_size, dtype=logits.dtype)
        assert torch.allclose(ppl, expected, rtol=1e-5)

    def test_perplexity_loss_correctness(self):
        """Verify loss is negative log probability."""
        batch_size = 1
        seq_len = 1
        vocab_size = 10

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[0]])

        loss = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True, loss=True)

        # Loss should be -log(1/10) = log(10)
        expected_loss = torch.tensor(np.log(vocab_size), dtype=logits.dtype)
        assert torch.allclose(loss, expected_loss, rtol=1e-4)

    def test_perplexity_batch_mean(self):
        """Test that batch mean is computed correctly."""
        batch_size = 3
        seq_len = 5
        vocab_size = 50

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([seq_len] * batch_size)
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        # Compute per-example
        ppl_per_example = perplexity_span(logits, None, input_lengths, labels, mean=False)

        # Compute mean
        ppl_mean = perplexity_span(logits, None, input_lengths, labels, mean=True)

        expected_mean = ppl_per_example.mean()
        assert torch.allclose(ppl_mean, expected_mean)


class TestPerplexityGradients:
    """Test that perplexity supports backpropagation (needed for EAP)."""

    def test_perplexity_backward(self):
        """Test that perplexity metric supports backpropagation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        input_lengths = torch.tensor([seq_len] * batch_size)
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)

        # Should be able to backprop
        ppl.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_perplexity_loss_backward(self):
        """Test that loss variant supports backpropagation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        input_lengths = torch.tensor([seq_len] * batch_size)
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        loss = perplexity_loss_span(logits, None, input_lengths, labels, mean=True)

        loss.backward()
        assert logits.grad is not None


class TestPerplexityEdgeCases:
    """Test edge cases and error handling."""

    def test_perplexity_empty_batch(self):
        """Test with batch size 1."""
        logits = torch.randn(1, 5, 50)
        input_lengths = torch.tensor([5])
        labels = torch.tensor([[25]])

        ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)
        assert ppl.shape == torch.Size([])

    def test_perplexity_high_vocab_size(self):
        """Test with large vocabulary."""
        batch_size = 2
        seq_len = 10
        vocab_size = 50000  # Large vocab

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([seq_len] * batch_size)
        labels = torch.randint(0, vocab_size, (batch_size, 1))

        ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)
        assert ppl > 0 and not torch.isnan(ppl)

    def test_perplexity_single_position_sequence(self):
        """Test with sequence of length 1."""
        logits = torch.randn(1, 1, 100)
        input_lengths = torch.tensor([1])
        labels = torch.tensor([[42]])

        ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)
        assert ppl > 0


class TestPerplexityPartialFunctions:
    """Test creating partial functions for integration."""

    def test_perplexity_as_partial(self):
        """Test creating a partial perplexity function for use in evaluate_graph."""
        metric_fn = partial(perplexity_legacy, loss=True, mean=True)

        logits = torch.randn(4, 10, 100)
        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.randint(0, 100, (4, 1))

        result = metric_fn(logits, None, input_lengths, labels)

        assert result.shape == torch.Size([])

    def test_perplexity_loss_as_partial(self):
        """Test creating a partial loss function."""
        metric_fn = partial(perplexity_loss_legacy, mean=True)

        logits = torch.randn(4, 10, 100)
        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.randint(0, 100, (4, 1))

        result = metric_fn(logits, None, input_lengths, labels)

        assert result.shape == torch.Size([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
