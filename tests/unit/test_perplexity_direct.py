"""
Direct tests for perplexity metric - no circuitkit imports.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def perplexity_span(
    logits: torch.Tensor,
    clean_logits: Optional[torch.Tensor],
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """
    Compute perplexity metric on answer span.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Determine answer span
        if answer_spans and answer_spans[i] is not None:
            start, end = answer_spans[i]
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            # Single-token: use last position
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1

        # Get logits for answer span [span_len, vocab_size]
        span_logits = logits[i, start:end, :]

        # Get target tokens for this span
        span_len = end - start
        if labels.shape[1] == 1:
            # Single-token case
            target_tokens = labels[i, :1]
        else:
            # Multi-token case
            target_tokens = labels[i, :span_len]

        # Compute cross-entropy loss over span
        log_probs = F.log_softmax(span_logits, dim=-1)

        # For each position in span, get log prob of target token
        token_log_probs = []
        for j in range(min(span_len, target_tokens.shape[0])):
            target_idx = target_tokens[j].item()
            target_idx = max(0, min(target_idx, log_probs.shape[1] - 1))
            token_log_probs.append(log_probs[j, target_idx])

        # Average log probability over span (cross-entropy loss)
        if token_log_probs:
            mean_log_prob = torch.stack(token_log_probs).mean()
        else:
            mean_log_prob = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Perplexity = exp(-mean_log_prob)
        # Loss = -mean_log_prob (cross-entropy)
        if loss:
            results.append(-mean_log_prob)
        else:
            perplexity = torch.exp(-mean_log_prob)
            results.append(perplexity)

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def test_perplexity_basic_single_token():
    """Test basic perplexity computation for single token."""
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.randint(0, vocab_size, (batch_size, 1))

    ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)

    assert ppl.shape == torch.Size([])
    assert ppl.dtype == logits.dtype
    assert ppl.item() > 0
    print("[PASS] test_perplexity_basic_single_token")


def test_perplexity_per_example():
    """Test perplexity returns per-example scores when mean=False."""
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.randint(0, vocab_size, (batch_size, 1))

    ppl = perplexity_span(logits, None, input_lengths, labels, mean=False)

    assert ppl.shape == torch.Size([batch_size])
    assert (ppl > 0).all()
    print("[PASS] test_perplexity_per_example")


def test_perplexity_loss_variant():
    """Test that loss variant returns cross-entropy."""
    batch_size = 2
    seq_len = 5
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([5, 5])
    labels = torch.randint(0, vocab_size, (batch_size, 1))

    ppl = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=False)
    loss = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=True)

    # Relationship: ppl = exp(-mean_log_prob), loss = -mean_log_prob
    # So: log(ppl) = loss
    assert torch.allclose(torch.log(ppl), loss, rtol=1e-5)
    print("[PASS] test_perplexity_loss_variant")


def test_perplexity_perfect_confidence():
    """Test perplexity with very confident predictions."""
    batch_size = 1
    seq_len = 5
    vocab_size = 100

    logits = torch.ones(batch_size, seq_len, vocab_size) * -100
    correct_token_idx = 42
    labels = torch.tensor([[correct_token_idx]])

    logits[0, -1, correct_token_idx] = 100

    ppl = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True)

    assert ppl < 2.0
    print("[PASS] test_perplexity_perfect_confidence")


def test_perplexity_two_token_span():
    """Test perplexity with 2-token answer span."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([10, 10])
    labels = torch.tensor([[42, 55], [10, 20]])
    answer_spans = [
        (8, 10),
        (7, 9),
    ]

    ppl = perplexity_span(logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True)

    assert ppl.shape == torch.Size([])
    assert ppl > 0
    print("[PASS] test_perplexity_two_token_span")


def test_perplexity_computation_correctness():
    """Verify perplexity formula: exp(-mean_log_prob)."""
    batch_size = 1
    seq_len = 1
    vocab_size = 10

    logits = torch.zeros(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[0]])

    ppl = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True)

    expected = torch.tensor(vocab_size, dtype=logits.dtype)
    assert torch.allclose(ppl, expected, rtol=1e-5)
    print("[PASS] test_perplexity_computation_correctness")


def test_perplexity_loss_correctness():
    """Verify loss is negative log probability."""
    batch_size = 1
    seq_len = 1
    vocab_size = 10

    logits = torch.zeros(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[0]])

    loss = perplexity_span(logits, None, torch.tensor([seq_len]), labels, mean=True, loss=True)

    expected_loss = torch.tensor(np.log(vocab_size), dtype=logits.dtype)
    assert torch.allclose(loss, expected_loss, rtol=1e-4)
    print("[PASS] test_perplexity_loss_correctness")


def test_perplexity_backward():
    """Test that perplexity metric supports backpropagation."""
    batch_size = 2
    seq_len = 5
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    input_lengths = torch.tensor([seq_len] * batch_size)
    labels = torch.randint(0, vocab_size, (batch_size, 1))

    ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)

    ppl.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    print("[PASS] test_perplexity_backward")


def test_perplexity_with_none_spans():
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
    print("[PASS] test_perplexity_with_none_spans")


def test_perplexity_boundary_clamping():
    """Test that out-of-bounds spans are clamped correctly."""
    batch_size = 1
    seq_len = 10
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([10])
    labels = torch.tensor([[42]])
    answer_spans = [(20, 30)]

    ppl = perplexity_span(logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True)

    assert ppl > 0
    print("[PASS] test_perplexity_boundary_clamping")


def test_perplexity_high_vocab_size():
    """Test with large vocabulary."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([seq_len] * batch_size)
    labels = torch.randint(0, vocab_size, (batch_size, 1))

    ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)
    assert ppl > 0 and not torch.isnan(ppl)
    print("[PASS] test_perplexity_high_vocab_size")


if __name__ == "__main__":
    print("Running perplexity metric tests...")
    print()

    test_perplexity_basic_single_token()
    test_perplexity_per_example()
    test_perplexity_loss_variant()
    test_perplexity_perfect_confidence()
    test_perplexity_two_token_span()
    test_perplexity_computation_correctness()
    test_perplexity_loss_correctness()
    test_perplexity_backward()
    test_perplexity_with_none_spans()
    test_perplexity_boundary_clamping()
    test_perplexity_high_vocab_size()

    print()
    print("All tests passed!")
