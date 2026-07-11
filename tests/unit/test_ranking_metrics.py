"""
Unit tests for ranking metrics.

Tests cover:
- Binary ranking (2 options): A vs B
- MCQ ranking (4 options): A, B, C, D
- Single-token and multi-token answer spans
- Ranking loss and accuracy
- Recall@k metric
- Edge cases: tied scores, partial credit
"""

import torch

from circuitkit.backends.eap.metrics import ranking_accuracy, ranking_loss, recall_at_k


class TestRankingLoss:
    """Test ranking loss metric for binary and multi-way comparisons."""

    def test_ranking_loss_basic_correct_higher(self):
        """Test ranking loss when correct option has higher logit than incorrect."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create logits where correct token (idx 10) has higher logit than incorrect (idx 20)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 10.0  # Correct token (idx 10) gets high logit
        logits[:, -1, 20] = 5.0  # Incorrect token (idx 20) gets lower logit

        input_lengths = torch.tensor([10, 10])
        labels = torch.tensor([[10, 20], [10, 20]])

        # Ranking loss should be 0 (no margin violation)
        loss = ranking_loss(logits, None, input_lengths, labels, margin=1.0, mean=False, loss=True)

        assert loss.shape == torch.Size([batch_size])
        assert torch.allclose(loss, torch.zeros(batch_size))

    def test_ranking_loss_basic_incorrect_higher(self):
        """Test ranking loss when incorrect option has higher logit than correct."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create logits where incorrect token (idx 20) has higher logit than correct (idx 10)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 5.0  # Correct token (idx 10) gets lower logit
        logits[:, -1, 20] = 10.0  # Incorrect token (idx 20) gets high logit

        input_lengths = torch.tensor([10, 10])
        labels = torch.tensor([[10, 20], [10, 20]])

        # Ranking loss = max(0, margin + score(incorrect) - score(correct))
        #              = max(0, 1.0 + 10.0 - 5.0) = 6.0
        loss = ranking_loss(logits, None, input_lengths, labels, margin=1.0, mean=False, loss=True)

        assert loss.shape == torch.Size([batch_size])
        assert torch.allclose(loss, torch.tensor([6.0, 6.0]))

    def test_ranking_loss_with_margin(self):
        """Test ranking loss respects margin parameter."""
        batch_size = 1
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 5.0  # Correct
        logits[:, -1, 20] = 7.0  # Incorrect

        input_lengths = torch.tensor([5])
        labels = torch.tensor([[10, 20]])

        # Test with different margins
        loss_m1 = ranking_loss(
            logits, None, input_lengths, labels, margin=1.0, mean=False, loss=True
        )
        loss_m2 = ranking_loss(
            logits, None, input_lengths, labels, margin=2.0, mean=False, loss=True
        )

        # loss_m1 = max(0, 1.0 + 7.0 - 5.0) = 3.0
        # loss_m2 = max(0, 2.0 + 7.0 - 5.0) = 4.0
        assert torch.allclose(loss_m1, torch.tensor([3.0]))
        assert torch.allclose(loss_m2, torch.tensor([4.0]))

    def test_ranking_loss_per_example(self):
        """Test ranking loss returns per-example scores when mean=False."""
        batch_size = 3
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[0, -1, 10] = 10.0
        logits[0, -1, 20] = 5.0
        logits[1, -1, 15] = 3.0
        logits[1, -1, 25] = 8.0
        logits[2, -1, 30] = 7.0
        logits[2, -1, 40] = 7.0

        input_lengths = torch.tensor([8, 8, 8])
        labels = torch.tensor([[10, 20], [15, 25], [30, 40]])

        loss = ranking_loss(logits, None, input_lengths, labels, margin=1.0, mean=False, loss=True)

        assert loss.shape == torch.Size([batch_size])
        # loss[0] = max(0, 1.0 + 5.0 - 10.0) = 0.0
        # loss[1] = max(0, 1.0 + 8.0 - 3.0) = 6.0
        # loss[2] = max(0, 1.0 + 7.0 - 7.0) = 1.0
        assert torch.allclose(loss, torch.tensor([0.0, 6.0, 1.0]))

    def test_ranking_loss_with_answer_spans(self):
        """Test ranking loss with multi-token answer spans."""
        batch_size = 2
        seq_len = 20
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Set logits for multi-token span [5:8] and [10:12]
        logits[0, 5:8, 10] = torch.tensor([10.0, 11.0, 9.0])
        logits[0, 5:8, 20] = torch.tensor([5.0, 6.0, 4.0])
        logits[1, 10:12, 15] = torch.tensor([2.0, 3.0])
        logits[1, 10:12, 25] = torch.tensor([4.0, 5.0])

        input_lengths = torch.tensor([20, 20])
        labels = torch.tensor([[10, 20], [15, 25]])
        # answer_spans is a list of spans, one per example
        answer_spans = [(5, 8), (10, 12)]

        loss = ranking_loss(
            logits,
            None,
            input_lengths,
            labels,
            answer_spans=answer_spans,
            margin=1.0,
            mean=False,
            loss=True,
        )

        # For example 0: mean logit correct = (10+11+9)/3 = 10.0
        #                mean logit incorrect = (5+6+4)/3 = 5.0
        #                loss = max(0, 1.0 + 5.0 - 10.0) = 0.0
        # For example 1: mean logit correct = (2+3)/2 = 2.5
        #                mean logit incorrect = (4+5)/2 = 4.5
        #                loss = max(0, 1.0 + 4.5 - 2.5) = 3.0
        assert torch.allclose(loss, torch.tensor([0.0, 3.0]), atol=1e-5)


class TestRankingAccuracy:
    """Test ranking accuracy metric."""

    def test_ranking_accuracy_correct_higher(self):
        """Test ranking accuracy when correct option scores higher."""
        batch_size = 3
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 10.0  # Correct
        logits[:, -1, 20] = 5.0  # Incorrect

        input_lengths = torch.tensor([10, 10, 10])
        labels = torch.tensor([[10, 20], [10, 20], [10, 20]])

        acc = ranking_accuracy(logits, None, input_lengths, labels, mean=False)

        assert acc.shape == torch.Size([batch_size])
        assert torch.allclose(acc, torch.ones(batch_size))

    def test_ranking_accuracy_incorrect_higher(self):
        """Test ranking accuracy when incorrect option scores higher."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 5.0  # Correct
        logits[:, -1, 20] = 10.0  # Incorrect

        input_lengths = torch.tensor([10, 10])
        labels = torch.tensor([[10, 20], [10, 20]])

        acc = ranking_accuracy(logits, None, input_lengths, labels, mean=False)

        assert acc.shape == torch.Size([batch_size])
        assert torch.allclose(acc, torch.zeros(batch_size))

    def test_ranking_accuracy_mean(self):
        """Test ranking accuracy returns mean when mean=True."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[0, -1, 10] = 10.0  # Correct
        logits[0, -1, 20] = 5.0
        logits[1, -1, 15] = 10.0  # Correct
        logits[1, -1, 25] = 5.0
        logits[2, -1, 30] = 5.0  # Incorrect
        logits[2, -1, 40] = 10.0
        logits[3, -1, 35] = 5.0  # Incorrect
        logits[3, -1, 45] = 10.0

        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.tensor([[10, 20], [15, 25], [30, 40], [35, 45]])

        acc = ranking_accuracy(logits, None, input_lengths, labels, mean=True)

        # 2 correct, 2 incorrect -> mean = 0.5
        assert acc.shape == torch.Size([])
        assert torch.allclose(acc, torch.tensor(0.5))


class TestRecallAtK:
    """Test recall@k metric for multi-candidate ranking."""

    def test_recall_at_1_correct_ranked_first(self):
        """Test recall@1 when correct candidate is ranked first."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Candidate 0 (correct): max logit = 10.0
        # Candidate 1: max logit = 5.0
        # Candidate 2: max logit = 3.0
        # Candidate 3: max logit = 2.0
        logits[:, -1, 0] = 10.0
        logits[:, -1, 1] = 5.0
        logits[:, -1, 2] = 3.0
        logits[:, -1, 3] = 2.0

        input_lengths = torch.tensor([10, 10])
        labels = torch.tensor([0, 0])  # Correct candidate is 0

        # answer_spans: 4 candidates, each at last position
        answer_spans = [
            [(9, 10), (9, 10), (9, 10), (9, 10)],
            [(9, 10), (9, 10), (9, 10), (9, 10)],
        ]

        recall = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=False
        )

        # Candidate 0 is ranked first -> recall@1 = 1.0
        assert recall.shape == torch.Size([batch_size])
        assert torch.allclose(recall, torch.ones(batch_size))

    def test_recall_at_k_correct_in_top_k(self):
        """Test recall@k when correct candidate is in top-k but not first."""
        batch_size = 1
        seq_len = 15
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Each candidate at a different position with different max logits
        # Candidate 0: max logit = 5.0 (at pos 5)
        # Candidate 1: max logit = 10.0 (at pos 7, ranked first)
        # Candidate 2: max logit = 8.0  (at pos 9, ranked second - correct answer)
        # Candidate 3: max logit = 3.0  (at pos 11, ranked last)
        logits[0, 5, 0] = 5.0
        logits[0, 7, 1] = 10.0
        logits[0, 9, 2] = 8.0
        logits[0, 11, 3] = 3.0

        input_lengths = torch.tensor([15])
        labels = torch.tensor([2])  # Correct candidate is 2 (ranked 2nd)

        # Each candidate has its own span/position
        answer_spans = [[(5, 6), (7, 8), (9, 10), (11, 12)]]

        # recall@1: correct is ranked 2nd -> 0.0
        recall_1 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=False
        )
        assert torch.allclose(recall_1, torch.tensor([0.0]))

        # recall@2: correct is in top-2 -> 1.0
        recall_2 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=2, mean=False
        )
        assert torch.allclose(recall_2, torch.tensor([1.0]))

        # recall@3: correct is in top-3 -> 1.0
        recall_3 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=3, mean=False
        )
        assert torch.allclose(recall_3, torch.tensor([1.0]))

    def test_recall_at_k_batch(self):
        """Test recall@k with batch of examples."""
        batch_size = 3
        seq_len = 15
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Example 0: candidate 0 (correct) is ranked 1st
        logits[0, 3, 0] = 10.0  # Candidate 0 (correct) - max logit 10
        logits[0, 5, 1] = 5.0  # Candidate 1 - max logit 5
        logits[0, 7, 2] = 3.0  # Candidate 2 - max logit 3

        # Example 1: candidate 1 (correct) is ranked 2nd
        logits[1, 3, 0] = 10.0  # Candidate 0 - max logit 10 (ranked 1st)
        logits[1, 5, 1] = 8.0  # Candidate 1 (correct) - max logit 8 (ranked 2nd)
        logits[1, 7, 2] = 5.0  # Candidate 2 - max logit 5

        # Example 2: candidate 2 (correct) is ranked 3rd
        logits[2, 3, 0] = 10.0  # Candidate 0 - max logit 10 (ranked 1st)
        logits[2, 5, 1] = 8.0  # Candidate 1 - max logit 8 (ranked 2nd)
        logits[2, 7, 2] = 6.0  # Candidate 2 (correct) - max logit 6 (ranked 3rd)

        input_lengths = torch.tensor([15, 15, 15])
        labels = torch.tensor([0, 1, 2])

        answer_spans = [
            [(3, 4), (5, 6), (7, 8)],
            [(3, 4), (5, 6), (7, 8)],
            [(3, 4), (5, 6), (7, 8)],
        ]

        # recall@1: [1.0, 0.0, 0.0]
        recall_1 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=False
        )
        assert torch.allclose(recall_1, torch.tensor([1.0, 0.0, 0.0]))

        # recall@2: [1.0, 1.0, 0.0]
        recall_2 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=2, mean=False
        )
        assert torch.allclose(recall_2, torch.tensor([1.0, 1.0, 0.0]))

        # recall@3: [1.0, 1.0, 1.0]
        recall_3 = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=3, mean=False
        )
        assert torch.allclose(recall_3, torch.tensor([1.0, 1.0, 1.0]))

    def test_recall_at_k_with_mean(self):
        """Test recall@k returns mean when mean=True."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Ex 0: candidate 0 (correct) ranked first
        logits[0, 3, 0] = 10.0
        logits[0, 5, 1] = 5.0
        # Ex 1: candidate 0 (correct) ranked first
        logits[1, 3, 0] = 10.0
        logits[1, 5, 1] = 5.0
        # Ex 2: candidate 1 (correct) ranked second
        logits[2, 3, 0] = 10.0
        logits[2, 5, 1] = 5.0
        # Ex 3: candidate 1 (correct) ranked second
        logits[3, 3, 0] = 10.0
        logits[3, 5, 1] = 5.0

        input_lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.tensor([0, 0, 1, 1])

        answer_spans = [
            [(3, 4), (5, 6)],
            [(3, 4), (5, 6)],
            [(3, 4), (5, 6)],
            [(3, 4), (5, 6)],
        ]

        recall = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=True
        )

        # Correct is ranked first: [1.0, 1.0, 0.0, 0.0] -> mean = 0.5
        assert recall.shape == torch.Size([])
        assert torch.allclose(recall, torch.tensor(0.5))

    def test_recall_at_k_with_multi_token_spans(self):
        """Test recall@k with multi-token answer spans."""
        batch_size = 1
        seq_len = 20
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Candidate 0 (correct): mean logit = (10 + 11 + 9) / 3 = 10
        # Candidate 1: mean logit = (5 + 6 + 4) / 3 = 5
        logits[0, 5:8, 0] = torch.tensor([10.0, 11.0, 9.0])
        logits[0, 5:8, 1] = torch.tensor([5.0, 6.0, 4.0])
        logits[0, 10:12, 0] = torch.tensor([2.0, 3.0])
        logits[0, 10:12, 1] = torch.tensor([8.0, 7.0])

        input_lengths = torch.tensor([20])
        labels = torch.tensor([0])

        answer_spans = [[(5, 8), (5, 8)]]

        recall = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=False
        )

        # Candidate 0 max logit = 11.0, Candidate 1 max logit = 6.0
        # Candidate 0 is ranked first -> recall@1 = 1.0
        assert torch.allclose(recall, torch.tensor([1.0]))

    def test_recall_at_k_tied_scores(self):
        """Test recall@k with tied candidate scores."""
        batch_size = 1
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # All candidates have the same max logit
        logits[:, -1, 0] = 10.0
        logits[:, -1, 1] = 10.0
        logits[:, -1, 2] = 10.0

        input_lengths = torch.tensor([10])
        labels = torch.tensor([1])  # Correct is candidate 1

        answer_spans = [[(9, 10), (9, 10), (9, 10)]]

        # With ties, candidate 1 is not ranked strictly first, but tied
        # rank = 1 + number of candidates with strictly higher score
        # Since all tied, rank should be 1 (no one strictly higher)
        recall = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=1, mean=False
        )

        # Should be 1.0 since rank = 1
        assert torch.allclose(recall, torch.tensor([1.0]))


class TestRankingEdgeCases:
    """Test edge cases and special scenarios."""

    def test_ranking_loss_zero_margin(self):
        """Test ranking loss with zero margin."""
        batch_size = 1
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 5.0
        logits[:, -1, 20] = 5.0  # Same logit

        input_lengths = torch.tensor([5])
        labels = torch.tensor([[10, 20]])

        loss = ranking_loss(logits, None, input_lengths, labels, margin=0.0, mean=False, loss=True)

        # loss = max(0, 0.0 + 5.0 - 5.0) = 0.0
        assert torch.allclose(loss, torch.tensor([0.0]))

    def test_ranking_metrics_with_variable_lengths(self):
        """Test ranking metrics with variable sequence lengths."""
        batch_size = 2
        seq_len = 15
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[0, 4, 10] = 10.0  # Shorter sequence (length 5)
        logits[0, 4, 20] = 5.0
        logits[1, 9, 10] = 10.0  # Longer sequence (length 10)
        logits[1, 9, 20] = 5.0

        input_lengths = torch.tensor([5, 10])
        labels = torch.tensor([[10, 20], [10, 20]])

        loss = ranking_loss(logits, None, input_lengths, labels, mean=False, loss=True)

        assert loss.shape == torch.Size([batch_size])
        assert torch.allclose(loss, torch.zeros(batch_size))

    def test_ranking_accuracy_single_example(self):
        """Test ranking accuracy with single example."""
        batch_size = 1
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 10] = 10.0
        logits[:, -1, 20] = 5.0

        input_lengths = torch.tensor([5])
        labels = torch.tensor([[10, 20]])

        acc = ranking_accuracy(logits, None, input_lengths, labels, mean=True)

        assert acc.shape == torch.Size([])
        assert torch.allclose(acc, torch.tensor(1.0))

    def test_recall_at_k_k_larger_than_candidates(self):
        """Test recall@k when k is larger than number of candidates."""
        batch_size = 1
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 0] = 5.0
        logits[:, -1, 1] = 10.0

        input_lengths = torch.tensor([10])
        labels = torch.tensor([0])

        answer_spans = [[(9, 10), (9, 10)]]

        # k=10 but only 2 candidates
        recall = recall_at_k(
            logits, None, input_lengths, labels, answer_spans=answer_spans, k=10, mean=False
        )

        # Correct candidate is ranked 2nd, k=10 includes all -> 1.0
        assert torch.allclose(recall, torch.tensor([1.0]))


class TestRankingBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_ranking_loss_equiv_to_ranking_accuracy(self):
        """Test that ranking_loss with loss=False is equivalent to ranking_accuracy."""
        batch_size = 5
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        for i in range(batch_size):
            logits[i, -1, i * 10] = torch.randn(1).item() + 10.0
            logits[i, -1, i * 10 + 5] = torch.randn(1).item()

        input_lengths = torch.tensor([8] * batch_size)
        labels = torch.tensor([[i * 10, i * 10 + 5] for i in range(batch_size)])

        loss_as_acc = ranking_loss(logits, None, input_lengths, labels, loss=False, mean=False)
        acc = ranking_accuracy(logits, None, input_lengths, labels, mean=False)

        assert torch.allclose(loss_as_acc, acc)

    def test_metrics_return_correct_dtype(self):
        """Test that metrics return same dtype as input logits."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        for dtype in [torch.float32, torch.float64]:
            logits = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype)
            logits[:, -1, 10] = 10.0
            logits[:, -1, 20] = 5.0

            input_lengths = torch.tensor([10, 10])
            labels = torch.tensor([[10, 20], [10, 20]])

            loss = ranking_loss(logits, None, input_lengths, labels, mean=False, loss=True)
            acc = ranking_accuracy(logits, None, input_lengths, labels, mean=False)

            assert loss.dtype == dtype
            assert acc.dtype == dtype
