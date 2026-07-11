"""
Multi-token aware metrics for circuit discovery evaluation.

This module provides metric functions that can handle both single-token and
multi-token answers via answer spans.

Includes:
- logit_diff_span: Logit difference metric for classification
- kl_div_span: KL divergence metric comparing distributions
- accuracy_span: Accuracy metric for answer prediction
- perplexity_span: Perplexity metric for language modeling tasks
- perplexity_loss_span: Cross-entropy loss variant for optimization
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def logit_diff_span(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute logit difference metric over answer spans.

    Supports both single-token and multi-token answers. For multi-token answers,
    averages the logits over the answer span before computing the difference.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Label indices [batch_size, 2] where first is correct,
            rest are incorrect
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span. If None or contains
            None, uses the last token position for that example (backward compatible).
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Logit difference scores. Shape [batch_size] if not mean,
            scalar if mean=True.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Determine answer position for this example
        if answer_spans and answer_spans[i] is not None:
            # Multi-token answer: use provided span
            start, end = answer_spans[i]
            # Clamp to valid range
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            # Single-token answer: use last position (input_length - 1)
            # This is the standard behavior for single-token answers
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1

        # Get logits for answer span, averaging if multi-token
        span_logits = logits[i, start:end, :].mean(dim=0)  # [vocab_size]

        # Get correct and incorrect token indices
        correct_idx = labels[i, 0].item()
        incorrect_idx = labels[i, 1].item()

        # Compute difference
        correct_logit = span_logits[correct_idx]
        incorrect_logit = span_logits[incorrect_idx]
        diff = correct_logit - incorrect_logit

        results.append(diff)

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def kl_div_span(
    logits: Tensor,
    clean_logits: Tensor,
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute KL divergence metric over answer spans.

    For multi-token answers, averages the logits over the span before computing
    KL divergence.

    Args:
        logits (Tensor): Ablated model logits [batch_size, seq_len, vocab_size]
        clean_logits (Tensor): Clean model logits [batch_size, seq_len, vocab_size]
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Label indices [batch_size, 2] (unused for KL, kept for API)
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span.
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: KL divergence scores. Shape [batch_size] if not mean,
            scalar if mean=True.
    """
    import torch.nn.functional as F

    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Determine answer position for this example
        if answer_spans and answer_spans[i] is not None:
            start, end = answer_spans[i]
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1

        # Get logits for answer span, averaging if multi-token
        span_logits = logits[i, start:end, :].mean(dim=0)
        clean_span_logits = clean_logits[i, start:end, :].mean(dim=0)

        # Compute KL divergence: KL(clean || ablated)
        p = F.softmax(clean_span_logits, dim=-1)
        q = F.softmax(span_logits, dim=-1)
        kl = F.kl_div(q.log(), p, reduction="sum")

        results.append(kl)

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def accuracy_span(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute accuracy metric over answer spans.

    For multi-token answers, uses the average logits over the span to predict
    the correct answer.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Label indices [batch_size, 2] where first is correct
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span.
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Accuracy scores (0 or 1 per example). Shape [batch_size] if not mean,
            scalar if mean=True.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Determine answer position for this example
        if answer_spans and answer_spans[i] is not None:
            start, end = answer_spans[i]
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1

        # Get logits for answer span, averaging if multi-token
        span_logits = logits[i, start:end, :].mean(dim=0)

        # Get correct token index and check if it has max logit
        correct_idx = labels[i, 0].item()
        predicted_idx = span_logits.argmax().item()

        correct = 1.0 if predicted_idx == correct_idx else 0.0
        results.append(torch.tensor(correct, device=logits.device, dtype=logits.dtype))

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def perplexity_span(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
    loss: bool = False,
) -> Tensor:
    """
    Compute perplexity metric on answer span.

    Perplexity = exp(cross_entropy_loss) measures model confidence on a sequence.
    Lower perplexity = model more confident. Works for language modeling and ranking tasks.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of input [batch_size]
        labels (Tensor): Target token indices for the span [batch_size, span_len].
            For single-token answers (backward compat), shape is [batch_size, 1].
            For multi-token spans, contains all target tokens in order.
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span. If None or contains
            None, uses the last token position for that example.
        mean (bool): If True, return mean of batch. Otherwise return per-example.
        loss (bool): If True, return -log(prob) (cross-entropy) instead of perplexity.
            Useful for optimization since lower loss is better.

    Returns:
        Tensor: Perplexity scores (or loss if loss=True). Shape [batch_size] if not mean,
            scalar if mean=True.
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
        # Handle both single-token [batch_size, 1] and multi-token [batch_size, span_len]
        span_len = end - start
        if labels.shape[1] == 1:
            # Single-token case: just one label
            target_tokens = labels[i, :1]  # [1]
        else:
            # Multi-token case: get span_len tokens
            target_tokens = labels[i, :span_len]  # [span_len]

        # Compute cross-entropy loss over span
        # log_softmax + gather to get log prob of target token at each position
        log_probs = F.log_softmax(span_logits, dim=-1)  # [span_len, vocab_size]

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
            # Return cross-entropy loss (negative log prob)
            results.append(-mean_log_prob)
        else:
            # Return perplexity
            perplexity = torch.exp(-mean_log_prob)
            results.append(perplexity)

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def perplexity_loss_span(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute cross-entropy loss (negative log probability) for answer span.

    This is the optimization variant of perplexity_span. Returns -log(prob) directly,
    which is better for optimization since lower loss = higher circuit importance.

    Useful for tasks where you want to optimize the circuit to minimize loss
    rather than minimize perplexity.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of input [batch_size]
        labels (Tensor): Target token indices [batch_size, span_len]
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): Answer span positions
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Cross-entropy loss scores. Shape [batch_size] if not mean, scalar if mean=True.
    """
    return perplexity_span(
        logits, clean_logits, input_lengths, labels, answer_spans=answer_spans, mean=mean, loss=True
    )


# Backward-compatible wrappers for old API (without answer_spans)
def logit_diff_legacy(logits, clean_logits, input_lengths, labels, mean=True):
    """Legacy wrapper for logit_diff_span without answer_spans parameter."""
    return logit_diff_span(
        logits, clean_logits, input_lengths, labels, answer_spans=None, mean=mean
    )


def kl_div_legacy(logits, clean_logits, input_lengths, labels, mean=True):
    """Legacy wrapper for kl_div_span without answer_spans parameter."""
    return kl_div_span(logits, clean_logits, input_lengths, labels, answer_spans=None, mean=mean)


def accuracy_legacy(logits, clean_logits, input_lengths, labels, mean=True):
    """Legacy wrapper for accuracy_span without answer_spans parameter."""
    return accuracy_span(logits, clean_logits, input_lengths, labels, answer_spans=None, mean=mean)


def perplexity_legacy(logits, clean_logits, input_lengths, labels, mean=True, loss=False):
    """Legacy wrapper for perplexity_span without answer_spans parameter."""
    return perplexity_span(
        logits, clean_logits, input_lengths, labels, answer_spans=None, mean=mean, loss=loss
    )


def perplexity_loss_legacy(logits, clean_logits, input_lengths, labels, mean=True):
    """Legacy wrapper for perplexity_loss_span without answer_spans parameter."""
    return perplexity_loss_span(
        logits, clean_logits, input_lengths, labels, answer_spans=None, mean=mean
    )


def ranking_loss(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    margin: float = 1.0,
    mean: bool = True,
    loss: bool = True,
) -> Tensor:
    """
    Compute ranking loss for ranking tasks.

    Ranking loss measures whether correct options score higher than incorrect options.
    Loss = max(0, margin + score(incorrect) - score(correct))

    Useful for:
    - Multiple choice questions (MCQ): rank 4 options, pick the correct one
    - Retrieval: rank documents, pick the most relevant one
    - Comparison: rank 2 candidates, pick the better one

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Label indices [batch_size, 2] where:
            - labels[:, 0] = index of correct option
            - labels[:, 1] = index of incorrect option to compare against
            For ranking with multiple candidates, only the first "incorrect" is used
            (to get the hardest negative or a specific comparison).
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span. If None or contains
            None, uses the last token position for that example.
        margin (float): Margin between correct and incorrect scores. Defaults to 1.0.
        mean (bool): If True, return mean of batch. Otherwise return per-example.
        loss (bool): If True, return loss (for optimization). If False, return
            1.0 if correct ranks higher, 0.0 otherwise (ranking accuracy).

    Returns:
        Tensor: Ranking loss or accuracy. Shape [batch_size] if not mean,
            scalar if mean=True.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Determine answer position for this example
        if answer_spans and answer_spans[i] is not None:
            start, end = answer_spans[i]
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1

        # Get logits for answer span, averaging if multi-token
        span_logits = logits[i, start:end, :].mean(dim=0)  # [vocab_size]

        # Get correct and incorrect token indices
        correct_idx = labels[i, 0].item()
        incorrect_idx = labels[i, 1].item()

        # Compute scores (logits of the answer tokens)
        correct_score = span_logits[correct_idx]
        incorrect_score = span_logits[incorrect_idx]

        # Ranking loss: max(0, margin + score(incorrect) - score(correct))
        if loss:
            ranking_loss_val = torch.clamp(margin + incorrect_score - correct_score, min=0.0)
            results.append(ranking_loss_val)
        else:
            # Ranking accuracy: 1.0 if correct > incorrect, else 0.0
            correct = 1.0 if correct_score > incorrect_score else 0.0
            results.append(torch.tensor(correct, device=logits.device, dtype=logits.dtype))

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def ranking_accuracy(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute ranking accuracy for ranking tasks.

    Ranking accuracy measures whether the correct option scores higher than the
    incorrect option. Returns 1.0 if correct > incorrect, else 0.0.

    This is equivalent to ranking_loss with loss=False.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Label indices [batch_size, 2] where:
            - labels[:, 0] = index of correct option
            - labels[:, 1] = index of incorrect option to compare against
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): List of
            (start, end) tuples for each example's answer span.
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Ranking accuracy scores (0 or 1 per example). Shape [batch_size]
            if not mean, scalar if mean=True.
    """
    return ranking_loss(
        logits,
        clean_logits,
        input_lengths,
        labels,
        answer_spans=answer_spans,
        margin=1.0,
        mean=mean,
        loss=False,
    )


def recall_at_k(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[List[Optional[Tuple[int, int]]]]] = None,
    k: int = 1,
    mean: bool = True,
) -> Tensor:
    """
    Compute recall@k metric for ranking tasks with multiple candidates.

    Recall@k measures whether the correct answer is in the top-k ranked candidates.
    Useful for retrieval, search, and multi-candidate ranking tasks.

    Scores each candidate and returns 1.0 if the correct candidate is in top-k,
    else 0.0.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): [batch_size] = index of correct candidate (0 to num_candidates-1)
        answer_spans (Optional[List[List[Optional[Tuple[int, int]]]]]): List of lists
            of (start, end) tuples. For each example i, answer_spans[i] is a list of
            spans for each candidate j. Shape: [batch_size][num_candidates].
            If None, uses the last token position for all candidates.
        k (int): Number of top candidates to consider. Defaults to 1.
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Recall@k scores (0 or 1 per example). Shape [batch_size] if not mean,
            scalar if mean=True.

    Example:
        # 4-choice MCQ: question + 4 answer options
        # answer_spans[i] = [(start1, end1), (start2, end2), (start3, end3), (start4, end4)]
        # labels[i] = 2 (correct answer is option C, the 3rd option)
        # recall_at_k with k=2 checks if option C is in top-2 ranked options
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Get list of candidate spans for this example
        if answer_spans and answer_spans[i] is not None:
            candidate_spans = answer_spans[i]
        else:
            # Default: all candidates end at the last token
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            num_candidates = max(
                1, labels.max().item() + 1
            )  # infer number of candidates from max label
            candidate_spans = [(pos, pos + 1) for _ in range(num_candidates)]

        # Compute scores for all candidates
        # Each candidate's score is computed from its answer span
        candidate_scores = []
        for candidate_idx, span in enumerate(candidate_spans):
            start, end = span
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))

            # Get mean logit over span (average of all logits at that position)
            span_logits = logits[i, start:end, :].mean(dim=0)  # [vocab_size]
            # Score is the maximum logit value at this span position
            # This represents model confidence for this candidate
            score = span_logits.max().item()
            candidate_scores.append(score)

        # Get correct candidate index and its score
        correct_idx = labels[i].item()
        correct_score = candidate_scores[correct_idx]

        # Rank candidates by score (descending)
        # rank = 1 + number of candidates with strictly higher score
        scores_array = np.array(candidate_scores)
        num_better = np.sum(scores_array > correct_score)
        rank = num_better + 1

        # Recall@k: 1.0 if rank <= k, else 0.0
        in_top_k = 1.0 if rank <= k else 0.0
        results.append(torch.tensor(in_top_k, device=logits.device, dtype=logits.dtype))

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def span_f1(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
    loss: bool = False,
) -> Tensor:
    """
    Compute F1 score for predicted answer spans vs ground truth spans.

    F1 score measures token-level overlap between predicted and ground truth answer spans.
    Useful for reading comprehension tasks like SQuAD where the answer is a span of tokens.

    Predicted span is extracted using argmax on the logits:
    - Predicted start: argmax(logits[:, :, START_TOKEN_ID])
    - Predicted end: argmax(logits[:, start:, END_TOKEN_ID])
    Or, if provided via answer_spans, uses those directly.

    Token overlap:
    - Predicted span: [pred_start, pred_end)
    - Ground truth: [true_start, true_end)
    - Overlap = intersection length
    - Precision = overlap / predicted_length (or 0 if predicted_length == 0)
    - Recall = overlap / true_length (or 0 if true_length == 0)
    - F1 = 2 * (precision * recall) / (precision + recall) (or 0 if denominator == 0)

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Ground truth [batch_size, 2] where:
            - labels[:, 0] = true_start_idx
            - labels[:, 1] = true_end_idx (exclusive, so true span is [start, end))
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): Predicted answer spans.
            If None, uses argmax-based span prediction. If provided, uses these spans
            instead. Each element is (start_idx, end_idx) or None.
        mean (bool): If True, return mean F1 of batch. Otherwise return per-example.
        loss (bool): If True, return negative F1 (for optimization where lower is better).

    Returns:
        Tensor: F1 scores (0-1) or negative F1 (if loss=True). Shape [batch_size]
            if not mean, scalar if mean=True.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Get ground truth span
        true_start = labels[i, 0].item()
        true_end = labels[i, 1].item()

        # Clamp to valid range
        true_start = max(0, min(true_start, logits.shape[1] - 1))
        true_end = max(true_start, min(true_end, logits.shape[1]))

        # Get predicted span
        if answer_spans and answer_spans[i] is not None:
            # Use provided predicted span
            pred_start, pred_end = answer_spans[i]
        else:
            # Default: use last token as both start and end (single-token prediction)
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            pred_start, pred_end = pos, pos + 1

        # Clamp predicted span to valid range
        pred_start = max(0, min(pred_start, logits.shape[1] - 1))
        pred_end = max(pred_start, min(pred_end, logits.shape[1]))

        # Compute token-level F1
        # Overlap: intersection of [pred_start, pred_end) and [true_start, true_end)
        overlap_start = max(pred_start, true_start)
        overlap_end = min(pred_end, true_end)
        overlap_length = max(0, overlap_end - overlap_start)

        # Lengths
        pred_length = pred_end - pred_start
        true_length = true_end - true_start

        # Precision and recall
        precision = overlap_length / pred_length if pred_length > 0 else 0.0
        recall = overlap_length / true_length if true_length > 0 else 0.0

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Return loss or F1
        if loss:
            results.append(torch.tensor(-f1, device=logits.device, dtype=logits.dtype))
        else:
            results.append(torch.tensor(f1, device=logits.device, dtype=logits.dtype))

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def exact_match(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute exact match metric for answer spans.

    Exact match is a binary score: 1.0 if predicted span exactly matches ground truth,
    0.0 otherwise. Useful for evaluation alongside F1 score.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of each sequence [batch_size]
        labels (Tensor): Ground truth [batch_size, 2] where:
            - labels[:, 0] = true_start_idx
            - labels[:, 1] = true_end_idx (exclusive)
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): Predicted answer spans.
            If None, uses argmax-based span prediction. Each element is (start_idx, end_idx) or None.
        mean (bool): If True, return mean exact match of batch. Otherwise return per-example.

    Returns:
        Tensor: Exact match scores (0 or 1 per example). Shape [batch_size] if not mean,
            scalar if mean=True.
    """
    batch_size = logits.shape[0]
    results = []

    for i in range(batch_size):
        # Get ground truth span
        true_start = labels[i, 0].item()
        true_end = labels[i, 1].item()

        # Clamp to valid range
        true_start = max(0, min(true_start, logits.shape[1] - 1))
        true_end = max(true_start, min(true_end, logits.shape[1]))

        # Get predicted span
        if answer_spans and answer_spans[i] is not None:
            # Use provided predicted span
            pred_start, pred_end = answer_spans[i]
        else:
            # Default: use last token as both start and end
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            pred_start, pred_end = pos, pos + 1

        # Clamp predicted span to valid range
        pred_start = max(0, min(pred_start, logits.shape[1] - 1))
        pred_end = max(pred_start, min(pred_end, logits.shape[1]))

        # Check exact match
        match = 1.0 if (pred_start == true_start and pred_end == true_end) else 0.0
        results.append(torch.tensor(match, device=logits.device, dtype=logits.dtype))

    result_tensor = torch.stack(results)

    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


def span_f1_loss(
    logits: Tensor,
    clean_logits: Optional[Tensor],
    input_lengths: Tensor,
    labels: Tensor,
    answer_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
    mean: bool = True,
) -> Tensor:
    """
    Compute negative F1 score (loss variant) for answer span prediction.

    This is the optimization variant of span_f1. Returns -F1 directly,
    which is better for optimization since lower loss = higher circuit importance.

    Args:
        logits (Tensor): Model logits [batch_size, seq_len, vocab_size]
        clean_logits (Optional[Tensor]): Unused, kept for API compatibility
        input_lengths (Tensor): Length of input [batch_size]
        labels (Tensor): Ground truth [batch_size, 2]: [start_idx, end_idx]
        answer_spans (Optional[List[Optional[Tuple[int, int]]]]): Predicted spans
        mean (bool): If True, return mean of batch. Otherwise return per-example.

    Returns:
        Tensor: Negative F1 (loss) scores. Shape [batch_size] if not mean, scalar if mean=True.
    """
    return span_f1(
        logits, clean_logits, input_lengths, labels, answer_spans=answer_spans, mean=mean, loss=True
    )
