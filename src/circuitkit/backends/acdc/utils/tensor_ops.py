import math

import torch as t

from ..types import PromptPairBatch, PruneScores

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
left, right, temp = -0.1, 1.1, 2 / 3


def sample_hard_concrete(mask: t.Tensor, batch_size: int, mask_expanded: bool = False) -> t.Tensor:
    """
    Sample from the hard concrete distribution.

    Args:
        mask: The mask whose values parameterize the distribution.
        batch_size: The number of samples to draw.
        mask_expanded: Whether the mask has a batch dimension at the start.

    Returns:
        A sample for each element in the mask for each batch element. The returned
        tensor has shape `(batch_size, *mask.shape)`.
    """
    if not mask_expanded:
        mask = mask.repeat(batch_size, *([1] * mask.ndim))
    else:
        assert mask.size(0) == batch_size
    u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
    s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
    s_bar = s * (right - left) + left
    return s_bar.clamp(min=0.0, max=1.0)


def indices_vals(vals: t.Tensor, indices: t.Tensor) -> t.Tensor:
    assert vals.ndim == indices.ndim
    return t.gather(vals, dim=-1, index=indices)


def vocab_avg_val(vals: t.Tensor, indices: t.Tensor) -> t.Tensor:
    return indices_vals(vals, indices).mean()


def batch_avg_answer_val(
    vals: t.Tensor, batch: PromptPairBatch, wrong_answer: bool = False
) -> t.Tensor:
    """Get the average value of the answer logits for the batch."""
    answers = batch.answers if not wrong_answer else batch.wrong_answers
    if isinstance(answers, t.Tensor):
        return vocab_avg_val(vals, answers)
    else:
        assert isinstance(answers, list)
        return t.stack([vocab_avg_val(v, a) for v, a in zip(vals, answers)]).mean()


def batch_answer_diffs(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """Difference between correct and wrong answer logits for each prompt."""
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        ans_avgs = t.gather(vals, dim=-1, index=answers).mean(dim=-1)
        wrong_avgs = t.gather(vals, dim=-1, index=wrong_answers).mean(dim=-1)
        return ans_avgs - wrong_avgs
    else:
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        ans_avgs = [vocab_avg_val(v, a) for v, a in zip(vals, answers)]
        wrong_avgs = [vocab_avg_val(v, w) for v, w in zip(vals, wrong_answers)]
        return t.stack(ans_avgs) - t.stack(wrong_avgs)


def batch_avg_answer_diff(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """Mean of `batch_answer_diffs` over the batch."""
    return batch_answer_diffs(vals, batch).mean()


def multibatch_kl_div(input_logprobs: t.Tensor, target_logprobs: t.Tensor) -> t.Tensor:
    """
    Compute the average KL divergence between two sets of log probabilities.
    Assumes the last dimension is the log probability of each class.
    """
    assert input_logprobs.shape == target_logprobs.shape
    kl_div_sum = t.nn.functional.kl_div(
        input_logprobs,
        target_logprobs,
        reduction="sum",
        log_target=True,
    )
    n_batch = math.prod(input_logprobs.shape[:-1])
    return kl_div_sum / n_batch


def flat_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    """Flatten the prune scores into a single, 1-dimensional tensor."""
    return t.cat([ps.flatten() for _, ps in prune_scores.items()])


def desc_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    """Flatten prune scores and sort them in descending order."""
    return flat_prune_scores(prune_scores).abs().sort(descending=True).values


def prune_scores_threshold(prune_scores: PruneScores | t.Tensor, edge_count: int) -> t.Tensor:
    """
    Return the minimum absolute value of the top `edge_count` prune scores.
    """
    if edge_count == 0:
        return t.tensor(float("inf"))

    if isinstance(prune_scores, t.Tensor):
        assert prune_scores.ndim == 1
        # Handle case where edge_count is out of bounds
        if edge_count > len(prune_scores):
            return t.tensor(0.0)
        return prune_scores[edge_count - 1]
    else:
        desc_scores = desc_prune_scores(prune_scores)
        if edge_count > len(desc_scores):
            return t.tensor(0.0)
        return desc_scores[edge_count - 1]
