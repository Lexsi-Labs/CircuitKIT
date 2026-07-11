"""
Utility functions for IBCircuit training.
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def extract_logits_at_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Extract logits at specific sequence positions for each batch element.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        positions: Position indices [batch_size]
                  positions[i] indicates which token position to extract
                  from batch element i

    Returns:
        Extracted logits [batch_size, vocab_size]
    """
    batch_size = logits.shape[0]
    batch_idx = torch.arange(batch_size, device=logits.device)
    extracted = logits[batch_idx, positions]

    return extracted


def compute_baseline_reference(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    answer_positions: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Run the frozen model once to compute reference log-probabilities at answer positions.

    Used to establish the target distribution for KL-mode task loss. The IB training
    minimises divergence from this distribution while maximising sparsity.

    Args:
        model (HookedTransformer): Frozen base model (weights unchanged by IB training).
        input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
        answer_positions (torch.Tensor): Per-example positions at which to extract
            logits [batch_size]. Typically the last real token position.

    Returns:
        torch.Tensor: Log-softmax probabilities at answer positions [batch_size, vocab_size].
    """
    model.eval()

    with torch.no_grad():
        # Forward pass through frozen model
        outputs = model(input_ids)

        if isinstance(outputs, torch.Tensor):
            outputs = type("ModelOutput", (), {"logits": outputs})()

        # Extract logits at answer positions
        baseline_logits = extract_logits_at_positions(outputs.logits, answer_positions)

        # Convert to log probabilities for KL divergence
        baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    return baseline_logprobs


def compute_task_loss(
    ib_logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    baseline_logprobs: Optional[torch.Tensor],
    loss_mode: Literal["kl", "ce"],
    baseline_ce_loss: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute task preservation loss between IB model output and baseline.

    Two modes:
    - 'kl': "KL divergence of the baseline distribution from the IB distribution, i.e. KL(baseline || IB).
            Encourages matching the full output distribution, not just the top prediction.
    - 'ce': Absolute deviation of cross-entropy from the baseline scalar loss.
            Less stable when baseline loss is near zero.

    Args:
        ib_logits (torch.Tensor): IB model logits at answer positions [batch_size, vocab_size].
        answer_tokens (torch.Tensor): Ground-truth answer token IDs [batch_size].
            Only used in 'ce' mode.
        baseline_logprobs (torch.Tensor | None): Baseline log-probabilities
            [batch_size, vocab_size]. Required for 'kl' mode.
        loss_mode (str): Loss computation mode — 'kl' or 'ce'.
        baseline_ce_loss (float | None): Baseline cross-entropy scalar.
            Required for 'ce' mode.

    Returns:
        torch.Tensor: Scalar task loss.

    Raises:
        ValueError: If a required argument for the chosen mode is None, or if
            loss_mode is not 'kl' or 'ce'.
    """
    if loss_mode == "kl":
        if baseline_logprobs is None:
            raise ValueError("baseline_logprobs required for KL mode")

        ib_logprobs = F.log_softmax(ib_logits.float(), dim=-1)

        # Clamp log-probs to a sane finite range. F.kl_div with log_target=True
        # computes exp(target) * (target - input); if either side has -inf
        # the product becomes 0 * -inf = NaN. Clamping to [-50, 0] keeps the
        # KL well-defined for the rare-token tails that GPT-2 produces on
        # multi-choice tasks.
        ib_logprobs_c = ib_logprobs.clamp(min=-50.0)
        baseline_c = baseline_logprobs.float().clamp(min=-50.0)

        task_loss = F.kl_div(
            ib_logprobs_c,
            baseline_c,
            log_target=True,
            reduction="batchmean",
        )

    elif loss_mode == "ce":
        # Cross-Entropy mode (alternative approach)
        if baseline_ce_loss is None:
            raise ValueError("baseline_ce_loss required for CE mode")

        # Compute current cross-entropy loss
        current_ce_loss = F.cross_entropy(ib_logits, answer_tokens)

        # Penalize deviation from baseline task loss
        task_loss = torch.abs(current_ce_loss - baseline_ce_loss)

    else:
        raise ValueError(f"Invalid loss_mode: {loss_mode}. Must be 'kl' or 'ce'")

    return task_loss


def validate_batch_data(batch: dict, required_keys: list = None) -> None:
    """
    Validate IBCircuit batch structure and tensor shape consistency.

    Checks that all required keys are present. When using the default key set
    ('tokens', 'labels', 'answer_positions'), also validates that batch sizes
    are consistent and that all answer positions are valid sequence indices.

    Args:
        batch (dict): Batch dictionary from a DataLoader.
        required_keys (list | None): Keys that must be present in batch.
            Defaults to ['tokens', 'labels', 'answer_positions'].

    Raises:
        KeyError: If any required key is missing from batch.
        ValueError: If batch sizes are inconsistent across keys, or if any
            answer position is out of bounds for the token sequence length.
    """
    if required_keys is None:
        required_keys = ["tokens", "labels", "answer_positions"]

    # Check all required keys present
    missing_keys = [key for key in required_keys if key not in batch]
    if missing_keys:
        raise KeyError(
            f"Batch missing required keys: {missing_keys}. " f"Available keys: {list(batch.keys())}"
        )

    # Only perform detailed validation for default keys
    # Custom keys might have different structures that we can't validate
    if required_keys == ["tokens", "labels", "answer_positions"]:
        # Validate shapes are consistent
        batch_size = len(batch["tokens"])

        if len(batch["labels"]) != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: tokens={batch_size}, " f"labels={len(batch['labels'])}"
            )

        if len(batch["answer_positions"]) != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: tokens={batch_size}, "
                f"answer_positions={len(batch['answer_positions'])}"
            )

        # Validate answer_positions are within sequence bounds
        seq_len = batch["tokens"].shape[1]
        max_pos = batch["answer_positions"].max().item()

        if max_pos >= seq_len:
            raise ValueError(
                f"Invalid answer_positions: max position {max_pos} >= " f"sequence length {seq_len}"
            )
