"""
Healing Quality Metrics: Measure recovery of pruned model performance.

This module provides metrics for evaluating the effectiveness of soft healing
(fine-tuning with LoRA) in recovering performance of pruned circuits.

Key metrics:
- Recovery Rate: fraction of lost accuracy recovered
- Convergence: speed of LoRA adaptation
- Generalization: performance on held-out data
"""

import warnings
from circuitkit.utils.device import get_device, empty_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


@dataclass
class HealingMetrics:
    """Structured healing metrics report."""

    recovery_rate: float
    """Fraction of lost accuracy recovered by healing. Range [0, 1]."""

    original_accuracy: float
    """Baseline accuracy of unpruned model."""

    pruned_accuracy: float
    """Accuracy after pruning (before healing)."""

    healed_accuracy: float
    """Accuracy after healing with LoRA."""

    convergence_epoch: int
    """Epoch at which validation loss stabilized."""

    convergence_speed: float
    """Recovery speed: (healed_acc - pruned_acc) / epochs_to_convergence."""

    generalization_gap: float
    """Difference between validation and training loss at convergence."""

    lora_parameter_count: int
    """Total number of LoRA parameters."""

    model_parameter_count: int
    """Total model parameters."""

    efficiency_ratio: float
    """LoRA params / model params (percentage)."""

    def __post_init__(self):
        """Validate metrics."""
        if not (0 <= self.recovery_rate <= 1):
            warnings.warn(f"Recovery rate {self.recovery_rate} outside [0, 1]")

        if self.model_parameter_count > 0:
            self.efficiency_ratio = (self.lora_parameter_count / self.model_parameter_count) * 100

    def __repr__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 60,
            "HEALING METRICS REPORT",
            "=" * 60,
            f"Recovery Rate:        {self.recovery_rate:.2%}",
            f"Original Accuracy:    {self.original_accuracy:.4f}",
            f"Pruned Accuracy:      {self.pruned_accuracy:.4f}",
            f"Healed Accuracy:      {self.healed_accuracy:.4f}",
            f"Accuracy Delta:       {self.healed_accuracy - self.pruned_accuracy:+.4f}",
            "",
            f"Convergence Epoch:    {self.convergence_epoch}",
            f"Convergence Speed:    {self.convergence_speed:.4f} acc/epoch",
            f"Generalization Gap:   {self.generalization_gap:.4f}",
            "",
            f"LoRA Parameters:      {self.lora_parameter_count:,}",
            f"Model Parameters:     {self.model_parameter_count:,}",
            f"Efficiency Ratio:     {self.efficiency_ratio:.3f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class HealingEvaluator:
    """Evaluate healing effectiveness. Supports both multi-model and single-model (hook-based) modes."""

    def __init__(
        self,
        original_model: HookedTransformer,
        pruned_model: Optional[HookedTransformer] = None,
        healed_model: Optional[HookedTransformer] = None,
        pruning_hooks: Optional[List[Tuple[str, callable]]] = None,
        lora_hooks: Optional[List[Tuple[str, callable]]] = None,
    ):
        """
        Initialize evaluator.
        Pass pruned_model and healed_model for legacy multi-model evaluation.
        Pass pruning_hooks and lora_hooks for VRAM-efficient single-model evaluation.
        """
        self.model = original_model

        self.single_model_mode = pruned_model is None and healed_model is None

        if self.single_model_mode:
            self.pruning_hooks = pruning_hooks or []
            self.lora_hooks = lora_hooks or []
        else:
            self.pruned_model = pruned_model
            self.healed_model = healed_model

    def evaluate_accuracy(
        self, dataloader: DataLoader, device: str = "auto", return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate accuracy across states.

        Args:
            dataloader: Evaluation dataloader
            device: Target device
            return_logits: If True, returns the raw logits for custom metrics (e.g. MMLU logit diff)
                           instead of standard argmax accuracy.
        """
        results = {}

        if self.single_model_mode:
            states = [
                ("original", self.model, []),
                ("pruned", self.model, self.pruning_hooks),
                ("healed", self.model, self.pruning_hooks + self.lora_hooks),
            ]
        else:
            states = [
                ("original", self.model, None),
                ("pruned", self.pruned_model, None),
                ("healed", self.healed_model, None),
            ]

        for state_name, eval_model, hooks in states:
            eval_model.to(device)
            eval_model.eval()

            total_correct = 0
            total_samples = 0
            state_logits = []

            from contextlib import nullcontext


            context = eval_model.hooks(fwd_hooks=hooks) if hooks is not None else nullcontext()

            with context:
                with torch.no_grad():
                    for batch in dataloader:
                        if isinstance(batch, dict):
                            input_ids = batch.get("input_ids", batch.get("clean"))
                            labels = batch.get("labels", input_ids)
                        else:
                            input_ids = batch
                            labels = batch

                        input_ids = input_ids.to(device)
                        labels = labels.to(device)

                        logits = eval_model(input_ids)

                        if return_logits:
                            state_logits.append(logits.detach().cpu().to(torch.float16))

                        if len(logits.shape) == 3:
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()

                            preds = shift_logits.argmax(dim=-1)
                            valid_mask = shift_labels != -100

                            correct = ((preds == shift_labels) & valid_mask).sum().item()
                            valid_tokens = valid_mask.sum().item()
                        else:
                            preds = logits.argmax(dim=-1)
                            correct = (preds == labels).sum().item()
                            valid_tokens = labels.numel()

                        total_correct += correct
                        total_samples += valid_tokens

            if return_logits:
                results[state_name] = torch.cat(state_logits, dim=0)
            else:
                results[state_name] = total_correct / max(total_samples, 1)

        return results

    def compute_recovery_rate(self, accuracies: Dict[str, float]) -> float:
        """
        Compute recovery rate: how much of lost accuracy was recovered?

        Recovery Rate = (healed_acc - pruned_acc) / (original_acc - pruned_acc)

        Args:
            accuracies: Dict from evaluate_accuracy()

        Returns:
            Recovery rate in [0, 1]. 1.0 = full recovery, 0.0 = no recovery
        """
        orig = accuracies["original"]
        pruned = accuracies["pruned"]
        healed = accuracies["healed"]

        loss = orig - pruned
        if loss <= 0:
            return 0.0

        recovery = (healed - pruned) / loss
        return min(1.0, max(0.0, recovery))

    def compute_metrics(
        self,
        eval_dataloader: DataLoader,
        train_losses: List[float],
        val_losses: List[float],
        lora_param_count: int,
        device: str = "auto",
    ) -> HealingMetrics:
        """
        Compute comprehensive healing metrics.

        Args:
            eval_dataloader: Evaluation dataset
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            lora_param_count: Total LoRA parameters
            device: Device to use

        Returns:
            HealingMetrics dataclass
        """
        accuracies = self.evaluate_accuracy(eval_dataloader, device)
        recovery_rate = self.compute_recovery_rate(accuracies)

        best_epoch = torch.argmin(torch.tensor(val_losses)).item()

        # FIX: generalization gap is how much worse the model is on validation
        # vs. training (val - train). The original formula was inverted (train - val),
        # which is typically negative and thus always clamped to 0.
        generalization_gap = max(0.0, val_losses[best_epoch] - train_losses[best_epoch])

        # FIX: convergence speed should be accuracy recovered per epoch *until*
        # convergence, not per total training epoch. Use best_epoch + 1.
        convergence_speed = (accuracies["healed"] - accuracies["pruned"]) / max(best_epoch + 1, 1)

        # FIX: use self.model, which is always set regardless of mode.
        # self.healed_model is only set in multi-model mode and raises
        # AttributeError in single-model mode.
        model_params = sum(p.numel() for p in self.model.parameters())

        return HealingMetrics(
            recovery_rate=recovery_rate,
            original_accuracy=accuracies["original"],
            pruned_accuracy=accuracies["pruned"],
            healed_accuracy=accuracies["healed"],
            convergence_epoch=best_epoch,
            convergence_speed=convergence_speed,
            generalization_gap=generalization_gap,
            lora_parameter_count=lora_param_count,
            model_parameter_count=model_params,
            efficiency_ratio=0.0,  # Computed in __post_init__
        )


def compute_recovery_metrics(
    original_accuracy: float, pruned_accuracy: float, healed_accuracy: float
) -> Dict[str, float]:
    """
    Quick utility to compute recovery metrics from accuracies.

    Args:
        original_accuracy: Baseline accuracy
        pruned_accuracy: After pruning
        healed_accuracy: After healing

    Returns:
        Dict with 'recovery_rate', 'absolute_recovery', 'relative_recovery'
    """
    loss = original_accuracy - pruned_accuracy

    if loss <= 0:
        return {
            "recovery_rate": 0.0,
            "absolute_recovery": healed_accuracy - pruned_accuracy,
            "relative_recovery": 0.0,
        }

    recovery_rate = min(1.0, max(0.0, (healed_accuracy - pruned_accuracy) / loss))

    return {
        "recovery_rate": recovery_rate,
        "absolute_recovery": healed_accuracy - pruned_accuracy,
        "relative_recovery": (healed_accuracy - pruned_accuracy) / original_accuracy,
    }
